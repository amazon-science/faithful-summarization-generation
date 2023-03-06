import inspect
from sys import prefix
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

import copy

from transformers.generation_beam_constraints import Constraint, DisjunctiveConstraint, PhrasalConstraint
from transformers.generation_beam_search import BeamScorer, BeamSearchScorer, ConstrainedBeamSearchScorer
from transformers.generation_logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    NoBadWordsLogitsProcessor,
    NoRepeatNGramLogitsProcessor,
    PrefixConstrainedLogitsProcessor,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    TypicalLogitsWarper,
)
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.pytorch_utils import torch_int_div
from transformers.utils import ModelOutput, logging

from transformers.generation_utils import (
    GreedySearchEncoderDecoderOutput,
    GreedySearchDecoderOnlyOutput,
    BeamSearchEncoderDecoderOutput,
    BeamSearchDecoderOnlyOutput,
    SampleEncoderDecoderOutput,
    SampleDecoderOnlyOutput,
)

logger = logging.get_logger(__name__)

class Lookahead:
    """
    Object that performs the lookahead. This is very similar to GenerationMixin, since it needs to decode the sequence as well,
    but this contains the additional function to compute heuristics score.
    """

    def __init__(
        self,
        model,
        tokenizer,
        scorer,
        lookahead_length=1,
        lookahead_lambda=1.0,
        lookahead_top_k=5,
        decoding_type="greedy",
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        early_stopping: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        typical_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        bad_words_ids: Optional[Iterable[int]] = None,
        force_words_ids: Optional[Union[Iterable[int], Iterable[Iterable[int]]]] = None,
        bos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        length_penalty: Optional[float] = None,
        no_repeat_ngram_size: Optional[int] = None,
        encoder_no_repeat_ngram_size: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        decoder_start_token_id: Optional[int] = None,
        use_cache: Optional[bool] = None,
        num_beam_groups: Optional[int] = None,
        diversity_penalty: Optional[float] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
        renormalize_logits: Optional[bool] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = StoppingCriteriaList(),
        constraints: Optional[List[Constraint]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        synced_gpus: Optional[bool] = False,
        exponential_decay_length_penalty: Optional[Tuple[Union[int, float]]] = None,
    ):
        """
        model: The Huggingface Model
        tokenizer: The tokenizer for decoding the summaries
        scorer: Scorer object that calculates the score given document and summary
        lookahead_length: The number of tokens to look ahead
        lookahead_lambda: The weight for the score
        lookahead_top_k: The number of top tokens to consider for expansion
        decoding_type: The decoding type for lookahead. [greedy, beam, sample]

        Other parameters are the same arguments expected for GenerationMixin to control the generation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.scorer = scorer

        if lookahead_length == -1:
            assert max_length is not None
            self.lookahead_length = max_length
            self.lookahead_until_sent = True
        else:
            self.lookahead_length = lookahead_length
            self.lookahead_until_sent = False
        
        self.lookahead_lambda = lookahead_lambda
        self.lookahead_top_k = lookahead_top_k
        self.decoding_type = decoding_type

        if self.decoding_type == "greedy":
            self.decoding_func = self.greedy_search
        elif self.decoding_type == "beam":
            self.decoding_func = self.beam_search
        elif self.decoding_type == "sample":
            self.decoding_func = self.sample

        # generation parameters from generate()
        self.bos_token_id = self.model.config.bos_token_id
        self.num_beams = num_beams if num_beams is not None else self.model.config.num_beams
        self.length_penalty = length_penalty if length_penalty is not None else self.model.config.length_penalty
        self.early_stopping = early_stopping if early_stopping is not None else self.model.config.early_stopping
        self.num_beam_groups = num_beam_groups if num_beam_groups is not None else self.model.config.num_beam_groups
        self.num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.model.config.num_return_sequences
        )

        self.pad_token_id = self.model.config.pad_token_id
        self.eos_token_id = self.model.config.eos_token_id

        if self.eos_token_id is None and hasattr(self.model.config, "decoder"):
            self.eos_token_id = self.model.config.decoder.eos_token_id

        if self.pad_token_id is None and self.eos_token_id is not None:
            # special case if pad_token_id is not defined
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{self.eos_token_id} for open-end generation.")
            self.pad_token_id = self.eos_token_id
        self.max_length =  max_length
        self.min_length = min_length
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.typical_p = typical_p
        self.reptition_penality = repetition_penalty
        self.bad_words_ids = bad_words_ids
        self.force_words_ids = force_words_ids
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.encoder_no_repeat_ngram_size = encoder_no_repeat_ngram_size
        self.max_new_tokens = max_new_tokens
        self.decoder_start_token_id = decoder_start_token_id
        self.use_cache = use_cache
        self.diversity_penalty = diversity_penalty
        self.prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self.renormalize_logits = renormalize_logits
        self.contraints = constraints
        self.forced_bos_token_id = forced_bos_token_id
        self.forced_eos_token_id = forced_eos_token_id
        self.remove_invalid_values = remove_invalid_values
        self.exponential_decay_length_penalty = exponential_decay_length_penalty
        self.synced_gpus = synced_gpus

        # self.return_dict_in_generate = return_dict_in_generate
        self.return_dict_in_generate = True
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.output_scores = output_scores

        # If not provided, logits processor will be prepared later since it requires input_tensor
        self.logits_processor = logits_processor

        # prepare stopping criteria
        self.stopping_criteria = self.model._get_stopping_criteria(
            max_length=max_length, max_time=max_time, stopping_criteria=stopping_criteria
        )

        self.logits_warper = self.model._get_logits_warper(
            top_k=self.top_k,
            top_p=self.top_p,
            typical_p=self.typical_p,
            temperature=self.temperature,
            num_beams=self.num_beams,
            renormalize_logits=self.renormalize_logits,
        )

    
    def score(
        self,
        input_ids,
        next_token_scores,
        num_beams=1,
        **model_kwargs,
    ):
        """
        Main function to call for the lookahead. This function generates the sequences and return the calculated heurstics
        """

        # prepare for generation
        if self.logits_processor is None:
            input_ids_seq_length = input_ids.size(1)
            inputs_tensor =  model_kwargs["encoder_outputs"][self.model.main_input_name]

            self.logits_processor = self.model._get_logits_processor(
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                encoder_no_repeat_ngram_size=self.encoder_no_repeat_ngram_size,
                input_ids_seq_length=input_ids_seq_length,
                encoder_input_ids=inputs_tensor,
                bad_words_ids=self.bad_words_ids,
                min_length=self.min_length,
                max_length=self.max_length,
                eos_token_id=self.eos_token_id,
                forced_bos_token_id=self.forced_bos_token_id,
                forced_eos_token_id=self.forced_eos_token_id,
                prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
                num_beams=self.num_beams,
                num_beam_groups=self.num_beam_groups,
                diversity_penalty=self.diversity_penalty,
                remove_invalid_values=self.remove_invalid_values,
                exponential_decay_length_penalty=self.exponential_decay_length_penalty,
                logits_processor=self.logits_processor,
                renormalize_logits=self.renormalize_logits,
            )
        
        do_sample = "sample" in self.decoding_type
        use_beam = "beam" in self.decoding_type
        beam_scorer = None

        if use_beam:
            batch_size = input_ids.shape[0] * self.lookahead_top_k
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=self.num_beams,
                max_length=self.stopping_criteria.max_length,
                device=input_ids.device,
                length_penalty=self.length_penalty,
                do_early_stopping=self.early_stopping,
                num_beam_hyps_to_keep=self.num_return_sequences,
                num_beam_groups=self.num_beam_groups,
            )
        
        indices = torch.arange(input_ids.size(0), dtype=input_ids.dtype, device=input_ids.device)
        
        # expand for top k tokens to use with scorer
        _, top_k_indices = torch.topk(next_token_scores, k=self.lookahead_top_k, dim=-1)
        top_k_indices = top_k_indices.reshape(-1)

        indices = indices.repeat_interleave(self.lookahead_top_k)
        input_ids = torch.cat([input_ids[indices],top_k_indices.unsqueeze(1)], dim=1)

        # adjust model_kwargs
        model_kwargs = self.expand_model_kwargs(model_kwargs, indices)
        
        # expand if necssary for beam, currently ignoring sampling with multiple num sequences
        if use_beam:
            input_ids, model_kwargs = self.model._expand_inputs_for_generation(
                input_ids,
                expand_size=self.num_beams,
                is_encoder_decoder=self.model.config.is_encoder_decoder,
                **model_kwargs,
            )
            indices = indices.repeat_interleave(self.num_beams)
            # exapand inputs for generation but does not expand past
            if "past" in model_kwargs:
                model_kwargs["past"] = tuple([tuple([p.repeat_interleave(self.num_beams, dim=0) for p in past]) for past in model_kwargs["past"]])
        
        # calling the respective decoding function
        # the only difference between this implementation and the original is the addition of lookahead length and breaking once that is reached
        if self.lookahead_length == 0:
            seq = input_ids
        else:
            dec_out = self.decoding_func(input_ids, beam_scorer, **model_kwargs)
            seq = dec_out["sequences"]

        # generate the actual summary
        dec_seq = self.tokenizer.batch_decode(seq, skip_special_tokens=True)
        
        # calculate score given the heuristics, need to account for different indices when doing beam search
        _lookahead_scores = self.scorer.score(dec_seq, torch.div(indices, num_beams, rounding_mode="trunc"))
        _lookahead_scores = torch.clamp(_lookahead_scores,min=1e-9).log()

        _lookahead_scores = _lookahead_scores.view(-1,  self.lookahead_top_k, self.num_beams)
        _lookahead_scores, _ = _lookahead_scores.max(-1)

        lookahead_scores = torch.ones_like(next_token_scores, dtype=_lookahead_scores.dtype, device=next_token_scores.device) * 1e-9
        lookahead_scores = lookahead_scores.log()
        
        next_token_scores = F.log_softmax(next_token_scores, dim=-1)

        if use_beam:
            # remove repat interleave for beams
            indices = indices.view(-1,self.num_beams)[:,0]
        
        lookahead_scores[indices, top_k_indices] = _lookahead_scores.view(-1)

        return self.lookahead_lambda * lookahead_scores
    
    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer = None,
        **model_kwargs,
    ):  
        # init attention / hidden states / scores tuples
        scores = () if (self.return_dict_in_generate and self.output_scores) else None
        decoder_attentions = () if (self.return_dict_in_generate and self.output_attentions) else None
        cross_attentions = () if (self.return_dict_in_generate and self.output_attentions) else None
        decoder_hidden_states = () if (self.return_dict_in_generate and self.output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if self.return_dict_in_generate and self.model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if self.output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if self.output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]
       
        lookahead_length = self.lookahead_length + cur_len

        this_peer_finished = False  # used by synced_gpus only
        while True:
            
            if self.synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_attentions=self.output_attentions,
                output_hidden_states=self.output_hidden_states,
            )

            if self.synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # Store scores, attentions and hidden_states when required
            if self.return_dict_in_generate:
                if self.output_scores:
                    scores += (next_token_logits,)
                if self.output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.model.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.model.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if self.output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.model.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # pre-process distribution
            next_tokens_scores = self.logits_processor(input_ids, next_token_logits)

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if self.eos_token_id is not None:
                if self.pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + self.pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self.model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder
            )
            cur_len = cur_len + 1
            
            # Lookahead break
            if cur_len >= lookahead_length:
                break

            # if eos_token was found in one sentence, set sentence to finished
            if self.eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != self.eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or self.stopping_criteria(input_ids, scores):
                if not self.synced_gpus:
                    break
                else:
                    this_peer_finished = True

        if self.return_dict_in_generate:
            if self.model.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids
        
    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer = None,
        **model_kwargs,
    ):  
        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        lookahead_length = self.lookahead_length + cur_len

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (self.return_dict_in_generate and self.output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (self.return_dict_in_generate and self.output_scores) else None
        )
        decoder_attentions = () if (self.return_dict_in_generate and self.output_attentions) else None
        cross_attentions = () if (self.return_dict_in_generate and self.output_attentions) else None
        decoder_hidden_states = () if (self.return_dict_in_generate and self.output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if self.return_dict_in_generate and self.model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if self.output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if self.output_hidden_states else None
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:

            if self.synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_attentions=self.output_attentions,
                output_hidden_states=self.output_hidden_states,
            )

            if self.synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.model.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = self.logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if self.return_dict_in_generate:
                if self.output_scores:
                    scores += (next_token_scores_processed,)
                if self.output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.model.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.model.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if self.output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.model.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self.model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self.model._reorder_cache(model_kwargs["past"], beam_idx)

            if self.return_dict_in_generate and self.output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if cur_len >= lookahead_length:
                break

            if beam_scorer.is_done or self.stopping_criteria(input_ids, scores):
                if not self.synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            max_length=self.stopping_criteria.max_length,
        )

        if self.return_dict_in_generate:
            if not self.output_scores:
                sequence_outputs["sequence_scores"] = None
            else:
                num_return_sequences = beam_scorer.num_beam_hyps_to_keep
                # return only as many indices as sequences
                beam_indices = tuple(
                    (beam_indices[i * num_beams : i * num_beams + num_return_sequences] for i in range(batch_size))
                )
                beam_indices = sum(beam_indices, ())

            if self.model.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=beam_indices,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=beam_indices,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]
    
    def sample(
        self,
        input_ids: torch.LongTensor,
        beam_scorer = None,
        **model_kwargs,
    ):
        scores = () if (self.return_dict_in_generate and self.output_scores) else None
        decoder_attentions = () if (self.return_dict_in_generate and self.output_attentions) else None
        cross_attentions = () if (self.return_dict_in_generate and self.output_attentions) else None
        decoder_hidden_states = () if (self.return_dict_in_generate and self.output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if self.return_dict_in_generate and self.model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if self.output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if self.output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

        lookahead_length = self.lookahead_length + cur_len
        
        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        while True:

            if self.synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self.model(
                **model_inputs,
                return_dict=True,
                output_attentions=self.output_attentions,
                output_hidden_states=self.output_hidden_states,
            )

            if self.synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = self.logits_processor(input_ids, next_token_logits)
            next_token_scores = self.logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if self.return_dict_in_generate:
                if self.output_scores:
                    scores += (next_token_scores,)
                if self.output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.model.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.model.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if self.output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.model.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if self.eos_token_id is not None:
                if self.pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + self.pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self.model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.model.config.is_encoder_decoder
            )
            cur_len = cur_len + 1

            if cur_len >= lookahead_length:
                break

            # if eos_token was found in one sentence, set sentence to finished
            if self.eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != self.eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or self.stopping_criteria(input_ids, scores):
                if not self.synced_gpus:
                    break
                else:
                    this_peer_finished = True

        if self.return_dict_in_generate:
            if self.model.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids

    
    def expand_model_kwargs(self, model_kwargs, indices):
        model_kwargs = copy.deepcopy(model_kwargs)
        if "attention_mask" in model_kwargs:
            model_kwargs["attention_mask"] = model_kwargs["attention_mask"][indices]
        if "encoder_outputs" in model_kwargs:
            for k,v in model_kwargs["encoder_outputs"].items():
                if v is not None:
                    model_kwargs["encoder_outputs"][k] = v[indices]
        if "past" in model_kwargs:
            model_kwargs["past"] = tuple([tuple([p[indices] for p in past]) for past in model_kwargs["past"]])
        return model_kwargs