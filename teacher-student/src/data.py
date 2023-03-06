import random
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

from transformers import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

@dataclass
class DataCollatorForSeq2SeqWithMultipleReferences:
    """
    This is similar to DataCollatorForSeq2Seq except that it also accounts for additional output summaries.

    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*
            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        additional_labels = [feature["additional_labels"] for feature in features] if "additional_labels" in features[0].keys() else None
        additional_candidates = [feature["additional_candidates"] for feature in features] if "additional_candidates" in features[0].keys() else None

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
        
        if additional_labels is not None:
            max_label_length = max(len(l) for l in additional_labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["additional_labels"]))
                if isinstance(feature["additional_labels"], list):
                    feature["additional_labels"] = (
                        feature["additional_labels"] + remainder if padding_side == "right" else remainder + feature["additional_labels"]
                    )
                elif padding_side == "right":
                    feature["additional_labels"] = np.concatenate([feature["additional_labels"], remainder]).astype(np.int64)
                else:
                    feature["additional_labels"] = np.concatenate([remainder, feature["additional_labels"]]).astype(np.int64)

        if additional_candidates is not None:
            max_label_length = max(max([len(l) for l in ll]) for ll in additional_candidates)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side

            for feature in features:
                padded_feature = []
                for feat in feature["additional_candidates"]:
                    remainder = [self.label_pad_token_id] * (max_label_length - len(feat))
                    if isinstance(feat, list):
                        _feat = (
                            feat + remainder if padding_side == "right" else remainder + feat
                        )
                    elif padding_side == "right":
                        _feat = np.concatenate([feat, remainder]).astype(np.int64)
                    else:
                        _feat = np.concatenate([remainder, feat]).astype(np.int64)
                    padded_feature.append(_feat)
                feature["additional_candidates"] = padded_feature
        
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
        
        if (
            additional_labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["additional_labels"])
            features["additional_decoder_input_ids"] = decoder_input_ids

        # additional candidates
        if additional_candidates is not None:
            additional_candidates = features.pop("additional_candidates")
            num_candidates = additional_candidates.size(1)

            # additional_candidates.masked_fill_(additional_candidates == -100, self.tokenizer.pad_token_id)
            # features["candidates_decoder_input_ids"] = additional_candidates
            
            # no need to shift
            additional_candidates = additional_candidates.view(-1, additional_candidates.size(-1))
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=additional_candidates)
            decoder_input_ids = decoder_input_ids.view(-1, num_candidates, decoder_input_ids.size(-1))
            features["candidates_decoder_input_ids"] = decoder_input_ids

        return features