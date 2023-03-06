import torch
from torch import nn
import torch.nn.functional as F
from transformers import Seq2SeqTrainer


class CustomTrainer(Seq2SeqTrainer):
    """
    Custom trainer for multiple Cross Entropy Loss. Adapted from original huggingface trainer code.
    """
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        alpha = 1.0,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        additional_decoder_input_ids = inputs.pop("additional_decoder_input_ids", None)
        additional_labels = inputs.pop("additional_labels", None)

        # first get encoder outputs to save computation
        encoder_outputs = model.get_encoder()(
            input_ids = inputs["input_ids"],
            attention_mask = inputs["attention_mask"]
        )
        inputs["encoder_outputs"] = encoder_outputs

        # Cross Entropy Loss

        # original XE
        orig_loss = super().compute_loss(model, inputs, return_outputs)
        loss = orig_loss
        
        # additional labels
        if additional_labels is not None:
            # compute loss for labels and additional_labels separaetly

            inputs["decoder_input_ids"] = additional_decoder_input_ids
            inputs["labels"] = additional_labels
            additional_loss = super().compute_loss(model, inputs, return_outputs)
            
            loss += self.alpha * additional_loss
        
        return loss