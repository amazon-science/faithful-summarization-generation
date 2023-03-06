import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class BERTScoreScorer:
    """
    Scorer using BS-Fact, code adapted from bertscore official repo: https://github.com/Tiiiger/bert_score
    """
    def __init__(self, model_name="roberta-large", device="cuda", num_layers=17, cache_dir=".cache"):
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        # We assume we are using roberta-large, please reference https://github.com/Tiiiger/bert_score/blob/dbcf6db37e8bd6ff68446f06b0ba5d0763b62d20/bert_score/utils.py#L247
        # if you wish to use other model and select the recommended layer
        model.encoder.layer = torch.nn.ModuleList([layer for layer in model.encoder.layer[:num_layers]])

        self.model = model.to(device)
        self.device = device
    
    def prepare_document(self, input_str):
        """
        Prepare anything that requires processing on document.
        This is called each iteration only once to save computation.
        """
        self.bertscore_input_embedding, self.bertscore_input_attention_mask, self.bertscore_input_idf = self.encode_text(input_str)

    def score(self, summaries, index):
        """
        Output the score for each example.
        summaries: The summary strings
        index: The indice of example (document that it should be compared to). IT should ideally be just range() except for beam search.
        """
        bertscore_output_embedding, bertscore_output_attention_mask, bertscore_output_idf = self.encode_text(summaries)

        bertscore_input_embedding = self.bertscore_input_embedding[index]
        bertscore_input_attention_mask = self.bertscore_input_attention_mask[index]
        bertscore_input_idf = self.bertscore_input_idf[index]

        bertscore_scores = self.compute_bertscore(
            bertscore_input_embedding,
            bertscore_input_attention_mask,
            bertscore_input_idf,
            bertscore_output_embedding,
            bertscore_output_attention_mask,
            bertscore_output_idf,
        )
        return bertscore_scores

    def encode_text(self, input_str):
        """
        Helper function to encode any string to tensor using the tokenizer
        """
        inputs = self.tokenizer(input_str, padding=True, truncation=True, return_tensors="pt")
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # idf
        idf = torch.clone(inputs["attention_mask"]).float()
        idf[idf == self.tokenizer.sep_token_id] = 0
        idf[idf == self.tokenizer.cls_token_id] = 0
        idf.div_(idf.sum(dim=1, keepdim=True))

        return F.normalize(outputs[0], dim=-1), inputs["attention_mask"], idf

    def compute_bertscore(self, doc_embedding, doc_masks, doc_idf, summ_embedding, summ_masks, summ_idf):
        """
        Helper function that is modified from the official code (greedy_cos_idf() method) https://github.com/Tiiiger/bert_score/blob/dbcf6db37e8bd6ff68446f06b0ba5d0763b62d20/bert_score/utils.py#L469
        """
        
        batch_size = doc_embedding.size(0)
        sim = torch.bmm(summ_embedding, doc_embedding.transpose(1, 2))
        masks = torch.bmm(summ_masks.unsqueeze(2).float(), doc_masks.unsqueeze(1).float())
        masks = masks.expand(batch_size, -1, -1).contiguous().view_as(sim)

        masks = masks.float().to(sim.device)
        sim = sim * masks

        precision = sim.max(dim=2)[0]
        precision_scale = summ_idf.to(precision.device)
        P = (precision * precision_scale).sum(dim=1)
        
        summ_zero_mask = summ_masks.sum(dim=1).eq(2)
        if torch.any(summ_zero_mask):
            P = P.masked_fill(summ_zero_mask, 0.0)

        doc_zero_mask = doc_masks.sum(dim=1).eq(2)
        if torch.any(doc_zero_mask):
            P = P.masked_fill(doc_zero_mask, 0.0)
        
        return P