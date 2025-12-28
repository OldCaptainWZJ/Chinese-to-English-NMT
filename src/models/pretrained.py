import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM

class mT5Wrapper(nn.Module):
    def __init__(self, model_name='mT5-base', device='cpu'):
        super().__init__()
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        self.device = device
        
    def forward(self, src, src_len, tgt, tgt_len, teacher_forcing_ratio=None):
        # mT5 forward expects: input_ids, attention_mask, labels
        # src: [batch, src_len]
        # tgt: [batch, tgt_len] (contains sos/eos usually, but mT5 handles labels internally shifting)

        labels = tgt.clone()
        labels[labels == 0] = -100 
        
        outputs = self.model(
            input_ids=src,
            attention_mask=(src != 0).long(),
            labels=labels
        )
        
        return outputs.logits, outputs.loss

    def decode(self, src, src_len, tgt=None, method='greedy', max_len=512):
        # Inference using generate()
        outputs = self.model.generate(
            input_ids=src,
            max_length=max_len,
            num_beams=1 if method=='greedy' else 3,
            early_stopping=True
        )
        # Output is [1, seq_len]
        return outputs[0].cpu().tolist()

def build_pretrained_model(model_name, device):
    return mT5Wrapper(model_name, device)