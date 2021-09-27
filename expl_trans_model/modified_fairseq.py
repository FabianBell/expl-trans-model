import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import FSMTForConditionalGeneration
import torch

class DecoderNoCacheWrapper(torch.nn.Module):
    """
    Wrapper Module that deactivates the cache for the given decoder.
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        # the following is requested by the upper layers
        self.embed_tokens = self.decoder.embed_tokens
    
    def forward(self, *args, use_cache=None, **kwargs):
        return self.decoder(*args, use_cache=False, **kwargs)

def make_label(tokens):
    """
    Transforms given tokens to labels
    """
    labels = torch.empty_like(tokens)
    labels[:, 1:] = tokens[:, :-1]
    labels[:, 0] = tokens[:, -1]
    mask = (labels != 1).int()
    return labels, mask

class Fairseq(torch.nn.Module):
    """
    A modified Fairseq (FSMT) model that properly computes the attention scores.
    """

    def __init__(self, name):
        super().__init__()
        self.model = FSMTForConditionalGeneration.from_pretrained(name)
        self.model.model.decoder = DecoderNoCacheWrapper(self.model.model.decoder)
        
    def forward(self, input_ids, labels, *args, use_cache=None, decoder_input_ids=None, decoder_attention_mask=None, output_attentions=None, **kwargs):
        labels, label_mask = make_label(labels)
        return self.model(input_ids=input_ids, *args, use_cache=False, decoder_input_ids=labels, decoder_attention_mask=label_mask, output_attentions=True, **kwargs)
    
    def generate(self, *args, **kwargs):
        return self.model(*args, **kwargs)
