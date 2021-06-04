#from modified_fairseq import FirstDerivativeGradientFairseq 
from transformers import FSMTTokenizer
from transformers import MarianMTModel, MarianTokenizer
import torch

class GradientModel:

    def __init__(self):
        trans_model_name = 'Helsinki-NLP/opus-mt-de-en'
        self.tokenizer = MarianTokenizer.from_pretrained(trans_model_name)
        self.model = MarianMTModel.from_pretrained(trans_model_name, return_dict=True)
        self.embeddings_layer = self.model.model.encoder.embed_tokens
        self.scale = self.model.model.encoder.embed_scale
        self.model.eval()

    def to(self, device):
        """
        Move model to device
        """
        self.model.to(device)
        self.embeddings_layer.to(device)

    def _map_tokens(self, inp_ids, inp_mask, out_ids, out_mask, token_positions):
        """
        Maps the tokens by finding the token with the 
        largest first-derivative saliency
        """
        embeddings = self.embeddings_layer(inp_ids) * self.scale
        embeddings = embeddings.detach()  # drop graph 
        embeddings.requires_grad = True
       
        labels = out_ids.clone()
        labels[out_mask == 0] = -100
        logits = self.model(inputs_embeds=embeddings, attention_mask=inp_mask, labels=out_ids).logits
        graph = logits.sum(dim=-1)
       
        eos_pos = inp_mask.ne(0).min(-1)[1] - 1

        out_pos = []
        for i, token_pos in enumerate(token_positions):
            out_pos.append([])
            for pos in token_pos:
                embeddings.grad = None  # set gradients to zero
                graph[i, pos].backward(retain_graph=True)
                # ignore eos token embeddings
                grad = embeddings.grad[0, :eos_pos[i], :]
                pred_pos = grad.pow(2).sum(dim=-1).sqrt().argmax(-1).item()
                out_pos[-1].append(pred_pos)
        return out_pos
    
    def __call__(self, word_positions, token_word_mapping, word_token_mapping, *args):
        token_ids, token_mask, trans_token_ids, trans_token_mask = [elem.to(self.model.device) for elem in args]
        token_pos = []
        for i, row in enumerate(word_positions):
            token_pos.append([])
            for word_pos in row:
                for pos in word_token_mapping[i][word_pos]:
                    token_pos[-1].append(pos)
        
        mapped = self._map_tokens(token_ids, token_mask, trans_token_ids, trans_token_mask, token_pos)
        words = [set([token_word_mapping[i][pos] for pos in token_pos]) for i, token_pos in enumerate(mapped)]
        return words
