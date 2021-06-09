from transformers import FSMTTokenizer, FSMTForConditionalGeneration, MarianTokenizer, MarianMTModel
import torch
from typing import List
from .gradient_model import GradientModel 

class TranslationModel:

    def __init__(self, back_mapping=True):
        translation_name = 'facebook/wmt19-de-en'
        self.trans_tokenizer = FSMTTokenizer.from_pretrained(translation_name)
        self.trans_model = FSMTForConditionalGeneration.from_pretrained(translation_name)
        self.trans_model.eval()
        if back_mapping is True:
            self.mapping_model = GradientModel()
            self.backward = self._backward
        
    def _get_word_pos(self, sent, pos):
        """
        Returns the word positions for the given character ranges
        """
        char_pos = 0
        word_pos = [list() for _ in pos]
        for j, word in enumerate(sent.split()):
            for i, (l, r) in enumerate(pos):
                if l <= char_pos < r or l <= char_pos + len(word) < r or (char_pos <= l and r <= char_pos + len(word)):
                    word_pos[i].append(j)
            char_pos += len(word) + 1
        assert all([len(elem) != 0 for elem in word_pos])
        return word_pos

    def _get_word_pos_batch(self, sentences, positions):
        return [self._get_word_pos(sent, pos) for sent, pos in zip(sentences, positions)]

    def _get_token2word(self, seq):
        """
        Generates the token to word mapping
        """
        mapping = [word_id for i, elem in enumerate(seq.split()) for word_id in [i for _ in self.mapping_model.tokenizer.tokenize(elem)]]
        return mapping

    def _get_token2word_batch(self, sequences):
        return [self._get_token2word(seq) for seq in sequences]

    def _get_word2token(self, seq):
        """
        Generates the word to token mapping
        """
        num_tokens = [len(self.mapping_model.tokenizer.tokenize(elem)) for elem in seq.split()]
        num_gen = iter(range(sum(num_tokens)))
        return [[next(num_gen) for _ in range(i)] for i in num_tokens]
    
    def _get_word2token_batch(self, sequences):
        return [self._get_word2token(seq) for seq in sequences]

    def translate(self, sentence, max_length=512):
        inp = self.trans_tokenizer(sentence, padding=True, return_tensors='pt')
        if inp.input_ids.shape[-1] > 512:
            # TODO what to do here?
            raise NotImplemented()
        inp = {k : v.to(self.trans_model.device) for k, v in inp.items()}
        out = self.trans_model.generate(**inp, max_length=max_length)
        out_seq = self.trans_tokenizer.batch_decode(out, skip_special_tokens=True)
        return out_seq
        
    def _backward(self, sentences: List[str], trans_sentences: List[str], char_positions: List[List[List[int]]]):
        tokens = self.mapping_model.tokenizer(sentences, padding=True, return_tensors='pt')
        trans_tokens = self.mapping_model.tokenizer(trans_sentences, padding=True, return_tensors='pt')
        if tokens.input_ids.shape[-1] > 512 or trans_tokens.input_ids.shape[-1] > 512:
            # TODO what to do here?
            raise NotImplementedError

        word2token = self._get_word2token_batch(trans_sentences)
        token2word = self._get_token2word_batch(sentences)
        word_pos = self._get_word_pos_batch(trans_sentences, char_positions)
        pred_word_pos = self.mapping_model(word_pos, token2word, word2token, *tokens.values(), *trans_tokens.values())
        
        return [list(elem) for elem in pred_word_pos]

    def to(self, device):
        """
        Move models to given device
        """
        self.trans_model.to(device)
        self.mapping_model.to(device)
