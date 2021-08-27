from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List
import re
from langdetect import detect
from .modified_fairseq import Fairseq

class TranslationModel:

    def __init__(self, back_mapping=True, translation_name='facebook/wmt19-de-en'):
        if translation_name.startswith('facebook/wmt19'):
            self.config = 'fairseq'
            self.trans_tokenizer = AutoTokenizer.from_pretrained(translation_name)
        elif translation_name.startswith('facebook/m2m100_418M') and re.fullmatch('facebook/m2m100_418M-[a-zA-Z]{2}', translation_name):
            self.config = 'm2m100_418M'
            translation_name, trg_code = translation_name.split('-')
            self.trans_tokenizer = AutoTokenizer.from_pretrained(translation_name)
            self.trg_code = self.trans_tokenizer.get_lang_id(trg_code) 
        else:
            raise Exception(f'Model {translation_name} not supported')
        self.trans_model = AutoModelForSeq2SeqLM.from_pretrained(translation_name)
        self.trans_model.eval()
        self.back_mapping = back_mapping
        if back_mapping is True:
            if self.config == 'fairseq':
                self.mapping_model = Fairseq(translation_name)
            else:
                raise Exception(f'Backward path is not supported for model {translation_name}')
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
        if self.config == 'm2m100_418M':
            src_lang = detect(sentence[0])
            if '-' in src_lang:
                src_lang = src_lang.split('-')[0]
            self.trans_tokenizer.src_lang = src_lang 
        inp = self.trans_tokenizer(sentence, padding=True, return_tensors='pt')
        if inp.input_ids.shape[-1] > 512:
            # TODO what to do here?
            raise NotImplemented()
        inp = {k : v.to(self.trans_model.device) for k, v in inp.items()}
        if self.config == 'm2m100_418M':
            out = self.trans_model.generate(**inp, max_length=max_length, forced_bos_token_id=self.trg_code)
        else:
            out = self.trans_model.generate(**inp, max_length=max_length)
        out_seq = self.trans_tokenizer.batch_decode(out, skip_special_tokens=True)
        return out_seq
    
    def _get_char_sequences(self, sentences, positions):
        word_ranges = []
        for sentence in sentences:
            word_ranges.append([])
            pos = 0
            for word in sentence.split():
                word_ranges[-1].append((pos, pos+len(word)))
                pos += len(word) + 1
        results = []
        for i, batch in enumerate(positions):
            results.append([])
            for match in batch:
                results[-1].append([])
                for word_pos in match:
                    results[-1][-1].append(word_ranges[i][word_pos])
        return results
        

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
        
        words = [list(elem) for elem in pred_word_pos]
        char_seq = self._get_char_sequences(sentences, words)
        return char_seq
        

    def to(self, device):
        """
        Move models to given device
        """
        self.trans_model.to(device)
        if self.back_mapping is True:
            self.mapping_model.to(device)
