from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Tuple
import re
from langdetect import detect
from .modified_fairseq import Fairseq

class TranslationModel:

    def __init__(self, back_mapping=True, translation_name='facebook/wmt19-de-en', word_level=True):
        if translation_name.startswith('facebook/wmt19'):
            self.config = 'wmt19'
            self.tokenizer = AutoTokenizer.from_pretrained(translation_name)
        elif translation_name.startswith('facebook/m2m100_418M') and re.fullmatch('facebook/m2m100_418M-[a-zA-Z]{2}', translation_name):
            self.config = 'm2m100_418M'
            translation_name, trg_code_name = translation_name.split('-')
            self.tokenizer = AutoTokenizer.from_pretrained(translation_name)
            self.trg_code_name = trg_code_name
            self.trg_code = self.tokenizer.get_lang_id(trg_code_name) 
        else:
            raise Exception(f'Model {translation_name} not supported')
        self.trans_model = AutoModelForSeq2SeqLM.from_pretrained(translation_name)
        self.trans_model.eval()
        self.back_mapping = back_mapping
        if back_mapping is True:
            self.word_level = word_level
            if self.config == 'wmt19':
                self.mapping_model = Fairseq(translation_name)
            elif self.config == 'm2m100_418M':
                self.mapping_model = self.trans_model
            else:
                raise Exception(f'Backward path is not supported for model {translation_name}')
            self.backward = self._backward
        
    def _get_word_pos(self, sent, pos) -> List[List[int]]:
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

    def _get_word_pos_batch(self, sentences, positions) -> List[List[List[int]]]:
        return [self._get_word_pos(sent, pos) for sent, pos in zip(sentences, positions)]

    def _get_token2word(self, seq) -> List[int]:
        """
        Generates the token to word mapping
        """
        mapping = [word_id for i, elem in enumerate(seq.split()) for word_id in [i for _ in self.tokenizer.tokenize(elem)]]
        return mapping

    def _get_token2word_batch(self, sequences) -> List[List[int]]:
        return [self._get_token2word(seq) for seq in sequences]

    def _get_word2token(self, seq : str) -> List[List[int]]:
        """
        Generates the word to token mapping
        """
        num_tokens = [len(self.tokenizer.tokenize(elem)) for elem in seq.split()]
        num_gen = iter(range(sum(num_tokens)))
        return [[next(num_gen) for _ in range(i)] for i in num_tokens]
    
    def _get_word2token_batch(self, sequences) -> List[List[List[int]]]:
        return [self._get_word2token(seq) for seq in sequences]

    def translate(self, sentence, max_length=512):
        if self.config == 'm2m100_418M':
            src_lang = detect(sentence[0])
            if '-' in src_lang:
                src_lang = src_lang.split('-')[0]
            self.tokenizer.src_lang = src_lang 
        inp = self.tokenizer(sentence, padding=True, return_tensors='pt')
        if inp.input_ids.shape[-1] > 512:
            # TODO what to do here?
            raise NotImplemented()
        inp = {k : v.to(self.trans_model.device) for k, v in inp.items()}
        if self.config == 'm2m100_418M':
            out = self.trans_model.generate(**inp, max_length=max_length, forced_bos_token_id=self.trg_code)
        else:
            out = self.trans_model.generate(**inp, max_length=max_length)
        out_seq = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        return out_seq
    
    def _get_char_sequences(self, sentences : List[str], positions : List[List[List[int]]]) -> List[List[List[Tuple[int, int]]]]:
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
    
    def _get_token_pos(self, trans_tokens : List[int], trans_sentence : str, char_positions : List[Tuple[int, int]]) -> List[List[int]]:
        upper_char_pos = [len(self.tokenizer.decode(trans_tokens[:i], skip_special_tokens=True)) for i in range(1, len(trans_tokens))]
        char_ranges = [(upper_char_pos[i], upper_char_pos[i+1]) for i in range(len(upper_char_pos) - 1)]
        token_pos = []
        for l, r in char_positions:
            token_pos.append([])
            for pos, (t, k) in enumerate(char_ranges):
                if t <= l < k or t < r <= k or (l <= t and k <= r):
                    token_pos[-1].append(pos)
                if r <= t:
                    break
        return token_pos

    def _get_char_sequences_from_tokens(self, tokens : List[int], pred : List[List[int]]) -> List[List[Tuple[int, int]]]:
        results : List[List[Tuple[int, int]]] = []
        for entry in pred:
            results.append([])
            for pos in entry:
                l = len(self.tokenizer.decode(tokens[:pos], skip_special_tokens=True))
                r = len(self.tokenizer.decode(tokens[:pos+1], skip_special_tokens=True))
                results[-1].append((l, r))
        return results
                

    def _get_char_sequences_from_tokens_batch(self, token_batch : List[List[int]], pred : List[List[List[int]]]) -> List[List[List[Tuple[int, int]]]]:
        return [self._get_char_sequences_from_tokens(tokens, pred_entry) for tokens, pred_entry in zip(token_batch, pred)]
        
    
    def _get_token_pos_batch(self, trans_token_batch : List[List[int]], trans_sentences : List[str], char_positions : List[List[Tuple[int, int]]]) -> List[List[List[int]]]:
        return [self._get_token_pos(trans_tokens, trans_sentence, char_position) for trans_tokens, trans_sentence, char_position in zip(trans_token_batch, trans_sentences, char_positions)]

    def _backward(self, sentences: List[str], trans_sentences: List[str], char_positions: List[List[List[int]]]) -> List[List[List[Tuple[int, int]]]]:
        if self.config == 'm2m100_418M':
            src_lang = detect(sentences[0])
            if '-' in src_lang:
                src_lang = src_lang.split('-')[0]
            self.tokenizer.src_lang = src_lang
            inp = self.tokenizer(sentences, padding=True, return_tensors='pt')
            self.tokenizer.src_lang = self.trg_code_name
            trans_tokens = self.tokenizer(trans_sentences, padding=True, return_tensors='pt').input_ids
        else:
            inp = self.tokenizer(sentences, padding=True, return_tensors='pt')
            trans_tokens = self.tokenizer(trans_sentences, padding=True, return_tensors='pt').input_ids
        if inp.input_ids.shape[-1] > 512 or trans_tokens.shape[-1] > 512:
            # TODO what to do here?
            raise NotImplementedError

        if self.word_level is True:
            word2token : List[List[List[int]]]= self._get_word2token_batch(trans_sentences)
            token2word : List[List[int]] = self._get_token2word_batch(sentences)
            word_pos : List[List[List[int]]] = self._get_word_pos_batch(trans_sentences, char_positions)
            key_pos : List[List[List[int]]] = [[[token for word in words for token in word2token[i][word]] for words in entry] for i, entry in enumerate(word_pos)]
        else:
            key_pos : List[List[List[int]]] = self._get_token_pos_batch(trans_tokens, trans_sentences, char_positions)

        attentions = self.mapping_model(**inp, labels=trans_tokens, output_attentions=True).cross_attentions
        mapping = attentions[-2][..., :-1, :-1].sum(dim=1).softmax(-1).argmax(-1)

        pred : List[List[List[int]]] = [[list(set([mapping[i, j].item() for j in pos])) for pos in key_pos_entry] for i, key_pos_entry in enumerate(key_pos)]
        
        if self.word_level is True:
            pred = [[list(set([token2word[i][j] for j in pos])) for pos in pred_entry] for i, pred_entry in enumerate(pred)]
            char_seq : List[List[List[Tuple[int, int]]]] = self._get_char_sequences(sentences, pred)
        else:
            char_seq : List[List[List[Tuple[int, int]]]] = self._get_char_sequences_from_tokens_batch(inp.input_ids, pred)
        return char_seq
        

    def to(self, device):
        """
        Move models to given device
        """
        self.trans_model.to(device)
        if self.back_mapping is True:
            self.mapping_model.to(device)
