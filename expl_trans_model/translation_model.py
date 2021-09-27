from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Tuple
import re
from langdetect import detect
from .modified_fairseq import Fairseq


class SequenceTooLongException(Exception):

    def __init__(self):
        super().__init__('Target or source sequence exceeded the maximum token sequence length of 512 tokens. Did you check the sequences before? (check_length(seq))')


class TranslationModel:
    """
    Translation model that supports the back mapping of word labels.
    """

    def __init__(
            self,
            back_mapping=True,
            translation_name='facebook/wmt19-de-en'):
        # init forward path / translation
        if translation_name.startswith('facebook/wmt19'):
            self.config = 'wmt19'
            self.tokenizer = AutoTokenizer.from_pretrained(translation_name)
        elif translation_name.startswith('Helsinki-NLP/opus-mt'):
            self.config = 'marianmt'
            self.tokenizer = AutoTokenizer.from_pretrained(translation_name)
        elif translation_name.startswith('facebook/m2m100_418M') and re.fullmatch('facebook/m2m100_418M-[a-zA-Z]{2}', translation_name):
            self.config = 'm2m100_418M'
            translation_name, trg_code_name = translation_name.split('-')
            self.tokenizer = AutoTokenizer.from_pretrained(translation_name)
            self.trg_code_name = trg_code_name
            self.trg_code = self.tokenizer.get_lang_id(trg_code_name)
        else:
            raise Exception(f'Model {translation_name} not supported')
        self.trans_model = AutoModelForSeq2SeqLM.from_pretrained(
            translation_name)
        self.trans_model.eval()

        # init backward path / mapping
        self.back_mapping = back_mapping
        if back_mapping is True:
            if self.config == 'wmt19':
                self.mapping_model = Fairseq(translation_name)
            elif self.config in ['m2m100_418M', 'marianmt']:
                self.mapping_model = self.trans_model
            else:
                raise Exception(
                    f'Backward path is not supported for model {translation_name}')
            self.backward = self._backward

    def translate(self, sentence, max_length=512):
        """
        Translates the given sentence.
        Stops when the translation reaches the maximum length.
        """
        if self.config == 'm2m100_418M':
            # detect language
            src_lang = detect(sentence[0])
            if '-' in src_lang:
                src_lang = src_lang.split('-')[0]
            self.tokenizer.src_lang = src_lang

        inp = self.tokenizer(sentence, padding=True, return_tensors='pt')
        if inp.input_ids.shape[-1] > 512:
            raise SequenceTooLongException()
        inp = {k: v.to(self.trans_model.device) for k, v in inp.items()}
        if self.config == 'm2m100_418M':
            out = self.trans_model.generate(
                **inp, max_length=max_length, forced_bos_token_id=self.trg_code)
        else:
            out = self.trans_model.generate(**inp, max_length=max_length)
        out_seq = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        return out_seq

    def _get_char_sequences(
            self, sentences: List[str], positions: List[List[List[int]]]) -> List[List[List[Tuple[int, int]]]]:
        """
        Returns the character ranges for the given word positions in the given sentences.
        """
        # compute the character ranges of the words
        word_ranges = []
        for sentence in sentences:
            word_ranges.append([])
            pos = 0
            for word in sentence.split():
                word_ranges[-1].append((pos, pos + len(word)))
                pos += len(word) + 1

        # map word positions to character ranges
        results = []
        for i, batch in enumerate(positions):
            results.append([])
            for match in batch:
                results[-1].append([])
                for word_pos in match:
                    results[-1][-1].append(word_ranges[i][word_pos])
        return results

    def _get_token_pos(self,
                       trans_tokens: List[int],
                       trans_sentence: str,
                       char_positions: List[Tuple[int,
                                                  int]]) -> List[List[int]]:
        """
        Get token positions for the given character sequences in the sentence.
        """
        # compute token's character ranges
        upper_char_pos = [len(self.tokenizer.decode(
            trans_tokens[:i], skip_special_tokens=True)) for i in range(1, len(trans_tokens))]
        char_ranges = [(upper_char_pos[i], upper_char_pos[i + 1])
                       for i in range(len(upper_char_pos) - 1)]
        # search for tokens that match given character ranges
        token_pos = []
        for l, r in char_positions:
            token_pos.append([])
            for pos, (t, k) in enumerate(char_ranges):
                if t <= l < k or t < r <= k or (l <= t and k <= r):
                    token_pos[-1].append(pos)
                if r <= t:
                    break
        return token_pos

    def _get_char_sequences_from_tokens(
            self, tokens: List[int], pred: List[List[int]]) -> List[List[Tuple[int, int]]]:
        """
        Compute the character ranges for the given tokens.
        """
        results: List[List[Tuple[int, int]]] = []
        for entry in pred:
            results.append([])
            for pos in entry:
                l = len(self.tokenizer.decode(
                    tokens[:pos], skip_special_tokens=True))
                r = len(self.tokenizer.decode(
                    tokens[:pos + 1], skip_special_tokens=True))
                results[-1].append((l, r))
        return results

    def _get_char_sequences_from_tokens_batch(
            self, token_batch: List[List[int]], pred: List[List[List[int]]]) -> List[List[List[Tuple[int, int]]]]:
        """
        Compute _get_char_sequences_from_tokens for batches
        """
        return [
            self._get_char_sequences_from_tokens(
                tokens, pred_entry) for tokens, pred_entry in zip(
                token_batch, pred)]

    def _get_token_pos_batch(self,
                             trans_token_batch: List[List[int]],
                             trans_sentences: List[str],
                             char_positions: List[List[Tuple[int,
                                                             int]]]) -> List[List[List[int]]]:
        """
        Compute _get_token_pos for batches
        """
        return [
            self._get_token_pos(
                trans_tokens,
                trans_sentence,
                char_position) for trans_tokens,
            trans_sentence,
            char_position in zip(
                trans_token_batch,
                trans_sentences,
                char_positions)]

    def check_length(self, seq):
        """
        Checks if the given sequence meets the maximum length requirement (512 tokens)
        """
        return len(self.tokenizer(seq).input_ids) <= 512

    def _backward(self,
                  sentences: List[str],
                  trans_sentences: List[str],
                  char_positions: List[List[List[int]]]) -> List[List[List[Tuple[int,
                                                                                 int]]]]:
        """
        Maps the given character sequence in the translated sequence back to the source sequence.
        """
        if self.config == 'm2m100_418M':
            # detect language
            src_lang = detect(sentences[0])
            if '-' in src_lang:
                src_lang = src_lang.split('-')[0]
            self.tokenizer.src_lang = src_lang
            inp = self.tokenizer(sentences, padding=True, return_tensors='pt')
            self.tokenizer.src_lang = self.trg_code_name
            trans_tokens = self.tokenizer(
                trans_sentences,
                padding=True,
                return_tensors='pt').input_ids
        else:
            # use normal tokenizer
            inp = self.tokenizer(sentences, padding=True, return_tensors='pt')
            trans_tokens = self.tokenizer(
                trans_sentences,
                padding=True,
                return_tensors='pt').input_ids
        if inp.input_ids.shape[-1] > 512 or trans_tokens.shape[-1] > 512:
            raise SequenceTooLongException()

        # get token positions
        key_pos: List[List[List[int]]] = self._get_token_pos_batch(
            trans_tokens, trans_sentences, char_positions)

        # compute attention score and apply reduction from thesis
        attentions = self.mapping_model(
            **inp, labels=trans_tokens, output_attentions=True).cross_attentions[-1]
        attentions = attentions.sum(dim=1)

        # ignore eos (end of sequence) tokens
        eos_pos = inp.attention_mask.argmin(-1) - 1
        attentions[torch.arange(trans_tokens.shape[0]),
                   :, eos_pos] = float('-inf')

        if self.config == 'm2m100_418M':
            # ignore first language token
            attentions[..., 0] = float('-inf')

        # apply the maximum score selection method
        mapping = attentions.softmax(-1).argmax(-1)

        # apply mapping
        pred: List[List[List[int]]] = [[list(set([mapping[i, j].item(
        ) for j in pos])) for pos in key_pos_entry] for i, key_pos_entry in enumerate(key_pos)]

        # map tokens to character ranges
        char_seq: List[List[List[Tuple[int, int]]]
                       ] = self._get_char_sequences_from_tokens_batch(inp.input_ids, pred)
        return char_seq

    def to(self, device):
        """
        Move models to given device
        """
        self.trans_model.to(device)
        if self.back_mapping is True:
            self.mapping_model.to(device)
