from copy import deepcopy

import numpy as np
from transformers import GPT2TokenizerFast

_MAX_VOCAB_SIZE_FOR_UINT16 = np.iinfo('uint16').max + 1


class SongsEncoder:
    def __init__(self, tokenizer: GPT2TokenizerFast, max_n_tokens, artist_tokens):
        self._tokenizer = deepcopy(tokenizer)
        self._max_n_tokens = max_n_tokens
        self._artist_tokens = set(artist_tokens)
        self._tokenizer.add_special_tokens({'additional_special_tokens': sorted(self._artist_tokens)})
        self._vocab_size = len(self._tokenizer.get_vocab())
        self._dtype = np.dtype('uint16') if self._vocab_size < _MAX_VOCAB_SIZE_FOR_UINT16 else np.dtype('uint32')

    def encode(self, text, artist_token):
        text = self._prepare_text(text, artist_token)
        input_ids = self._tokenizer.encode(text)
        return input_ids[:self._max_n_tokens]

    def batch_encode(self, texts, artist_tokens):
        texts = []
        for text, artist_token in zip(texts, artist_tokens):
            texts.append(self._prepare_text(text, artist_token))
        encoded = self._tokenizer.batch_encode_plus(texts, return_attention_mask=False)
        return encoded.input_ids

    def _prepare_text(self, text, artist_token):
        if artist_token not in self._artist_tokens:
            raise ValueError(f'Unknown artist token: {artist_token}')
        return artist_token + text
