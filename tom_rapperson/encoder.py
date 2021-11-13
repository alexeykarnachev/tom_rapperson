import json
import pickle
import re
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

_MAX_VOCAB_SIZE_FOR_UINT16 = np.iinfo('uint16').max + 1
_UNKNOWN_ARTIST_TOKEN = '[UNKNOWN_ARTIST]'


class SongsEncoder:
    _ENCODER_FILE_NAME = 'encoder.json'

    def __init__(self, tokenizer_name_or_path, max_n_tokens, artist_names):
        self._tokenize_name_or_path = tokenizer_name_or_path
        self._max_n_tokens = max_n_tokens
        self._artist_names = set(artist_names)
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self._artist_name_to_token = {name: _convert_artist_name_to_token(name) for name in self._artist_names}
        artist_tokens = set(self._artist_name_to_token.values())
        artist_tokens.add(_UNKNOWN_ARTIST_TOKEN)
        self._tokenizer.add_special_tokens({'additional_special_tokens': sorted(artist_tokens)})
        self._vocab_size = len(self._tokenizer.get_vocab())
        self._dtype = np.dtype('uint16') if self._vocab_size < _MAX_VOCAB_SIZE_FOR_UINT16 else np.dtype('uint32')

    @property
    def vocab_size(self):
        return max(self._tokenizer.all_special_ids) + 1

    def encode(self, text, artist_name):
        text = self._prepare_text(text, artist_name)
        input_ids = self._tokenizer.encode(text)
        return np.array(input_ids[:self._max_n_tokens], dtype=self._dtype)

    def decode(self, input_ids):
        return self._tokenizer.decode(input_ids)

    def _prepare_text(self, text, artist_name):
        if artist_name not in self._artist_names:
            artist_token = _UNKNOWN_ARTIST_TOKEN
        else:
            artist_token = self._artist_name_to_token[artist_name]
        return artist_token + text

    def batch_encode(self, texts, artist_names):
        prepared_texts = []
        for text, artist_name in zip(texts, artist_names):
            prepared_texts.append(self._prepare_text(text, artist_name))
        encoded = self._tokenizer.batch_encode_plus(prepared_texts, return_attention_mask=False)
        return [np.array(input_ids[:self._max_n_tokens], dtype=self._dtype) for input_ids in encoded.input_ids]

    def save(self, out_dir):
        params = {
            'tokenizer_name_or_path': self._tokenize_name_or_path,
            'max_n_tokens': self._max_n_tokens,
            'artist_names': list(self._artist_names),
        }
        with open(Path(out_dir) / self._ENCODER_FILE_NAME, 'w') as out_file:
            json.dump(params, out_file, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, dir_):
        with open(Path(dir_) / cls._ENCODER_FILE_NAME) as inp_file:
            params = json.load(inp_file)
        return cls(**params)


def _convert_artist_name_to_token(artist_name):
    return '[' + re.sub(r'\s+', '_', artist_name) + ']'
