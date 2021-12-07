import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

_MAX_VOCAB_SIZE_FOR_UINT16 = np.iinfo('uint16').max + 1
_END_OF_PREFIX_TOKEN = '[END_OF_PREFIX]'


class SongsEncoder:
    _ENCODER_FILE_NAME = 'encoder.json'

    def __init__(self, tokenizer_name_or_path, max_n_tokens, max_n_prefix_tokens):
        self._tokenize_name_or_path = tokenizer_name_or_path
        self._max_n_tokens = max_n_tokens
        self._max_n_prefix_tokens = max_n_prefix_tokens
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self._tokenizer.add_special_tokens({'additional_special_tokens': [_END_OF_PREFIX_TOKEN]})
        self._vocab_size = len(self._tokenizer.get_vocab())
        self._dtype = np.dtype('uint16') if self._vocab_size < _MAX_VOCAB_SIZE_FOR_UINT16 else np.dtype('uint32')
        self._new_line_token_id = self._tokenizer.encode('\n')[0]

    @property
    def vocab_size(self):
        return max(self._tokenizer.all_special_ids) + 1

    @property
    def new_line_token_id(self):
        return self._new_line_token_id

    def batch_encode_train(self, prefixes, contexts, targets):
        prefixes = [prefix + _END_OF_PREFIX_TOKEN for prefix in prefixes]
        contexts = [context.strip() + '\n' for context in contexts]
        targets = [target.strip() + '\n' for target in targets]
        prefixes_input_ids = self._tokenizer.batch_encode_plus(list(prefixes), return_attention_mask=False)
        contexts_input_ids = self._tokenizer.batch_encode_plus(list(contexts), return_attention_mask=False)
        targets_input_ids = self._tokenizer.batch_encode_plus(list(targets), return_attention_mask=False)
        samples = []
        for prefix_input_ids, context_input_ids, target_input_ids in zip(
                prefixes_input_ids.input_ids,
                contexts_input_ids.input_ids,
                targets_input_ids.input_ids,
        ):
            prefix_input_ids = prefix_input_ids[-self._max_n_prefix_tokens:]
            prefix_n_tokens = len(prefix_input_ids)
            input_ids = (context_input_ids + target_input_ids)[-(self._max_n_tokens - prefix_n_tokens):]
            target_n_tokens = len(target_input_ids) if len(input_ids) >= len(target_input_ids) else len(input_ids)
            input_ids = prefix_input_ids + input_ids
            assert len(input_ids) <= self._max_n_tokens
            samples.append((np.array(input_ids, dtype=self._dtype), target_n_tokens))
        return samples

    def encode_inference(self, prefix, context):
        prefix += _END_OF_PREFIX_TOKEN
        context = context.strip() + '\n'
        prefix_input_ids = self._tokenizer.encode(prefix)[-self._max_n_prefix_tokens:]
        prefix_n_tokens = len(prefix_input_ids)
        context_input_ids = self._tokenizer.encode(context)
        input_ids = prefix_input_ids + context_input_ids[-(self._max_n_tokens - prefix_n_tokens):]
        return input_ids

    def decode(self, input_ids):
        return self._tokenizer.decode(input_ids)

    def save(self, out_dir):
        params = {
            'tokenizer_name_or_path': self._tokenize_name_or_path,
            'max_n_tokens': self._max_n_tokens,
            'max_n_prefix_tokens': self._max_n_tokens,
        }
        with open(Path(out_dir) / self._ENCODER_FILE_NAME, 'w') as out_file:
            json.dump(params, out_file, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, dir_):
        with open(Path(dir_) / cls._ENCODER_FILE_NAME) as inp_file:
            params = json.load(inp_file)
        return cls(**params)
