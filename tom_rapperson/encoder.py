import inspect
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from transformers import AutoTokenizer

_MAX_VOCAB_SIZE_FOR_UINT16 = np.iinfo('uint16').max + 1
_END_OF_PREFIX_TOKEN = '[END_OF_PREFIX]'
_SPECIAL_TOKENS = [_END_OF_PREFIX_TOKEN]


class SongsEncoder:
    _ENCODER_FILE_NAME = 'encoder.json'

    def __init__(
            self,
            tokenizer_name_or_path,
            max_n_context_lines,
            max_n_prefix_tokens,
            max_n_context_tokens,
            max_n_target_tokens,
    ):
        self._arg_to_value = {}
        for arg in inspect.getargspec(self.__init__).args:
            if arg != 'self':
                self._arg_to_value[arg] = locals()[arg]
        self._tokenize_name_or_path = tokenizer_name_or_path
        self._max_n_context_lines = max_n_context_lines
        self._max_n_prefix_tokens = max_n_prefix_tokens
        self._max_n_context_tokens = max_n_context_tokens
        self._max_n_target_tokens = max_n_target_tokens
        self._max_n_tokens = max_n_prefix_tokens + max_n_context_tokens + max_n_target_tokens
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self._tokenizer.add_special_tokens({'additional_special_tokens': _SPECIAL_TOKENS})
        self._vocab_size = len(self._tokenizer.get_vocab())
        self._dtype = np.dtype('uint16') if self._vocab_size < _MAX_VOCAB_SIZE_FOR_UINT16 else np.dtype('uint32')
        self._new_line_token_id = self._tokenizer.encode('\n')[0]

    @property
    def vocab_size(self):
        return max(self._tokenizer.all_special_ids) + 1

    @property
    def new_line_token_id(self):
        return self._new_line_token_id

    @property
    def max_n_target_toknes(self):
        return self._max_n_target_tokens

    def iterate_on_train_samples(
            self,
            prefixes: Iterable[str],
            contexts: Iterable[Sequence[str]],
            targets: Iterable[str],
    ):
        prefixes = [self._prepare_prefix(prefix) for prefix in prefixes]
        contexts = [self._prepare_context(context) for context in contexts]
        targets = [self._prepare_target(target) for target in targets]
        prefixes_input_ids = self._batch_encode(prefixes)
        contexts_input_ids = self._batch_encode(contexts)
        targets_input_ids = self._batch_encode(targets)
        for prefix_input_ids, context_input_ids, target_input_ids in zip(
                prefixes_input_ids,
                contexts_input_ids,
                targets_input_ids,
        ):
            if len(target_input_ids) > self._max_n_target_tokens:
                continue
            input_ids = self._concat_input_ids(
                prefix_input_ids=prefix_input_ids,
                context_input_ids=context_input_ids,
                target_input_ids=target_input_ids,
            )
            target_n_tokens = len(target_input_ids)
            yield (input_ids, target_n_tokens)

    def encode_inference(self, prefix, context: Sequence[str]):
        prefix = self._prepare_prefix(prefix)
        context = self._prepare_context(context)
        prefix_input_ids = self._tokenizer.encode(prefix)
        context_input_ids = self._tokenizer.encode(context)
        input_ids = self._concat_input_ids(
            prefix_input_ids=prefix_input_ids,
            context_input_ids=context_input_ids,
            target_input_ids=[],
        )
        return input_ids

    def decode(self, input_ids):
        return self._tokenizer.decode(input_ids)

    def save(self, out_dir):
        with open(Path(out_dir) / self._ENCODER_FILE_NAME, 'w') as out_file:
            json.dump(self._arg_to_value, out_file, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, dir_):
        with open(Path(dir_) / cls._ENCODER_FILE_NAME) as inp_file:
            params = json.load(inp_file)
        return cls(**params)

    def _concat_input_ids(self, prefix_input_ids, context_input_ids, target_input_ids):
        prefix_input_ids = prefix_input_ids[-self._max_n_prefix_tokens:]
        context_input_ids = context_input_ids[-self._max_n_context_tokens:]
        input_ids = prefix_input_ids + context_input_ids + target_input_ids
        assert len(input_ids) <= self._max_n_tokens
        input_ids = np.array(input_ids, dtype=self._dtype)
        return input_ids

    def _batch_encode(self, texts):
        return self._tokenizer.batch_encode_plus(list(texts), return_attention_mask=False).input_ids

    def _prepare_prefix(self, prefix):
        return prefix + _END_OF_PREFIX_TOKEN

    def _prepare_context(self, context: Sequence[str]) -> str:
        context = context[-self._max_n_context_lines:]
        context = '\n'.join(line.strip() for line in context) + '\n'
        return context

    def _prepare_target(self, target: str) -> str:
        return target.strip() + '\n'
