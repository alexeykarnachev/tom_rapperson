import inspect
import json
import re
from itertools import chain
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from more_itertools import chunked, windowed
from transformers import AutoTokenizer

_TOKENIZER_CHUNK_SIZE = 10000
_MAX_VOCAB_SIZE_FOR_UINT16 = np.iinfo('uint16').max + 1
_END_OF_PREFIX_TOKEN = '[END_OF_PREFIX]'
_END_OF_TARGET_TOKEN = '[END_OF_TARGET]'
_SPECIAL_TOKENS = [_END_OF_PREFIX_TOKEN, _END_OF_TARGET_TOKEN]


class SongsEncoder:
    _ENCODER_FILE_NAME = 'encoder.json'

    def __init__(
            self,
            tokenizer_name_or_path,
            max_n_context_lines,
            max_n_prefix_tokens,
            max_n_context_tokens,
            max_n_post_tokens,
    ):
        self._arg_to_value = {}
        for arg in inspect.getargspec(self.__init__).args:
            if arg != 'self':
                self._arg_to_value[arg] = locals()[arg]
        self._tokenize_name_or_path = tokenizer_name_or_path
        self._max_n_context_lines = max_n_context_lines
        self._max_n_prefix_tokens = max_n_prefix_tokens
        self._max_n_context_tokens = max_n_context_tokens
        self._max_n_post_tokens = max_n_post_tokens
        self._max_n_tokens = max_n_prefix_tokens + max_n_context_tokens + max_n_post_tokens
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        self._tokenizer.add_special_tokens({'additional_special_tokens': _SPECIAL_TOKENS})
        self._vocab_size = len(self._tokenizer.get_vocab())
        self._dtype = np.dtype('uint16') if self._vocab_size < _MAX_VOCAB_SIZE_FOR_UINT16 else np.dtype('uint32')
        self._new_line_token_id = self._tokenizer.encode('\n')[0]
        self._space_token_id = self._tokenizer.encode(' ')[0]
        self._end_of_target_token_id = self._tokenizer.convert_tokens_to_ids(_END_OF_TARGET_TOKEN)
        self._end_of_prefix_token_id = self._tokenizer.convert_tokens_to_ids(_END_OF_PREFIX_TOKEN)

    @property
    def vocab_size(self):
        return max(self._tokenizer.all_special_ids) + 1

    @property
    def new_line_token_id(self):
        return self._new_line_token_id

    @property
    def end_of_target_token_id(self):
        return self._end_of_target_token_id

    @property
    def end_of_prefix_token_id(self):
        return self._end_of_prefix_token_id

    @property
    def max_n_post_tokens(self):
        return self._max_n_post_tokens

    def iterate_on_train_samples(self, songs: Iterable[str]):
        for parts in chunked(self._iterate_on_train_sample_parts(songs), n=_TOKENIZER_CHUNK_SIZE):
            prefixes, contexts, targets, postfixes = zip(*parts)
            prefixes_input_ids = self._batch_encode(prefixes)
            contexts_input_ids = self._batch_encode(contexts)
            targets_input_ids = self._batch_encode(targets)
            postfixes_input_ids = self._batch_encode(postfixes)
            for prefix_input_ids, context_input_ids, target_input_ids, postfix_input_ids in zip(
                    prefixes_input_ids,
                    contexts_input_ids,
                    targets_input_ids,
                    postfixes_input_ids,
            ):
                if len(target_input_ids) + len(postfix_input_ids) > self._max_n_post_tokens:
                    continue
                input_ids = self._concat_input_ids(
                    prefix_input_ids=prefix_input_ids,
                    context_input_ids=context_input_ids,
                    target_input_ids=target_input_ids,
                    postfix_input_ids=postfix_input_ids,
                )
                prefix_n_tokens = len(prefix_input_ids)
                post_n_tokens = len(target_input_ids) + len(postfix_input_ids)
                yield input_ids, prefix_n_tokens, post_n_tokens

    def encode_inference(self, prefix, context: Sequence[str]):
        prefix = self._prepare_prefix(prefix)
        context = self._prepare_context(context)
        prefix_input_ids = self._tokenizer.encode(prefix)
        context_input_ids = self._tokenizer.encode(context)
        input_ids = self._concat_input_ids(
            prefix_input_ids=prefix_input_ids,
            context_input_ids=context_input_ids,
            target_input_ids=[],
            postfix_input_ids=[],
        )
        return input_ids

    def decode(self, input_ids):
        input_ids = input_ids.cpu().numpy().tolist()
        try:
            end_of_target_pos = input_ids.index(self.end_of_target_token_id)
            input_ids = list(input_ids)
            input_ids[end_of_target_pos] = self._space_token_id
        except ValueError:
            pass
        return self._tokenizer.decode(input_ids, skip_special_tokens=True)

    def save(self, out_dir):
        with open(Path(out_dir) / self._ENCODER_FILE_NAME, 'w') as out_file:
            json.dump(self._arg_to_value, out_file, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, dir_):
        with open(Path(dir_) / cls._ENCODER_FILE_NAME) as inp_file:
            params = json.load(inp_file)
        return cls(**params)

    def _concat_input_ids(
            self,
            prefix_input_ids,
            context_input_ids,
            target_input_ids,
            postfix_input_ids,
    ):
        context_input_ids = context_input_ids[-self._max_n_context_tokens:]
        prefix_input_ids = prefix_input_ids[-self._max_n_prefix_tokens:]
        input_ids = prefix_input_ids + context_input_ids + target_input_ids + postfix_input_ids
        assert len(input_ids) <= self._max_n_tokens
        input_ids = np.array(input_ids, dtype=self._dtype)
        return input_ids

    def _batch_encode(self, texts):
        return self._tokenizer.batch_encode_plus(list(texts), return_attention_mask=False).input_ids

    def _prepare_prefix(self, prefix):
        return prefix.lower().strip() + _END_OF_PREFIX_TOKEN

    def _prepare_context(self, context: Sequence[str]) -> str:
        context = context[-self._max_n_context_lines:]
        context = '\n'.join(line.strip() for line in context) + '\n'
        return context

    def _prepare_target(self, target: str) -> str:
        return target.strip() + _END_OF_TARGET_TOKEN

    def _prepare_postfix(self, postfix: str) -> str:
        return postfix.strip() + '\n'

    def _iterate_on_train_sample_parts(self, songs: Iterable[str]):
        for song in songs:
            lines = song.split('\n')
            lines = [line.strip() for line in lines]
            lines = [line for line in lines if len(line) > 1]
            lines = chain(('' for _ in range(self._max_n_context_lines)), lines)

            for lines_window in windowed(lines, self._max_n_context_lines + 1):
                *context, target = lines_window
                if target is None:
                    continue
                context = [line for line in context if line.strip()]
                target_split_idx = _get_target_split_idx(target)
                if target_split_idx is not None:
                    target, postfix = target[:target_split_idx], target[target_split_idx:]
                    prefix = re.search(r'\w+', postfix.lower()).group(0)
                    prefix = self._prepare_prefix(prefix)
                    context = self._prepare_context(context)
                    target = self._prepare_target(target)
                    postfix = self._prepare_postfix(postfix)
                    yield prefix, context, target, postfix


def _get_target_split_idx(target):
    words = re.findall(r'\w+', target)
    if len(words) <= 3:
        return None
    search_res = re.search(r'\w+\W*?$', target)
    split_idx = search_res.span()[0]
    return split_idx
