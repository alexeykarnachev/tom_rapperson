import re
import random
from argparse import ArgumentParser
from itertools import chain
from pathlib import Path

import numpy as np
import orjson
import tqdm
from more_itertools import chunked, windowed

from tom_rapperson.dataset import INPUT_IDS_FILE_NAME, SEQUENCE_LENGTHS_FILE_NAME, TARGET_LENGTHS_FILE_NAME
from tom_rapperson.encoder import SongsEncoder

_TOKENIZER_CHUNK_SIZE = 10000


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument('--train-songs-file-path', '-t', type=str, required=True)
    parser.add_argument('--valid-songs-file-path', '-v', type=str, required=True)
    parser.add_argument('--tokenizer-name-or-path', '-n', type=str, required=True)
    parser.add_argument('--max-n-context-lines', '-l', type=int, required=True)
    parser.add_argument('--max-n-prefix-tokens', '-p', type=int, required=True)
    parser.add_argument('--max-n-context-tokens', '-x', type=int, required=True)
    parser.add_argument('--max-n-target-tokens', '-g', type=int, required=True)
    parser.add_argument('--out-dir', '-o', type=str, required=True)
    return parser.parse_args()


def main(
        train_songs_file_path,
        valid_songs_file_path,
        tokenizer_name_or_path,
        max_n_context_lines,
        max_n_prefix_tokens,
        max_n_context_tokens,
        max_n_target_tokens,
        out_dir,
):
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    encoder = SongsEncoder(
        tokenizer_name_or_path=tokenizer_name_or_path,
        max_n_context_lines=max_n_context_lines,
        max_n_prefix_tokens=max_n_prefix_tokens,
        max_n_context_tokens=max_n_context_tokens,
        max_n_target_tokens=max_n_target_tokens,
    )
    encoder.save(out_dir)
    for data_name, songs_file_path in (('train', train_songs_file_path), ('valid', valid_songs_file_path)):
        text_samples = _iterate_on_samples(songs_file_path, max_n_context_lines)
        input_ids = []
        target_lengths = []
        for text_samples_chunk in tqdm.tqdm(chunked(text_samples, _TOKENIZER_CHUNK_SIZE), desc=data_name):
            prefixes, contexts, targets = zip(*text_samples_chunk)
            samples = encoder.iterate_on_train_samples(prefixes, contexts, targets)
            input_ids_, target_lengths_ = zip(*samples)
            input_ids.extend(input_ids_)
            target_lengths.extend(target_lengths_)

        sequence_lengths = np.array([len(x) for x in input_ids], dtype=np.uint16)
        input_ids = np.hstack(input_ids)
        target_lengths = np.array(target_lengths, dtype=np.uint16)
        data_out_dir = Path(out_dir) / data_name
        data_out_dir.mkdir(exist_ok=True, parents=True)
        np.save(data_out_dir / INPUT_IDS_FILE_NAME, input_ids)
        np.save(data_out_dir / SEQUENCE_LENGTHS_FILE_NAME, sequence_lengths)
        np.save(data_out_dir / TARGET_LENGTHS_FILE_NAME, target_lengths)


def _iterate_on_samples(file_path, max_n_context_lines):
    with open(file_path) as inp_file:
        for line in inp_file:
            data = orjson.loads(line)
            song_lines = data['text'].split('\n')
            song_lines = chain(('' for _ in range(max_n_context_lines)), song_lines)
            for song_lines_window in windowed(song_lines, max_n_context_lines + 1):
                *context, target = song_lines_window
                context = [line for line in context if line.strip()]
                prefix = _get_prefix_from_target(target)
                yield prefix, context, target


def _get_prefix_from_target(target):
    words = re.findall(r'\w+', target)
    if len(words) <= 2:
        n_words = random.randint(0, 1)
    elif len(words) <= 5:
        n_words = random.randint(0, 2)
    else:
        n_words = random.randint(0, 3)

    if n_words > 0:
        prefix = ' '.join(word.lower() for word in words[-n_words:])
    else:
        prefix = ''
    return prefix


if __name__ == '__main__':
    args = _parse_args()
    main(**vars(args))
