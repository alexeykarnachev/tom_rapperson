from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import numpy as np
import orjson
import tqdm
from more_itertools import chunked

from tom_rapperson.dataset import INPUT_IDS_FILE_NAME, SEQUENCE_LENGTHS_FILE_NAME
from tom_rapperson.encoder import SongsEncoder

_UNKNOWN_ARTIST_NAME = 'UNKNOWN_ARTIST'
_TOKENIZER_CHUNK_SIZE = 10000


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument('--train-songs-file-path', type=str, required=True)
    parser.add_argument('--valid-songs-file-path', type=str, required=True)
    parser.add_argument('--tokenizer-name-or-path', type=str, required=True)
    parser.add_argument('--max-n-tokens', type=int, required=True)
    parser.add_argument('--out-dir', type=str, required=True)
    return parser.parse_args()


def _count_artist_names(songs_file_path):
    names_counter = Counter()
    with open(songs_file_path) as inp_file:
        for line in tqdm.tqdm(inp_file, desc='Names parsed'):
            data = orjson.loads(line)
            name = data['artist_name']
            names_counter.update([name])
    return names_counter


def _filter_artist_names(artist_names_counter, min_freq):
    names = set()
    for name, count in artist_names_counter.items():
        if count >= min_freq:
            names.add(name)
    return names


def _iterate_on_text_and_artist_names_pairs(songs_file_path):
    with open(songs_file_path) as inp_file:
        for line in inp_file:
            data = orjson.loads(line)
            artist_name = data['artist_name']
            text = data['text']
            yield text, artist_name


def main(train_songs_file_path, valid_songs_file_path, tokenizer_name_or_path, max_n_tokens, out_dir):
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    artist_names_counter = _count_artist_names(train_songs_file_path)
    artist_names = _filter_artist_names(artist_names_counter, min_freq=50)
    encoder = SongsEncoder(
        tokenizer_name_or_path=tokenizer_name_or_path,
        max_n_tokens=max_n_tokens,
        artist_names=artist_names,
    )
    encoder.save(out_dir)
    for data_name, songs_file_path in (('train', train_songs_file_path), ('valid', valid_songs_file_path)):
        input_ids = []
        for text_and_artist_name_pairs in tqdm.tqdm(
                chunked(
                    _iterate_on_text_and_artist_names_pairs(songs_file_path),
                    n=_TOKENIZER_CHUNK_SIZE,
                ),
                desc='Chunks encoded',
        ):
            texts, artist_names = zip(*text_and_artist_name_pairs)
            input_ids_ = encoder.batch_encode(texts=texts, artist_names=artist_names)
            input_ids.extend(input_ids_)
        sequence_lengths = np.array([len(x) for x in input_ids], dtype=np.uint16)
        input_ids = np.hstack(input_ids)
        data_out_dir = Path(out_dir) / data_name
        data_out_dir.mkdir(exist_ok=True, parents=True)
        np.save(data_out_dir / INPUT_IDS_FILE_NAME, input_ids)
        np.save(data_out_dir / SEQUENCE_LENGTHS_FILE_NAME, sequence_lengths)


if __name__ == '__main__':
    args = _parse_args()
    main(**vars(args))
