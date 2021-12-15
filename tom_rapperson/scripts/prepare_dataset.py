from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import tqdm

from tom_rapperson.dataset import INPUT_IDS_FILE_NAME, POST_LENGTHS_FILE_NAME, \
    PREFIX_LENGTHS_FILE_NAME, SEQUENCE_LENGTHS_FILE_NAME
from tom_rapperson.encoder import SongsEncoder
from tom_rapperson.utils import iterate_on_songs


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument('--train-songs-file-path', '-t', type=str, required=True)
    parser.add_argument('--valid-songs-file-path', '-v', type=str, required=True)
    parser.add_argument('--tokenizer-name-or-path', '-n', type=str, required=True)
    parser.add_argument('--max-n-context-lines', '-l', type=int, required=True)
    parser.add_argument('--max-n-prefix-tokens', '-p', type=int, required=True)
    parser.add_argument('--max-n-context-tokens', '-x', type=int, required=True)
    parser.add_argument('--max-n-post-tokens', '-g', type=int, required=True)
    parser.add_argument('--out-dir', '-o', type=str, required=True)
    return parser.parse_args()


def main(
        train_songs_file_path,
        valid_songs_file_path,
        tokenizer_name_or_path,
        max_n_context_lines,
        max_n_prefix_tokens,
        max_n_context_tokens,
        max_n_post_tokens,
        out_dir,
):
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    encoder = SongsEncoder(
        tokenizer_name_or_path=tokenizer_name_or_path,
        max_n_context_lines=max_n_context_lines,
        max_n_prefix_tokens=max_n_prefix_tokens,
        max_n_context_tokens=max_n_context_tokens,
        max_n_post_tokens=max_n_post_tokens,
    )
    encoder.save(out_dir)
    for data_name, songs_file_path in (('train', train_songs_file_path), ('valid', valid_songs_file_path)):
        songs = tqdm.tqdm(iterate_on_songs(songs_file_path), desc='Songs')
        samples = encoder.iterate_on_train_samples(songs)
        input_ids, prefix_n_tokens, post_n_tokens = zip(*samples)

        sequence_lengths = np.array([len(x) for x in input_ids], dtype=np.uint16)
        input_ids = np.hstack(input_ids)
        prefix_n_tokens = np.array(prefix_n_tokens, dtype=np.uint16)
        data_out_dir = Path(out_dir) / data_name
        data_out_dir.mkdir(exist_ok=True, parents=True)

        np.save(data_out_dir / INPUT_IDS_FILE_NAME, input_ids)
        np.save(data_out_dir / SEQUENCE_LENGTHS_FILE_NAME, sequence_lengths)
        np.save(data_out_dir / PREFIX_LENGTHS_FILE_NAME, prefix_n_tokens)
        np.save(data_out_dir / POST_LENGTHS_FILE_NAME, post_n_tokens)


if __name__ == '__main__':
    args = _parse_args()
    main(**vars(args))
