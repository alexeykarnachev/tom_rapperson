import argparse
import re
from multiprocessing import Pool

import orjson
import tqdm


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp-songs-file-path', '-i', type=str, required=True)
    parser.add_argument('--out-songs-file-path', '-o', type=str, required=True)
    return parser.parse_args()


def main(inp_file_path, out_file_path):
    with open(inp_file_path) as inp_file, open(out_file_path, 'wb') as out_file:
        with Pool() as pool:
            out_lines = pool.imap_unordered(_preprocess_song, inp_file, chunksize=5000)
            for line in tqdm.tqdm(out_lines):
                out_file.write(line)
                out_file.write(b'\n')


def _preprocess_song(raw_song_line):
    data = orjson.loads(raw_song_line)
    data['text'] = _preprocess_text(data['text'])
    return orjson.dumps(data)


def _preprocess_text(song_text):
    song_lines = _get_song_lines(song_text)
    song_lines = _clean_noise_in_lines(song_lines)
    song_lines = _remove_header_lines(song_lines)
    song_lines = _remove_repetitive_lines(song_lines)
    return '\n'.join(song_lines)


def _get_song_lines(song_text):
    lines = song_text.split('\n')
    lines = [line for line in lines if len(line) > 0]
    return lines


def _clean_noise_in_lines(song_lines):
    lines = []
    for line in song_lines:
        line = re.sub(r'<.*?>', '', line)
        line = re.sub(r'}\".+?\">', '', line)
        line = re.sub(r'\(.*?\)', '', line)
        line = re.sub(r'\s+', ' ', line).strip()
        lines.append(line)
    return lines


def _remove_header_lines(song_lines):
    lines = []
    for line in song_lines:
        if re.search(r'[\[\]]', line):
            continue
        lines.append(line)
    return lines


def _remove_repetitive_lines(song_lines):
    appeared_lines = set()
    filtered_lines = []
    for line in song_lines:
        words = re.findall(r'\w+', line)
        line_signature = ' '.join(sorted(set(word.lower() for word in words)))
        if line_signature in appeared_lines:
            continue
        appeared_lines.add(line_signature)
        filtered_lines.append(line)
    return filtered_lines


if __name__ == '__main__':
    args = _parse_args()
    main(
        inp_file_path=args.inp_songs_file_path,
        out_file_path=args.out_songs_file_path,
    )
