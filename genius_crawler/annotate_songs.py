import logging
import re
from argparse import ArgumentParser
from itertools import cycle
from multiprocessing import Lock, Process

import orjson
from langdetect import detect_langs
from langdetect.lang_detect_exception import LangDetectException
from more_itertools import chunked

_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def _parse_args():
    parser = ArgumentParser()
    parser.add_argument('--songs_file_path', type=str, required=True)
    parser.add_argument('--out_file_path', type=str, required=True)
    parser.add_argument('--n_workers', type=int, required=True)
    parser.add_argument('--min_n_words_in_song', type=int, required=True)
    return parser.parse_args()


def main(songs_file_path, out_file_path, n_workers, min_n_words_in_song):
    annotator = _Annotator(
        songs_file_path=songs_file_path,
        out_file_path=out_file_path,
        n_workers=n_workers,
        min_n_words_in_song=min_n_words_in_song,
    )
    annotator.start()


class _Annotator:
    _WRITE_CHUNK_SIZE = 10000
    _LOG_EACH_N_LINES = 10000

    def __init__(self, songs_file_path, out_file_path, n_workers, min_n_words_in_song):
        self._songs_file_path = songs_file_path
        self._out_file_path = out_file_path
        self._n_workers = n_workers
        self._min_n_words_in_song = min_n_words_in_song
        self._out_file_lock = Lock()

    def start(self):
        workers = []
        for worker_id in range(self._n_workers):
            worker = Process(target=self._run, args=(worker_id, ))
            worker.start()
            workers.append(worker)
        for worker in workers:
            worker.join()

    def _run(self, worker_id):
        worker_out_lines = self._iterate_on_worker_out_lines(worker_id)
        for out_lines in chunked(worker_out_lines, n=self._WRITE_CHUNK_SIZE):
            self._write_out_lines(out_lines)

    def _write_out_lines(self, out_lines):
        _logger.info(f'Writing {len(out_lines)} lines...')
        with self._out_file_lock:
            with open(self._out_file_path, 'wb') as out_file:
                for line in out_lines:
                    out_file.write(line)
                    out_file.write(b'\n')
        _logger.info('Written!')

    def _iterate_on_worker_out_lines(self, worker_id):
        worker_file_lines = self._iterate_on_worker_file_lines(worker_id)
        for line in worker_file_lines:
            line_data = orjson.loads(line)
            text = line_data['text']
            for i_word, _ in enumerate(re.finditer(r'\w+', text), start=1):
                if i_word == self._min_n_words_in_song:
                    break
            else:
                _logger.debug(f"Song has not enough words to be annotated, song will be skipped: {text}")
                continue
            try:
                langs = {x.lang: x.prob for x in detect_langs(text)}
            except LangDetectException:
                langs = None
                _logger.debug(f"Can't detect language, song will be skipped: {text[:50]}...")
            out_data = {
                'full_title': line_data['full_title'],
                'title': line_data['title'],
                'title_with_featured': line_data['title_with_featured'],
                'artist_name': line_data['primary_artist']['name'],
                'langs': langs,
                'text': text,
            }
            out_line = orjson.dumps(out_data)
            yield out_line

    def _iterate_on_worker_file_lines(self, worker_id):
        worker_ids = cycle(range(self._n_workers))
        with open(self._songs_file_path) as inp_file:
            for n_lines_done, (line_worker_id, line) in enumerate(zip(worker_ids, inp_file), start=1):
                if line_worker_id == worker_id:
                    yield line
                if worker_id == 0 and n_lines_done % self._LOG_EACH_N_LINES == 0:
                    _logger.info(f'Lines done: {n_lines_done}')


if __name__ == '__main__':
    args = _parse_args()
    main(**vars(args))
