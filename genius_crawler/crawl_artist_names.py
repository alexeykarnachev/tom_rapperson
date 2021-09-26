import asyncio
import logging
import re
import string
from argparse import ArgumentParser
from itertools import chain, count

from common import Requester

_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def _parse_args():
    parser = ArgumentParser(description='Parses all artist names from the genius.com site.')
    parser.add_argument(
        '--out_file_path',
        required=True,
        type=str,
        help='Path to the output text file to store artist names.',
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    requester = Requester(
        concurrency=10,
        timeout=5,
        n_retries=5,
    )
    loop = asyncio.get_event_loop()
    artist_names = loop.run_until_complete(_crawl_all_artist_names(requester))
    _save_artist_names(artist_names, args.out_file_path)


async def _crawl_all_artist_names(requester: Requester):
    coroutines = []
    for start_letter in string.ascii_lowercase + '0':
        coroutine = _crawl_start_letter_artist_names(requester, start_letter)
        coroutines.append(coroutine)
    return list(chain(*await asyncio.gather(*coroutines)))


async def _crawl_start_letter_artist_names(requester: Requester, start_letter):
    artist_names = []
    for page_number in count(start=1):
        url = _get_artists_index_url(start_letter=start_letter, page_number=page_number)
        page_text = await requester.get(url)
        artist_names_ = re.findall(r'<a href="https://genius.com/artists/(.+)">', page_text)
        if not artist_names_:
            break
        _logger.info(f'Parsed letter {start_letter}, page {page_number}, artists: {len(artist_names_)}')
        artist_names.extend(artist_names_)
    return artist_names


def _save_artist_names(artist_names, out_file_path):
    artist_names = sorted(set(artist_names))
    with open(out_file_path, 'w') as out_file:
        for artist_name in artist_names:
            out_file.write(artist_name)
            out_file.write('\n')


def _get_artists_index_url(start_letter, page_number):
    start_letter = start_letter.lower()
    return f'https://genius.com/artists-index/{start_letter}/all?page={page_number}'


if __name__ == '__main__':
    main()
