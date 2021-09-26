import asyncio
import logging
import re
from argparse import ArgumentParser
from itertools import count
from pathlib import Path

import aiofiles
import orjson
from common import Requester, RequesterError

_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


def _parse_args():
    parser = ArgumentParser(description='Parses songs meta information (url, description etc) from genius.com.')
    parser.add_argument(
        '--artist_names_file_path',
        type=str,
        required=True,
        help='Path to the artist names text file.',
    )
    parser.add_argument(
        '--songs_meta_file_path',
        type=str,
        required=True,
        help='Path to the output jsonl file to store songs meta.',
    )
    parser.add_argument(
        '--failed_artist_names_file_path',
        type=str,
        required=True,
        help='Path to the output text file to store artist names which failed to be parsed.',
    )
    parser.add_argument(
        '--concurrency',
        type=int,
        required=False,
        help='Max number of simultaneous requests to the genius.com.',
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    if Path(args.songs_meta_file_path).exists() or Path(args.failed_artist_names_file_path).exists():
        raise FileExistsError('Passed `songs_meta_file_path` or `failed_artist_names_file_path` exist.')
    artist_names = _read_artist_names(args.artist_names_file_path)
    requester = Requester(
        concurrency=args.concurrency,
        timeout=5,
        n_retries=5,
    )
    loop = asyncio.get_event_loop()
    coroutine = _crawl_all_artists_songs_meta(
        requester=requester,
        songs_meta_file_path=args.songs_meta_file_path,
        failed_artist_names_file_path=args.failed_artist_names_file_path,
        artist_names=artist_names,
    )
    loop.run_until_complete(coroutine)


def _read_artist_names(file_path):
    artist_names = []
    with open(file_path) as inp_file:
        for line in inp_file:
            line = line.strip()
            if line:
                artist_names.append(line)
    return artist_names


class _CantObtainArtistIDError(Exception):
    pass


async def _crawl_all_artists_songs_meta(
        requester: Requester,
        songs_meta_file_path,
        failed_artist_names_file_path,
        artist_names,
):
    coroutines = []
    for artist_name in artist_names:
        coroutine = _crawl_artist_songs_meta(
            requester=requester,
            songs_meta_file_path=songs_meta_file_path,
            failed_artist_names_file_path=failed_artist_names_file_path,
            artist_name=artist_name,
        )
        coroutines.append(coroutine)
    await asyncio.gather(*coroutines)


async def _crawl_artist_songs_meta(
        requester: Requester,
        songs_meta_file_path,
        failed_artist_names_file_path,
        artist_name,
):
    songs_meta = []
    try:
        artist_id = await _crawl_artist_id(requester, artist_name)
    except _CantObtainArtistIDError:
        _logger.warning(f'Can\'t obtain id for artist: "{artist_name}".')
    else:
        for page_number in count(start=1):
            url = f'https://genius.com/api/artists/{artist_id}/songs?page={page_number}'
            try:
                page_text = await requester.get(url)
            except RequesterError:
                _save_failed_artist(failed_artist_names_file_path, artist_name)
            response_data = orjson.loads(page_text)
            if response_data['meta']['status'] != 200:
                _logger.warning(f'Can\'t obtain songs meta from url (status != 200): {url}')
                break
            songs_meta.extend(response_data['response']['songs'])
            next_page_number = response_data['response']['next_page']
            if next_page_number is None:
                break
            elif next_page_number != page_number + 1:
                raise ValueError(f'Unexpected next page number for url: {url}')
    await _save_artist_songs_meta(songs_meta_file_path, songs_meta)
    _logger.info(f'Crawled {len(songs_meta)} songs meta for "{artist_name}"')


async def _save_artist_songs_meta(out_file_path, songs_meta):
    async with aiofiles.open(out_file_path, "ab") as out_file:
        for song_meta in songs_meta:
            await out_file.write(orjson.dumps(song_meta))
            await out_file.write(b'\n')
            await out_file.flush()


async def _save_failed_artist(out_file_path, artist_name):
    async with aiofiles.open(out_file_path, 'a') as out_file:
        await out_file.write(artist_name)
        await out_file.wite('\n')


async def _crawl_artist_id(requester: Requester, artist_name):
    url = f'https://genius.com/artists/{artist_name}'
    page_text = await requester.get(url)
    matches = re.finditer(r'meta content="/artists/(\d+)" name="newrelic-resource-path"', page_text)
    try:
        artist_id = int(next(matches).group(1))
    except StopIteration:
        raise _CantObtainArtistIDError
    return artist_id


if __name__ == '__main__':
    main()
