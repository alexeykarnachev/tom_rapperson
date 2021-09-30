import asyncio
import json
import logging
import re
from argparse import ArgumentParser
from itertools import count
from pathlib import Path

import aiofiles
import orjson
from common import Requester, RequesterError


def _parse_args():
    parser = ArgumentParser(description='Parses songs from genius.com.')
    parser.add_argument(
        '--songs_meta_file_path',
        type=str,
        required=True,
        help='Path to the file with songs meta information (crawled by crawl_songs_meta.py script).',
    )
    parser.add_argument(
        '--crawled_songs_file_path',
        type=str,
        required=True,
        help='Path to the file which contains already crawled songs (their urls).')
    parser.add_argument(
        '--failed_songs_file_path',
        type=str,
        required=True,
        help='Path to the file which contains failed ot be crawled songs (their urls).')
    parser.add_argument(
        '--concurrency',
        type=int,
        required=True,
        help='Max number of simultaneous requests to the genius.com.',
    )

    return parser.parse_args()


def main():
    args = _parse_args()
    _validate_args(args)
    requester = Requester(concurrency=args.concurrency)
    songs_meta_async_gen = _get_songs_meta_async_gen(args.songs_meta_file_path)
    loop = asyncio.get_event_loop()
    coroutine = _crawl_all_songs(
        requester=requester,
        songs_meta_async_gen=songs_meta_async_gen,
        crawled_songs_file_path=args.crawled_songs_file_path,
        failed_songs_file_path=args.failed_songs_file_path,
    )
    loop.run_until_complete(coroutine)


async def _get_songs_meta_async_gen(file_path):
    async with aiofiles.open(file_path, mode='r') as inp_file:
        async for line in inp_file:
            yield json.loads(line)


async def _crawl_all_songs(
        requester,
        songs_meta_async_gen,
        crawled_songs_file_path,
        failed_songs_file_path,
):
    async for song_meta in songs_meta_async_gen:
        _crawl_song(
            requester=requester,
            song_meta=song_meta,
            crawled_songs_file_path=crawled_songs_file_path,
            failed_songs_file_path=failed_songs_file_path,
        )


async def _crawl_song(
        requester,
        song_meta,
        crawled_songs_file_path,
        failed_songs_file_path,
):
    pass


def _validate_args(args):
    for file_path in (
            args.crawled_songs_file_path,
            args.failed_songs_file_path,
    ):
        if Path(file_path).exists():
            raise FileExistsError(file_path)


if __name__ == '__main__':
    pass
