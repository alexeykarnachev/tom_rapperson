import asyncio
import json
import logging
import re
from argparse import ArgumentParser
from json.decoder import JSONDecodeError
from pathlib import Path

import aiofiles
import orjson
from common import Requester, RequesterError
from more_itertools import chunked

_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

_HEADERS = {
    'authority':
        'genius.com',
    'cache-control':
        'max-age=0',
    'sec-ch-ua':
        '";Not A Brand";v="99", "Chromium";v="94"',
    'sec-ch-ua-mobile':
        '?0',
    'sec-ch-ua-platform':
        '"Linux"',
    'upgrade-insecure-requests':
        '1',
    'user-agent':
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36',
    'accept':
        'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'sec-fetch-site':
        'none',
    'sec-fetch-mode':
        'navigate',
    'sec-fetch-user':
        '?1',
    'sec-fetch-dest':
        'document',
    'accept-language':
        'en-US,en;q=0.9',
    'cookie':
        '_genius_ab_test_cohort=64; AMP_TOKEN=%24NOT_FOUND; _ga=GA1.2.2135689348.1634471605; _gid=GA1.2.782282800.1634471605; _fbp=fb.1.1634471605026.1883300496; genius_first_impression=1634471605083; _ab_tests_identifier=34155ff9-6963-4a3b-ba0e-0b5881cf707f; __gads=ID=e545b744adfaaff8-22efc0a9f8ca002f:T=1634471606:S=ALNI_MZWevpkQAs-OiJuUgm8IvB_wJGr8g; __qca=P0-165944924-1634471610019; mp_mixpanel__c=0; _cb_ls=1; _cb=CPTf03CQV5mPVrN0X; _cb_svref=null; mp_77967c52dc38186cc1aadebdd19e2a82_mixpanel=%7B%22distinct_id%22%3A%20%222135689348.1634471605%22%2C%22%24device_id%22%3A%20%2217c8e1972cb280-0f4b10a9244d83-1a2f1c08-384000-17c8e1972cc2ab%22%2C%22%24initial_referrer%22%3A%20%22%24direct%22%2C%22%24initial_referring_domain%22%3A%20%22%24direct%22%2C%22Logged%20In%22%3A%20false%2C%22Is%20Editor%22%3A%20null%2C%22Is%20Moderator%22%3A%20null%2C%22Mobile%20Site%22%3A%20false%2C%22AMP%22%3A%20false%2C%22Tag%22%3A%20%22rap%22%2C%22genius_platform%22%3A%20%22web%22%2C%22%24user_id%22%3A%20%222135689348.1634471605%22%2C%22provider%22%3A%20%22apple%22%2C%22provider_id%22%3A%20%221587792698%22%2C%22song%22%3A%20%22Last%20One%20Standing%22%2C%22song_id%22%3A%207241584%2C%22user_id%22%3A%20null%2C%22Song%20ID%22%3A%207241584%2C%22Title%22%3A%20%22Last%20One%20Standing%22%2C%22Primary%20Artist%22%3A%20%22Skylar%20Grey%2C%20Polo%20G%2C%20Mozzy%20%26%20Eminem%22%2C%22Primary%20Artist%20ID%22%3A%202898525%2C%22Primary%20Album%22%3A%20%22Last%20One%20Standing%20-%20Single%22%2C%22Primary%20Album%20ID%22%3A%20824572%2C%22Primary%20Tag%22%3A%20%22rap%22%2C%22Primary%20Tag%20ID%22%3A%201434%2C%22Music%3F%22%3A%20true%2C%22Annotatable%20Type%22%3A%20%22Song%22%2C%22Annotatable%20ID%22%3A%207241584%2C%22featured_video%22%3A%20false%2C%22cohort_ids%22%3A%20%5B%5D%2C%22has_verified_callout%22%3A%20false%2C%22has_featured_annotation%22%3A%20true%2C%22created_at%22%3A%20%222021-09-28T23%3A33%3A09Z%22%2C%22created_month%22%3A%20%222021-09-01%22%2C%22created_year%22%3A%202021%2C%22song_tier%22%3A%20%22B%22%2C%22Has%20Recirculated%20Articles%22%3A%20true%2C%22Lyrics%20Language%22%3A%20%22en%22%2C%22Has%20Apple%20Match%22%3A%20true%2C%22Release%20Date%22%3A%20%222021-09-30%22%2C%22NRM%20Tier%22%3A%20null%2C%22NRM%20Target%20Date%22%3A%20null%2C%22Has%20Description%22%3A%20true%2C%22Has%20Youtube%20URL%22%3A%20true%2C%22Has%20Translation%20Q%26A%22%3A%20true%2C%22Comment%20Count%22%3A%20120%2C%22react%22%3A%20false%2C%22amp_cache%22%3A%20false%2C%22containing_frame_is_fullbleed%22%3A%20true%2C%22AB%20Test%20-%20apple_desktop_static_cta%22%3A%20%22control%22%7D; _chartbeat2=.1634471612923.1634471620583.1.BZvNhzL1nQIBGuEGpBsSi2uB5AFFv.2',
    'if-none-match':
        'W/"2f14510b71733bdf4692865c43cd50da"',
}


def _parse_args():
    parser = ArgumentParser(description='Parses songs from genius.com.')
    parser.add_argument(
        '--out_file_path',
        type=str,
        required=True,
        help='Path to the output file with crawled songs.',
    )
    parser.add_argument(
        '--songs_meta_file_path',
        type=str,
        required=True,
        help='Path to the file with songs meta information (crawled by crawl_song_texts_meta.py script).',
    )
    parser.add_argument(
        '--crawled_song_ids_file_path',
        type=str,
        required=True,
        help='Path to the file which contains already crawled songs (their ids).')
    parser.add_argument(
        '--failed_song_ids_file_path',
        type=str,
        required=True,
        help='Path to the file which contains failed ot be crawled songs (their ids).')
    parser.add_argument(
        '--concurrency',
        type=int,
        required=True,
        help='Max number of simultaneous requests to the genius.com.',
    )
    parser.add_argument(
        '--max_n_simultaneous_tasks',
        type=int,
        required=False,
        default=1000,
        help='Max number of simultaneous tasks runned in the async loop.',
    )
    parser.add_argument(
        '--min_song_n_lines',
        type=int,
        required=False,
        default=10,
        help='Min number of lines in song to be saved. Smaller songs will be skiped.')
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
        out_file_path=args.out_file_path,
        crawled_song_ids_file_path=args.crawled_song_ids_file_path,
        failed_song_ids_file_path=args.failed_song_ids_file_path,
        max_n_simultaneous_tasks=args.max_n_simultaneous_tasks,
        min_song_n_lines=args.min_song_n_lines,
    )
    loop.run_until_complete(coroutine)


async def _get_songs_meta_async_gen(file_path):
    async with aiofiles.open(file_path, mode='r') as inp_file:
        async for line in inp_file:
            if len(line) < 10:
                continue
            try:
                yield json.loads(line)
            except JSONDecodeError:
                line_parts = re.split('("}})({")', line)
                lines = (''.join(parts) for parts in chunked(line_parts, n=2))
                for line in lines:
                    yield json.loads(line)


async def _crawl_all_songs(
        requester,
        songs_meta_async_gen,
        out_file_path,
        crawled_song_ids_file_path,
        failed_song_ids_file_path,
        max_n_simultaneous_tasks,
        min_song_n_lines,
):
    crawl_song_semaphore = asyncio.BoundedSemaphore(max_n_simultaneous_tasks)
    async for song_meta in songs_meta_async_gen:
        coroutine = _crawl_song(
            song_meta=song_meta,
            requester=requester,
            out_file_path=out_file_path,
            failed_song_ids_file_path=failed_song_ids_file_path,
            crawled_song_ids_file_path=crawled_song_ids_file_path,
            min_song_n_lines=min_song_n_lines,
            crawl_song_semaphore=crawl_song_semaphore,
        )
        async with crawl_song_semaphore:
            asyncio.ensure_future(coroutine)


async def _crawl_song(
        song_meta,
        requester,
        out_file_path,
        failed_song_ids_file_path,
        crawled_song_ids_file_path,
        min_song_n_lines,
        crawl_song_semaphore,
):
    async with crawl_song_semaphore:
        if song_meta['instrumental'] or song_meta['lyrics_state'] != 'complete':
            return
        url = song_meta['url']
        id_ = str(song_meta['id'])
        _logger.info(f'Crawling song: {url}')
        try:
            page_text = await requester.get(url, headers=_HEADERS)
            song_lines = _get_song_lines_from_page_text(page_text)
        except (RequesterError, AttributeError):
            await _append_line_to_file(failed_song_ids_file_path, id_ + '\n')
            _logger.info(f'Fail to crawl song: {url}')
            return
        if len(song_lines) < min_song_n_lines:
            return
        song_text = '\n'.join(song_lines)
        song_meta['text'] = song_text
        song_meta_payload = orjson.dumps(song_meta).decode() + '\n'
        await _append_line_to_file(out_file_path, song_meta_payload)
        await _append_line_to_file(crawled_song_ids_file_path, id_ + '\n')
        _logger.info(f'Song crawled ({len(song_text)} chars): {url}')


def _get_song_lines_from_page_text(page_text):
    page_text = re.search(r'<div class="lyrics">(.+)<!--/sse-->', page_text, flags=re.DOTALL).group(0)
    page_text = re.sub(r'<a href=.*?>', '', page_text, flags=re.DOTALL)
    song_line_matches = re.finditer(r'\n(.*?)(<br>|</p>|</a>)', page_text)
    song_lines = []
    for song_line_match in song_line_matches:
        song_line = song_line_match.group(1)
        song_line = re.sub(r'^\s+(<p>)*|\s+$', '', song_line)
        song_lines.append(song_line)
    return song_lines


def _validate_args(args):
    for file_path in (
            args.crawled_song_ids_file_path,
            args.failed_song_ids_file_path,
            args.out_file_path,
    ):
        if Path(file_path).exists():
            raise FileExistsError(file_path)


async def _append_line_to_file(out_file_path, line):
    assert line[-1] == '\n'
    async with aiofiles.open(out_file_path, 'a') as out_file:
        await out_file.write(line)


if __name__ == '__main__':
    main()
