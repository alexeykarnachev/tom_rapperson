import asyncio
import logging
from typing import Optional

import aiohttp
from aiohttp.client_exceptions import ClientConnectionError

_logger = logging.getLogger(__name__)


class RequesterError(Exception):
    pass


class Requester:
    def __init__(self, concurrency, timeout, n_retries):
        self._timeout = timeout
        self._n_retries = n_retries
        self._semaphore = asyncio.BoundedSemaphore(concurrency)

    async def get(self, url, headers=None) -> Optional[str]:
        """Requests a page and returns content."""

        _logger.debug(f'Requesting page: {url}')
        async with self._get_session(headers=headers) as session:
            for i_retry in range(self._n_retries):
                try:
                    async with self._semaphore, session.get(url, allow_redirects=False) as response:
                        text = await response.text()
                        _logger.debug(f'Page source obtained: {url}')
                        return text
                except (asyncio.TimeoutError, ClientConnectionError):
                    _logger.warning(f'Retrying [{i_retry + 1}/{self._n_retries}]: {url}')
            else:
                _logger.warning(f'Max number of retries exceeded for page: {url}')
                raise RequesterError

    def _get_session(self, headers):
        connector = aiohttp.TCPConnector()
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        session = aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers)
        return session
