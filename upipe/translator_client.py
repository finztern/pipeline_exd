import asyncio
from urllib.parse import urljoin

import aiohttp


class TranslatorBusy(Exception):
    pass


class TranslatorUnavailable(Exception):
    pass


class TranslatorRejected(Exception):
    pass


async def translate_remote(session: aiohttp.ClientSession, base_url: str, text: str, lang: str, timeout: float) -> str:
    url = urljoin(base_url, "translate")
    try:
        async with session.post(
            url,
            json={"lang": lang, "text": text},
            timeout=aiohttp.ClientTimeout(total=timeout + 5),
        ) as resp:
            if resp.status == 200:
                return (await resp.json())["translation"]
            body = (await resp.text())[:200]
            if resp.status == 503 and "queue_full" in body:
                raise TranslatorBusy(body)
            if resp.status in (413, 422):
                raise TranslatorRejected(body)
            raise TranslatorUnavailable(f"HTTP {resp.status}: {body}")
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        raise TranslatorUnavailable(f"{type(e).__name__}: {e}")
