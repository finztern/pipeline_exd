"""Загрузка и разбор подписки: base64-блок (или обычный текст) со ссылками vless:// hysteria2:// ...

Ссылка подписки содержит секретный токен, поэтому она нигде не логируется: в сообщениях только хост.
"""
import asyncio
import base64
import binascii
import re
from urllib.parse import urlparse

from links import SCHEMES

_WS = re.compile(r"\s+")


def extract_links(text: str) -> list:
    """Строки, начинающиеся с поддерживаемой схемы; дубликаты убираются, порядок сохраняется."""
    seen, out = set(), []
    for line in text.replace("\r", "\n").split("\n"):
        line = line.strip()
        if line.startswith(SCHEMES) and line not in seen:
            seen.add(line)
            out.append(line)
    return out


def _b64decode(blob: str) -> str:
    s = _WS.sub("", blob)
    s = s.replace("-", "+").replace("_", "/")  # urlsafe -> обычный алфавит
    s += "=" * (-len(s) % 4)
    return base64.b64decode(s, validate=False).decode("utf-8", "replace")


def decode_subscription(raw) -> list:
    """Подписка -> список ссылок. Понимает base64 (обычный/urlsafe, с переносами и без padding) и plain-текст."""
    text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else raw
    text = text.lstrip("\ufeff").strip()
    links = extract_links(text)
    if links:
        return links
    head = text[:200].lstrip()
    if head.startswith("{") or head.startswith("[") or "proxies:" in text[:2000] or "outbounds" in head:
        raise ValueError("формат подписки sing-box/Clash не поддерживается: выберите в панели провайдера "
                         "вариант «v2ray / base64» (список ссылок)")
    if head.lower().startswith(("<!doctype", "<html")):
        raise ValueError("вместо подписки пришла HTML-страница (проверьте ссылку или не заблокирован ли доступ)")
    try:
        links = extract_links(_b64decode(text))
    except (binascii.Error, ValueError):
        links = []
    if not links:
        raise ValueError("в подписке не найдено ни одной ссылки vless/hysteria2")
    return links


async def fetch(url: str, *, proxy: str = "", timeout: int = 30, user_agent: str = "v2rayN/7.0") -> bytes:
    """Скачивает подписку через curl (поддерживает http(s), file:// и SOCKS/HTTP-прокси)."""
    host = urlparse(url).hostname or "?"
    cmd = ["curl", "-fsSL", "--max-time", str(timeout), "--connect-timeout", "10", "-A", user_agent]
    if proxy:
        cmd += ["-x", proxy]
    cmd += ["--", url]
    proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await proc.communicate()
    if proc.returncode != 0:
        msg = err.decode("utf-8", "replace").strip().splitlines()[-1:] or ["curl error"]
        # curl не печатает путь с токеном, но на всякий случай вырезаем URL целиком
        raise RuntimeError(f"{host}: {msg[0].replace(url, '<url>')[:160]}")
    return out
