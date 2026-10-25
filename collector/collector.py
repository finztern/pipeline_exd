import asyncio
import hashlib
import logging
import os
import re
import sys
import time
from collections import Counter, OrderedDict, deque
from datetime import datetime, timezone
from urllib.parse import urljoin

import aiohttp
import orjson
from aiohttp import web
from langdetect import detect_langs, DetectorFactory, LangDetectException

DetectorFactory.seed = 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [collector] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

UPIPE_URL           = os.getenv("UPIPE_URL", "http://127.0.0.1:5981/")
COLLECTOR_PORT      = int(os.getenv("COLLECTOR_PORT", "9000"))
MIN_TEXT_LEN        = int(os.getenv("MIN_TEXT_LEN", "20"))
MAX_TEXT_LEN        = int(os.getenv("MAX_TEXT_LEN", "20000"))
MAX_OLDNESS_SECONDS = int(os.getenv("MAX_OLDNESS_SECONDS", "86400"))
LANG_FILTER         = os.getenv("LANG_FILTER", "en").strip().lower()
BPIPE_URLS          = [u.strip() for u in os.getenv("BPIPE_URLS", "http://bpipe:7995/").split(",") if u.strip()]
UPIPE_QUEUE_LIMIT   = int(os.getenv("UPIPE_QUEUE_LIMIT", "200"))
QUEUE_LOW_FILL      = float(os.getenv("QUEUE_LOW_FILL", "0.2"))
QUEUE_HIGH_FILL     = float(os.getenv("QUEUE_HIGH_FILL", "0.7"))
FOREIGN_FORWARD_URL = os.getenv("FOREIGN_FORWARD_URL", "").strip()

TRANSLATE             = os.getenv("TRANSLATE", "false").lower() == "true"
TRANSLATOR_URL        = os.getenv("TRANSLATOR_URL", "http://translator:8003/")
TRANSACTIONEER_URL    = os.getenv("TRANSACTIONEER_URL", "http://transactioneer:8002/")
TRANSLATE_MAX_CHARS   = int(os.getenv("TRANSLATE_MAX_CHARS", "2000"))
TRANSLATE_BUSY_FILL   = float(os.getenv("TRANSLATE_BUSY_FILL", "0.9"))
TRANSLATE_POLL_SEC    = float(os.getenv("TRANSLATE_POLL_SECONDS", "1.0"))
LANG_CONFIDENCE       = float(os.getenv("LANG_CONFIDENCE_THRESHOLD", "0.90"))

EN_RESCUE_MIN_HITS  = int(os.getenv("EN_RESCUE_MIN_HITS", "2"))
EN_RESCUE_RATIO     = float(os.getenv("EN_RESCUE_RATIO", "0.15"))

GPU_TRANSLATOR_URL  = os.getenv("GPU_TRANSLATOR_URL", "").strip()
GPU_ROUTE_REASONS   = {r.strip() for r in os.getenv("GPU_ROUTE_REASONS", "queue_full").split(",") if r.strip()}
GPU_RETRY_SECONDS   = float(os.getenv("GPU_RETRY_SECONDS", "30"))

URL_DEDUP_MODE      = os.getenv("URL_DEDUP_MODE", "shadow").strip().lower()
URL_DEDUP_API       = os.getenv("URL_DEDUP_API", "https://url-check.exorde.network").rstrip("/")
URL_DEDUP_BATCH     = max(1, int(os.getenv("URL_DEDUP_BATCH", "48")))
URL_DEDUP_WORKERS   = max(1, int(os.getenv("URL_DEDUP_WORKERS", "4")))
URL_DEDUP_FLUSH_SEC = float(os.getenv("URL_DEDUP_FLUSH_SECONDS", "0.1"))
URL_DEDUP_TIMEOUT   = float(os.getenv("URL_DEDUP_TIMEOUT_SECONDS", "5"))
URL_DEDUP_WAIT      = float(os.getenv("URL_DEDUP_WAIT_SECONDS", "3"))
URL_DEDUP_QUEUE_MAX = int(os.getenv("URL_DEDUP_QUEUE_MAX", "20000"))
URL_DEDUP_TTL       = float(os.getenv("URL_DEDUP_TTL_SECONDS", str(3 * 86400)))
URL_DEDUP_CACHE_MAX = int(os.getenv("URL_DEDUP_CACHE_MAX", "200000"))
URL_DEDUP_DOMAINS   = [d.strip().lower() for d in os.getenv("URL_DEDUP_DOMAINS", "").split(",") if d.strip()]

LANG_ALIASES = {"zh-cn": "zh", "zh-tw": "zt", "no": "nb"}

# Слова, не пересекающиеся с de/nl/af/da/no/sv/romance — langdetect часто принимает короткий en за cy/af/so/tl/nl и т.п.
_EN_WORDS = frozenset(
    "the and you that this with have but they what from your just like about would there their "
    "been were them because really when who its our how then also only which not are "
    "i'm it's don't that's can't you're i've i'll they're we're isn't didn't doesn't".split()
)
_WORD_RE = re.compile(r"[a-z']+")
_STRIP_RE = re.compile(r"https?://\S+|@\w+|#\w+")

_seen_ids: OrderedDict = OrderedDict()
_DEDUP_MAX_SIZE = 100_000
_started = time.time()

_stats = {
    "received": 0, "forwarded": 0, "filtered_old": 0, "filtered_dup": 0, "filtered_lang": 0,
    "foreign_sent": 0, "to_translate": 0, "truncated": 0, "errors": 0, "dropped": 0,
    "to_gpu": 0, "gpu_failed": 0, "rescued_en": 0, "filtered_url_dup": 0,
}
_dom_buckets: deque = deque()
_dom_totals = {"r": Counter(), "f": Counter()}
_DOM_MAX = 1000
_langs_seen: Counter = Counter()
_foreign_reasons: Counter = Counter()
_tr_state = {"ok": False, "fill": 1.0, "installed": frozenset()}
_session: aiohttp.ClientSession | None = None
_bg_tasks: set = set()
_gpu_down_until = 0.0

_url_pending: deque = deque()
_url_evt: asyncio.Event | None = None
_known_dups: OrderedDict = OrderedDict()
_sent_urls: OrderedDict = OrderedDict()
_url_dom: dict = {}
_url_stats = {
    "queued": 0, "checked": 0, "duplicates": 0, "requests": 0, "errors": 0, "format_errors": 0,
    "overflow": 0, "timeouts": 0, "cache_hits": 0, "self_dups": 0, "unmatched": 0,
    "total_ms": 0.0, "last_ms": 0.0, "sample": None, "last_error": None,
}


def _dom_record(domain: str, kind: str):
    domain = (domain or "unknown").strip().lower()[:100] or "unknown"
    if domain not in _dom_totals["r"] and len(_dom_totals["r"]) >= _DOM_MAX:
        domain = "other"
    minute = int(time.time() // 60)
    if not _dom_buckets or _dom_buckets[-1][0] != minute:
        _dom_buckets.append((minute, {"r": Counter(), "f": Counter()}))
        while _dom_buckets and _dom_buckets[0][0] <= minute - 60:
            _dom_buckets.popleft()
    _dom_buckets[-1][1][kind][domain] += 1
    _dom_totals[kind][domain] += 1


def _dom_snapshot() -> dict:
    now_min = int(time.time() // 60)
    up = max(60.0, time.time() - _started)
    wins = {}
    for name, m in (("10m", 10), ("30m", 30), ("60m", 60)):
        r, f = Counter(), Counter()
        for mi, d in _dom_buckets:
            if mi > now_min - m:
                r.update(d["r"])
                f.update(d["f"])
        wins[name] = {"r": dict(r), "f": dict(f), "window_s": int(min(m * 60, up))}
    return {"totals": {"r": dict(_dom_totals["r"]), "f": dict(_dom_totals["f"])}, "windows": wins}


def _norm_lang(lang: str) -> str:
    return LANG_ALIASES.get(lang, lang.split("-")[0])


def _hash_author(author: str) -> str:
    if not author:
        return ""
    return hashlib.sha1(author.encode("utf-8")).hexdigest()


def _looks_english(text: str) -> bool:
    words = _WORD_RE.findall(_STRIP_RE.sub(" ", text.lower()))
    if len(words) < 4:
        return False
    hits = sum(1 for w in words if w in _EN_WORDS)
    return hits >= EN_RESCUE_MIN_HITS and hits / len(words) >= EN_RESCUE_RATIO


async def forward_to_upipe(item: dict) -> bool:
    global _session
    if _session is None:
        return False
    try:
        async with _session.post(UPIPE_URL, data=orjson.dumps(item),
                                  headers={"Content-Type": "application/json"}) as resp:
            if 200 <= resp.status < 300:
                return True
            log.warning(f"upipe ответил {resp.status}")
            return False
    except aiohttp.ClientConnectorError:
        log.error(f"❌ upipe недоступен: {UPIPE_URL}")
        return False
    except Exception as e:
        log.debug(f"Ошибка пересылки: {e}")
        return False


async def _send_foreign(item: dict):
    try:
        async with _session.post(FOREIGN_FORWARD_URL, data=orjson.dumps(item),
                                  headers={"Content-Type": "application/json"},
                                  timeout=aiohttp.ClientTimeout(total=5)) as resp:
            await resp.read()
    except Exception as e:
        log.debug(f"foreign forward error: {e}")


def _forward_foreign(item: dict):
    if not FOREIGN_FORWARD_URL or _session is None:
        return
    _stats["foreign_sent"] += 1
    task = asyncio.create_task(_send_foreign(item))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


def _gpu_route_ok(reason: str | None) -> bool:
    return bool(GPU_TRANSLATOR_URL) and reason in GPU_ROUTE_REASONS and time.monotonic() >= _gpu_down_until


async def _send_gpu(item: dict, raw_item: dict):
    global _gpu_down_until
    try:
        async with _session.post(GPU_TRANSLATOR_URL, data=orjson.dumps(item),
                                  headers={"Content-Type": "application/json"},
                                  timeout=aiohttp.ClientTimeout(total=3)) as resp:
            await resp.read()
            if resp.status == 202:
                return
            if resp.status != 422:
                _gpu_down_until = time.monotonic() + GPU_RETRY_SECONDS
    except Exception:
        _gpu_down_until = time.monotonic() + GPU_RETRY_SECONDS
    _stats["gpu_failed"] += 1
    _stats["to_gpu"] -= 1
    _forward_foreign(raw_item)


def _forward_gpu(item: dict, raw_item: dict):
    _stats["to_gpu"] += 1
    task = asyncio.create_task(_send_gpu(item, raw_item))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


def _parse_created_at(created_at_str: str) -> datetime | None:
    if not created_at_str:
        return None
    try:
        dt = datetime.fromisoformat(str(created_at_str).strip().replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fmt_created_at(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _detect_lang(content: str) -> tuple[str | None, float]:
    try:
        res = detect_langs(content)
    except LangDetectException:
        return None, 0.0
    if not res:
        return None, 0.0
    return res[0].lang, float(res[0].prob)


def _translate_block_reason(lang: str, prob: float, length: int) -> str | None:
    if not TRANSLATE:
        return "translate_off"
    if not _tr_state["ok"]:
        return "translator_down"
    if prob < LANG_CONFIDENCE:
        return "low_confidence"
    if length > TRANSLATE_MAX_CHARS:
        return "too_long"
    if _norm_lang(lang) not in _tr_state["installed"]:
        return "no_pack"
    if _tr_state["fill"] >= TRANSLATE_BUSY_FILL:
        return "queue_full"
    return None


async def _get_json(url: str) -> dict | None:
    try:
        async with _session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
            if resp.status != 200:
                return None
            return await resp.json()
    except Exception:
        return None


async def _none():
    return None


async def _poll_translator():
    url = urljoin(TRANSLATOR_URL, "health")
    was_ok = None
    while True:
        data = await _get_json(url)
        ok = bool(data and data.get("ready") and data.get("enabled"))
        if ok:
            _tr_state["fill"] = float(data.get("fill", 0.0))
            _tr_state["installed"] = frozenset(data.get("installed", []))
        _tr_state["ok"] = ok
        if ok != was_ok:
            if ok:
                log.info(f"✅ translator доступен, языки: {sorted(_tr_state['installed'])}")
            else:
                log.error(f"❌ translator недоступен: {url} — non-en уходит на foreign")
            was_ok = ok
        await asyncio.sleep(TRANSLATE_POLL_SEC)


def _bpipe_summary(bpipe_data: list) -> dict:
    live = [b for b in bpipe_data if b]
    queue = sum(b.get("queue", 0) for b in live)
    mx = sum(b.get("max", 0) for b in live)
    return {
        "queue": queue,
        "max": mx,
        "fill": round(queue / mx, 3) if mx else 0.0,
        "inflight": sum(b.get("inflight", 0) for b in live),
        "dropped": sum(b.get("dropped", 0) for b in live),
        "instances": len(BPIPE_URLS),
        "instances_up": len(live),
    }


async def handle_queue(request: web.Request) -> web.Response:
    upipe_data, *bpipe_data = await asyncio.gather(
        _get_json(urljoin(UPIPE_URL, "health")),
        *[_get_json(urljoin(u, "queue")) for u in BPIPE_URLS],
    )

    degraded = upipe_data is None or any(b is None or not b.get("ready") for b in bpipe_data)

    upipe_queue = int((upipe_data or {}).get("queue", 0))
    upipe_fill = upipe_queue / UPIPE_QUEUE_LIMIT if UPIPE_QUEUE_LIMIT else 0.0

    live = [b for b in bpipe_data if b]
    bpipe_queue = sum(b.get("queue", 0) for b in live)
    bpipe_max = sum(b.get("max", 0) for b in live)
    bpipe_inflight = sum(b.get("inflight", 0) for b in live)
    bpipe_fill = bpipe_queue / bpipe_max if bpipe_max else 0.0

    fill = max(upipe_fill, bpipe_fill)

    if degraded or fill >= QUEUE_HIGH_FILL:
        action = "slow_down"
    elif fill <= QUEUE_LOW_FILL:
        action = "speed_up"
    else:
        action = "hold"

    return web.json_response({
        "action": action,
        "fill": round(fill, 3),
        "low": QUEUE_LOW_FILL,
        "high": QUEUE_HIGH_FILL,
        "degraded": degraded,
        "upipe_queue": upipe_queue,
        "upipe_limit": UPIPE_QUEUE_LIMIT,
        "bpipe_queue": bpipe_queue,
        "bpipe_max": bpipe_max,
        "bpipe_inflight": bpipe_inflight,
        "bpipe_instances": len(BPIPE_URLS),
        "languages": len(_langs_seen),
    })


async def handle_stats(request: web.Request) -> web.Response:
    tr, upipe_data, tx, *bpipe_data = await asyncio.gather(
        _get_json(urljoin(TRANSLATOR_URL, "stats")) if TRANSLATE else _none(),
        _get_json(urljoin(UPIPE_URL, "health")),
        _get_json(urljoin(TRANSACTIONEER_URL, "stats")),
        *[_get_json(urljoin(u, "queue")) for u in BPIPE_URLS],
    )
    tr = tr or {}
    up_stats = (upipe_data or {}).get("stats", {})

    return web.json_response({
        "ts": int(time.time()),
        "uptime_s": int(time.time() - _started),
        "translate": {
            "enabled": TRANSLATE,
            "translator_up": bool(tr) if TRANSLATE else None,
            "rate_per_sec": tr.get("rate_per_sec", 0.0),
            "rate_window_s": tr.get("rate_window_s", 60),
            "translated_total": tr.get("stats", {}).get("translated", 0),
            "errors_total": tr.get("stats", {}).get("errors", 0),
            "queue": tr.get("queue", 0),
            "queue_max": tr.get("queue_max", 0),
            "fill": tr.get("fill", 0.0),
            "workers": tr.get("workers"),
            "installed": tr.get("installed", []),
            "sent_to_translator": _stats["to_translate"],
            "upipe_fallback_to_foreign": up_stats.get("translate_fallback", 0),
            "upipe_errors": up_stats.get("translate_errors", 0),
            "identical_dropped": up_stats.get("translate_identical", 0),
            "foreign_reasons": dict(_foreign_reasons),
            "gpu_sent_collector": _stats["to_gpu"],
            "gpu_sent_upipe": up_stats.get("gpu_sent", 0),
        },
        "languages": {
            "seen_count": len(_langs_seen),
            "seen": dict(_langs_seen),
            "translated_count": tr.get("languages_translated_count", 0),
            "installed_count": tr.get("installed_count", 0),
        },
        "bpipe": _bpipe_summary(bpipe_data),
        "upipe": {"up": upipe_data is not None, "queue": (upipe_data or {}).get("queue", 0), "limit": UPIPE_QUEUE_LIMIT, "stats": up_stats},
        "exorde": tx or {"error": "transactioneer unavailable"},
        "collector": _stats,
        "domains": _dom_snapshot(),
    })


def _url_domain(url: str) -> str:
    try:
        host = url.split("/", 3)[2].lower()
    except IndexError:
        return ""
    return host[4:] if host.startswith("www.") else host


def _url_in_scope(domain: str) -> bool:
    if not URL_DEDUP_DOMAINS:
        return True
    d = (domain or "").lower()
    return any(d == x or d.endswith("." + x) for x in URL_DEDUP_DOMAINS)


def _lru_add(od: OrderedDict, key, value, limit: int):
    od[key] = value
    od.move_to_end(key)
    while len(od) > limit:
        od.popitem(last=False)


def _parse_dups(data):
    if isinstance(data, dict):
        for k in ("duplicates", "duplicate_urls", "duplicate", "urls", "data", "result"):
            if isinstance(data.get(k), list):
                data = data[k]
                break
        else:
            return None
    if not isinstance(data, list):
        return None
    return {u for u in data if isinstance(u, str)}


def _url_submit(url: str, want_result: bool):
    if len(_url_pending) >= URL_DEDUP_QUEUE_MAX:
        _url_stats["overflow"] += 1
        return None
    fut = asyncio.get_running_loop().create_future() if want_result else None
    _url_pending.append((url, fut))
    _url_stats["queued"] += 1
    _url_evt.set()
    return fut


async def _url_is_duplicate(url: str) -> bool:
    ts = _known_dups.get(url)
    if ts is not None and time.time() - ts < URL_DEDUP_TTL:
        _url_stats["cache_hits"] += 1
        return True
    fut = _url_submit(url, True)
    if fut is None:
        return False
    try:
        return await asyncio.wait_for(fut, URL_DEDUP_WAIT)
    except asyncio.TimeoutError:
        _url_stats["timeouts"] += 1
        return False


def _url_resolve(batch: list, dup_set: set):
    for url, fut in batch:
        if fut is not None and not fut.done():
            fut.set_result(url in dup_set)


async def _url_flush(batch: list):
    urls = list(dict.fromkeys(u for u, _ in batch))
    t0 = time.monotonic()
    dups = None
    try:
        async with _session.post(
            f"{URL_DEDUP_API}/v1/check",
            data=orjson.dumps({"urls": urls}),
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=URL_DEDUP_TIMEOUT),
        ) as resp:
            body = await resp.read()
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status}: {body[:120]!r}")
        if _url_stats["sample"] is None:
            _url_stats["sample"] = body[:400].decode("utf-8", "replace")
        dups = _parse_dups(orjson.loads(body))
        if dups is None:
            _url_stats["format_errors"] += 1
            _url_stats["last_error"] = f"unexpected response: {body[:120]!r}"
    except Exception as e:
        _url_stats["errors"] += 1
        _url_stats["last_error"] = f"{type(e).__name__}: {e}"[:200]
    if dups is None:
        _url_resolve(batch, set())
        return
    ms = (time.monotonic() - t0) * 1000
    norm = {d.rstrip("/") for d in dups}
    dup_set = {u for u in urls if u in dups or u.rstrip("/") in norm}
    _url_stats["unmatched"] += len(norm - {u.rstrip("/") for u in urls})
    now = time.time()
    for u in urls:
        d = _url_domain(u)
        if d not in _url_dom and len(_url_dom) >= 200:
            d = "other"
        rec = _url_dom.setdefault(d, [0, 0])
        rec[0] += 1
        if u in dup_set:
            rec[1] += 1
            if URL_DEDUP_MODE == "enforce":
                _lru_add(_known_dups, u, now, URL_DEDUP_CACHE_MAX)
            if u in _sent_urls:
                _url_stats["self_dups"] += 1
    _url_stats["requests"] += 1
    _url_stats["total_ms"] += ms
    _url_stats["last_ms"] = round(ms, 1)
    _url_stats["checked"] += len(urls)
    _url_stats["duplicates"] += len(dup_set)
    _url_resolve(batch, dup_set)
    if _url_stats["requests"] % 50 == 1:
        log.info(
            f"🔎 url-dedup {URL_DEDUP_MODE}: checked={_url_stats['checked']} dup={_url_stats['duplicates']} "
            f"pending={len(_url_pending)} last={ms:.0f}ms"
        )


async def _url_worker():
    while True:
        await _url_evt.wait()
        if len(_url_pending) < URL_DEDUP_BATCH:
            await asyncio.sleep(URL_DEDUP_FLUSH_SEC)
        batch = [_url_pending.popleft() for _ in range(min(URL_DEDUP_BATCH, len(_url_pending)))]
        if not _url_pending:
            _url_evt.clear()
        if batch:
            await _url_flush(batch)


def _url_dedup_snapshot() -> dict:
    st = _url_stats
    checked, reqs = st["checked"], st["requests"]
    top = sorted(_url_dom.items(), key=lambda kv: kv[1][0], reverse=True)[:15]
    return {
        "mode": URL_DEDUP_MODE,
        "workers": URL_DEDUP_WORKERS,
        "batch": URL_DEDUP_BATCH,
        "domains": URL_DEDUP_DOMAINS or "all",
        "pending": len(_url_pending),
        "queued": st["queued"],
        "checked": checked,
        "duplicates": st["duplicates"],
        "hit_rate": round(st["duplicates"] / checked, 4) if checked else None,
        "by_domain": {d: {"checked": c, "dups": k, "rate": round(k / c, 3) if c else None} for d, (c, k) in top},
        "self_dups": st["self_dups"],
        "cache_hits": st["cache_hits"],
        "requests": reqs,
        "errors": st["errors"],
        "format_errors": st["format_errors"],
        "timeouts": st["timeouts"],
        "overflow": st["overflow"],
        "unmatched": st["unmatched"],
        "avg_ms": round(st["total_ms"] / reqs, 1) if reqs else None,
        "last_ms": st["last_ms"],
        "last_error": st["last_error"],
        "sample": st["sample"],
    }


async def _process_item(item) -> str:
    if not isinstance(item, dict) or not isinstance(item.get("content", ""), str):
        return "invalid_item"

    _stats["received"] += 1
    content = item.get("content", "")
    domain = item.get("domain", "")
    _dom_record(domain, "r")

    raw_item = dict(item)
    item["author"] = _hash_author(item.get("author", ""))

    ext_id = item.get("external_id", "")
    if ext_id:
        if ext_id in _seen_ids:
            _stats["filtered_dup"] += 1
            return "duplicate"
        _seen_ids[ext_id] = True
        _seen_ids.move_to_end(ext_id)
        if len(_seen_ids) > _DEDUP_MAX_SIZE:
            for _ in range(_DEDUP_MAX_SIZE // 2):
                _seen_ids.popitem(last=False)

    if len(content) < MIN_TEXT_LEN:
        return "skipped_short"

    gpu_route = False
    if LANG_FILTER:
        lang, prob = _detect_lang(content)
        if lang == LANG_FILTER:
            item["lang_ok"] = True
        elif LANG_FILTER == "en" and _looks_english(content):
            _stats["rescued_en"] += 1
            if _stats["rescued_en"] % 50 == 1:
                log.info(f"♻️ rescued en (langdetect={lang} p={prob:.2f}): {content[:100]!r}")
            item["lang_ok"] = True
        else:
            reason = None
            if lang is not None:
                _langs_seen[lang] += 1
                reason = _translate_block_reason(lang, prob, len(content))
            if lang is not None and reason is None:
                item["detected_lang"] = lang
                item["foreign_author"] = raw_item.get("author", "")
                _stats["to_translate"] += 1
            elif lang is not None and _gpu_route_ok(reason):
                item["detected_lang"] = lang
                gpu_route = True
            else:
                _stats["filtered_lang"] += 1
                if lang is not None:
                    _foreign_reasons[reason] += 1
                    _forward_foreign(raw_item)
                return "filtered_lang"

    if len(content) > MAX_TEXT_LEN:
        content = content[:MAX_TEXT_LEN]
        item["content"] = content
        _stats["truncated"] += 1

    created_at_str = item.get("created_at", "")
    tweet_dt = _parse_created_at(created_at_str)
    if tweet_dt is not None:
        age_seconds = (datetime.now(timezone.utc) - tweet_dt).total_seconds()
        if age_seconds > MAX_OLDNESS_SECONDS:
            _stats["filtered_old"] += 1
            return "filtered_old"
        # exorde_data принимает только ...SS(.fff)Z; threads.net шлёт +00:00
        item["created_at"] = _fmt_created_at(tweet_dt)
    else:
        log.warning(f"⚠️ Не удалось распарсить created_at: {created_at_str!r}")

    url = item.get("url")
    if URL_DEDUP_MODE != "off" and isinstance(url, str) and url and _url_in_scope(domain):
        if URL_DEDUP_MODE == "enforce":
            if await _url_is_duplicate(url):
                _stats["filtered_url_dup"] += 1
                return "url_duplicate"
        else:
            _url_submit(url, False)

    if gpu_route:
        _forward_gpu(item, raw_item)
        _dom_record(domain, "f")
        return "to_gpu"

    if await forward_to_upipe(item):
        _stats["forwarded"] += 1
        if URL_DEDUP_MODE != "off" and isinstance(url, str) and url:
            _lru_add(_sent_urls, url, True, URL_DEDUP_CACHE_MAX)
        _dom_record(domain, "f")
        if _stats["forwarded"] % 50 == 0:
            log.info(
                f"📊 recv={_stats['received']} fwd={_stats['forwarded']} "
                f"old={_stats['filtered_old']} dup={_stats['filtered_dup']} lang={_stats['filtered_lang']} "
                f"rescued={_stats['rescued_en']} transl={_stats['to_translate']} gpu={_stats['to_gpu']} "
                f"foreign={_stats['foreign_sent']} dropped={_stats['dropped']}"
            )
        return "OK"

    _stats["errors"] += 1
    _stats["dropped"] += 1
    if _stats["dropped"] % 10 == 0:
        log.warning(f"⚠️  Потеряно элементов (upipe недоступен/503): {_stats['dropped']}")
    return "dropped"


async def handle_store_item(request: web.Request) -> web.Response:
    try:
        item = await request.json()
    except Exception as e:
        return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)

    status = await _process_item(item)
    if status == "invalid_item":
        return web.json_response({"error": "expected JSON object with string 'content'"}, status=400)
    return web.json_response({"message": "OK" if status == "dropped" else status}, status=200)


async def handle_store_items(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception as e:
        return web.json_response({"error": f"Invalid JSON: {e}"}, status=400)

    if isinstance(data, dict) and isinstance(data.get("items"), list):
        data = data["items"]
    if not isinstance(data, list):
        return web.json_response({"error": "expected JSON array or {\"items\": [...]}"}, status=400)

    statuses = await asyncio.gather(*(_process_item(i) for i in data))
    return web.json_response({
        "message": "OK",
        "count": len(data),
        "results": dict(Counter(statuses)),
    }, status=200)


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok",
        "stats": _stats,
        "url_dedup": _url_dedup_snapshot(),
        "upipe": UPIPE_URL,
        "lang_filter": LANG_FILTER or None,
        "translate": TRANSLATE,
        "gpu_translator": GPU_TRANSLATOR_URL or None,
        "foreign_forward_url": FOREIGN_FORWARD_URL or None,
    })


@web.middleware
async def cors_middleware(request: web.Request, handler):
    resp = await handler(request)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


async def on_startup(app: web.Application):
    global _session
    connector = aiohttp.TCPConnector(limit=20, keepalive_timeout=60)
    _session = aiohttp.ClientSession(connector=connector)
    if TRANSLATE:
        app["poller"] = asyncio.create_task(_poll_translator())
    global _url_evt
    _url_evt = asyncio.Event()
    if URL_DEDUP_MODE in ("shadow", "enforce"):
        app["url_workers"] = [asyncio.create_task(_url_worker()) for _ in range(URL_DEDUP_WORKERS)]
    log.info(f"🚀 Collector запущен на порту {COLLECTOR_PORT}")
    log.info(f"   Пересылает в upipe: {UPIPE_URL}")
    log.info(f"   Опрос очередей bpipe: {BPIPE_URLS}")
    log.info(
        f"   URL dedup: {URL_DEDUP_MODE} api={URL_DEDUP_API} batch={URL_DEDUP_BATCH} "
        f"workers={URL_DEDUP_WORKERS} domains={URL_DEDUP_DOMAINS or 'all'}"
    )
    log.info(f"   Языковой фильтр: {LANG_FILTER or 'выключен'}")
    log.info(f"   TRANSLATE={TRANSLATE} translator={TRANSLATOR_URL if TRANSLATE else '-'}")
    log.info(f"   GPU translator: {GPU_TRANSLATOR_URL or 'выключен'} (причины: {sorted(GPU_ROUTE_REASONS)})")
    log.info(f"   Не-{LANG_FILTER} без перевода → {FOREIGN_FORWARD_URL or 'выключено'}")
    log.info(f"   Макс. возраст твита: {MAX_OLDNESS_SECONDS}с ({MAX_OLDNESS_SECONDS/3600:.1f}ч)")


async def on_shutdown(app: web.Application):
    global _session
    if "poller" in app:
        app["poller"].cancel()
    for t in app.get("url_workers", []):
        t.cancel()
    if _session:
        await _session.close()
    log.info(f"📊 Итог: {_stats}")


app = web.Application(client_max_size=50 * 1024 * 1024, middlewares=[cors_middleware])
app.router.add_post("/store_item", handle_store_item)
app.router.add_post("/store_items", handle_store_items)
app.router.add_get("/queue", handle_queue)
app.router.add_get("/stats", handle_stats)
app.router.add_get("/health", handle_health)
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=COLLECTOR_PORT, print=None)
