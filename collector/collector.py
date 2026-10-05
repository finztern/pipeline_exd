import asyncio
import hashlib
import logging
import os
import sys
import time
from collections import Counter, OrderedDict
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
FOREIGN_FORWARD_URL = os.getenv("FOREIGN_FORWARD_URL", "http://192.168.0.101:9000/store_item").strip()

TRANSLATE             = os.getenv("TRANSLATE", "false").lower() == "true"
TRANSLATOR_URL        = os.getenv("TRANSLATOR_URL", "http://translator:8003/")
TRANSACTIONEER_URL    = os.getenv("TRANSACTIONEER_URL", "http://transactioneer:8002/")
TRANSLATE_MAX_CHARS   = int(os.getenv("TRANSLATE_MAX_CHARS", "2000"))
TRANSLATE_BUSY_FILL   = float(os.getenv("TRANSLATE_BUSY_FILL", "0.9"))
TRANSLATE_POLL_SEC    = float(os.getenv("TRANSLATE_POLL_SECONDS", "1.0"))
LANG_CONFIDENCE       = float(os.getenv("LANG_CONFIDENCE_THRESHOLD", "0.90"))

LANG_ALIASES = {"zh-cn": "zh", "zh-tw": "zt", "no": "nb"}

_seen_ids: OrderedDict = OrderedDict()
_DEDUP_MAX_SIZE = 100_000
_started = time.time()

_stats = {
    "received": 0, "forwarded": 0, "filtered_old": 0, "filtered_dup": 0, "filtered_lang": 0,
    "foreign_sent": 0, "to_translate": 0, "truncated": 0, "errors": 0, "dropped": 0,
}
_langs_seen: Counter = Counter()
_foreign_reasons: Counter = Counter()
_tr_state = {"ok": False, "fill": 1.0, "installed": frozenset()}
_session: aiohttp.ClientSession | None = None
_bg_tasks: set = set()


def _norm_lang(lang: str) -> str:
    return LANG_ALIASES.get(lang, lang.split("-")[0])


def _hash_author(author: str) -> str:
    if not author:
        return ""
    return hashlib.sha1(author.encode("utf-8")).hexdigest()


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


def _parse_created_at(created_at_str: str) -> datetime | None:
    if not created_at_str:
        return None
    try:
        s = created_at_str.rstrip("Z")
        if "." in s:
            dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%f")
        else:
            dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


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
    })


async def _process_item(item) -> str:
    if not isinstance(item, dict) or not isinstance(item.get("content", ""), str):
        return "invalid_item"

    _stats["received"] += 1
    content = item.get("content", "")

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

    if LANG_FILTER:
        lang, prob = _detect_lang(content)
        if lang != LANG_FILTER:
            reason = None
            if lang is not None:
                _langs_seen[lang] += 1
                reason = _translate_block_reason(lang, prob, len(content))
            if lang is not None and reason is None:
                item["detected_lang"] = lang
                item["foreign_author"] = raw_item.get("author", "")
                _stats["to_translate"] += 1
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
    else:
        log.warning(f"⚠️ Не удалось распарсить created_at: {created_at_str!r}")

    if await forward_to_upipe(item):
        _stats["forwarded"] += 1
        if _stats["forwarded"] % 50 == 0:
            log.info(
                f"📊 recv={_stats['received']} fwd={_stats['forwarded']} "
                f"old={_stats['filtered_old']} dup={_stats['filtered_dup']} lang={_stats['filtered_lang']} "
                f"transl={_stats['to_translate']} foreign={_stats['foreign_sent']} dropped={_stats['dropped']}"
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
        "upipe": UPIPE_URL,
        "lang_filter": LANG_FILTER or None,
        "translate": TRANSLATE,
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
    log.info(f"🚀 Collector запущен на порту {COLLECTOR_PORT}")
    log.info(f"   Пересылает в upipe: {UPIPE_URL}")
    log.info(f"   Опрос очередей bpipe: {BPIPE_URLS}")
    log.info(f"   Языковой фильтр: {LANG_FILTER or 'выключен'}")
    log.info(f"   TRANSLATE={TRANSLATE} translator={TRANSLATOR_URL if TRANSLATE else '-'}")
    log.info(f"   Не-{LANG_FILTER} без перевода → {FOREIGN_FORWARD_URL or 'выключено'}")
    log.info(f"   Макс. возраст твита: {MAX_OLDNESS_SECONDS}с ({MAX_OLDNESS_SECONDS/3600:.1f}ч)")


async def on_shutdown(app: web.Application):
    global _session
    if "poller" in app:
        app["poller"].cancel()
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
