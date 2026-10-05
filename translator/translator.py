import asyncio
import logging
import os
import sys
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor

from aiohttp import web

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [translator] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PORT          = int(os.getenv("TRANSLATOR_PORT", "8003"))
ENABLED       = os.getenv("TRANSLATE", "false").lower() == "true"
QUEUE_SIZE    = int(os.getenv("TRANSLATE_QUEUE_SIZE", "200"))
BATCH_SIZE    = max(1, int(os.getenv("TRANSLATE_BATCH_SIZE", "8")))
WORKERS       = max(1, int(os.getenv("TRANSLATE_WORKERS", "3")))
TIMEOUT       = float(os.getenv("TRANSLATE_TIMEOUT_SECONDS", "30"))
MAX_CHARS     = int(os.getenv("TRANSLATE_MAX_CHARS", "2000"))
LANG_PACKS    = [c.strip().lower() for c in os.getenv("ARGO_LANG_PACKS", "es,pt,fr,de,it,ru,ja").split(",") if c.strip()]
USE_PROXY     = os.getenv("USE_PROXY", "false").lower() == "true"
PROXY_URL     = os.getenv("PROXY_URL", "")
INSTALL_TRIES = 3

LANG_ALIASES = {"zh-cn": "zh", "zh-tw": "zt", "no": "nb"}

_translations: dict = {}
_queue: asyncio.Queue | None = None
_pool: ThreadPoolExecutor | None = None
_ready = False
_init_error = ""
_started = time.monotonic()
_done_ts: deque = deque(maxlen=50000)
_lang_counts: Counter = Counter()
_stats = {
    "translated": 0,
    "errors": 0,
    "timeouts": 0,
    "rejected_busy": 0,
    "rejected_no_pack": 0,
    "rejected_too_long": 0,
}


def _norm(lang: str) -> str:
    lang = lang.strip().lower()
    return LANG_ALIASES.get(lang, lang.split("-")[0])


def _install_and_load() -> dict:
    saved = {k: os.environ.get(k) for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy")}
    if USE_PROXY and PROXY_URL:
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ[k] = PROXY_URL
        os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost,127.0.0.1"

    try:
        import argostranslate.package as pkg
        import argostranslate.translate as tr

        installed = {(p.from_code, p.to_code) for p in pkg.get_installed_packages()}
        missing = [c for c in LANG_PACKS if (c, "en") not in installed]

        if missing:
            log.info(f"Устанавливаю языковые пакеты: {missing}")
            index_ok = False
            for attempt in range(INSTALL_TRIES):
                try:
                    pkg.update_package_index()
                    index_ok = True
                    break
                except Exception as e:
                    log.error(f"Индекс пакетов Argos недоступен (попытка {attempt + 1}/{INSTALL_TRIES}): {e}")
                    time.sleep(5 * (attempt + 1))
            if index_ok:
                available = pkg.get_available_packages()
                for code in missing:
                    cand = next((p for p in available if p.from_code == code and p.to_code == "en"), None)
                    if cand is None:
                        log.error(f"Пакета {code}->en нет в индексе Argos")
                        continue
                    try:
                        pkg.install_from_path(cand.download())
                        log.info(f"Установлен пакет {code}->en")
                    except Exception as e:
                        log.error(f"Не удалось установить {code}->en: {e}")

        langs = tr.get_installed_languages()
        en = next((l for l in langs if l.code == "en"), None)
        if en is None:
            raise RuntimeError("английский не найден среди установленных языков Argos")

        result = {}
        for l in langs:
            if l.code == "en" or l.code not in LANG_PACKS:
                continue
            t = l.get_translation(en)
            if t is None:
                continue
            t.translate("Hello")
            result[l.code] = t
        return result
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_group(lang: str, texts: list) -> list:
    t = _translations[lang]
    out = []
    for text in texts:
        try:
            out.append(t.translate(text))
        except Exception as e:
            out.append(e)
    return out


def _record_done(lang: str):
    now = time.monotonic()
    _done_ts.append(now)
    while _done_ts and _done_ts[0] < now - 60:
        _done_ts.popleft()
    _stats["translated"] += 1
    _lang_counts[lang] += 1


def _rate(window: float = 60.0) -> float:
    now = time.monotonic()
    cut = now - window
    n = sum(1 for t in _done_ts if t >= cut)
    return round(n / max(1.0, min(window, now - _started)), 2)


async def _worker(idx: int):
    loop = asyncio.get_running_loop()
    while True:
        try:
            first = await _queue.get()
            # доля очереди на воркера — иначе один воркер забирает весь батч и остальные простаивают
            take = min(BATCH_SIZE, -(-(_queue.qsize() + 1) // WORKERS))
            jobs = [first]
            while len(jobs) < take:
                try:
                    jobs.append(_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            groups: dict = {}
            for job in jobs:
                if not job[2].done():
                    groups.setdefault(job[0], []).append(job)

            for lang, items in groups.items():
                results = await loop.run_in_executor(_pool, _run_group, lang, [i[1] for i in items])
                for (_, _, fut), r in zip(items, results):
                    if fut.done():
                        continue
                    if isinstance(r, Exception):
                        _stats["errors"] += 1
                        log.error(f"Ошибка перевода [{lang}]: {r}")
                        fut.set_exception(r)
                    else:
                        _record_done(lang)
                        fut.set_result(r)
        except Exception as e:
            log.error(f"Воркер #{idx}: {e}", exc_info=True)
            await asyncio.sleep(1)


async def handle_translate(request: web.Request) -> web.Response:
    if not ENABLED:
        return web.json_response({"error": "disabled"}, status=503)
    if not _ready:
        return web.json_response({"error": "not_ready"}, status=503)

    try:
        data = await request.json()
        lang = _norm(str(data.get("lang", "")))
        text = data.get("text")
        if not lang or not isinstance(text, str) or not text.strip():
            raise ValueError
    except Exception:
        return web.json_response({"error": "bad_request"}, status=400)

    if len(text) > MAX_CHARS:
        _stats["rejected_too_long"] += 1
        return web.json_response({"error": "too_long"}, status=413)
    if lang not in _translations:
        _stats["rejected_no_pack"] += 1
        return web.json_response({"error": "no_pack"}, status=422)
    if _queue.full():
        _stats["rejected_busy"] += 1
        return web.json_response({"error": "queue_full"}, status=503)

    fut = asyncio.get_running_loop().create_future()
    _queue.put_nowait((lang, text, fut))
    try:
        out = await asyncio.wait_for(fut, TIMEOUT)
    except asyncio.TimeoutError:
        _stats["timeouts"] += 1
        return web.json_response({"error": "timeout"}, status=504)
    except Exception as e:
        return web.json_response({"error": f"translate_failed: {e}"}, status=500)
    return web.json_response({"translation": out, "lang": lang})


async def handle_health(request: web.Request) -> web.Response:
    size = _queue.qsize() if _queue else 0
    return web.json_response({
        "status": "ok" if _ready else "starting",
        "ready": _ready,
        "enabled": ENABLED,
        "queue": size,
        "max": QUEUE_SIZE,
        "fill": round(size / QUEUE_SIZE, 3) if QUEUE_SIZE else 0.0,
        "installed": sorted(_translations),
        "init_error": _init_error or None,
    }, status=200 if _ready else 503)


async def handle_stats(request: web.Request) -> web.Response:
    size = _queue.qsize() if _queue else 0
    return web.json_response({
        "enabled": ENABLED,
        "ready": _ready,
        "uptime_s": int(time.monotonic() - _started),
        "rate_per_sec": _rate(60.0),
        "rate_window_s": 60,
        "queue": size,
        "queue_max": QUEUE_SIZE,
        "fill": round(size / QUEUE_SIZE, 3) if QUEUE_SIZE else 0.0,
        "workers": WORKERS,
        "batch_size": BATCH_SIZE,
        "installed": sorted(_translations),
        "installed_count": len(_translations),
        "languages_translated_count": len(_lang_counts),
        "languages_translated": dict(_lang_counts),
        "stats": _stats,
        "init_error": _init_error or None,
    })


async def _init():
    global _translations, _ready, _init_error, _queue, _pool
    loop = asyncio.get_running_loop()
    if ENABLED:
        _pool = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="argos")
        try:
            _translations = await loop.run_in_executor(None, _install_and_load)
            log.info(f"✅ Языки готовы: {sorted(_translations)}")
        except Exception as e:
            _init_error = str(e)
            log.error(f"❌ Инициализация Argos упала: {e}", exc_info=True)
        _queue = asyncio.Queue(maxsize=QUEUE_SIZE)
        for i in range(WORKERS):
            asyncio.create_task(_worker(i))
    else:
        _queue = asyncio.Queue(maxsize=1)
        log.info("TRANSLATE=false — сервис в режиме простоя")
    _ready = True


async def on_startup(app: web.Application):
    app["init"] = asyncio.create_task(_init())


app = web.Application(client_max_size=5 * 1024 * 1024)
app.router.add_post("/translate", handle_translate)
app.router.add_get("/health", handle_health)
app.router.add_get("/stats", handle_stats)
app.on_startup.append(on_startup)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=PORT, print=None)
