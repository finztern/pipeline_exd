import asyncio
import itertools
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import orjson
from aiohttp import web

sys.path.insert(0, os.path.dirname(__file__))

from exorde_data import (
    Item, CreatedAt, Content, Domain, Url, Title, ExternalId, Author, ExternalParentId,
    Translation, Language, Translated,
)
from process import process
from translate import NonEnglishError
from lab_initialization import lab_initialization
from translator_client import (
    translate_remote, TranslatorBusy, TranslatorUnavailable, TranslatorRejected,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [upipe] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

UPIPE_PORT   = int(os.getenv("UPIPE_PORT", "5981"))
WORKERS      = int(os.getenv("UPIPE_WORKERS", "4"))
QUEUE_LIMIT  = int(os.getenv("UPIPE_QUEUE_LIMIT", "200"))
MAX_DEPTH_CLASSIFICATION = int(os.getenv("MAX_DEPTH_CLASSIFICATION", "2"))

TRANSLATE           = os.getenv("TRANSLATE", "false").lower() == "true"
TRANSLATOR_URL      = os.getenv("TRANSLATOR_URL", "http://translator:8003/")
TRANSLATE_TIMEOUT   = float(os.getenv("TRANSLATE_TIMEOUT_SECONDS", "30"))
FOREIGN_FORWARD_URL = os.getenv("FOREIGN_FORWARD_URL", "").strip()
GPU_TRANSLATOR_URL  = os.getenv("GPU_TRANSLATOR_URL", "").strip()
GPU_RETRY_SECONDS   = float(os.getenv("GPU_RETRY_SECONDS", "30"))

def _parse_bpipe_urls() -> list[str]:
    urls_env = os.getenv("BPIPE_URLS", "")
    if urls_env:
        return [u.strip() for u in urls_env.split(",") if u.strip()]
    single = os.getenv("BPIPE_URL", "http://127.0.0.1:7995/")
    return [single]

BPIPE_URLS: list[str] = _parse_bpipe_urls()

_session: aiohttp.ClientSession | None = None
_lab_config: dict | None = None
_process_queue: asyncio.Queue | None = None
_thread_pool: ThreadPoolExecutor | None = None

_bpipe_cycle: itertools.cycle | None = None
_bpipe_lock = asyncio.Lock()
_bg_tasks: set = set()
_gpu_down_until = 0.0

_stats = {
    "received": 0, "processed": 0, "forwarded": 0, "errors": 0, "dropped": 0, "filtered_lang": 0,
    "translated": 0, "translate_fallback": 0, "translate_errors": 0, "translate_identical": 0,
    "foreign_sent": 0, "gpu_sent": 0, "gpu_returned": 0,
}


def _process_sync(item: Item, lab_config: dict, translation: Translation | None = None) -> dict:
    processed = process(item, lab_config, MAX_DEPTH_CLASSIFICATION, translation)
    return {
        "item": {
            "created_at":         str(processed.item.get("created_at", "")),
            "title":              str(processed.item.get("title", "")),
            "content":            str(processed.item.get("content", "")),
            "domain":             str(processed.item.get("domain", "")),
            "url":                str(processed.item.get("url", "")),
            "external_id":        str(processed.item.get("external_id", "")),
            "external_parent_id": str(processed.item.get("external_parent_id", "")),
            "author":             str(processed.item.get("author", "")),
            "username":           str(processed.item.get("username", "")) if processed.item.get("username") else "",
        },
        "translation": {
            "language":    str(processed.translation.language),
            "translation": str(processed.translation.translation),
        },
        "top_keywords": list(processed.top_keywords),
        "classification": {
            "label": str(processed.classification.label),
            "score": float(processed.classification.score),
        },
    }


async def forward_to_bpipe(payload: dict) -> bool:
    global _session, _bpipe_cycle
    if _session is None or _bpipe_cycle is None:
        return False

    async with _bpipe_lock:
        url = next(_bpipe_cycle)

    try:
        async with _session.post(
            url,
            data=orjson.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            ok = 200 <= resp.status < 300
            if not ok:
                log.warning(f"⚠️  bpipe {url} вернул {resp.status}")
            return ok
    except aiohttp.ClientConnectorError:
        log.error(f"❌ bpipe недоступен: {url}")
        return False
    except Exception as e:
        log.debug(f"Ошибка отправки в bpipe: {e}")
        return False


async def _send_foreign(payload: dict):
    try:
        async with _session.post(
            FOREIGN_FORWARD_URL,
            data=orjson.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            await resp.read()
    except Exception as e:
        log.debug(f"foreign forward error: {e}")


def _forward_foreign(raw_item: dict):
    if not FOREIGN_FORWARD_URL or _session is None:
        return
    payload = {k: v for k, v in raw_item.items() if k not in ("detected_lang", "foreign_author")}
    payload["author"] = raw_item.get("foreign_author", raw_item.get("author", ""))
    _stats["foreign_sent"] += 1
    task = asyncio.create_task(_send_foreign(payload))
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


async def _forward_gpu(raw_item: dict) -> bool:
    global _gpu_down_until
    if not GPU_TRANSLATOR_URL or _session is None or time.monotonic() < _gpu_down_until:
        return False
    payload = {k: v for k, v in raw_item.items() if k != "foreign_author"}
    try:
        async with _session.post(
            GPU_TRANSLATOR_URL,
            data=orjson.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=3),
        ) as resp:
            await resp.read()
            if resp.status == 202:
                _stats["gpu_sent"] += 1
                return True
            if resp.status == 422:
                return False
    except Exception:
        pass
    _gpu_down_until = time.monotonic() + GPU_RETRY_SECONDS
    return False


async def _translate_or_fallback(item: Item, raw_item: dict, lang: str) -> Translation | None:
    text = str(item.get("content", ""))
    try:
        out = await translate_remote(_session, TRANSLATOR_URL, text, lang, TRANSLATE_TIMEOUT)
    except TranslatorBusy:
        _stats["translate_fallback"] += 1
        if not await _forward_gpu(raw_item):
            _forward_foreign(raw_item)
        return None
    except (TranslatorUnavailable, TranslatorRejected) as e:
        _stats["translate_errors"] += 1
        _stats["translate_fallback"] += 1
        if _stats["translate_errors"] % 20 == 1:
            log.error(f"❌ translator: {e} (ошибок всего: {_stats['translate_errors']})")
        _forward_foreign(raw_item)
        return None

    if out.strip() == text.strip():
        _stats["translate_identical"] += 1
        return None

    _stats["translated"] += 1
    return Translation(language=Language(lang.split("-")[0]), translation=Translated(out))


async def _translate_then_requeue(item: Item, raw_item: dict, lang: str):
    try:
        translation = await _translate_or_fallback(item, raw_item, lang)
        if translation is None:
            return
        raw_item["pretranslated"] = str(translation.translation)
        await _process_queue.put((item, raw_item))
    except Exception as e:
        _stats["errors"] += 1
        log.warning(f"⚠️ Ошибка фонового перевода: {e}")


def _spawn(coro):
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


async def worker_loop(worker_id: int):
    global _stats
    log.info(f"👷 Upipe-воркер #{worker_id} запущен")

    loop = asyncio.get_event_loop()
    while True:
        try:
            item, raw_item = await _process_queue.get()

            try:
                translation = None
                lang = (raw_item.get("detected_lang") or "").lower()
                pre = raw_item.get("pretranslated")
                if pre:
                    translation = Translation(
                        language=Language(lang.split("-")[0] if lang else "en"),
                        translation=Translated(pre),
                    )
                elif TRANSLATE and lang and lang != "en":
                    _spawn(_translate_then_requeue(item, raw_item, lang))
                    continue

                payload = await loop.run_in_executor(
                    _thread_pool,
                    _process_sync,
                    item,
                    _lab_config,
                    translation,
                )
                _stats["processed"] += 1

                if raw_item.get("username"):
                    payload["item"]["username"] = raw_item["username"]
                if raw_item.get("summary"):
                    payload["item"]["summary"] = raw_item["summary"]

                ok = await forward_to_bpipe(payload)
                if ok:
                    _stats["forwarded"] += 1
                    log.debug(
                        f"✅ [{worker_id}] → bpipe | {raw_item.get('url', '')[:60]}"
                    )
                else:
                    _stats["errors"] += 1

            except NonEnglishError as e:
                _stats["filtered_lang"] += 1
                log.debug(f"🌐 [{worker_id}] Не-английский текст отброшен: {e}")

            except Exception as e:
                _stats["errors"] += 1
                log.warning(f"⚠️ [{worker_id}] Ошибка обработки: {e}")

            finally:
                _process_queue.task_done()

            if _stats["forwarded"] % 25 == 0 and _stats["forwarded"] > 0:
                log.info(
                    f"📊 recv={_stats['received']} proc={_stats['processed']} "
                    f"fwd={_stats['forwarded']} err={_stats['errors']} "
                    f"lang={_stats['filtered_lang']} dropped={_stats['dropped']} "
                    f"transl={_stats['translated']} tr_fallback={_stats['translate_fallback']} "
                    f"gpu_sent={_stats['gpu_sent']} gpu_ret={_stats['gpu_returned']}"
                )

        except Exception as e:
            log.error(f"Ошибка воркера #{worker_id}: {e}")
            await asyncio.sleep(1)


async def handle_receive_item(request: web.Request) -> web.Response:
    global _stats
    try:
        raw_item = await request.json()
    except Exception as e:
        return web.Response(text=f"bad json: {e}", status=400)

    _stats["received"] += 1
    if raw_item.get("pretranslated"):
        _stats["gpu_returned"] += 1

    try:
        external_parent_id = raw_item.get("external_parent_id") or ""
        item = Item(
            created_at=CreatedAt(raw_item["created_at"]),
            title=Title(raw_item.get("title", "")),
            content=Content(raw_item["content"]),
            domain=Domain(raw_item["domain"]),
            url=Url(raw_item["url"]),
            external_id=ExternalId(raw_item.get("external_id", "")),
            external_parent_id=ExternalParentId(external_parent_id),
            author=Author(raw_item.get("author", "")),
        )
    except Exception as e:
        log.warning(f"Ошибка создания Item: {e} | данные: {raw_item}")
        return web.Response(text="invalid_item", status=400)

    if _process_queue.qsize() >= QUEUE_LIMIT:
        _stats["dropped"] += 1
        if _stats["dropped"] % 10 == 0:
            log.warning(
                f"🗑️  Очередь переполнена — дропнуто элементов: {_stats['dropped']} "
                f"(queue={_process_queue.qsize()}/{QUEUE_LIMIT})"
            )
        return web.Response(text="queue_full", status=503)

    await _process_queue.put((item, raw_item))
    return web.Response(text="received")


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok",
        "queue":  _process_queue.qsize() if _process_queue else 0,
        "bpipe_urls": BPIPE_URLS,
        "translate": {"enabled": TRANSLATE, "url": TRANSLATOR_URL if TRANSLATE else None},
        "gpu_translator": GPU_TRANSLATOR_URL or None,
        "stats":  _stats,
    })


async def on_startup(app: web.Application):
    global _session, _lab_config, _process_queue, _thread_pool, _bpipe_cycle

    log.info("🔬 Инициализация upipe...")
    log.info(f"   bpipe инстансов: {len(BPIPE_URLS)} → {BPIPE_URLS}")
    log.info(f"   TRANSLATE={TRANSLATE} translator={TRANSLATOR_URL if TRANSLATE else '-'}")
    log.info(f"   GPU translator: {GPU_TRANSLATOR_URL or 'выключен'}")

    loop = asyncio.get_event_loop()
    _thread_pool = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="upipe_worker")
    _lab_config = await loop.run_in_executor(None, lab_initialization)

    _process_queue = asyncio.Queue()
    _bpipe_cycle   = itertools.cycle(BPIPE_URLS)

    connector = aiohttp.TCPConnector(limit=40, keepalive_timeout=60)
    _session = aiohttp.ClientSession(connector=connector)

    for i in range(WORKERS):
        asyncio.create_task(worker_loop(i))

    log.info(f"✅ Upipe запущен на порту {UPIPE_PORT} ({WORKERS} воркеров)")
    log.info(f"   Round-robin → {BPIPE_URLS}")


async def on_shutdown(app: web.Application):
    global _session, _thread_pool
    if _session:
        await _session.close()
    if _thread_pool:
        _thread_pool.shutdown(wait=False)
    log.info(f"📊 Итог upipe: {_stats}")


app = web.Application(client_max_size=50 * 1024 * 1024)
app.router.add_post("/", handle_receive_item)
app.router.add_get("/", handle_health)
app.router.add_get("/health", handle_health)
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=UPIPE_PORT, print=None)
