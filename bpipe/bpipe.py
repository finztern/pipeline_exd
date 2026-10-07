"""
Bpipe (standalone) — принимает обработанные элементы от upipe,
запускает ML-пайплайн (эмбеддинги, сентимент, эмоции, тип текста, etc.),
формирует батчи и отправляет в transactioneer.

Слушает: POST / на BPIPE_PORT (по умолчанию 7995)
Отправляет в: TRANSACTIONEER_URL (по умолчанию http://127.0.0.1:8002/commit)
"""
import asyncio
import gc
import itertools
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from timeit import default_timer as timerit
from urllib.parse import urlparse, urljoin

import aiohttp
import orjson
from aiohttp import web

sys.path.insert(0, os.path.dirname(__file__))

from exorde_data import (
    CreatedAt, Content, Domain, Url, Title,
    Item, ExternalId, Author, ExternalParentId,
)
from exorde_compat import (
    Classification, Translation, Language, Translated,
    Keywords, Processed, Username, UserProfileUrl,
)
try:
    from exorde_data.get_live_configuration import get_live_configuration, LiveConfiguration
except ImportError:
    get_live_configuration = None
    LiveConfiguration = None
from process_batch import process_batch
from lab_initialization import lab_initialization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [bpipe] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BPIPE_PORT           = int(os.getenv("BPIPE_PORT", "7995"))
TRANSACTIONEER_URL   = os.getenv("TRANSACTIONEER_URL", "http://127.0.0.1:8002")
FIXED_BATCH_SIZE     = int(os.getenv("FIXED_BATCH_SIZE", "6"))
BATCH_TIMEOUT_SECS   = float(os.getenv("BATCH_TIMEOUT_SECONDS", "3.0"))
MAX_QUEUE_SIZE       = int(os.getenv("MAX_QUEUE_SIZE", "100"))
BPIPE_MAX_AGE_SECS   = float(os.getenv("BPIPE_MAX_AGE_SECONDS", "300"))

_process_queue: asyncio.PriorityQueue | None = None
_seq = itertools.count()
_lab_config: dict | None = None
_live_config = None
_session: aiohttp.ClientSession | None = None
_inflight = 0
_stats = {
    "received": 0,
    "batches_processed": 0,
    "items_sent": 0,
    "errors": 0,
    "dropped": 0,
    "stale": 0,
    "evicted": 0,
}


class TooBigError(Exception):
    pass


def _created_ts(created_at: str) -> float:
    s = (created_at or "").rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return time.time()


def _purge_stale() -> int:
    # Прямой доступ к внутренней куче PriorityQueue: фильтр + heapify за O(n), n<=MAX_QUEUE_SIZE.
    # join()/task_done() не используются, поэтому счётчики Queue не нарушаются.
    import heapq
    heap = _process_queue._queue
    cutoff = time.time() - BPIPE_MAX_AGE_SECS
    fresh = [e for e in heap if -e[0] >= cutoff]
    removed = len(heap) - len(fresh)
    if removed:
        heap[:] = fresh
        heapq.heapify(heap)
    return removed


def _evict_oldest_if_older(new_ts: float) -> bool:
    import heapq
    heap = _process_queue._queue
    if not heap:
        return False
    oldest = max(heap, key=lambda e: e[0])
    if -oldest[0] >= new_ts:
        return False
    heap.remove(oldest)
    heapq.heapify(heap)
    return True


async def send_batch_to_transactioneer(processed_batch: dict):
    global _session, _stats
    if _session is None:
        return

    commit_url = urljoin(TRANSACTIONEER_URL.rstrip("/") + "/", "commit")
    items = processed_batch["items"]
    if not items:
        log.warning("Батч пустой — пропускаем отправку")
        return

    try:
        async with _session.post(
            commit_url,
            data=orjson.dumps(items, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)),
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status == 200:
                _stats["items_sent"] += len(items)
                _stats["batches_processed"] += 1
                log.info(
                    f"📤 Батч отправлен ({len(items)} items) | "
                    f"итого отправлено: {_stats['items_sent']}"
                )
            else:
                body = await resp.text()
                log.error(f"❌ transactioneer вернул {resp.status}: {body[:200]}")
                _stats["errors"] += 1
    except aiohttp.ClientConnectorError:
        log.error(f"❌ transactioneer недоступен: {commit_url}")
        _stats["errors"] += 1
    except Exception as e:
        log.error(f"Ошибка отправки батча: {e}")
        _stats["errors"] += 1


def _process_batch_sync(batch, lab_config: dict) -> dict:
    return process_batch(batch, lab_config)


async def batch_processing_loop():
    global _stats, _inflight

    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bpipe_ml")

    log.info(
        f"🔄 Batch loop запущен: размер батча={FIXED_BATCH_SIZE}, "
        f"таймаут={BATCH_TIMEOUT_SECS}с, макс. возраст={BPIPE_MAX_AGE_SECS}с"
    )

    batch_id = 0
    while True:
        batch = []
        first_item_time = None

        while len(batch) < FIXED_BATCH_SIZE:
            timeout = None
            if first_item_time is not None:
                remaining = BATCH_TIMEOUT_SECS - (time.monotonic() - first_item_time)
                if remaining <= 0:
                    break
                timeout = remaining

            try:
                neg_ts, _, item = await asyncio.wait_for(_process_queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                break

            if time.time() + neg_ts > BPIPE_MAX_AGE_SECS:
                _stats["stale"] += 1
                continue

            batch.append(item)
            _inflight = len(batch)
            if first_item_time is None:
                first_item_time = time.monotonic()

        if not batch:
            continue

        batch_id += 1
        log.info(f"[Batch-{batch_id}] Обрабатываем {len(batch)} элементов...")

        t0 = timerit()
        try:
            processed_batch = await loop.run_in_executor(
                executor,
                _process_batch_sync,
                batch,
                _lab_config,
            )
            t1 = timerit()
            log.info(f"[Batch-{batch_id}] ✅ ML готово за {t1-t0:.2f}с")

            await send_batch_to_transactioneer(processed_batch)

        except TooBigError as e:
            log.warning(f"[Batch-{batch_id}] TooBigError: {e}")
        except Exception as e:
            _stats["errors"] += 1
            log.error(f"[Batch-{batch_id}] ❌ Ошибка обработки: {e}", exc_info=True)
        finally:
            _inflight = 0


async def handle_receive_item(request: web.Request) -> web.Response:
    global _stats, _live_config
    try:
        raw_item = await request.json()
    except Exception as e:
        return web.Response(text=f"bad json: {e}", status=400)

    translation_text = raw_item.get("translation", {}).get("translation", "")
    if not translation_text or not translation_text.strip():
        return web.Response(text="skipped_empty")

    try:
        external_id_value    = raw_item["item"].get("external_id") or ""
        author_value         = raw_item["item"].get("author") or ""
        title_value          = raw_item["item"].get("title") or ""
        ext_parent_id        = raw_item["item"].get("external_parent_id") or ""

        processed_item = Processed(
            classification=Classification(
                label=raw_item["classification"]["label"],
                score=raw_item["classification"]["score"],
            ),
            translation=Translation(
                language=Language(raw_item["translation"]["language"]),
                translation=Translated(translation_text),
            ),
            top_keywords=Keywords(list(raw_item["top_keywords"])),
            item=Item(
                created_at=CreatedAt(raw_item["item"]["created_at"]),
                title=Title(title_value),
                content=Content(raw_item["item"]["content"]),
                domain=Domain(raw_item["item"]["domain"]),
                url=Url(raw_item["item"]["url"]),
                external_id=ExternalId(external_id_value),
                author=Author(author_value),
            ),
        )

        if ext_parent_id:
            processed_item.item["external_parent_id"] = ExternalParentId(ext_parent_id)
        if raw_item["item"].get("username"):
            processed_item.item["username"] = Username(raw_item["item"]["username"])

    except Exception as e:
        log.warning(f"Ошибка создания Processed: {e}")
        return web.Response(text="invalid_item", status=400)

    ts = _created_ts(raw_item["item"]["created_at"])
    if time.time() - ts > BPIPE_MAX_AGE_SECS:
        _stats["stale"] += 1
        return web.Response(text="stale")

    if _process_queue.qsize() >= MAX_QUEUE_SIZE:
        _stats["stale"] += _purge_stale()
        if _process_queue.qsize() >= MAX_QUEUE_SIZE:
            if _evict_oldest_if_older(ts):
                _stats["evicted"] += 1
            else:
                _stats["dropped"] += 1
                if _stats["dropped"] % 10 == 0:
                    log.warning(
                        f"🗑️  bpipe очередь переполнена — дропнуто: {_stats['dropped']} "
                        f"(queue={_process_queue.qsize()}/{MAX_QUEUE_SIZE})"
                    )
                return web.Response(text="queue_full", status=503)

    await _process_queue.put((-ts, next(_seq), (id(processed_item), processed_item)))
    _stats["received"] += 1

    if _stats["received"] % 100 == 0:
        log.info(
            f"📥 recv={_stats['received']} | "
            f"queue={_process_queue.qsize()} | "
            f"sent={_stats['items_sent']} | "
            f"dropped={_stats['dropped']} | stale={_stats['stale']} | evicted={_stats['evicted']}"
        )

    return web.Response(text="received")


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok",
        "queue": _process_queue.qsize() if _process_queue else 0,
        "stats": _stats,
        "batch_size": FIXED_BATCH_SIZE,
        "max_age_s": BPIPE_MAX_AGE_SECS,
    })


async def handle_queue(request: web.Request) -> web.Response:
    size = _process_queue.qsize() if _process_queue else 0
    return web.json_response({
        "ready": _process_queue is not None,
        "queue": size,
        "max": MAX_QUEUE_SIZE,
        "fill": round(size / MAX_QUEUE_SIZE, 3) if MAX_QUEUE_SIZE else 0.0,
        "inflight": _inflight,
        "batch_size": FIXED_BATCH_SIZE,
        "dropped": _stats["dropped"],
        "stale": _stats["stale"],
        "evicted": _stats["evicted"],
    })


async def on_startup(app: web.Application):
    global _session, _lab_config, _live_config, _process_queue

    log.info("🔬 Инициализация ML-моделей bpipe...")
    log.info("   (первая загрузка занимает 5-15 минут — скачиваются модели HuggingFace)")

    loop = asyncio.get_event_loop()

    try:
        _lab_config = await loop.run_in_executor(None, lab_initialization)
        log.info("✅ ML модели загружены")
    except Exception as e:
        log.error(f"❌ Ошибка загрузки моделей: {e}")
        raise

    try:
        _live_config = await get_live_configuration()
        _lab_config["live_configuration"] = _live_config
        log.info("✅ Live configuration получена")
    except Exception as e:
        log.warning(f"⚠️ Не удалось получить live configuration: {e}")

    _process_queue = asyncio.PriorityQueue()

    connector = aiohttp.TCPConnector(limit=5, keepalive_timeout=60)
    _session = aiohttp.ClientSession(connector=connector)

    asyncio.create_task(batch_processing_loop())

    log.info(f"✅ Bpipe запущен на порту {BPIPE_PORT}")
    log.info(f"   Батч: {FIXED_BATCH_SIZE} items | таймаут: {BATCH_TIMEOUT_SECS}с | макс. возраст: {BPIPE_MAX_AGE_SECS}с")
    log.info(f"   Отправляет в transactioneer: {TRANSACTIONEER_URL}")


async def on_shutdown(app: web.Application):
    global _session
    if _session:
        await _session.close()
    log.info(f"📊 Итог bpipe: {_stats}")


app = web.Application(client_max_size=500 * 1024 * 1024)
app.router.add_post("/", handle_receive_item)
app.router.add_get("/", handle_health)
app.router.add_get("/health", handle_health)
app.router.add_get("/queue", handle_queue)
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=BPIPE_PORT, print=None)
