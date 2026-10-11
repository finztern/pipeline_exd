"""
Bpipe (standalone) — принимает обработанные элементы от upipe,
запускает ML-пайплайн (эмбеддинги, сентимент, эмоции, тип текста, etc.),
формирует батчи и отправляет в transactioneer.

Слушает: POST / на BPIPE_PORT (по умолчанию 7995)
Отправляет в: TRANSACTIONEER_URL (по умолчанию http://127.0.0.1:8002/commit)

Адаптировано для локального запуска на RTX 3050 8GB.
"""
import asyncio
import gc
import json
import logging
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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

# ─── Конфигурация ─────────────────────────────────────────────
BPIPE_PORT           = int(os.getenv("BPIPE_PORT", "7995"))
TRANSACTIONEER_URL   = os.getenv("TRANSACTIONEER_URL", "http://127.0.0.1:8002")
FIXED_BATCH_SIZE     = int(os.getenv("FIXED_BATCH_SIZE", "6"))     # Меньше для RTX 3050
BATCH_TIMEOUT_SECS   = float(os.getenv("BATCH_TIMEOUT_SECONDS", "3.0"))

QUEUE_TRANSLATED_MAX      = int(os.getenv("BPIPE_QUEUE_TRANSLATED_MAX", "100"))
QUEUE_TRANSLATED_HARD_MAX = max(QUEUE_TRANSLATED_MAX, int(os.getenv("BPIPE_QUEUE_TRANSLATED_HARD_MAX", "1000")))
QUEUE_DEFAULT_MAX         = int(os.getenv("BPIPE_QUEUE_DEFAULT_MAX") or os.getenv("MAX_QUEUE_SIZE") or "200")
QUEUE_BUFFER_MAX          = int(os.getenv("BPIPE_QUEUE_BUFFER_MAX", "500"))
BUFFER_TTL_SECS           = float(os.getenv("BPIPE_BUFFER_TTL_SECONDS", "7200"))
BUFFER_DOMAINS            = [d.strip().lower() for d in os.getenv("BPIPE_BUFFER_DOMAINS", "").split(",") if d.strip()]

# ─── Глобальное состояние ──────────────────────────────────────
_q_translated: deque = deque()
_q_default: deque = deque()
_q_buffer: deque = deque()
_wake: asyncio.Event | None = None
_lab_config: dict | None = None
_live_config = None
_session: aiohttp.ClientSession | None = None
_inflight = 0
_send_tasks: set = set()
_stats = {
    "received": 0,
    "batches_processed": 0,
    "items_sent": 0,
    "errors": 0,
    "dropped": 0,
    "dropped_translated": 0,
    "dropped_default": 0,
    "dropped_buffer": 0,
    "expired_buffer": 0,
    "received_translated": 0,
    "received_default": 0,
    "received_buffer": 0,
}


class TooBigError(Exception):
    pass


# ─── Отправка в transactioneer ─────────────────────────────────

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


# ─── Обработка батча (в отдельном потоке) ─────────────────────

def _process_batch_sync(batch, lab_config: dict) -> dict:
    """Синхронная ML-обработка батча. Запускается в ThreadPoolExecutor."""
    return process_batch(batch, lab_config)


# ─── Приоритетные очереди ─────────────────────────────────────

def _domain_is_buffer(domain: str) -> bool:
    domain = (domain or "").strip().lower()
    if not domain:
        return False
    for d in BUFFER_DOMAINS:
        if "." in d:
            if domain == d or domain.endswith("." + d):
                return True
        elif d in domain:
            return True
    return False


def _classify(raw_item: dict) -> str:
    translation = raw_item.get("translation") or {}
    lang = str(translation.get("language") or "").strip().lower().split("-")[0]
    if raw_item.get("translated") is True or lang not in ("", "en"):
        return "translated"
    if (
        raw_item.get("priority") == "buffer"
        or raw_item.get("queue") == "buffer"
        or _domain_is_buffer((raw_item.get("item") or {}).get("domain", ""))
    ):
        return "buffer"
    return "default"


def _purge_expired_buffer():
    if BUFFER_TTL_SECS <= 0:
        return
    cutoff = time.monotonic() - BUFFER_TTL_SECS
    while _q_buffer and _q_buffer[0][0] < cutoff:
        _q_buffer.popleft()
        _stats["expired_buffer"] += 1


def _pop_next():
    if _q_translated:
        return _q_translated.popleft()[1]
    if _q_default:
        return _q_default.popleft()[1]
    _purge_expired_buffer()
    if _q_buffer:
        return _q_buffer.popleft()[1]
    return None


def _enqueue(kind: str, entry) -> bool:
    now = time.monotonic()
    if kind == "translated":
        if len(_q_translated) >= QUEUE_TRANSLATED_HARD_MAX:
            _stats["dropped_translated"] += 1
            return False
        if len(_q_translated) >= QUEUE_TRANSLATED_MAX and _stats["received_translated"] % 20 == 0:
            log.warning(
                f"translated сверх мягкого лимита: {len(_q_translated)}/{QUEUE_TRANSLATED_MAX} "
                f"(жёсткий {QUEUE_TRANSLATED_HARD_MAX})"
            )
        _q_translated.append((now, entry))
    elif kind == "buffer":
        if len(_q_buffer) >= QUEUE_BUFFER_MAX:
            _purge_expired_buffer()
        if len(_q_buffer) >= QUEUE_BUFFER_MAX:
            _stats["dropped_buffer"] += 1
            return False
        _q_buffer.append((now, entry))
    else:
        if len(_q_default) >= QUEUE_DEFAULT_MAX:
            _stats["dropped_default"] += 1
            return False
        _q_default.append((now, entry))
    _stats[f"received_{kind}"] += 1
    _wake.set()
    return True


# ─── Основной цикл формирования батчей ────────────────────────

async def batch_processing_loop():
    global _stats, _inflight

    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bpipe_ml")

    log.info(
        f"🔄 Batch loop запущен: размер батча={FIXED_BATCH_SIZE}, "
        f"таймаут={BATCH_TIMEOUT_SECS}с"
    )

    batch_id = 0
    while True:
        batch = []
        first_item_time = None

        while len(batch) < FIXED_BATCH_SIZE:
            item = _pop_next()
            if item is not None:
                batch.append(item)
                _inflight = len(batch)
                if first_item_time is None:
                    first_item_time = time.monotonic()
                continue

            timeout = None
            if first_item_time is not None:
                timeout = BATCH_TIMEOUT_SECS - (time.monotonic() - first_item_time)
                if timeout <= 0:
                    break

            # Между пустым _pop_next() и clear() нет await — wakeup не теряется
            _wake.clear()
            try:
                await asyncio.wait_for(_wake.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                break

        if not batch:
            continue

        batch_id += 1
        log.info(f"[Batch-{batch_id}] Обрабатываем {len(batch)} элементов...")

        # ML обработка в потоке (GPU)
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

            task = asyncio.create_task(send_batch_to_transactioneer(processed_batch))
            _send_tasks.add(task)
            task.add_done_callback(_send_tasks.discard)

        except TooBigError as e:
            log.warning(f"[Batch-{batch_id}] TooBigError: {e}")
        except Exception as e:
            _stats["errors"] += 1
            log.error(f"[Batch-{batch_id}] ❌ Ошибка обработки: {e}", exc_info=True)
        finally:
            _inflight = 0


# ─── HTTP обработчики ──────────────────────────────────────────

async def handle_receive_item(request: web.Request) -> web.Response:
    global _stats, _live_config
    try:
        raw_item = await request.json()
    except Exception as e:
        return web.Response(text=f"bad json: {e}", status=400)

    # Проверяем наличие translation
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

    kind = _classify(raw_item)
    if not _enqueue(kind, (id(processed_item), processed_item)):
        _stats["dropped"] += 1
        if _stats["dropped"] % 10 == 0:
            log.warning(
                f"🗑️  bpipe очередь {kind} переполнена — дропнуто всего: {_stats['dropped']} "
                f"(t={len(_q_translated)} d={len(_q_default)} b={len(_q_buffer)})"
            )
        return web.Response(text="queue_full", status=503)

    _stats["received"] += 1

    if _stats["received"] % 100 == 0:
        log.info(
            f"📥 recv={_stats['received']} | "
            f"t={len(_q_translated)} d={len(_q_default)} b={len(_q_buffer)} | "
            f"sent={_stats['items_sent']} | "
            f"dropped={_stats['dropped']}"
        )

    return web.Response(text="received")


def _queues_snapshot() -> dict:
    t, d, b = len(_q_translated), len(_q_default), len(_q_buffer)
    return {
        "translated": {
            "queue": t, "max": QUEUE_TRANSLATED_MAX, "hard_max": QUEUE_TRANSLATED_HARD_MAX,
            "fill": round(t / QUEUE_TRANSLATED_MAX, 3) if QUEUE_TRANSLATED_MAX else 0.0,
            "dropped": _stats["dropped_translated"],
        },
        "default": {
            "queue": d, "max": QUEUE_DEFAULT_MAX,
            "fill": round(d / QUEUE_DEFAULT_MAX, 3) if QUEUE_DEFAULT_MAX else 0.0,
            "dropped": _stats["dropped_default"],
        },
        "buffer": {
            "queue": b, "max": QUEUE_BUFFER_MAX,
            "fill": round(b / QUEUE_BUFFER_MAX, 3) if QUEUE_BUFFER_MAX else 0.0,
            "dropped": _stats["dropped_buffer"],
            "expired": _stats["expired_buffer"],
            "ttl_s": BUFFER_TTL_SECS,
        },
    }


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok",
        "queue": len(_q_translated) + len(_q_default),
        "queue_total": len(_q_translated) + len(_q_default) + len(_q_buffer),
        "queues": _queues_snapshot(),
        "inflight": _inflight,
        "stats": _stats,
        "batch_size": FIXED_BATCH_SIZE,
    })


async def handle_queue(request: web.Request) -> web.Response:
    # queue/max/fill = translated+default (контракт для collector); buffer — только в "queues"
    size = len(_q_translated) + len(_q_default)
    mx = QUEUE_TRANSLATED_MAX + QUEUE_DEFAULT_MAX
    return web.json_response({
        "ready": _wake is not None,
        "queue": size,
        "max": mx,
        "fill": round(size / mx, 3) if mx else 0.0,
        "queue_total": size + len(_q_buffer),
        "queues": _queues_snapshot(),
        "inflight": _inflight,
        "batch_size": FIXED_BATCH_SIZE,
        "dropped": _stats["dropped"],
    })


async def on_startup(app: web.Application):
    global _session, _lab_config, _live_config, _wake

    log.info("🔬 Инициализация ML-моделей bpipe...")
    log.info("   (первая загрузка занимает 5-15 минут — скачиваются модели HuggingFace)")

    loop = asyncio.get_event_loop()

    # Инициализация ML моделей
    try:
        _lab_config = await loop.run_in_executor(None, lab_initialization)
        log.info("✅ ML модели загружены")
    except Exception as e:
        log.error(f"❌ Ошибка загрузки моделей: {e}")
        raise

    # Live конфиг (категории/метки) от Exorde
    try:
        _live_config = await get_live_configuration()
        _lab_config["live_configuration"] = _live_config
        log.info("✅ Live configuration получена")
    except Exception as e:
        log.warning(f"⚠️ Не удалось получить live configuration: {e}")

    _wake = asyncio.Event()

    connector = aiohttp.TCPConnector(limit=5, keepalive_timeout=60)
    _session = aiohttp.ClientSession(connector=connector)

    # Запускаем основной цикл
    asyncio.create_task(batch_processing_loop())

    log.info(f"✅ Bpipe запущен на порту {BPIPE_PORT}")
    log.info(f"   Батч: {FIXED_BATCH_SIZE} items | таймаут: {BATCH_TIMEOUT_SECS}с")
    log.info(
        f"   Очереди: translated={QUEUE_TRANSLATED_MAX} (жёстк. {QUEUE_TRANSLATED_HARD_MAX}) "
        f"default={QUEUE_DEFAULT_MAX} buffer={QUEUE_BUFFER_MAX} ttl={BUFFER_TTL_SECS}с "
        f"buffer_domains={BUFFER_DOMAINS}"
    )
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
