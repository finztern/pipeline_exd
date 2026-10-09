import asyncio
import json
import logging
import os
import sys
import time
from collections import deque
from typing import Optional

from aiohttp import web, ClientSession, FormData, ClientTimeout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [transactioneer] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

TRANSACTIONEER_PORT = int(os.getenv("TRANSACTIONEER_PORT", "8002"))
UPLOAD_API_URL      = os.getenv("UPLOAD_API_URL", "http://upload.exorde.network/v1/upload")
MAIN_ADDRESS        = os.getenv("MAIN_ADDRESS", "")
UPLOAD_API_TIMEOUT  = int(os.getenv("UPLOAD_API_TIMEOUT", "120"))
USE_PROXY           = os.getenv("USE_PROXY", "false").lower() == "true"
PROXY_URL           = os.getenv("PROXY_URL", "")
MAX_RETRIES         = int(os.getenv("UPLOAD_MAX_RETRIES", "3"))

_started = time.monotonic()
_stats = {
    "batches_received": 0,
    "items_received": 0,
    "uploads_success": 0,
    "uploads_failed": 0,
    "items_accepted": 0,
}
_totals = {"accepted": 0, "rejected": 0, "duplicates": 0}
_events: deque = deque()
_active_tasks: list = []

RATE_WINDOWS = (("10m", 600), ("30m", 1800), ("60m", 3600))


def _record(accepted: int, rejected: int, duplicates: int):
    now = time.monotonic()
    _totals["accepted"] += accepted
    _totals["rejected"] += rejected
    _totals["duplicates"] += duplicates
    _events.append((now, accepted, rejected, duplicates))
    cut = now - RATE_WINDOWS[-1][1]
    while _events and _events[0][0] < cut:
        _events.popleft()


def _rates() -> dict:
    now = time.monotonic()
    uptime = now - _started
    out = {}
    for name, window in RATE_WINDOWS:
        win = max(1.0, min(float(window), uptime))
        cut = now - window
        a = r = d = 0
        for ts, x, y, z in _events:
            if ts >= cut:
                a += x
                r += y
                d += z
        out[name] = {
            "accepted_per_sec": round(a / win, 3),
            "rejected_per_sec": round(r / win, 3),
            "duplicates_per_sec": round(d / win, 3),
            "window_s": int(win),
        }
    return out


async def upload_to_api(items: list, main_address: str) -> Optional[dict]:
    batch_json = json.dumps({"items": items, "kind": "SPOTTING"}, default=str, ensure_ascii=False)

    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                delay = 2 ** attempt
                log.info(f"🔄 Повторная попытка {attempt+1}/{MAX_RETRIES} через {delay}с...")
                await asyncio.sleep(delay)

            form = FormData()
            form.add_field(
                "file",
                batch_json,
                filename=f"batch_{int(time.time())}.json",
                content_type="application/json",
            )

            connector_kwargs = {}
            if USE_PROXY and PROXY_URL:
                try:
                    from aiohttp_socks import ProxyConnector
                    connector_kwargs["connector"] = ProxyConnector.from_url(PROXY_URL)
                except ImportError:
                    log.warning("aiohttp_socks не установлен, прокси игнорируется")

            async with ClientSession(**connector_kwargs) as session:
                async with session.post(
                    UPLOAD_API_URL,
                    data=form,
                    headers={"MAIN_ADDRESS": main_address},
                    timeout=ClientTimeout(total=UPLOAD_API_TIMEOUT),
                ) as resp:
                    if resp.status == 200:
                        data       = await resp.json()
                        file_id    = data.get("file_id", "?")
                        filtered   = data.get("filtered_items", 0)
                        rejected   = data.get("rejected_items", 0)
                        duplicates = data.get("duplicate_items", 0)
                        proc_ms    = data.get("processing_time_ms", 0)

                        log.info(
                            f"✅ Загружено | file_id={file_id} | "
                            f"отправлено={len(items)} принято={filtered} "
                            f"отклонено={rejected} дубликаты={duplicates} | "
                            f"API обработка={proc_ms}ms"
                        )

                        if rejected > 0:
                            log.warning(f"⚠️ Полный ответ API: {json.dumps(data, ensure_ascii=False)[:2000]}")
                            try:
                                with open(f"/tmp/rejected_batch_{file_id}.json", "w") as dbf:
                                    json.dump({"items": items[:3], "kind": "SPOTTING"}, dbf, default=str, ensure_ascii=False, indent=2)
                            except Exception as de:
                                log.warning(f"Не удалось сохранить debug файл: {de}")

                        return data
                    err = await resp.text()
                    log.error(f"❌ HTTP {resp.status}: {err[:300]}")

        except asyncio.TimeoutError:
            log.error(f"⏱️ Таймаут (попытка {attempt+1}/{MAX_RETRIES})")
        except Exception as e:
            log.exception(f"❌ Ошибка при загрузке (попытка {attempt+1}/{MAX_RETRIES}): {e}")

    log.error(f"❌ Все {MAX_RETRIES} попытки загрузки исчерпаны")
    return None


async def _upload_task(items: list):
    result = await upload_to_api(items, MAIN_ADDRESS)
    if result:
        accepted = result.get("filtered_items", 0)
        _stats["uploads_success"] += 1
        _stats["items_accepted"] += accepted
        _record(accepted, result.get("rejected_items", 0), result.get("duplicate_items", 0))
    else:
        _stats["uploads_failed"] += 1


async def handle_commit(request: web.Request) -> web.Response:
    global _active_tasks
    try:
        items = await request.json()
    except Exception as e:
        return web.Response(text=f"bad json: {e}", status=400)

    if not isinstance(items, list):
        return web.Response(text="expected array", status=400)

    _stats["batches_received"] += 1
    _stats["items_received"] += len(items)
    log.info(f"📥 Батч получен: {len(items)} элементов")

    _active_tasks = [t for t in _active_tasks if not t.done()]
    _active_tasks.append(asyncio.create_task(_upload_task(items)))
    return web.Response(text="received", status=200)


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok",
        "main_address": MAIN_ADDRESS or "NOT SET",
        "upload_api": UPLOAD_API_URL,
        "stats": _stats,
        "active_tasks": len(_active_tasks),
    })


async def handle_stats(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok",
        "uptime_s": int(time.monotonic() - _started),
        "totals": _totals,
        "rate": _rates(),
        "stats": _stats,
        "active_tasks": len(_active_tasks),
    })


async def on_startup(app: web.Application):
    if not MAIN_ADDRESS:
        log.error("❌ MAIN_ADDRESS не задан! Установи его в .env файле")
    else:
        log.info(f"✅ Transactioneer запущен | address={MAIN_ADDRESS[:10]}...")
    log.info(f"   Upload API: {UPLOAD_API_URL}")
    log.info(f"   Порт: {TRANSACTIONEER_PORT}")


app = web.Application(client_max_size=500 * 1024 * 1024)
app.router.add_post("/commit", handle_commit)
app.router.add_get("/", handle_health)
app.router.add_get("/health", handle_health)
app.router.add_get("/stats", handle_stats)
app.on_startup.append(on_startup)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=TRANSACTIONEER_PORT, print=None)