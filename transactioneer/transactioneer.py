import asyncio
import json
import logging
import os
import sys
import time
from collections import Counter, deque
from typing import Optional

from aiohttp import web, FormData, ClientTimeout

from route_manager import RouteManager

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
MAX_RETRIES         = int(os.getenv("UPLOAD_MAX_RETRIES", "3"))
UPLOAD_CONCURRENCY  = max(1, int(os.getenv("UPLOAD_CONCURRENCY", "6")))

ROUTE_FAIL_STATUS = {403, 407, 451, 500, 502, 503, 504}
routes = RouteManager(UPLOAD_API_URL)
_upload_sem = asyncio.Semaphore(UPLOAD_CONCURRENCY)

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
_dom_buckets: deque = deque()
_dom_totals = {"s": Counter(), "x": Counter()}
_DOM_MAX = 1000


def _dom_record(items: list, kind: str):
    minute = int(time.time() // 60)
    if not _dom_buckets or _dom_buckets[-1][0] != minute:
        _dom_buckets.append((minute, {"s": Counter(), "x": Counter()}))
        while _dom_buckets and _dom_buckets[0][0] <= minute - 60:
            _dom_buckets.popleft()
    for it in items:
        try:
            domain = str(it["item"]["domain"]).strip().lower()[:100] or "unknown"
        except Exception:
            domain = "unknown"
        if domain not in _dom_totals[kind] and len(_dom_totals[kind]) >= _DOM_MAX:
            domain = "other"
        _dom_buckets[-1][1][kind][domain] += 1
        _dom_totals[kind][domain] += 1


def _dom_snapshot() -> dict:
    now_min = int(time.time() // 60)
    up = max(60.0, time.monotonic() - _started)
    wins = {}
    for name, m in (("10m", 10), ("30m", 30), ("60m", 60)):
        s, x = Counter(), Counter()
        for mi, d in _dom_buckets:
            if mi > now_min - m:
                s.update(d["s"])
                x.update(d["x"])
        wins[name] = {"s": dict(s), "x": dict(x), "window_s": int(min(m * 60, up))}
    return {"totals": {"s": dict(_dom_totals["s"]), "x": dict(_dom_totals["x"])}, "windows": wins}


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
        if attempt > 0:
            delay = 2 ** attempt
            log.info(f"🔄 Повторная попытка {attempt+1}/{MAX_RETRIES} через {delay}с...")
            await asyncio.sleep(delay)

        route = routes.pick()
        if route is None:
            log.error("❌ нет доступных маршрутов выгрузки")
            continue

        await _upload_sem.acquire()
        t0 = time.monotonic()
        try:
            form = FormData()
            form.add_field(
                "file",
                batch_json,
                filename=f"batch_{int(time.time())}.json",
                content_type="application/json",
            )
            async with route.request(
                "POST",
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
                    routes.report_success(route)

                    log.info(
                        f"✅ Загружено | file_id={file_id} | "
                        f"отправлено={len(items)} принято={filtered} "
                        f"отклонено={rejected} дубликаты={duplicates} | "
                        f"API обработка={proc_ms}ms | {route.name} {(time.monotonic()-t0)*1000:.0f}ms"
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
                log.error(f"❌ HTTP {resp.status} via {route.name}: {err[:300]}")
                if resp.status in ROUTE_FAIL_STATUS:
                    routes.report_failure(route)

        except asyncio.TimeoutError:
            log.error(f"⏱️ Таймаут via {route.name} (попытка {attempt+1}/{MAX_RETRIES})")
            routes.report_failure(route)
        except Exception as e:
            log.error(f"❌ Ошибка загрузки via {route.name} (попытка {attempt+1}/{MAX_RETRIES}): {type(e).__name__}: {e}")
            routes.report_failure(route)
        finally:
            _upload_sem.release()

    log.error(f"❌ Все {MAX_RETRIES} попытки загрузки исчерпаны")
    return None


async def _upload_task(items: list):
    result = await upload_to_api(items, MAIN_ADDRESS)
    if result:
        accepted = result.get("filtered_items", 0)
        _stats["uploads_success"] += 1
        _stats["items_accepted"] += accepted
        _record(accepted, result.get("rejected_items", 0), result.get("duplicate_items", 0))
        _dom_record(items, "s")
    else:
        _stats["uploads_failed"] += 1
        _dom_record(items, "x")


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
        "route": routes.active_name(),
        "stats": _stats,
        "active_tasks": len(_active_tasks),
    })


async def handle_stats(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok",
        "uptime_s": int(time.monotonic() - _started),
        "totals": _totals,
        "rate": _rates(),
        "domains": _dom_snapshot(),
        "route": routes.active_name(),
        "stats": _stats,
        "active_tasks": len(_active_tasks),
    })


async def handle_routes(request: web.Request) -> web.Response:
    return web.json_response(routes.snapshot())


async def on_startup(app: web.Application):
    try:
        await asyncio.wait_for(routes.probe_all(), 60)
    except Exception as e:
        log.error(f"❌ первичная проверка маршрутов не завершена: {e}")
    app["route_task"] = asyncio.create_task(routes.run())
    if not MAIN_ADDRESS:
        log.error("❌ MAIN_ADDRESS не задан! Установи его в .env файле")
    else:
        log.info(f"✅ Transactioneer запущен | address={MAIN_ADDRESS[:10]}...")
    log.info(f"   Upload API: {UPLOAD_API_URL}")
    log.info(f"   Порт: {TRANSACTIONEER_PORT}")


async def on_cleanup(app: web.Application):
    task = app.get("route_task")
    if task:
        task.cancel()
    await routes.close()


app = web.Application(client_max_size=500 * 1024 * 1024)
app.router.add_post("/commit", handle_commit)
app.router.add_get("/", handle_health)
app.router.add_get("/health", handle_health)
app.router.add_get("/stats", handle_stats)
app.router.add_get("/routes", handle_routes)
app.on_startup.append(on_startup)
app.on_cleanup.append(on_cleanup)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=TRANSACTIONEER_PORT, print=None)