#!/usr/bin/env python3
"""Менеджер sing-box по подписке.

Что делает:
  1. скачивает подписку (SUBSCRIPTION_URL) и периодически обновляет её; при изменении набора узлов перестраивает конфиг;
  2. замеряет КАЖДЫЙ узел реальным запросом к API (тем же, каким ходит транзакционер): через узел должен проходить
     запрос к API без блокировки (403/407/451/5xx) и без порчи multipart;
  3. оставляет TOP_K самых быстрых рабочих узлов «про запас» и включает лучший (selector `proxy` через Clash API);
  4. следит за активным узлом через общий вход (:SOCKS_PORT) и при отказе мгновенно переключает на следующий из запаса;
  5. раз в MEASURE_INTERVAL перемеряет всё и переключается только при заметном выигрыше (SWITCH_MARGIN).
Весь трафик сервисов идёт через общий SOCKS/HTTP вход; прямого выхода в конфиге нет (утечек мимо VPN нет).
"""
import asyncio
import json
import logging
import os
import random
import secrets
import signal
import statistics
import sys
import time
import uuid
from urllib.parse import urlsplit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subscription  # noqa: E402
from sbconfig import make_nodes, build_managed_config  # noqa: E402


def _int(name, default):
    return int(os.getenv(name, str(default)) or default)


def _float(name, default):
    return float(os.getenv(name, str(default)) or default)


SUB_URL = os.getenv("SUBSCRIPTION_URL", "").strip()
SUB_INTERVAL = _int("SUB_INTERVAL", 1800)
SUB_UA = os.getenv("SUB_USER_AGENT", "v2rayN/7.0")
FALLBACK_FILE = os.getenv("VPN_FILE", "/vpn/vpn.txt")

API_URL = os.getenv("API_URL", "http://upload.exorde.network/v1/upload").strip()
_u = urlsplit(API_URL)
PROBE_URL = os.getenv("PROBE_URL", "").strip() or f"{_u.scheme}://{_u.netloc}/"
MAIN_ADDRESS = os.getenv("MAIN_ADDRESS", "")

MEASURE_INTERVAL = _int("MEASURE_INTERVAL", 600)
MEASURE_PARALLEL = max(1, _int("MEASURE_PARALLEL", 24))
MEASURE_TIMEOUT = _int("MEASURE_TIMEOUT", 8)
TOP_M = max(1, _int("TOP_M", 12))          # сколько лучших по быстрой проверке проверяем реальной загрузкой
TOP_K = max(1, _int("TOP_K", 5))           # размер запаса
POST_BYTES = _int("MEASURE_POST_BYTES", 131072)
POST_SAMPLES = max(1, _int("MEASURE_POST_SAMPLES", 2))
SWITCH_MARGIN = _float("SWITCH_MARGIN", 0.8)   # переключаемся на лучший, только если он быстрее активного минимум на 20%
WATCH_INTERVAL = _int("WATCH_INTERVAL", 5)
WATCH_FAILS = max(1, _int("WATCH_FAILS", 2))
NODE_COOLDOWN = _int("NODE_COOLDOWN", 300)
POOL_CHECK_INTERVAL = _int("POOL_CHECK_INTERVAL", 60)

SOCKS_PORT = _int("SOCKS_PORT", 1080)
HTTP_PORT = _int("HTTP_PORT", 1081)
BASE_PORT = _int("LINK_BASE_PORT", 10800)
CLASH_PORT = _int("CLASH_PORT", 9090)
STATUS_PORT = _int("STATUS_PORT", 9091)
LOG_LEVEL = os.getenv("XRAY_LOG_LEVEL", "warning")
STATE_DIR = os.getenv("STATE_DIR", "/state")
CONFIG_PATH = os.getenv("CONFIG_PATH", "/vpn/config.json")
SING_BOX_BIN = os.getenv("SING_BOX_BIN", "sing-box")

BAD_STATUS = {403, 407, 451, 500, 502, 503, 504}  # как в transactioneer/route_manager.py

logging.basicConfig(level=logging.INFO, format="%(asctime)s [vpn] %(levelname)s %(message)s", stream=sys.stdout)
log = logging.getLogger("vpn")


# ----------------------------------------------------------------------------- утилиты

class Result:
    __slots__ = ("ok", "status", "ms", "err")

    def __init__(self, ok, status=0, ms=None, err=""):
        self.ok, self.status, self.ms, self.err = ok, status, ms, err


def multipart_file(filename: str, content: str) -> tuple:
    b = uuid.uuid4().hex
    head = (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f'Content-Type: application/json\r\n\r\n').encode()
    return head + content.encode("utf-8") + f"\r\n--{b}--\r\n".encode(), f"multipart/form-data; boundary={b}"


async def run_cmd(cmd, stdin=None, timeout=30):
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(proc.communicate(stdin), timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
        return -1, b"", b"timeout"
    return proc.returncode, out, err


async def probe(port: int, *, post: bool, timeout: int = None) -> Result:
    """Запрос к API через SOCKS-вход на localhost:port. post=False: быстрая проверка (HEAD), True: реальная загрузка."""
    timeout = timeout or MEASURE_TIMEOUT
    cmd = ["curl", "-sS", "--max-time", str(timeout), "--connect-timeout", str(min(timeout, 6)),
           "-x", f"socks5h://127.0.0.1:{port}"]
    stdin = None
    if post and POST_BYTES > 0:
        payload, ctype = multipart_file("probe.json", '{"items":[],"kind":"SPOTTING","pad":"' + "x" * POST_BYTES + '"}')
        cmd += ["-X", "POST", "-H", f"MAIN_ADDRESS: {MAIN_ADDRESS}", "-H", f"Content-Type: {ctype}", "-H", "Expect:",
                "--data-binary", "@-", "-w", "\n%{http_code} %{time_total}", API_URL]
        stdin = payload
    else:
        post = False
        cmd += ["-I", "-o", "/dev/null", "-w", "%{http_code} %{time_total}", PROBE_URL]
    rc, out, err = await run_cmd(cmd, stdin, timeout + 3)
    text = out.decode("utf-8", "replace")
    body, _, last = text.rpartition("\n")
    last = last.strip() if post else text.strip()
    try:
        code_s, t_s = last.split()[:2]
        status, secs = int(code_s), float(t_s)
    except (ValueError, IndexError):
        status, secs = 0, 0.0
    if rc != 0 or status == 0:
        msg = (err.decode("utf-8", "replace").strip().splitlines() or ["нет ответа"])[-1]
        return Result(False, 0, None, msg[:100])
    if status in BAD_STATUS:
        return Result(False, status, None, f"HTTP {status}")
    if post and status == 400 and "multipart" in body.lower():
        return Result(False, status, None, "узел ломает multipart")
    return Result(True, status, secs * 1000.0)


async def http_local(port: int, method: str, path: str, body=None, secret: str = "", timeout: float = 5.0):
    """Минимальный HTTP-клиент для Clash API sing-box (без зависимостей)."""
    r, w = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), timeout)
    try:
        data = json.dumps(body).encode() if body is not None else b""
        head = (f"{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n"
                f"Authorization: Bearer {secret}\r\nContent-Length: {len(data)}\r\n"
                + ("Content-Type: application/json\r\n" if data else "") + "\r\n")
        w.write(head.encode() + data)
        await w.drain()
        resp = await asyncio.wait_for(r.read(), timeout)
    finally:
        w.close()
    h, _, rest = resp.partition(b"\r\n\r\n")
    status = int(h.split(b" ", 2)[1])
    return status, rest


def _atomic_write(path: str, text: str, mode: int = 0o600) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------- sing-box

class SingBox:
    def __init__(self):
        self.proc = None
        self.expected_stop = False
        self.starts = 0

    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def start(self):
        self.expected_stop = False
        self.proc = await asyncio.create_subprocess_exec(SING_BOX_BIN, "run", "-c", CONFIG_PATH)
        self.starts += 1

    async def stop(self):
        self.expected_stop = True
        if not self.alive():
            return
        self.proc.terminate()
        try:
            await asyncio.wait_for(self.proc.wait(), 8)
        except asyncio.TimeoutError:
            self.proc.kill()
            await self.proc.wait()


# ----------------------------------------------------------------------------- менеджер

class Manager:
    def __init__(self):
        self.nodes, self.by_key, self.links = [], {}, []
        self.sb = SingBox()
        self.secret = secrets.token_hex(12)
        self.metrics = {}     # key -> {ok1, ms1, ok, score, status, err, ts}
        self.pool = []        # ключи лучших узлов (запас), лучший первым
        self.active = None
        self.cooldown = {}    # key -> monotonic-время, до которого узел исключён
        self.stats = {"measures": 0, "failovers": 0, "switches": 0, "sub_updates": 0, "sub_errors": 0, "restarts": 0}
        self.last_sub = 0.0
        self.last_measure = 0.0
        self.last_measure_summary = ""
        self.last_error = ""
        self.last_ok = 0.0
        self.watch_ok = False
        self.watch_fails = 0
        self.restarting = False
        self.sel_lock = asyncio.Lock()
        self.measure_lock = asyncio.Lock()
        self.rebuild_lock = asyncio.Lock()
        self.wake = asyncio.Event()
        self.stopping = False
        self._cached_pool, self._cached_active = [], None

    # ---- состояние на диске
    def load_state(self):
        try:
            with open(os.path.join(STATE_DIR, "state.json"), encoding="utf-8") as f:
                st = json.load(f)
            self._cached_pool = list(st.get("pool_keys", []))
            self._cached_active = st.get("active")
            log.info(f"💾 состояние восстановлено: запас {len(self._cached_pool)} узлов")
        except (OSError, ValueError):
            pass

    def save_state(self):
        try:
            _atomic_write(os.path.join(STATE_DIR, "state.json"), json.dumps({
                "ts": time.time(), "active": self.active, "pool_keys": self.pool,
                "scores": {k: self.metrics.get(k, {}).get("score") for k in self.pool},
            }))
        except OSError as e:
            log.warning(f"не удалось сохранить состояние: {e}")

    def cached_links(self):
        try:
            with open(os.path.join(STATE_DIR, "links.txt"), encoding="utf-8") as f:
                return subscription.extract_links(f.read())
        except OSError:
            return []

    def cooling(self, key) -> bool:
        return self.cooldown.get(key, 0) > time.monotonic()

    # ---- подписка
    async def fetch_links(self):
        """Список ссылок из подписки или None. Сначала напрямую, затем (если sing-box уже работает) через сам прокси."""
        if not SUB_URL:
            return None
        attempts = [""]
        if self.sb.alive() and self.active:
            attempts.append(f"socks5h://127.0.0.1:{SOCKS_PORT}")
        last = ""
        for proxy in attempts:
            try:
                raw = await subscription.fetch(SUB_URL, proxy=proxy, user_agent=SUB_UA)
                links = subscription.decode_subscription(raw)
                self.last_sub = time.time()
                try:
                    _atomic_write(os.path.join(STATE_DIR, "links.txt"), "\n".join(links) + "\n")
                except OSError:
                    pass
                log.info(f"📥 подписка получена{' через VPN' if proxy else ''}: ссылок {len(links)}")
                return links
            except Exception as e:  # noqa: BLE001
                last = str(e)
        self.stats["sub_errors"] += 1
        self.last_error = f"подписка: {last}"
        log.warning(f"⚠️ подписка не обновилась: {last}")
        return None

    def fallback_links(self):
        try:
            with open(FALLBACK_FILE, encoding="utf-8") as f:
                return subscription.extract_links(f.read())
        except OSError:
            return []

    # ---- sing-box: конфиг, запуск, перезапуск
    async def check_config(self) -> bool:
        rc, out, err = await run_cmd([SING_BOX_BIN, "check", "-c", CONFIG_PATH], timeout=60)
        if rc != 0:
            log.error("конфиг не прошёл sing-box check: " + (err or out).decode("utf-8", "replace").strip()[-300:])
        return rc == 0

    async def wait_clash(self, timeout=45) -> bool:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if not self.sb.alive():
                return False
            try:
                st, _ = await http_local(CLASH_PORT, "GET", "/version", secret=self.secret, timeout=2)
                if st == 200:
                    return True
            except (OSError, asyncio.TimeoutError, ValueError, IndexError):
                pass
            await asyncio.sleep(0.5)
        return False

    async def apply_links(self, links, force=False) -> bool:
        """Применяет набор ссылок; sing-box перезапускается только если набор узлов изменился."""
        nodes, skipped = make_nodes(links)
        for s in skipped[:5]:
            log.warning(f"⚠️ ссылка пропущена: {s}")
        if skipped:
            log.warning(f"⚠️ пропущено ссылок: {len(skipped)} из {len(links)}")
        if not nodes:
            log.error("в подписке нет пригодных узлов, оставляю прежний набор")
            return False
        new_keys = {n.key for n in nodes}
        if not force and self.sb.alive() and new_keys == set(self.by_key):
            self.links = links
            return True
        async with self.measure_lock, self.rebuild_lock:
            self.restarting = True
            try:
                self.links = links
                keep = (self.active if self.active in new_keys else None) \
                    or next((k for k in self.pool if k in new_keys), None) \
                    or (self._cached_active if self._cached_active in new_keys else None) \
                    or next((k for k in self._cached_pool if k in new_keys), None)
                cfg = build_managed_config(nodes, socks_port=SOCKS_PORT, http_port=HTTP_PORT, base_port=BASE_PORT,
                                           log_level=LOG_LEVEL, clash_port=CLASH_PORT, clash_secret=self.secret,
                                           initial_key=keep)
                old = None
                if os.path.exists(CONFIG_PATH):
                    with open(CONFIG_PATH, encoding="utf-8") as f:
                        old = f.read()
                _atomic_write(CONFIG_PATH, json.dumps(cfg), 0o600)
                if not await self.check_config():
                    if old is not None:
                        _atomic_write(CONFIG_PATH, old, 0o600)
                    return False
                added = len(new_keys - set(self.by_key))
                removed = len(set(self.by_key) - new_keys)
                self.nodes, self.by_key = nodes, {n.key: n for n in nodes}
                self.pool = [k for k in self.pool if k in new_keys]
                self.active = keep
                for d in (self.metrics, self.cooldown):
                    for k in [k for k in d if k not in new_keys]:
                        d.pop(k, None)
                await self.sb.stop()
                await self.sb.start()
                if not await self.wait_clash():
                    log.error("sing-box не поднялся после применения конфига")
                    return False
                self.stats["restarts"] += 1
                log.info(f"🔧 sing-box запущен: узлов {len(nodes)} (+{added}/-{removed}), старт с узла "
                         f"{self.by_key[keep].name if keep else 'первого по списку'}")
                self.watch_fails = 0
                return True
            finally:
                self.restarting = False

    async def select(self, key, reason: str) -> bool:
        node = self.by_key.get(key)
        if node is None:
            return False
        async with self.sel_lock:
            try:
                st, _ = await http_local(CLASH_PORT, "PUT", "/proxies/proxy", {"name": node.tag}, self.secret)
            except (OSError, asyncio.TimeoutError, ValueError, IndexError) as e:
                log.warning(f"не удалось переключить узел: {e}")
                return False
            if st not in (200, 204):
                log.warning(f"переключение узла вернуло HTTP {st}")
                return False
            prev = self.active
            self.active = key
            if prev != key:
                self.stats["switches"] += 1
                sc = self.metrics.get(key, {}).get("score")
                log.info(f"🔀 активный узел: {node.name} ({node.kind})"
                         f"{f', {sc:.0f} мс' if sc else ''} — {reason}")
            self.save_state()
            return True

    # ---- замеры
    async def measure(self, reason: str):
        async with self.measure_lock:
            if self.restarting or not self.sb.alive() or not self.nodes:
                return
            t0 = time.monotonic()
            self.stats["measures"] += 1
            nodes = list(self.nodes)
            sem = asyncio.Semaphore(MEASURE_PARALLEL)
            stage1 = {}

            async def s1(n):
                async with sem:
                    res = await probe(n.port, post=False)
                m = self.metrics.setdefault(n.key, {})
                m.update(ok1=res.ok, ms1=res.ms, status=res.status, err=res.err, ts=time.time())
                if res.ok:
                    stage1[n.key] = res.ms
                    if self.active is None:  # первый ответивший сразу в дело, остальное доизмерим
                        await self.select(n.key, "первый ответивший узел (предварительно)")

            await asyncio.gather(*(s1(n) for n in nodes))

            ranked = sorted(stage1, key=stage1.get)
            cands = [k for k in ranked if not self.cooling(k)]
            if self.active in stage1 and self.active not in cands[:TOP_M]:
                cands.insert(0, self.active)  # текущий узел всегда сравниваем с остальными
            sem2 = asyncio.Semaphore(6)

            async def s2(key):
                n = self.by_key[key]
                times, err = [], ""
                async with sem2:
                    for _ in range(POST_SAMPLES):
                        res = await probe(n.port, post=True, timeout=MEASURE_TIMEOUT + 4)
                        if not res.ok:
                            err = res.err
                            break
                        times.append(res.ms)
                m = self.metrics.setdefault(key, {})
                if times and not err:
                    m.update(ok=True, score=statistics.median(times), err="")
                else:
                    m.update(ok=False, score=None, err=err or "нет ответов")
                return key, m["ok"]

            verified, i = [], 0
            while i < len(cands) and len(verified) < TOP_K:
                chunk = cands[i:i + TOP_M]
                i += TOP_M
                for key, ok in await asyncio.gather(*(s2(k) for k in chunk)):
                    if ok:
                        verified.append(key)
            verified.sort(key=lambda k: self.metrics[k]["score"])
            pool = verified[:TOP_K]
            self.last_measure = time.time()
            ok1 = len(stage1)
            summary = (f"узлов {len(nodes)}, ответили {ok1}, API доступен через {len(verified)}"
                       + (": " + " | ".join(f"{self.by_key[k].name[:28]}={self.metrics[k]['score']:.0f}мс" for k in pool[:5])
                          if pool else ""))
            self.last_measure_summary = summary
            log.info(f"🛣 замер ({reason}, {time.monotonic() - t0:.0f} с): {summary}")

            if not pool:
                self.last_error = "ни через один узел не доступен API"
                log.error("❌ ни через один узел не доступен API; повтор через 60 с")
                return
            cur = self.active
            cur_m = self.metrics.get(cur, {}) if cur else {}
            cur_valid = bool(cur) and cur_m.get("ok") and not self.cooling(cur)
            best = pool[0]
            if not cur_valid:
                await self.select(best, "прежний узел недоступен или не замерен")
            elif best != cur and self.metrics[best]["score"] < cur_m["score"] * SWITCH_MARGIN:
                await self.select(best, f"быстрее прежнего ({self.metrics[best]['score']:.0f} < {cur_m['score']:.0f} мс)")
            if self.active in pool:
                pool.remove(self.active)
            elif self.active is not None:
                pass
            self.pool = ([self.active] if self.active else []) + pool
            self.pool = self.pool[:TOP_K]
            self.last_error = ""
            self.save_state()

    async def failover(self):
        bad = self.active
        self.stats["failovers"] += 1
        if bad:
            self.cooldown[bad] = time.monotonic() + NODE_COOLDOWN
            self.metrics.setdefault(bad, {}).update(ok=False, err="отказ при работе")
            log.warning(f"🚨 активный узел не отвечает: {self.by_key[bad].name if bad in self.by_key else bad}")
        for key in list(self.pool):
            node = self.by_key.get(key)
            if key == bad or node is None or self.cooling(key):
                continue
            if (await probe(node.port, post=False, timeout=6)).ok and await self.select(key, "аварийное переключение"):
                if (await probe(SOCKS_PORT, post=False, timeout=6)).ok:
                    self.watch_fails = 0
                    self.pool = [k for k in self.pool if k != bad]
                    return
        self.pool = [k for k in self.pool if k != bad]
        log.error("в запасе нет живых узлов — запускаю полный замер")
        self.wake.set()

    # ---- фоновые циклы
    async def measure_loop(self):
        first = True
        while not self.stopping:
            if not first:
                wait = MEASURE_INTERVAL if self.pool else 60
                try:
                    await asyncio.wait_for(self.wake.wait(), wait)
                    await asyncio.sleep(5)  # не чаще раза в несколько секунд
                except asyncio.TimeoutError:
                    pass
            first = False
            self.wake.clear()
            try:
                await self.measure("плановый")
            except Exception:  # noqa: BLE001
                log.exception("ошибка замера")

    async def watchdog_loop(self):
        while not self.stopping:
            await asyncio.sleep(WATCH_INTERVAL)
            if self.restarting or not self.sb.alive() or self.active is None:
                self.watch_ok = False
                continue
            try:
                res = await probe(SOCKS_PORT, post=False, timeout=6)
                if res.ok:
                    self.watch_ok, self.watch_fails, self.last_ok = True, 0, time.time()
                    continue
                self.watch_ok = False
                self.watch_fails += 1
                log.warning(f"⚠️ проверка API через общий вход не прошла ({res.err}), {self.watch_fails}/{WATCH_FAILS}")
                if self.watch_fails >= WATCH_FAILS:
                    await self.failover()
            except Exception:  # noqa: BLE001
                log.exception("ошибка watchdog")

    async def pool_check_loop(self):
        while not self.stopping:
            await asyncio.sleep(POOL_CHECK_INTERVAL)
            if self.restarting or not self.sb.alive():
                continue
            try:
                for key in [k for k in self.pool if k != self.active]:
                    node = self.by_key.get(key)
                    if node is None or not (await probe(node.port, post=False, timeout=6)).ok:
                        log.warning(f"запасной узел {node.name if node else key} не отвечает, исключаю из запаса")
                        self.pool = [k for k in self.pool if k != key]
                if len(self.pool) < 2:
                    self.wake.set()
            except Exception:  # noqa: BLE001
                log.exception("ошибка проверки запаса")

    async def sub_loop(self):
        while not self.stopping:
            await asyncio.sleep(SUB_INTERVAL * (0.9 + random.random() * 0.2))
            try:
                links = await self.fetch_links()
                if links:
                    before = set(self.by_key)
                    if await self.apply_links(links):
                        self.stats["sub_updates"] += 1
                        if set(self.by_key) != before:
                            self.wake.set()  # набор узлов изменился — сразу перемеряем
            except Exception:  # noqa: BLE001
                log.exception("ошибка обновления подписки")

    async def sb_supervisor(self):
        backoff = 2
        while not self.stopping:
            await asyncio.sleep(2)
            if self.restarting or self.sb.expected_stop or self.sb.alive() or not self.nodes:
                backoff = 2 if self.sb.alive() else backoff
                continue
            log.error(f"sing-box остановился (код {self.sb.proc.returncode if self.sb.proc else '?'}), перезапуск через {backoff} с")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)
            try:
                async with self.rebuild_lock:
                    self.restarting = True
                    await self.sb.start()
                    if await self.wait_clash():
                        if self.active:
                            await self.select(self.active, "после перезапуска sing-box")
                        self.stats["restarts"] += 1
            except Exception:  # noqa: BLE001
                log.exception("ошибка перезапуска sing-box")
            finally:
                self.restarting = False

    # ---- статус
    def status(self) -> dict:
        def node_info(k):
            n, m = self.by_key.get(k), self.metrics.get(k, {})
            return {"name": n.name if n else k, "kind": n.kind if n else "", "key": k,
                    "score_ms": round(m["score"], 1) if m.get("score") else None}
        return {
            "ok": self.healthy(),
            "singbox_alive": self.sb.alive(),
            "nodes": len(self.nodes),
            "active": node_info(self.active) if self.active else None,
            "pool": [node_info(k) for k in self.pool],
            "api_probe_url": API_URL,
            "last_subscription_ago_s": round(time.time() - self.last_sub) if self.last_sub else None,
            "last_measure_ago_s": round(time.time() - self.last_measure) if self.last_measure else None,
            "last_measure": self.last_measure_summary,
            "last_error": self.last_error,
            "watch_fails": self.watch_fails,
            "stats": self.stats,
            "cooldown_nodes": sum(1 for k in self.cooldown if self.cooling(k)),
        }

    def healthy(self) -> bool:
        return bool(self.sb.alive() and self.active and (self.watch_ok or time.time() - self.last_ok < 120))

    async def handle_http(self, reader, writer):
        try:
            line = (await asyncio.wait_for(reader.readline(), 5)).decode("latin-1")
            path = line.split(" ")[1] if " " in line else "/"
            if path.startswith("/health"):
                code, body = (200 if self.healthy() else 503), {"ok": self.healthy()}
            elif path.startswith("/remeasure"):
                self.wake.set()
                code, body = 202, {"scheduled": True}
            else:
                code, body = 200, self.status()
            data = json.dumps(body, ensure_ascii=False).encode()
            writer.write(f"HTTP/1.1 {code} X\r\nContent-Type: application/json; charset=utf-8\r\n"
                         f"Content-Length: {len(data)}\r\nConnection: close\r\n\r\n".encode() + data)
            await writer.drain()
        except Exception:  # noqa: BLE001
            pass
        finally:
            writer.close()

    # ---- запуск
    async def run(self):
        self.load_state()
        host = urlsplit(SUB_URL).hostname or "?"
        log.info(f"🚀 менеджер VPN: подписка {host}, API {API_URL}, запас {TOP_K}, замер каждые {MEASURE_INTERVAL} с, "
                 f"обновление подписки каждые {SUB_INTERVAL} с")
        server = await asyncio.start_server(self.handle_http, "0.0.0.0", STATUS_PORT)

        links = None
        for attempt in range(3):
            links = await self.fetch_links()
            if links:
                break
            await asyncio.sleep(5)
        if not links:
            links = self.cached_links()
            if links:
                log.warning(f"используются сохранённые ссылки ({len(links)})")
        if not links:
            links = self.fallback_links()
            if links:
                log.warning(f"используется {FALLBACK_FILE} ({len(links)} ссылок)")
        while not links and not self.stopping:
            log.error("нет ни подписки, ни сохранённых ссылок; повтор через 30 с")
            await asyncio.sleep(30)
            links = await self.fetch_links()
        if self.stopping:
            return
        if not await self.apply_links(links, force=True):
            log.error("не удалось запустить sing-box с полученными узлами")
            await asyncio.sleep(10)
            raise SystemExit(1)

        tasks = [asyncio.create_task(c()) for c in (self.measure_loop, self.watchdog_loop, self.pool_check_loop,
                                                    self.sub_loop, self.sb_supervisor)]
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
        await stop.wait()
        log.info("получен сигнал остановки")
        self.stopping = True
        for t in tasks:
            t.cancel()
        server.close()
        await self.sb.stop()


if __name__ == "__main__":
    if not SUB_URL and not os.path.exists(FALLBACK_FILE):
        print("SUBSCRIPTION_URL не задан", file=sys.stderr)
        sys.exit(2)
    asyncio.run(Manager().run())
