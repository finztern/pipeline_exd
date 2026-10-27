import asyncio
import logging
import os
import statistics
import time
from urllib.parse import quote, unquote, urlsplit

from aiohttp import BasicAuth, ClientSession, ClientTimeout, FormData, TCPConnector
from aiohttp_socks import ProxyConnector

log = logging.getLogger("routes")

MODE            = os.getenv("ROUTE_MODE", "auto").strip().lower()
ALLOW_DIRECT    = os.getenv("ROUTE_ALLOW_DIRECT", "true").lower() == "true"
VPN_FILE        = os.getenv("ROUTE_VPN_FILE", "/data/vpn.txt")
PROXY_FILE      = os.getenv("ROUTE_PROXY_FILE", "/data/proxy.txt")
VLESS_URL       = os.getenv("VLESS_URL", "").strip()
VPN_HOST        = os.getenv("ROUTE_VPN_HOST", "vpn")
VPN_BASE_PORT   = int(os.getenv("ROUTE_VPN_BASE_PORT", "10800"))
PROBE_INTERVAL  = float(os.getenv("ROUTE_PROBE_INTERVAL", "300"))
PROBE_SAMPLES   = max(1, int(os.getenv("ROUTE_PROBE_SAMPLES", "3")))
PROBE_TIMEOUT   = float(os.getenv("ROUTE_PROBE_TIMEOUT", "15"))
PROXY_MAX_MS    = float(os.getenv("ROUTE_PROXY_MAX_MS", "3000"))
PROBE_POST_BYTES = int(os.getenv("ROUTE_PROBE_POST_BYTES", "262144"))
MAIN_ADDRESS    = os.getenv("MAIN_ADDRESS", "")
SWITCH_MARGIN   = float(os.getenv("ROUTE_SWITCH_MARGIN", "0.85"))
FAIL_COOLDOWN   = float(os.getenv("ROUTE_FAIL_COOLDOWN", "60"))
FAIL_COOLDOWN_MAX = 1800.0
PROBE_PARALLEL  = 8
MIN_PROBE_GAP   = 10.0
BAD_STATUS      = {403, 407, 451, 500, 502, 503, 504}
SOCKS_SCHEMES   = {"socks4", "socks5", "socks5h"}


def _read_lines(path: str) -> list:
    try:
        with open(path, encoding="utf-8") as f:
            return [l.strip() for l in f.read().splitlines() if l.strip() and not l.strip().startswith("#")]
    except OSError:
        return []


def _vpn_links() -> list:
    links = [l for l in _read_lines(VPN_FILE) if l.startswith("vless://")]
    if not links and VLESS_URL.startswith("vless://"):
        links = [VLESS_URL]
    return links


def _parse_proxy(line: str) -> list:
    if "://" in line:
        scheme, rest = line.split("://", 1)
        scheme = scheme.lower()
        schemes = [{"socks": "socks5"}.get(scheme, scheme)]
    else:
        rest, schemes = line, ["http", "socks5"]
    u = urlsplit("//" + rest.rstrip("/"))
    try:
        host, port = u.hostname, u.port
    except ValueError:
        return []
    if not host or not port:
        return []
    user, pwd = unquote(u.username or ""), unquote(u.password or "")
    return [(s, host, port, user, pwd) for s in schemes if s == "http" or s in SOCKS_SCHEMES]


class Route:
    def __init__(self, name, kind, scheme="", host="", port=0, user="", pwd=""):
        self.name, self.kind, self.scheme = name, kind, scheme
        self.host, self.port, self.user, self.pwd = host, port, user, pwd
        self.ok = False
        self.latency = None
        self.error = "не проверен"
        self.fail_until = 0.0
        self.fail_streak = 0
        self.uploads = 0
        self.failures = 0
        self._session = None

    def _session_get(self) -> ClientSession:
        if self._session is None or self._session.closed:
            if self.scheme in SOCKS_SCHEMES:
                auth = f"{quote(self.user, safe='')}:{quote(self.pwd, safe='')}@" if self.user else ""
                conn = ProxyConnector.from_url(
                    f"{self.scheme}://{auth}{self.host}:{self.port}", rdns=True, keepalive_timeout=30,
                )
            else:
                conn = TCPConnector(keepalive_timeout=30)
            self._session = ClientSession(connector=conn)
        return self._session

    def request(self, method: str, url: str, **kw):
        if self.scheme == "http":
            kw["proxy"] = f"http://{self.host}:{self.port}"
            if self.user:
                kw["proxy_auth"] = BasicAuth(self.user, self.pwd)
        return self._session_get().request(method, url, **kw)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


class RouteManager:
    def __init__(self, upload_url: str):
        u = urlsplit(upload_url)
        self.upload_url = upload_url
        self.probe_url = os.getenv("ROUTE_PROBE_URL", "").strip() or f"{u.scheme}://{u.netloc}/"
        self.routes = self._build()
        self.current = None
        self.probed_at = 0.0
        self._wake = asyncio.Event()

    def _build(self) -> list:
        routes = []
        if MODE in ("auto", "vpn"):
            for i, link in enumerate(_vpn_links()):
                u = urlsplit(link)
                remark = unquote(u.fragment) or u.hostname or ""
                routes.append(Route(f"vpn-{i}:{remark}", "vpn", "socks5", VPN_HOST, VPN_BASE_PORT + i))
        if MODE == "direct" or (MODE == "auto" and ALLOW_DIRECT):
            routes.append(Route("direct", "direct"))
        if MODE in ("auto", "proxy"):
            for line in _read_lines(PROXY_FILE):
                for scheme, host, port, user, pwd in _parse_proxy(line):
                    routes.append(Route(f"proxy:{host}:{port}/{scheme}", "proxy", scheme, host, port, user, pwd))
        return routes

    def _usable(self, r: Route) -> bool:
        if not r.ok or r.fail_until > time.monotonic():
            return False
        return r.kind != "proxy" or r.latency <= PROXY_MAX_MS

    async def _probe_one(self, r: Route, sem: asyncio.Semaphore):
        async with sem:
            times, err = [], ""
            for _ in range(PROBE_SAMPLES):
                t0 = time.perf_counter()
                try:
                    if PROBE_POST_BYTES > 0:
                        form = FormData()
                        body = '{"items":[],"kind":"SPOTTING","pad":"' + "x" * PROBE_POST_BYTES + '"}'
                        form.add_field("file", body, filename="probe.json", content_type="application/json")
                        req = r.request("POST", self.upload_url, data=form, headers={"MAIN_ADDRESS": MAIN_ADDRESS},
                                        timeout=ClientTimeout(total=PROBE_TIMEOUT))
                    else:
                        req = r.request("HEAD", self.probe_url, timeout=ClientTimeout(total=PROBE_TIMEOUT),
                                        allow_redirects=False)
                    async with req as resp:
                        await resp.read()
                        status = resp.status
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"[:120]
                    break
                if status in BAD_STATUS:
                    err = f"HTTP {status}"
                    break
                times.append((time.perf_counter() - t0) * 1000)
            r.ok = bool(times) and not err
            r.latency = statistics.median(times) if r.ok else None
            r.error = "" if r.ok else (err or "нет ответов")

    async def probe_all(self):
        sem = asyncio.Semaphore(PROBE_PARALLEL)
        await asyncio.gather(*(self._probe_one(r, sem) for r in self.routes))
        self.probed_at = time.time()
        self._choose()
        log.info("🛣 маршруты: " + " | ".join(
            f"{r.name}={r.latency:.0f}ms" if r.ok else f"{r.name}=✗({r.error})" for r in self.routes
        ) + f" → активный: {self.active_name()}")

    def _choose(self):
        good = sorted((r for r in self.routes if self._usable(r)), key=lambda r: r.latency)
        if not good:
            if self.current is not None:
                log.error("❌ ни один маршрут не прошёл проверку")
            self.current = None
            return
        best = good[0]
        cur = self.current
        if cur in good and best is not cur and best.latency >= cur.latency * SWITCH_MARGIN:
            best = cur
        if best is not cur:
            log.info(f"🔀 маршрут: {cur.name if cur else '-'} → {best.name} ({best.latency:.0f}ms)")
        self.current = best

    def pick(self):
        now = time.monotonic()
        if self.current and self.current.fail_until <= now:
            return self.current
        live = [r for r in self.routes if r.fail_until <= now]
        good = sorted((r for r in live if self._usable(r)), key=lambda r: r.latency)
        if good:
            return good[0]
        pool = live or self.routes
        return min(pool, key=lambda r: (r.kind == "proxy", r.fail_until), default=None)

    def report_success(self, r: Route):
        r.uploads += 1
        r.fail_streak = 0

    def report_failure(self, r: Route):
        r.failures += 1
        r.fail_streak += 1
        r.fail_until = time.monotonic() + min(FAIL_COOLDOWN * 2 ** (r.fail_streak - 1), FAIL_COOLDOWN_MAX)
        log.warning(f"⚠️ маршрут {r.name}: сбой #{r.fail_streak}, пауза до повторного использования")
        if self.current is r:
            self.current = None
        self._wake.set()

    async def run(self):
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), PROBE_INTERVAL)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            try:
                await self.probe_all()
            except Exception as e:
                log.error(f"probe error: {e}", exc_info=True)
            await asyncio.sleep(MIN_PROBE_GAP)

    def active_name(self) -> str:
        r = self.current
        return r.name if r else "-"

    def snapshot(self) -> dict:
        now = time.monotonic()
        return {
            "mode": MODE,
            "active": self.active_name(),
            "probe_url": self.upload_url if PROBE_POST_BYTES > 0 else self.probe_url,
            "probe_post_bytes": PROBE_POST_BYTES,
            "probed_at": int(self.probed_at),
            "proxy_max_ms": PROXY_MAX_MS,
            "routes": [
                {
                    "name": r.name,
                    "kind": r.kind,
                    "ok": r.ok,
                    "latency_ms": round(r.latency, 1) if r.latency is not None else None,
                    "error": r.error,
                    "cooldown_s": max(0, int(r.fail_until - now)),
                    "uploads": r.uploads,
                    "failures": r.failures,
                    "active": r is self.current,
                }
                for r in sorted(self.routes, key=lambda r: (not r.ok, r.latency or 1e9))
            ],
        }

    async def close(self):
        await asyncio.gather(*(r.close() for r in self.routes), return_exceptions=True)
