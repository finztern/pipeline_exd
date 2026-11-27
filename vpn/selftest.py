#!/usr/bin/env python3
"""Самопроверка управляемого режима VPN.

  python3 vpn/selftest.py                          # разбор подписки и ссылок
  SING_BOX_BIN=/path/to/sing-box python3 vpn/selftest.py   # + проверка конфига и сквозной тест с мок-узлами

Сквозной тест поднимает настоящий sing-box, пять поддельных SOCKS5-узлов (быстрый, средний, медленный,
«блокирует API» = отдаёт 403, мёртвый) и поддельный API; запускает manager.py и проверяет:
выбор быстрейших рабочих узлов, исключение заблокированных/мёртвых, аварийное переключение и обновление подписки.
"""
import asyncio
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import links as L  # noqa: E402
import sbconfig  # noqa: E402
import subscription as S  # noqa: E402

SB = os.environ.get("SING_BOX_BIN", "")
FAILED = []


def check(name, cond, extra=""):
    print(f"  {'✓' if cond else '✗'} {name}{(' — ' + str(extra)) if extra and not cond else ''}")
    if not cond:
        FAILED.append(name)


def fake_links():
    import random
    import os as _os
    random.seed(7)

    def b64u(n):
        return base64.urlsafe_b64encode(_os.urandom(n)).decode().rstrip("=")

    out = []
    for i in range(3):
        out.append(f"vless://{uuid.UUID(int=random.getrandbits(128))}@10.0.0.{i + 1}:443?encryption=none&flow=xtls-rprx-vision"
                   f"&type=tcp&headerType=none&security=reality&fp=edge&pbk={b64u(32)}&sid=ab12cd34&sni=cdn.example.com#%F0%9F%87%A9%F0%9F%87%AA%20Node%20{i}")
    for i in range(2):
        out.append(f"vless://{uuid.UUID(int=random.getrandbits(128))}@10.0.1.{i + 1}:443?encryption=none&type=grpc&serviceName=svc{i}"
                   f"&mode=gun&security=tls&fp=qq&sni=x{i}.example.com&pcs=AA:BB:CC&alpn=h2#grpc%20{i}")
    for i in range(2):
        out.append(f"hysteria2://{uuid.UUID(int=random.getrandbits(128))}@10.0.2.{i + 1}:30443?sni=hy.example.com&alpn=h3"
                   f"&obfs=salamander&obfs-password=pw{i}&insecure=0#hy2%20obfs%20{i}")
    out.append(f"hysteria2://{uuid.UUID(int=random.getrandbits(128))}@10.0.2.9:33443?sni=hy.example.com&alpn=h3&insecure=0#hy2%20plain")
    return out


# ----------------------------------------------------------------------------- 1. разбор
def test_parsing():
    print("\n[1] разбор подписки и ссылок")
    ls = fake_links()
    plain = "\n".join(ls) + "\n"
    b64 = base64.b64encode(plain.encode()).decode()
    check("base64 (обычный алфавит)", S.decode_subscription(b64) == ls)
    wrapped = "\n".join(b64[i:i + 76] for i in range(0, len(b64), 76))
    check("base64 с переносами строк", S.decode_subscription(wrapped) == ls)
    check("base64 urlsafe без padding", S.decode_subscription(base64.urlsafe_b64encode(plain.encode()).decode().rstrip("=")) == ls)
    check("обычный текст", S.decode_subscription(plain) == ls)
    check("CRLF и BOM", S.decode_subscription("\ufeff" + plain.replace("\n", "\r\n")) == ls)
    check("мусор среди ссылок игнорируется", S.decode_subscription("# comment\nfoo\n" + plain + "\nbar") == ls)
    check("дубликаты убираются", S.decode_subscription(plain + plain) == ls)
    for bad, why in (("<html><body>403</body></html>", "HTML"), ('{"outbounds": []}', "JSON sing-box"),
                     ("proxies:\n - name: a", "Clash"), ("", "пусто"), ("!!!!", "не base64")):
        try:
            S.decode_subscription(bad)
            check(f"ошибка на «{why}»", False)
        except ValueError:
            check(f"понятная ошибка на «{why}»", True)

    nodes, skipped = sbconfig.make_nodes(ls)
    check("все 8 ссылок разобраны", len(nodes) == 8 and not skipped, skipped)
    check("порядок не зависит от порядка в подписке", [n.key for n in nodes] == [n.key for n in sbconfig.make_nodes(ls[::-1])[0]])
    renamed = [x.split("#")[0] + "#другое-имя" for x in ls]
    check("ключ не зависит от названия узла", {n.key for n in sbconfig.make_nodes(renamed)[0]} == {n.key for n in nodes})
    kinds = {n.name: n.kind for n in nodes}
    check("gRPC+TLS с pcs= помечен и без строгой проверки", any(k.endswith("+pcs") for k in kinds.values()))
    L.PCS_INSECURE = False
    ob, _ = L.parse_vless(ls[3])
    check("VPN_PCS_INSECURE=false оставляет проверку", ob["tls"]["insecure"] is False)
    L.PCS_INSECURE = True
    try:
        L.parse_vless("vless://u@h:1?type=quic&security=none")
        check("неподдерживаемый транспорт отклоняется", False)
    except ValueError:
        check("неподдерживаемый транспорт отклоняется", True)
    n2, sk = sbconfig.make_nodes(ls + ["vless://битая"])
    check("битая ссылка пропускается, остальные работают", len(n2) == 8 and len(sk) == 1)


# ----------------------------------------------------------------------------- 2. конфиг на настоящем sing-box
def test_singbox_check():
    print("\n[2] конфиг на настоящем sing-box")
    nodes, _ = sbconfig.make_nodes(fake_links())
    cfg = sbconfig.build_managed_config(nodes, socks_port=1080, http_port=1081, base_port=10800, log_level="warning",
                                        clash_port=9090, clash_secret="s", initial_key=nodes[2].key)
    check("selector: default = запрошенный узел", [o for o in cfg["outbounds"] if o["tag"] == "proxy"][0]["default"] == nodes[2].tag)
    check("нет прямого выхода (нет утечек мимо VPN)", not any(o["type"] == "direct" for o in cfg["outbounds"]))
    check("пробные входы слушают только localhost",
          all(i["listen"] == "127.0.0.1" for i in cfg["inbounds"] if i["tag"].startswith("in-")))
    d = tempfile.mkdtemp()
    p = os.path.join(d, "c.json")
    json.dump(cfg, open(p, "w"))
    r = subprocess.run([SB, "check", "-c", p], capture_output=True, text=True)
    check("sing-box check принимает конфиг", r.returncode == 0, r.stderr.strip()[-300:])
    shutil.rmtree(d, ignore_errors=True)


# ----------------------------------------------------------------------------- 3. сквозной тест
def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class Mock:
    """Поддельный API и поддельные SOCKS5-узлы в одном asyncio-цикле."""

    def __init__(self):
        self.servers, self.conns = {}, {}

    async def api(self, status):
        async def handle(r, w):
            try:
                head = await r.readuntil(b"\r\n\r\n")
                n = 0
                for line in head.split(b"\r\n"):
                    if line.lower().startswith(b"content-length:"):
                        n = int(line.split(b":")[1])
                if n:
                    await r.readexactly(n)
                body = b"{}"
                w.write(f"HTTP/1.1 {status} X\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + (b"" if head.startswith(b"HEAD") else body))
                await w.drain()
            except Exception:
                pass
            finally:
                w.close()
        srv = await asyncio.start_server(handle, "127.0.0.1", 0)
        return srv, srv.sockets[0].getsockname()[1]

    async def node(self, name, delay, override=None, dead=False):
        async def pipe(a, b):
            try:
                while data := await a.read(65536):
                    b.write(data)
                    await b.drain()
            except Exception:
                pass
            finally:
                try:
                    b.close()
                except Exception:
                    pass

        async def handle(r, w):
            self.conns.setdefault(name, []).append(w)
            try:
                if dead:
                    w.close()
                    return
                hdr = await r.readexactly(2)
                await r.readexactly(hdr[1])
                w.write(b"\x05\x00")
                req = await r.readexactly(4)
                atyp = req[3]
                if atyp == 1:
                    host = socket.inet_ntoa(await r.readexactly(4))
                elif atyp == 3:
                    host = (await r.readexactly((await r.readexactly(1))[0])).decode()
                else:
                    host = socket.inet_ntop(socket.AF_INET6, await r.readexactly(16))
                port = int.from_bytes(await r.readexactly(2), "big")
                if override:
                    host, port = override
                await asyncio.sleep(delay)
                ur, uw = await asyncio.open_connection(host, port)
                w.write(b"\x05\x00\x00\x01\x00\x00\x00\x00\x00\x00")
                await w.drain()
                await asyncio.gather(pipe(r, uw), pipe(ur, w))
            except Exception:
                try:
                    w.close()
                except Exception:
                    pass
        srv = await asyncio.start_server(handle, "127.0.0.1", 0)
        self.servers[name] = srv
        return srv.sockets[0].getsockname()[1]

    def kill(self, name):
        self.servers[name].close()
        for w in self.conns.get(name, []):
            try:
                w.close()
            except Exception:
                pass


async def get_json(port, path="/status"):
    try:
        r, w = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), 3)
        w.write(f"GET {path} HTTP/1.0\r\n\r\n".encode())
        await w.drain()
        data = await asyncio.wait_for(r.read(), 5)
        w.close()
        head, _, body = data.partition(b"\r\n\r\n")
        return int(head.split()[1]), json.loads(body or b"{}")
    except Exception:
        return 0, {}


async def curl_via(socks_port, url, scheme="socks5h"):
    p = await asyncio.create_subprocess_exec("curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "8",
                                             "-x", f"{scheme}://127.0.0.1:{socks_port}", url,
                                             stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, _ = await p.communicate()
    return out.decode().strip()


async def wait_for(pred, timeout, step=0.5):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        v = await pred()
        if v:
            return v
        await asyncio.sleep(step)
    return None


async def test_e2e():
    print("\n[3] сквозной тест: настоящий sing-box + мок-узлы + мок-API")
    mock = Mock()
    api_srv, api_port = await mock.api(200)
    waf_srv, waf_port = await mock.api(403)
    ports = {
        "fast": await mock.node("fast", 0.0, (("127.0.0.1", api_port))),
        "fast2": await mock.node("fast2", 0.03, ("127.0.0.1", api_port)),
        "mid": await mock.node("mid", 0.25, ("127.0.0.1", api_port)),
        "slow": await mock.node("slow", 0.9, ("127.0.0.1", api_port)),
        "waf": await mock.node("waf", 0.0, ("127.0.0.1", waf_port)),   # быстрый, но API за ним отдаёт 403
        "dead": await mock.node("dead", 0.0, dead=True),
    }
    tmp = tempfile.mkdtemp()
    sub_path = os.path.join(tmp, "sub.txt")

    def write_sub(names):
        ls = [f"socks5://127.0.0.1:{ports[n]}#{n}" for n in names]
        open(sub_path, "w").write(base64.b64encode(("\n".join(ls) + "\n").encode()).decode())

    write_sub(["fast", "fast2", "mid", "slow", "waf", "dead"])
    sp, hp, bp, cp, stp = free_port(), free_port(), 21800, free_port(), free_port()
    env = dict(os.environ, SUBSCRIPTION_URL=f"file://{sub_path}", SING_BOX_BIN=SB, API_URL=f"http://127.0.0.1:{api_port}/v1/upload",
               STATE_DIR=os.path.join(tmp, "state"), CONFIG_PATH=os.path.join(tmp, "config.json"),
               SOCKS_PORT=str(sp), HTTP_PORT=str(hp), LINK_BASE_PORT=str(bp), CLASH_PORT=str(cp), STATUS_PORT=str(stp),
               MEASURE_INTERVAL="25", WATCH_INTERVAL="2", WATCH_FAILS="2", TOP_K="3", TOP_M="4", MEASURE_POST_BYTES="20000",
               SUB_INTERVAL="12", MEASURE_TIMEOUT="4", POOL_CHECK_INTERVAL="5", MAIN_ADDRESS="0xTEST", NODE_COOLDOWN="120")
    mgr = subprocess.Popen([sys.executable, os.path.join(HERE, "manager.py")], env=env, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
    log_lines = []

    async def drain_log():
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, mgr.stdout.readline)
            if not line:
                return
            log_lines.append(line.rstrip())
    log_task = asyncio.create_task(drain_log())
    try:
        async def ready():
            st, d = await get_json(stp)
            return d if d.get("active") and len(d.get("pool", [])) >= 2 and d.get("last_measure") else None
        d = await wait_for(ready, 60)
        check("менеджер выбрал узел и собрал запас", bool(d), "\n".join(log_lines[-12:]))
        if not d:
            return
        names = {n["name"] for n in d["pool"]}
        check("в запасе только быстрые рабочие узлы", names <= {"fast", "fast2", "mid"} and len(names) == 3, names)
        check("узел с блокировкой API (403) исключён", "waf" not in names)
        check("мёртвый и медленный узлы не в запасе", not names & {"dead", "slow"}, names)
        check("активный — один из двух быстрых", d["active"]["name"] in ("fast", "fast2"), d["active"])
        check("через общий вход (SOCKS) API доступен", await curl_via(sp, f"http://127.0.0.1:{api_port}/") == "200")
        r = await curl_via(hp, f"http://127.0.0.1:{api_port}/", "http")
        check("через HTTP-вход API доступен", r == "200", r)
        st, _ = await get_json(stp, "/health")
        check("/health = 200", st == 200)

        # --- аварийное переключение
        first = d["active"]["name"]
        t_kill = time.monotonic()
        mock.kill(first)
        async def switched():
            st, dd = await get_json(stp)
            return dd if dd.get("active") and dd["active"]["name"] != first and dd.get("stats", {}).get("failovers", 0) >= 1 else None
        dd = await wait_for(switched, 40)
        check(f"отказ узла «{first}» → автоматически переключился (за {time.monotonic() - t_kill:.0f} с)", bool(dd), "\n".join(log_lines[-10:]))
        check("после переключения трафик идёт", await curl_via(sp, f"http://127.0.0.1:{api_port}/") == "200")

        # --- обновление подписки: убираем mid и slow, добавляем новый быстрый узел
        ports["new"] = await mock.node("new", 0.0, ("127.0.0.1", api_port))
        before = (await get_json(stp))[1]["stats"]["restarts"]
        write_sub(["fast", "fast2", "new", "waf", "dead"])
        async def updated():
            st, x = await get_json(stp)
            return x if x.get("stats", {}).get("restarts", 0) > before and x.get("nodes") == 5 else None
        x = await wait_for(updated, 40)
        check("подписка обновилась: набор узлов 5, sing-box перезапущен", bool(x), "\n".join(log_lines[-8:]))
        async def usable():
            st, y = await get_json(stp)
            return y if y.get("active") and y.get("last_measure_ago_s") is not None and "new" in {n["name"] for n in y["pool"] + [y["active"]]} else None
        y = await wait_for(usable, 50)
        check("новый узел замерен и попал в запас", bool(y), "\n".join(log_lines[-8:]))
        check("после обновления трафик идёт", await curl_via(sp, f"http://127.0.0.1:{api_port}/") == "200")
        leaked = [l for l in log_lines if "socks5://127.0.0.1" in l and "sub.txt" in l]
        check("ссылка подписки не попала в логи", not leaked and not any(f"file://{sub_path}" in l for l in log_lines))
    finally:
        mgr.terminate()
        try:
            mgr.wait(10)
        except subprocess.TimeoutExpired:
            mgr.kill()
        log_task.cancel()
        subprocess.run(["pkill", "-f", f"{tmp}/config.json"], capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_parsing()
    if SB and os.path.exists(SB):
        test_singbox_check()
        asyncio.run(test_e2e())
    else:
        print("\n(SING_BOX_BIN не задан: проверка конфига и сквозной тест пропущены)")
    print(f"\n{'✅ ВСЁ ПРОШЛО' if not FAILED else '❌ ПРОВАЛЕНО: ' + ', '.join(FAILED)}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
