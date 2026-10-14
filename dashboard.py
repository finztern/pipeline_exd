#!/usr/bin/env python3
import json
import os
import re
import select
import shutil
import subprocess
import sys
import termios
import time
import tty
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))


def read_env():
    env = {}
    try:
        with open(os.path.join(ROOT, ".env")) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return env


ENV = read_env()
COLLECTOR = os.getenv("COLLECTOR_URL", "http://localhost:9000").rstrip("/")
GPU_URL = (os.getenv("GPU_TRANSLATOR_URL") or ENV.get("GPU_TRANSLATOR_URL") or "http://192.168.0.100:8004/").rstrip("/") + "/"
try:
    N_BPIPE = max(1, min(3, int(ENV.get("BPIPE_REPLICAS", "1"))))
except ValueError:
    N_BPIPE = 1
BPIPES = [("bpipe", 7995), ("bpipe_2", 7996), ("bpipe_3", 7997)][:N_BPIPE]
SVC_ORDER = ["vpn", "translator", "transactioneer"] + [n for n, _ in BPIPES] + ["upipe", "collector"]

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

R, G, Y, C, M, D = "\033[31m", "\033[32m", "\033[33m", "\033[36m", "\033[35m", "\033[90m"
BD, RS = "\033[1m", "\033[0m"
ANSI = re.compile(r"\033\[[0-9;?]*[A-Za-z]")
SP = "▁▂▃▄▅▆▇█"


def col(s, c):
    return f"{c}{s}{RS}"


def clip(s, w):
    out, n, i = [], 0, 0
    while i < len(s):
        m = ANSI.match(s, i)
        if m:
            out.append(m.group())
            i = m.end()
            continue
        if n >= w:
            break
        out.append(s[i])
        n += 1
        i += 1
    return "".join(out) + RS


def cell(s, w, a="<", c=""):
    s = str(s)
    if len(s) > w:
        s = s[: w - 1] + "…"
    s = s.ljust(w) if a == "<" else s.rjust(w)
    return f"{c}{s}{RS}" if c else s


def head(title, W):
    return BD + C + "━━ " + title + " " + "━" * max(0, W - len(title) - 4) + RS


def bar(frac, w=20, color=None):
    frac = max(0.0, min(1.0, frac or 0.0))
    n = int(round(frac * w))
    c = color or (G if frac < 0.5 else (Y if frac < 0.8 else R))
    return c + "█" * n + D + "░" * (w - n) + RS


def spark(vals, w, c=C):
    vals = list(vals)[-w:]
    if not vals:
        return D + "·" * w + RS
    mx = max(vals) or 1.0
    s = "".join(SP[min(7, int(v / mx * 7.999))] for v in vals)
    return c + " " * (w - len(vals)) + s + RS


def fn(x, d=0):
    if x is None:
        return "—"
    x = float(x)
    if abs(x) >= 1e6:
        return f"{x / 1e6:.1f}M"
    if abs(x) >= 1e5:
        return f"{x / 1e3:.0f}k"
    if d and abs(x) < 100:
        return f"{x:.{d}f}"
    return f"{x:,.0f}".replace(",", " ")


def dur(sec):
    sec = int(max(0, sec))
    d, r = divmod(sec, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    if d:
        return f"{d}д{h}ч"
    if h:
        return f"{h}ч{m}м"
    if m:
        return f"{m}м{s}с"
    return f"{s}с"


def g(d, *p, default=0):
    for k in p:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
        if d is None:
            return default
    return d


def http_json(url, timeout=2.5):
    try:
        with OPENER.open(url, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read())
        except Exception:
            return None
    except Exception:
        return None


def sh(cmd, timeout=8):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout
    except Exception:
        return 1, ""


def docker_json(container, url):
    rc, out = sh(["docker", "exec", container, "curl", "-sf", "--noproxy", "*", "--max-time", "3", url])
    if rc != 0 or not out:
        return None
    try:
        return json.loads(out)
    except ValueError:
        return None


def parse_ts(s):
    if not s or s.startswith("0001"):
        return None
    try:
        return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def inspect(names):
    fmt = ("{{.Name}}|{{.State.Status}}|{{if .State.Health}}{{.State.Health.Status}}{{else}}-{{end}}|"
           "{{.RestartCount}}|{{.State.StartedAt}}|{{.State.OOMKilled}}|{{.State.ExitCode}}")
    rc, out = sh(["docker", "inspect", "--format", fmt] + names)
    res = {}
    for line in out.splitlines():
        p = line.split("|")
        if len(p) < 7:
            continue
        try:
            res[p[0].lstrip("/")] = dict(status=p[1], health=p[2], restarts=int(p[3] or 0),
                                         started=parse_ts(p[4]), oom=p[5] == "true", exit=int(p[6] or 0))
        except ValueError:
            continue
    return res


def nvidia():
    rc, out = sh(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                  "--format=csv,noheader,nounits"], 5)
    if rc != 0 or not out.strip():
        return None
    try:
        n, u, mu, mt, t = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        return dict(name=n, util=float(u), mem=float(mu), mem_total=float(mt), temp=float(t))
    except ValueError:
        return None


def host_info():
    info = {}
    try:
        info["load"] = os.getloadavg()
        mem = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                mem[k] = int(v.split()[0])
        info["ram_total"] = mem["MemTotal"] / 1048576
        info["ram_used"] = (mem["MemTotal"] - mem["MemAvailable"]) / 1048576
        info["cpus"] = os.cpu_count() or 1
    except Exception:
        pass
    return info


class Counters:
    def __init__(self):
        self.h = {}

    def push(self, key, val, t):
        q = self.h.setdefault(key, deque())
        q.append((t, val))
        while q and q[0][0] < t - 3700:
            q.popleft()

    def delta(self, key, secs):
        q = self.h.get(key)
        if not q or len(q) < 2:
            return 0.0, 0.0
        t1, v1 = q[-1]
        base = q[0]
        for s in q:
            if s[0] >= t1 - secs:
                base = s
                break
        t0, v0 = base
        if v1 < v0:
            return float(v1), t1 - t0
        return float(v1 - v0), t1 - t0

    def rate(self, key, secs):
        d, span = self.delta(key, secs)
        return d / span if span > 0 else 0.0

    def series(self, key, n):
        q = list(self.h.get(key, []))[-(n + 1):]
        out = []
        for a, b in zip(q, q[1:]):
            dt = b[0] - a[0]
            out.append(max(0.0, (b[1] - a[1]) / dt) if dt > 0 and b[1] >= a[1] else 0.0)
        return out


ERR_SPEC = [
    ("collector", [("errors", "e"), ("dropped", "e"), ("gpu_failed", "e"), ("filtered_lang", "i"),
                   ("filtered_old", "i"), ("filtered_dup", "i"), ("foreign_sent", "i")]),
    ("upipe", [("errors", "e"), ("dropped", "e"), ("translate_errors", "e"), ("filtered_lang", "i"),
               ("translate_fallback", "i"), ("translate_identical", "i")]),
    ("bpipe", [("errors", "e"), ("dropped", "e"), ("dropped_translated", "i"), ("dropped_default", "i"),
               ("dropped_buffer", "i"), ("expired_buffer", "i")]),
    ("tx", [("uploads_failed", "e"), ("rejected", "i"), ("duplicates", "i"), ("uploads_success", "i")]),
    ("trcpu", [("errors", "e"), ("timeouts", "e"), ("rejected_busy", "i"), ("rejected_no_pack", "i"),
               ("rejected_too_long", "i")]),
    ("trgpu", [("errors", "e"), ("send_errors", "e"), ("rejected_busy", "i"), ("unsupported", "i"),
               ("identical", "i")]),
]
COMP_NAMES = {"collector": "collector", "upipe": "upipe", "bpipe": "bpipe", "tx": "transactioneer",
              "trcpu": "translator CPU", "trgpu": "translator GPU"}


class State:
    def __init__(self, interval):
        self.interval = interval
        self.cnt = Counters()
        self.gauge = {}
        self.events = deque(maxlen=14)
        self.col = None
        self.bp = {}
        self.tr = None
        self.gs = None
        self.gh = None
        self.ins = {}
        self.prev_ins = {}
        self.nv = None
        self.host = {}
        self.cur = {}
        self.poll_ms = 0
        self.t = 0
        self.polled = False
        self.prev_col_up = None
        self.prev_gpu_up = None
        self.prev_gpu_uptime = None
        self.ev_base = {}
        self.ev_t = {}

    def add_event(self, t, lvl, text):
        self.events.appendleft((t, lvl, text))

    def pd(self, prefix, dct, t):
        for k, v in (dct or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                key = f"{prefix}.{k}"
                self.cnt.push(key, v, t)
                self.cur[key] = v

    def pv(self, key, val, t):
        self.cnt.push(key, val, t)
        self.cur[key] = val

    def pg(self, key, val):
        self.gauge.setdefault(key, deque(maxlen=24)).append(val)

    def poll(self):
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=16) as ex:
            f_col = ex.submit(http_json, COLLECTOR + "/stats")
            f_bp = {n: ex.submit(docker_json, n, f"http://localhost:{p}/health") for n, p in BPIPES}
            f_tr = ex.submit(docker_json, "translator", "http://localhost:8003/stats")
            f_gs = ex.submit(http_json, GPU_URL + "stats")
            f_gh = ex.submit(http_json, GPU_URL + "health")
            f_ins = ex.submit(inspect, SVC_ORDER)
            f_nv = ex.submit(nvidia)
            self.col = f_col.result()
            self.bp = {n: f.result() for n, f in f_bp.items()}
            self.tr = f_tr.result()
            self.gs = f_gs.result()
            self.gh = f_gh.result()
            self.ins = f_ins.result()
            self.nv = f_nv.result()
        self.host = host_info()
        t = time.time()
        self.t = t
        self.poll_ms = (t - t0) * 1000
        self.cur = {}
        self.push_all(t)
        self.detect(t)
        self.polled = True

    def push_all(self, t):
        c = self.col or {}
        self.pd("collector", c.get("collector"), t)
        self.pd("upipe", g(c, "upipe", "stats", default={}), t)
        tx = g(c, "exorde", default={})
        self.pd("tx", tx.get("stats"), t)
        self.pd("tx", tx.get("totals"), t)
        td = g(tx, "domains", "totals", default={})
        self.pv("tx.sent_ok", sum((td.get("s") or {}).values()), t)
        self.pv("tx.failed_items", sum((td.get("x") or {}).values()), t)
        bsum = {}
        for h in self.bp.values():
            for k, v in (h or {}).get("stats", {}).items():
                if isinstance(v, (int, float)):
                    bsum[k] = bsum.get(k, 0) + v
        if any(self.bp.values()):
            self.pd("bpipe", bsum, t)
        if self.tr:
            self.pd("trcpu", self.tr.get("stats"), t)
        if self.gs:
            self.pd("trgpu", self.gs.get("stats"), t)
        up = g(c, "upipe", default={})
        if up.get("limit"):
            self.pg("upipe", up.get("queue", 0) / up["limit"])
        for n, h in self.bp.items():
            if h:
                q = h.get("queues", {}).get("default", {})
                self.pg(n, q.get("fill", 0))
        if self.tr and self.tr.get("queue_max"):
            self.pg("trcpu", self.tr.get("fill", 0))
        if self.gh and self.gh.get("max"):
            self.pg("trgpu", self.gh.get("queue", 0) / self.gh["max"])
        if self.nv:
            self.pg("gpu_util", self.nv["util"] / 100)
            self.pg("gpu_mem", self.nv["mem"] / max(1, self.nv["mem_total"]))

    def detect(self, t):
        add = lambda lvl, text: self.add_event(t, lvl, text)
        col_up = self.col is not None
        if self.prev_col_up is not None and col_up != self.prev_col_up:
            add("G" if col_up else "R", "collector " + ("снова доступен" if col_up else "недоступен"))
        self.prev_col_up = col_up
        for n, s in self.ins.items():
            p = self.prev_ins.get(n)
            if not p:
                continue
            if s["restarts"] > p["restarts"]:
                add("R", f"{n}: перезапуск #{s['restarts']} (exit={s['exit']}, OOM={s['oom']})")
            if s["status"] != p["status"]:
                add("G" if s["status"] == "running" else "R", f"{n}: {p['status']} → {s['status']}")
            if s["health"] != p["health"]:
                add("G" if s["health"] == "healthy" else ("Y" if s["health"] == "starting" else "R"),
                    f"{n}: health {p['health']} → {s['health']}")
        self.prev_ins = self.ins
        gpu_up = self.gs is not None
        if self.prev_gpu_up is not None and gpu_up != self.prev_gpu_up:
            add("G" if gpu_up else "R", "GPU-переводчик " + ("online" if gpu_up else "offline"))
        self.prev_gpu_up = gpu_up
        gu = g(self.gs, "uptime_s", default=None)
        if gu is not None:
            if self.prev_gpu_uptime is not None and gu + 5 < self.prev_gpu_uptime:
                add("R", "GPU-переводчик перезапущен")
            self.prev_gpu_uptime = gu
        for comp, metrics in ERR_SPEC:
            for m, kind in metrics:
                if kind != "e":
                    continue
                key = f"{comp}.{m}"
                if key not in self.cur:
                    continue
                v = self.cur[key]
                base = self.ev_base.setdefault(key, v)
                if v > base and t - self.ev_t.get(key, 0) > 20:
                    add("Y", f"{COMP_NAMES[comp]}: {m} +{fn(v - base)} (всего {fn(v)})")
                    self.ev_base[key] = v
                    self.ev_t[key] = t
                elif v < base:
                    self.ev_base[key] = v

    def hw(self, key, mult=60):
        out = []
        for s in (600, 1800, 3600):
            d, span = self.cnt.delta(key, s)
            if span <= 0:
                out.append((None, True))
            else:
                out.append((d / span * mult, span < s * 0.9))
        return out

    def problems(self):
        bad = []
        if self.col is None:
            bad.append("collector недоступен")
        for n in SVC_ORDER:
            s = self.ins.get(n)
            if not s:
                continue
            if s["status"] != "running":
                bad.append(f"{n}:{s['status']}")
            elif s["health"] == "unhealthy":
                bad.append(f"{n}:unhealthy")
        if self.gs is None:
            bad.append("translator_gpu offline")
        return bad


def dom_win_sum(dd, kind):
    if not dd:
        return [(None, False)] * 3
    out = []
    for w in ("10m", "30m", "60m"):
        wd = g(dd, "windows", w, default={})
        span = max(1, g(wd, "window_s", default=1))
        out.append((sum((wd.get(kind) or {}).values()) / span * 60, False))
    return out


def dom_val(dd, kind, w, dom):
    wd = g(dd, "windows", w, default={})
    span = max(1, g(wd, "window_s", default=1))
    return (wd.get(kind) or {}).get(dom, 0) / span * 60


def sec_header(st, W):
    now = datetime.now().strftime("%H:%M:%S")
    bad = st.problems()
    status = col("● ВСЁ В НОРМЕ", G) if not bad else col("● ПРОБЛЕМЫ: " + ", ".join(bad), R)
    return [BD + "EXORDE PIPELINE" + RS + D + f"  {now}  опрос {st.poll_ms:.0f}мс  интервал {st.interval:g}с" + RS + "   " + status]


def sec_svc(st, W):
    L = [head("СЕРВИСЫ", W)]
    L.append(D + cell("СЕРВИС", 18) + cell("СТАТУС", 10) + cell("HEALTH", 10) + cell("UPTIME", 9)
             + cell("РЕСТАРТЫ", 9) + cell("OOM", 5) + cell("EXIT", 5) + "ИНФО" + RS)
    now = time.time()
    c = st.col or {}
    for name in SVC_ORDER:
        s = st.ins.get(name)
        if not s:
            L.append(cell(name, 18, c=R) + col("контейнер не найден", R))
            continue
        up = (now - s["started"]) if s["started"] and s["status"] == "running" else 0
        sc = G if s["status"] == "running" else R
        hc = {"healthy": G, "starting": Y, "unhealthy": R}.get(s["health"], D)
        rc = G if s["restarts"] == 0 else (R if up < 600 else Y)
        info = ""
        if name.startswith("bpipe"):
            h = st.bp.get(name)
            info = f"batch={h.get('batch_size')} inflight={h.get('inflight')} recv={fn(g(h, 'stats', 'received'))}" if h else col("API не отвечает", R)
        elif name == "transactioneer":
            info = f"активных загрузок={g(c, 'exorde', 'active_tasks')}"
        elif name == "translator":
            info = ("вкл" if g(c, "translate", "enabled", default=False) else "выкл (TRANSLATE=false)") + f" языков={len(g(c, 'translate', 'installed', default=[]))}"
        elif name == "upipe":
            info = f"queue={g(c, 'upipe', 'queue')}/{g(c, 'upipe', 'limit')}"
        elif name == "collector":
            info = f"uptime(api)={dur(g(c, 'uptime_s'))}"
        L.append(cell(name, 18) + cell(s["status"], 10, c=sc) + cell(s["health"], 10, c=hc) + cell(dur(up), 9)
                 + cell(s["restarts"], 9, c=rc) + cell("ДА" if s["oom"] else "нет", 5, c=R if s["oom"] else D)
                 + cell(s["exit"], 5, c=G if s["exit"] == 0 else R) + info)
    gs, gh = st.gs, st.gh
    if gs is None:
        L.append(cell("translator_gpu*", 18, c=R) + cell("offline", 10, c=R) + col(GPU_URL, D))
    else:
        ready = g(gh, "ready", default=False)
        L.append(cell("translator_gpu*", 18) + cell("online", 10, c=G)
                 + cell("ready" if ready else "starting", 10, c=G if ready else Y)
                 + cell(dur(g(gs, "uptime_s")), 9) + cell("—", 9, c=D) + cell("—", 5, c=D) + cell("—", 5, c=D)
                 + f"queue={g(gh, 'queue')}/{g(gh, 'max')} моделей opus={len(g(gh, 'opus', default=[]))}  {GPU_URL}")
    return L


def sec_flow(st, W):
    c = st.col or {}
    cd = g(c, "domains", default={})
    tx = g(c, "exorde", default={})
    td = g(tx, "domains", default={})
    L = [head("ПОТОК ДАННЫХ  (шт/мин; ~ = среднее с момента запуска панели)", W)]
    L.append(D + cell("ЭТАП", 30) + cell("ВСЕГО", 10, ">") + cell("СЕЙЧАС", 9, ">") + cell("10м", 9, ">")
             + cell("30м", 9, ">") + cell("60м", 9, ">") + "  ТРЕНД" + RS)

    def row(label, total, key, wins, color):
        now = st.cnt.rate(key, 60) * 60 if key in st.cnt.h else None
        cs = [cell(label, 30, c=color), cell(fn(total), 10, ">"), cell(fn(now, 1), 9, ">")]
        for v, ap in wins:
            cs.append(cell(("~" if ap else "") + fn(v, 1) if v is not None else "—", 9, ">"))
        return " ".join(cs) + "  " + spark(st.cnt.series(key, 24), 24, color)

    cur = st.cur
    rate_w = lambda k: [(g(tx, "rate", w, k) * 60, False) for w in ("10m", "30m", "60m")]
    L.append(row("Поступило в collector", cur.get("collector.received"), "collector.received", dom_win_sum(cd, "r"), C))
    L.append(row("Прошло фильтры → upipe", cur.get("collector.forwarded"), "collector.forwarded", dom_win_sum(cd, "f"), C))
    L.append(row("upipe → bpipe", cur.get("upipe.forwarded"), "upipe.forwarded", st.hw("upipe.forwarded"), M))
    L.append(row("bpipe → transactioneer", cur.get("bpipe.items_sent"), "bpipe.items_sent", st.hw("bpipe.items_sent"), M))
    L.append(row("Отправлено в Exorde", cur.get("tx.sent_ok"), "tx.sent_ok", dom_win_sum(td, "s"), Y))
    L.append(row("  принято", cur.get("tx.accepted"), "tx.accepted", rate_w("accepted_per_sec"), G))
    L.append(row("  отклонено", cur.get("tx.rejected"), "tx.rejected", rate_w("rejected_per_sec"), R))
    L.append(row("  дубликаты", cur.get("tx.duplicates"), "tx.duplicates", rate_w("duplicates_per_sec"), Y))
    rec = cur.get("collector.received") or 0
    acc = cur.get("tx.accepted") or 0
    fwd = cur.get("collector.forwarded") or 0
    pct = lambda a, b: f"{a / b * 100:.1f}%" if b else "—"
    L.append(D + f"Конверсия за всё время: прошло фильтры {pct(fwd, rec)} · принято Exorde {pct(acc, rec)} от поступившего · "
             f"загрузок ок/ошибок {fn(cur.get('tx.uploads_success'))}/{fn(cur.get('tx.uploads_failed'))}" + RS)
    return L


def sec_dom(st, W, rows):
    c = st.col or {}
    cd = g(c, "domains", default={})
    td = g(c, "exorde", "domains", default={})
    doms = set(g(cd, "totals", "r", default={})) | set(g(td, "totals", "s", default={})) | set(g(td, "totals", "x", default={}))
    L = [head("ДОМЕНЫ  (шт/мин, среднее за окно)", W)]
    wide = W >= 135
    hdr = (cell("ДОМЕН", 24) + cell("IN всего", 9, ">") + cell("IN 10м", 8, ">") + cell("IN 30м", 8, ">")
           + cell("IN 60м", 8, ">") + cell("→PIPE", 7, ">") + cell("OUT всего", 10, ">") + cell("OUT 10м", 8, ">")
           + cell("OUT 30м", 8, ">") + cell("OUT 60м", 8, ">") + cell("FAIL", 7, ">"))
    L.append(D + hdr + ("  ДОЛЯ IN 60м" if wide else "") + RS)
    data = []
    for d in doms:
        r = g(cd, "totals", "r", d, default=0)
        f = g(cd, "totals", "f", d, default=0)
        s = g(td, "totals", "s", d, default=0)
        x = g(td, "totals", "x", d, default=0)
        data.append(dict(d=d, r=r, f=f, s=s, x=x,
                         r10=dom_val(cd, "r", "10m", d), r30=dom_val(cd, "r", "30m", d), r60=dom_val(cd, "r", "60m", d),
                         s10=dom_val(td, "s", "10m", d), s30=dom_val(td, "s", "30m", d), s60=dom_val(td, "s", "60m", d)))
    data.sort(key=lambda z: (z["r60"], z["r"]), reverse=True)
    tot = {k: sum(z[k] for z in data) for k in ("r", "f", "s", "x", "r10", "r30", "r60", "s10", "s30", "s60")}
    if not data:
        L.append(D + "нет данных (collector не вернул домены — пересобери collector/transactioneer)" + RS)
        return L
    mx60 = max(tot["r60"], 1e-9)

    def line(z, name, color="", bold=False):
        pct = z["f"] / z["r"] * 100 if z["r"] else 0
        pc = G if pct >= 50 else (Y if pct >= 20 else R)
        s = (cell(name, 24, c=color) + cell(fn(z["r"]), 9, ">") + cell(fn(z["r10"], 1), 8, ">")
             + cell(fn(z["r30"], 1), 8, ">") + cell(fn(z["r60"], 1), 8, ">")
             + cell(f"{pct:.0f}%" if z["r"] else "—", 7, ">", pc) + cell(fn(z["s"]), 10, ">")
             + cell(fn(z["s10"], 1), 8, ">") + cell(fn(z["s30"], 1), 8, ">") + cell(fn(z["s60"], 1), 8, ">")
             + cell(fn(z["x"]), 7, ">", R if z["x"] else D))
        if wide and not bold:
            s += "  " + bar(z["r60"] / mx60, 12, C)
        return BD + s + RS if bold else s

    for z in data[:rows]:
        L.append(line(z, z["d"]))
    if len(data) > rows:
        L.append(D + f"… ещё {len(data) - rows} доменов (клавиша d — больше строк)" + RS)
    L.append(line(tot, f"ИТОГО ({len(data)})", Y, True))
    L.append(D + "OUT = элементы, отправленные в Upload API (по ответу API известны только итоги батча, поэтому принято/отклонено по доменам не разбивается). FAIL = не загрузилось после ретраев." + RS)
    return L


def sec_queues(st, W):
    c = st.col or {}
    L = [head("ОЧЕРЕДИ И НАГРУЗКА", W)]

    def qrow(label, q, mx, gkey=None):
        frac = q / mx if mx else 0
        s = (cell(label, 24) + bar(frac, 20) + " " + cell(f"{fn(q)}/{fn(mx)}", 11, ">") + " "
             + cell(f"{frac * 100:.0f}%", 5, ">"))
        if gkey:
            s += "  " + spark(st.gauge.get(gkey, []), 24, G if frac < 0.5 else (Y if frac < 0.8 else R))
        return s

    up = g(c, "upipe", default={})
    if up:
        L.append(qrow("upipe очередь", up.get("queue", 0), up.get("limit", 0), "upipe"))
    else:
        L.append(cell("upipe", 24, c=R) + col("недоступен", R))
    for n, _ in BPIPES:
        h = st.bp.get(n)
        if not h:
            L.append(cell(n, 24, c=R) + col("API не отвечает", R))
            continue
        qs = h.get("queues", {})
        for kind in ("translated", "default", "buffer"):
            q = qs.get(kind, {})
            L.append(qrow(f"{n} {kind}", q.get("queue", 0), q.get("max", 0), n if kind == "default" else None)
                     + (D + f"  inflight={h.get('inflight')} batch={h.get('batch_size')}" + RS if kind == "default" else ""))
    if st.tr and g(st.tr, "enabled", default=False):
        L.append(qrow("translator CPU", st.tr.get("queue", 0), st.tr.get("queue_max", 0), "trcpu"))
    if st.gh:
        L.append(qrow("translator GPU", st.gh.get("queue", 0), st.gh.get("max", 0), "trgpu"))
    return L


def sec_host(st, W):
    L = [head("ХОСТ", W)]
    nv = st.nv
    if nv:
        L.append(cell("GPU " + nv["name"][:20], 24) + bar(nv["util"] / 100, 20) + " " + cell(f"{nv['util']:.0f}%", 5, ">")
                 + f"  temp {nv['temp']:.0f}°C  " + spark(st.gauge.get("gpu_util", []), 24))
        L.append(cell("VRAM", 24) + bar(nv["mem"] / max(1, nv["mem_total"]), 20) + " "
                 + cell(f"{nv['mem']:.0f}/{nv['mem_total']:.0f}MiB", 11, ">") + "       "
                 + spark(st.gauge.get("gpu_mem", []), 24))
    else:
        L.append(D + "nvidia-smi недоступен" + RS)
    h = st.host
    if h.get("load"):
        l1 = h["load"][0]
        L.append(cell("CPU load 1/5/15", 24) + bar(l1 / h["cpus"], 20) + " "
                 + cell(f"{h['load'][0]:.1f} {h['load'][1]:.1f} {h['load'][2]:.1f}", 11, ">") + f"  ядер: {h['cpus']}")
    if h.get("ram_total"):
        L.append(cell("RAM", 24) + bar(h["ram_used"] / h["ram_total"], 20) + " "
                 + cell(f"{h['ram_used']:.1f}/{h['ram_total']:.1f}GB", 11, ">"))
    return L


def top_langs(d, n=8):
    items = sorted((d or {}).items(), key=lambda kv: kv[1], reverse=True)[:n]
    return " ".join(f"{k}:{fn(v)}" for k, v in items) or "—"


def sec_tr(st, W):
    c = st.col or {}
    L = [head("ПЕРЕВОД  (шт/с)", W)]
    L.append(D + cell("ДВИЖОК", 14) + cell("СТАТУС", 14) + cell("СЕЙЧАС", 8, ">") + cell("10м", 8, ">") + cell("30м", 8, ">")
             + cell("60м", 8, ">") + cell("ПЕРЕВЕДЕНО", 11, ">") + cell("ОШИБКИ", 8, ">") + cell("ОЧЕРЕДЬ", 10, ">") + "  ТРЕНД" + RS)
    en = g(c, "translate", "enabled", default=False)
    if st.tr and en:
        status, sc = "работает", G
    elif st.tr:
        status, sc = "выкл", D
    else:
        status, sc = "недоступен", R
    wins = st.hw("trcpu.translated", 1)
    now = st.cnt.rate("trcpu.translated", 60) if "trcpu.translated" in st.cnt.h else None
    cs = [cell("Argos CPU", 14), cell(status, 14, c=sc), cell(fn(now, 2), 8, ">")]
    for v, ap in wins:
        cs.append(cell(("~" if ap else "") + fn(v, 2) if v is not None else "—", 8, ">"))
    cs += [cell(fn(g(st.tr, "stats", "translated")), 11, ">"), cell(fn(g(st.tr, "stats", "errors")), 8, ">", R if g(st.tr, "stats", "errors") else D),
           cell(f"{g(st.tr, 'queue')}/{g(st.tr, 'queue_max')}", 10, ">")]
    L.append("".join(x for x in [" ".join(cs)]) + "  " + spark(st.cnt.series("trcpu.translated", 24), 24, C))
    gs = st.gs
    if gs is None:
        L.append(cell("NLLB/Opus GPU", 14) + cell("offline", 14, c=R) + col(GPU_URL, D))
    else:
        r = gs.get("rate_per_sec", {}) or {}
        now = st.cnt.rate("trgpu.translated", 60) if "trgpu.translated" in st.cnt.h else None
        cs = [cell("NLLB/Opus GPU", 14), cell("работает", 14, c=G), cell(fn(now, 2), 8, ">")]
        for w in ("10m", "30m", "60m"):
            cs.append(cell(fn(r.get(w), 2), 8, ">"))
        er = g(gs, "stats", "errors") + g(gs, "stats", "send_errors")
        cs += [cell(fn(g(gs, "stats", "translated")), 11, ">"), cell(fn(er), 8, ">", R if er else D),
               cell(f"{g(st.gh, 'queue')}/{g(st.gh, 'max')}", 10, ">")]
        L.append(" ".join(cs) + "  " + spark(st.cnt.series("trgpu.translated", 24), 24, M))
    L.append(D + "CPU: " + RS + f"timeouts={g(st.tr, 'stats', 'timeouts')} busy={g(st.tr, 'stats', 'rejected_busy')} "
             f"no_pack={g(st.tr, 'stats', 'rejected_no_pack')} too_long={g(st.tr, 'stats', 'rejected_too_long')}  языки: {top_langs(g(st.tr, 'languages_translated', default={}))}")
    L.append(D + "GPU: " + RS + f"получено={fn(g(gs, 'stats', 'received'))} возвращено в upipe={fn(g(gs, 'stats', 'returned'))} "
             f"identical={fn(g(gs, 'stats', 'identical'))} unsupported={fn(g(gs, 'stats', 'unsupported'))} busy={fn(g(gs, 'stats', 'rejected_busy'))}  языки: {top_langs(g(gs, 'languages', default={}))}")
    L.append(D + "collector: " + RS + f"на перевод={fn(g(c, 'translate', 'sent_to_translator'))} на GPU={fn(g(c, 'translate', 'gpu_sent_collector'))} "
             f"foreign-причины: {top_langs(g(c, 'translate', 'foreign_reasons', default={}), 5)}  замечены не-en: {top_langs(g(c, 'languages', 'seen', default={}), 8)}")
    L.append(D + "CPU-окна считаются панелью с момента её запуска; GPU-окна — самим переводчиком." + RS)
    return L


def sec_err(st, W):
    L = [head("ОШИБКИ  (красное — ошибки; Δ — прирост за 5 минут)", W)]
    L.append(D + cell("КОМПОНЕНТ", 16) + cell("ОШИБОК", 8, ">") + cell("Δ5м", 7, ">") + "  ПОДРОБНО" + RS)
    for comp, metrics in ERR_SPEC:
        if not any(f"{comp}.{m}" in st.cur for m, _ in metrics):
            L.append(cell(COMP_NAMES[comp], 16, c=D) + col("  нет данных", D))
            continue
        tot = dtot = 0
        parts = []
        for m, kind in metrics:
            key = f"{comp}.{m}"
            v = st.cur.get(key, 0)
            d = st.cnt.delta(key, 300)[0] if key in st.cnt.h else 0
            if kind == "e":
                tot += v
                dtot += d
                txt = f"{m} {fn(v)}" + (f"(+{fn(d)})" if d > 0 else "")
                parts.append(col(txt, R if v > 0 else D))
            else:
                parts.append(col(f"{m} {fn(v)}", D if not v else ""))
        bc = R if dtot > 0 else (Y if tot > 0 else G)
        L.append(cell(COMP_NAMES[comp], 16, c=BD) + cell(fn(tot), 8, ">", bc) + cell(("+" + fn(dtot)) if dtot else "0", 7, ">", bc)
                 + "  " + "  ".join(parts))
    return L


def sec_ev(st, W):
    L = [head("СОБЫТИЯ", W)]
    if not st.events:
        L.append(D + "событий с момента запуска панели нет" + RS)
    for t, lvl, text in list(st.events)[:8]:
        L.append(col(datetime.fromtimestamp(t).strftime("%H:%M:%S"), D) + " " + col({"R": "● ", "Y": "● ", "G": "● "}[lvl], {"R": R, "Y": Y, "G": G}[lvl]) + text)
    return L


PAGES = {
    "1": ("Обзор", ["svc", "flow", "queues", "host", "ev"]),
    "2": ("Домены", ["dom"]),
    "3": ("Переводы и ошибки", ["tr", "err", "ev"]),
    "0": ("Всё подряд", ["svc", "flow", "dom", "queues", "tr", "err", "host", "ev"]),
}


def render(st, page, W, H, dom_rows, clip_h=True):
    L = sec_header(st, W)
    if not st.polled:
        L.append(D + "загрузка…" + RS)
    else:
        for name in PAGES[page][1]:
            rows = max(3, H - 12) if (page == "2" and name == "dom") else dom_rows
            fn_ = {"svc": sec_svc, "flow": sec_flow, "queues": sec_queues, "host": sec_host, "tr": sec_tr,
                   "err": sec_err, "ev": sec_ev}.get(name)
            L += sec_dom(st, W, rows) if name == "dom" else fn_(st, W)
            L.append("")
    tabs = "  ".join((BD + C + f"[{k}] {v[0]}" + RS) if k == page else D + f"[{k}] {v[0]}" + RS for k, v in sorted(PAGES.items()))
    foot = tabs + D + "   Tab — след. страница · d — строк доменов · +/- интервал · r — обновить · q — выход" + RS
    if clip_h and len(L) > H - 1:
        L = L[: H - 2] + [D + f"… обрезано ({len(L) - (H - 2)} строк) — переключи страницу клавишами 1/2/3" + RS]
    elif clip_h:
        L += [""] * (H - 1 - len(L))
    L.append(foot)
    return L


def draw(lines, W, H):
    sys.stdout.write("\033[H" + "\n".join(clip(l, W) + "\033[K" for l in lines[:H]) + "\033[J")
    sys.stdout.flush()


def main():
    args = sys.argv[1:]
    interval = 3.0
    once = "--once" in args
    if "-i" in args:
        try:
            interval = max(1.0, float(args[args.index("-i") + 1]))
        except (IndexError, ValueError):
            pass
    st = State(interval)
    if once:
        st.poll()
        time.sleep(min(interval, 3))
        st.poll()
        W = shutil.get_terminal_size((140, 50)).columns
        page = "0"
        for k in ("1", "2", "3"):
            if f"--page={k}" in args:
                page = k
        print("\n".join(clip(l, W) for l in render(st, page, W, 999, 15, clip_h=False)))
        return
    if not sys.stdin.isatty():
        print("нужен интерактивный терминал (или запусти с --once)")
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    page, dom_rows = "1", 12
    try:
        tty.setcbreak(fd)
        sys.stdout.write("\033[?1049h\033[?25l")
        next_poll = 0.0
        while True:
            if time.time() >= next_poll:
                st.interval = interval
                st.poll()
                next_poll = time.time() + interval
            W, H = shutil.get_terminal_size((140, 45))
            draw(render(st, page, W, H, dom_rows), W, H)
            r, _, _ = select.select([sys.stdin], [], [], 0.5)
            if not r:
                continue
            k = sys.stdin.read(1)
            if k in ("q", "Q", "\x03"):
                break
            if k in PAGES:
                page = k
            elif k == "\t":
                order = ["1", "2", "3", "0"]
                page = order[(order.index(page) + 1) % 4]
            elif k == "r":
                next_poll = 0
            elif k == "+":
                interval = min(60.0, interval + 1)
            elif k == "-":
                interval = max(1.0, interval - 1)
            elif k == "d":
                dom_rows = {12: 25, 25: 50, 50: 8, 8: 12}.get(dom_rows, 12)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
