#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(ROOT, ".env")
COLLECTOR = os.getenv("COLLECTOR_URL", "http://localhost:9000").rstrip("/")
BPIPES = [("bpipe", 7995), ("bpipe_2", 7996), ("bpipe_3", 7997)]
CANDIDATES = [2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64]
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
RE_START = re.compile(r"\[Batch-(\d+)\] Обрабатываем (\d+) элементов")
RE_DONE = re.compile(r"\[Batch-(\d+)\].*?ML готово за ([\d.]+)с")
DEFAULTS = {
    "FIXED_BATCH_SIZE": "6", "BATCH_TIMEOUT_SECONDS": "3.0", "BPIPE_QUEUE_DEFAULT_MAX": "200",
    "BPIPE_QUEUE_TRANSLATED_MAX": "100", "BPIPE_QUEUE_TRANSLATED_HARD_MAX": "1000",
    "UPIPE_QUEUE_LIMIT": "200", "UPIPE_WORKERS": "4", "TAG_HF_BATCH_SIZE": "64", "TAG_MAX_LENGTH": "512",
}


def p(s=""):
    print(s, flush=True)


def val(d, *path):
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
        if d is None:
            return None
    return d


def sh(cmd, timeout=10):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except Exception as e:
        return 1, str(e)


def http_json(url, timeout=3):
    try:
        with OPENER.open(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception:
        return None


def docker_json(container, url):
    rc, out = sh(["docker", "exec", container, "curl", "-sf", "--noproxy", "*", "--max-time", "3", url], 8)
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except ValueError:
        return None


def nvidia():
    rc, out = sh(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                  "--format=csv,noheader,nounits"], 5)
    if rc != 0:
        return None
    try:
        u, mu, mt = [float(x) for x in out.strip().splitlines()[0].split(",")]
        return {"util": u, "mem": mu, "total": mt}
    except ValueError:
        return None


def cpu_times():
    try:
        v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
        return sum(v), v[3] + v[4]
    except Exception:
        return None


def upipe_cpu():
    rc, out = sh(["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}}", "upipe"], 20)
    try:
        return float(out.strip().strip("%")) / 100.0
    except ValueError:
        return None


def poll(bp):
    t = time.time()
    with ThreadPoolExecutor(8) as ex:
        fc = ex.submit(http_json, COLLECTOR + "/stats")
        fb = [ex.submit(docker_json, n, f"http://localhost:{port}/health") for n, port in bp]
        fg = ex.submit(nvidia)
        return {"t": t, "c": fc.result(), "b": [f.result() for f in fb], "gpu": fg.result(), "cpu": cpu_times()}


def read_env():
    env = {}
    try:
        for line in open(ENV_PATH, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return env


def write_env(updates):
    lines = []
    if os.path.exists(ENV_PATH):
        lines = open(ENV_PATH, encoding="utf-8").read().splitlines()
        shutil.copy(ENV_PATH, ENV_PATH + ".bak_autotune_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    seen, out = set(), []
    for l in lines:
        k = l.split("=", 1)[0].strip()
        if "=" in l and not l.lstrip().startswith("#") and k in updates:
            out.append(f"{k}={updates[k]}")
            seen.add(k)
        else:
            out.append(l)
    extra = [k for k in updates if k not in seen]
    if extra:
        out += ["", "# autotune " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")] + [f"{k}={updates[k]}" for k in extra]
    tmp = ENV_PATH + ".tmp"
    open(tmp, "w", encoding="utf-8").write("\n".join(out) + "\n")
    os.replace(tmp, ENV_PATH)


def series(samples, fn):
    pts = []
    for s in samples:
        v = fn(s)
        if v is not None:
            pts.append((s["t"], v))
    return pts


def rate(samples, fn):
    pts = series(samples, fn)
    if len(pts) < 2 or pts[-1][0] <= pts[0][0]:
        return 0.0
    return max(0.0, (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0]))


def delta(samples, fn):
    pts = series(samples, fn)
    return max(0.0, pts[-1][1] - pts[0][1]) if len(pts) >= 2 else 0.0


def avg_max(samples, fn):
    v = [x for _, x in series(samples, fn)]
    return (statistics.mean(v), max(v)) if v else (0.0, 0.0)


def bsum(s, *path):
    live = [b for b in s["b"] if b]
    if not live:
        return None
    return sum(val(b, *path) or 0 for b in live)


def bavg(s, *paths):
    live = [b for b in s["b"] if b]
    if not live:
        return None
    return sum(sum(val(b, *pa) or 0 for pa in paths) for b in live) / len(live)


def ml_points(bp, since):
    pts = []
    for name, _ in bp:
        rc, out = sh(["docker", "logs", "--since", f"{int(since)}s", name], 30)
        if rc != 0:
            continue
        pending, cur = {}, []
        for line in out.splitlines():
            m = RE_START.search(line)
            if m:
                pending[m.group(1)] = int(m.group(2))
                continue
            m = RE_DONE.search(line)
            if m and m.group(1) in pending:
                cur.append((pending.pop(m.group(1)), float(m.group(2))))
        pts += cur[2:] if len(cur) > 6 else cur
    return pts


class Svc:
    def __init__(self, a, b, src):
        self.a, self.b, self.src = a, b, src

    def t(self, n):
        return self.a + self.b * n


def fit(points, src):
    by = {}
    for n, t in points:
        by.setdefault(n, []).append(t)
    xs = sorted(by)
    ys = [statistics.median(by[x]) for x in xs]
    if len(xs) >= 3 and xs[-1] - xs[0] >= 4:
        xm, ym = statistics.mean(xs), statistics.mean(ys)
        var = sum((x - xm) ** 2 for x in xs)
        b = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / var
        a = ym - b * xm
        if b > 0 and a >= 0:
            return Svc(a, b, src)
        if b > 0:
            return Svc(0.0, sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs), src + " (через 0)")
    if points:
        n_obs = statistics.mean(n for n, _ in points)
        t_obs = statistics.median(t for _, t in points)
        return Svc(0.35 * t_obs, 0.65 * t_obs / n_obs, "прайор 35% фикс. накладных, один размер батча в логах")
    return None


def load_sweep(path, hf, maxlen):
    rows = json.load(open(path))
    pts = [(r["n"], r["p50"]) for r in rows if r.get("hf") == hf and r.get("maxlen") == maxlen]
    return fit(pts, "bench_tag sweep") if len(pts) >= 3 else None


def plan(lam_r, svc, a, sat, vram_frac, n_cur):
    rows = [{"n": n, "t": svc.t(n), "cap": n / svc.t(n)} for n in CANDIDATES]
    ok = [r for r in rows if r["t"] <= a.max_age * 0.25] or rows[:1]
    if vram_frac > 0.9:
        ok = [r for r in ok if r["n"] <= n_cur] or ok[:1]
    if sat:
        top = max(r["cap"] for r in ok)
        best = min((r for r in ok if r["cap"] >= 0.95 * top), key=lambda r: r["n"])
    else:
        good = [r for r in ok if r["cap"] >= lam_r * a.headroom]
        best = min(good, key=lambda r: r["n"]) if good else max(ok, key=lambda r: r["cap"])
    return rows, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=90)
    ap.add_argument("--max-age", type=float, default=15.0)
    ap.add_argument("--headroom", type=float, default=1.3)
    ap.add_argument("--fill-budget", type=float, default=1.5)
    ap.add_argument("--rate", type=float, default=0.0)
    ap.add_argument("--sweep")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--restart", action="store_true")
    a = ap.parse_args()

    env = read_env()
    cur = {k: env.get(k, d) for k, d in DEFAULTS.items()}
    try:
        n_rep = max(1, min(3, int(env.get("BPIPE_REPLICAS", "1") or 1)))
    except ValueError:
        n_rep = 1
    bp = BPIPES[:n_rep]
    cores = os.cpu_count() or 4

    p(f"замер {a.seconds}с, реплик bpipe: {n_rep}, логических ядер: {cores}")
    samples, up_cpu = [], []
    t_end = time.time() + a.seconds
    next_ds = time.time() + 3
    while True:
        s = poll(bp)
        samples.append(s)
        if time.time() >= next_ds:
            v = upipe_cpu()
            if v is not None:
                up_cpu.append(v)
            next_ds = time.time() + 15
        left = t_end - time.time()
        print(f"\r  осталось {max(0, left):.0f}с, сэмплов {len(samples)}   ", end="", flush=True)
        if left <= 0:
            break
        time.sleep(max(0.0, 2.0 - (time.time() - s["t"])))
    print()

    if not any(s["c"] for s in samples):
        sys.exit(f"collector недоступен ({COLLECTOR}); запусти систему: ./run.sh up -d")

    live_rep = max(1, max(sum(1 for b in s["b"] if b) for s in samples))
    r_col_in = rate(samples, lambda s: val(s["c"], "collector", "received"))
    r_col_fwd = rate(samples, lambda s: val(s["c"], "collector", "forwarded"))
    r_up_proc = rate(samples, lambda s: val(s["c"], "upipe", "stats", "processed"))
    r_up_fwd = rate(samples, lambda s: val(s["c"], "upipe", "stats", "forwarded"))
    r_bp_in = rate(samples, lambda s: bsum(s, "stats", "received"))
    r_bp_out = rate(samples, lambda s: bsum(s, "stats", "items_sent"))
    r_acc = rate(samples, lambda s: val(s["c"], "exorde", "totals", "accepted"))
    r_rej = rate(samples, lambda s: val(s["c"], "exorde", "totals", "rejected"))
    r_dup = rate(samples, lambda s: val(s["c"], "exorde", "totals", "duplicates"))

    up_limit = val(next((s["c"] for s in samples if s["c"]), {}), "upipe", "limit") or int(cur["UPIPE_QUEUE_LIMIT"])
    up_q_avg, up_q_max = avg_max(samples, lambda s: val(s["c"], "upipe", "queue"))
    bp_q_avg, bp_q_max = avg_max(samples, lambda s: bavg(s, ("queues", "default", "queue"), ("queues", "translated", "queue")))
    bp_cap_cur = next((val(b, "queues", "default", "max") for s in samples for b in s["b"] if b), None) or int(cur["BPIPE_QUEUE_DEFAULT_MAX"])
    bp_drop = delta(samples, lambda s: bsum(s, "stats", "dropped"))
    up_drop = delta(samples, lambda s: val(s["c"], "upipe", "stats", "dropped"))
    col_drop = delta(samples, lambda s: val(s["c"], "collector", "dropped"))

    gpu = [s["gpu"] for s in samples if s["gpu"]]
    gpu_util = statistics.mean(g["util"] for g in gpu) if gpu else None
    vram_frac = max(g["mem"] / g["total"] for g in gpu) if gpu else 0.0
    cpu_pts = [s["cpu"] for s in samples if s["cpu"]]
    host_cpu = None
    if len(cpu_pts) >= 2 and cpu_pts[-1][0] > cpu_pts[0][0]:
        host_cpu = 100.0 * (1 - (cpu_pts[-1][1] - cpu_pts[0][1]) / (cpu_pts[-1][0] - cpu_pts[0][0]))
    up_cpu_cores = statistics.mean(up_cpu) if up_cpu else None

    thr_r = r_bp_out / live_rep
    q_wait_cur = bp_q_avg / thr_r if thr_r > 0.05 else 0.0
    sat_b = bp_q_avg >= 0.5 * bp_cap_cur or bp_drop > 0
    sat_u = up_q_avg >= 0.5 * up_limit or up_drop > 0

    lam = max(r_bp_in, a.rate)
    if lam < 0.2 and not sat_b:
        sys.exit("поток в bpipe ≈ 0: запусти при работающем парсере или задай --rate N")

    pts = ml_points(bp, max(a.seconds + 60, 600))
    svc = None
    if a.sweep:
        svc = load_sweep(a.sweep, int(cur["TAG_HF_BATCH_SIZE"]), int(cur["TAG_MAX_LENGTH"]))
    if svc is None:
        svc = fit(pts, "логи bpipe")
    if svc is None:
        sys.exit("нет данных о времени батчей: нужен трафик в bpipe или --sweep bench_out/<ts>/bpipe_sweep.json")

    n_cur = int(cur["FIXED_BATCH_SIZE"])
    lam_r = lam / live_rep
    rows, best = plan(lam_r, svc, a, sat_b, vram_frac, n_cur)
    n = best["n"]
    cap_r = best["cap"]
    cap_total = cap_r * live_rep

    fill = n / lam_r if lam_r > 0 else a.fill_budget
    timeout = round(min(a.fill_budget, 1.0), 2) if sat_b else round(min(a.fill_budget, max(0.2, 1.5 * fill)), 2)
    budget = max(a.max_age - best["t"] - timeout, 1.0)
    q = max(2 * n, min(int(cap_r * budget), 1000))
    up_new_limit = max(20, min(int(math.ceil(cap_total * min(3.0, 0.2 * a.max_age))), 400))
    hf = min(64, max(16, n))
    if vram_frac > 0.9:
        hf = min(hf, 32)

    workers = int(cur["UPIPE_WORKERS"])
    feed = cap_total if (sat_b or sat_u) else lam * a.headroom
    if up_cpu_cores is not None and r_up_proc >= 1.0:
        per_item = up_cpu_cores / r_up_proc
        workers = int(math.ceil(feed * per_item * 1.3))
    workers = max(2, min(workers, max(2, cores - 2)))

    new = {
        "FIXED_BATCH_SIZE": n,
        "BATCH_TIMEOUT_SECONDS": timeout,
        "BPIPE_QUEUE_DEFAULT_MAX": q,
        "BPIPE_QUEUE_TRANSLATED_MAX": q,
        "BPIPE_QUEUE_TRANSLATED_HARD_MAX": q * 3,
        "UPIPE_QUEUE_LIMIT": up_new_limit,
        "UPIPE_WORKERS": workers,
        "TAG_HF_BATCH_SIZE": hf,
    }

    p("\n== ПОТОК (шт/с) ==")
    p(f"  collector in {r_col_in:.1f} → прошло фильтры {r_col_fwd:.1f} → upipe done {r_up_proc:.1f} → bpipe in {r_bp_in:.1f} → bpipe out {r_bp_out:.1f}")
    p(f"  Exorde: принято {r_acc:.1f} · отклонено {r_rej:.1f} · дубликаты {r_dup:.1f}")
    p(f"  дропы за окно: collector {col_drop:.0f} · upipe {up_drop:.0f} · bpipe {bp_drop:.0f}")

    p("\n== ОЧЕРЕДИ ==")
    p(f"  upipe: avg {up_q_avg:.1f} / max {up_q_max:.0f} из {up_limit}")
    p(f"  bpipe (на реплику): avg {bp_q_avg:.1f} / max {bp_q_max:.0f} из {bp_cap_cur}")
    p(f"  ожидание в очереди bpipe сейчас ≈ {q_wait_cur:.1f}с" + ("  ← ПРОТУХАНИЕ (> max-age)" if q_wait_cur > a.max_age else ""))
    if pts:
        p(f"  батчи в логах: средний размер {statistics.mean(x for x, _ in pts):.1f}, среднее время ML {statistics.mean(t for _, t in pts):.2f}с, n={len(pts)}")

    p("\n== РЕСУРСЫ ==")
    p(f"  GPU util {'-' if gpu_util is None else f'{gpu_util:.0f}%'} · VRAM max {vram_frac * 100:.0f}% · CPU хоста {'-' if host_cpu is None else f'{host_cpu:.0f}%'} · upipe {'-' if up_cpu_cores is None else f'{up_cpu_cores:.1f} ядра'}")

    p(f"\n== МОДЕЛЬ ML: t(n) = {svc.a:.3f} + {svc.b:.4f}·n  [{svc.src}] ==")
    p("     n     t,с   емк./с  ")
    for r in rows:
        mark = "  ←" if r["n"] == n else ""
        p(f"  {r['n']:>4} {r['t']:>7.3f} {r['cap']:>8.1f}{mark}")

    p("\n== ДИАГНОЗ ==")
    if sat_b and gpu_util is not None and gpu_util < 70:
        p(f"  - bpipe упёрся при GPU {gpu_util:.0f}%: узкое место CPU/Python внутри bpipe, не GPU")
    elif sat_b:
        p("  - bpipe упёрся в GPU: настройка на максимум пропускной способности")
    if sat_u and not sat_b:
        p("  - upipe упёрся в CPU воркеров, bpipe простаивает")
    if not sat_b and not sat_u:
        p("  - очереди не копятся: лимит потока задаёт источник, а не пайплайн")
    if sat_b and gpu_util is not None and gpu_util < 70 and vram_frac < 0.6 and n_rep == 1:
        p("  - возможно помогут BPIPE_REPLICAS=2 (проверь VRAM и ./bench_all.sh)")
    if lam_r > 0 and best["cap"] < lam_r * a.headroom and not sat_b:
        p(f"  - ёмкость {cap_r:.1f}/с на реплику ниже нужной {lam_r * a.headroom:.1f}/с: потолок по GPU")
    p(f"  - расчётная задержка в bpipe: типично ≈ {min(timeout, fill) / 2 + best['t']:.1f}с, худшая ≈ {timeout + best['t'] + q / cap_r:.1f}с (цель ≤ {a.max_age:.0f}с)")

    p("\n== ИЗМЕНЕНИЯ .env ==")
    for k, v in new.items():
        old = cur.get(k, "-")
        p(f"  {k}: {old} → {v}" + ("" if str(old) != str(v) else "  (без изменений)"))

    if a.dry:
        p("\n--dry: .env не изменён")
        return
    write_env(new)
    p(f"\n.env обновлён, бэкап: {ENV_PATH}.bak_autotune_*")
    if a.restart:
        cmd = [os.path.join(ROOT, "run.sh"), "up", "-d", "--force-recreate", "--no-deps"] + [n_ for n_, _ in bp] + ["upipe", "collector"]
        p("перезапуск: " + " ".join(cmd[1:]))
        subprocess.run(cmd, cwd=ROOT)
    else:
        p("применить: ./run.sh up -d --force-recreate --no-deps bpipe upipe collector")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        p("\nпрервано")
