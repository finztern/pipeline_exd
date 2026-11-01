#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(ROOT, "run.sh")
ENV_WHITELIST = (
    "BPIPE_REPLICAS", "FIXED_BATCH_SIZE", "BATCH_TIMEOUT_SECONDS", "TAG_HF_BATCH_SIZE", "TAG_MAX_LENGTH",
    "TAG_SKIP_DEPRECATED", "CLASSIFICATION_BACKEND", "UPIPE_WORKERS", "UPIPE_QUEUE_LIMIT", "TRANSLATE",
    "MAX_MODEL_TOKENS", "MAX_OLDNESS_SECONDS", "UPLOAD_CONCURRENCY", "LANG_FILTER", "URL_DEDUP_MODE",
    "BPIPE_QUEUE_DEFAULT_MAX", "BPIPE_QUEUE_TRANSLATED_MAX", "USE_PROXY_BPIPE", "USE_PROXY_UPIPE",
)
GAUGE_WORDS = ("queue", "fill", "inflight", "max", "rate", "uptime", "workers", "size", "ts", "ttl", "active", "qsize")
SKIP_KEYS = ("sample", "last_error", "installed", "route")

report_lines = []


def out(msg=""):
    print(msg, flush=True)
    report_lines.append(msg)


def sh(cmd, timeout=15):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout
    except Exception as e:
        return 1, str(e)


def run_logged(cmd, logpath, timeout=None):
    t0 = time.time()
    with open(logpath, "w") as lf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
        timer = None
        if timeout:
            timer = threading.Timer(timeout, p.kill)
            timer.start()
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
        p.wait()
        if timer:
            timer.cancel()
    return p.returncode, time.time() - t0


def http_json(url, timeout=5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


def dexec_json(container, port, path):
    rc, o = sh(["docker", "exec", container, "curl", "-s", "-m", "5", f"http://localhost:{port}/{path}"], timeout=10)
    if rc != 0 or not o.strip():
        return None
    try:
        return json.loads(o)
    except Exception:
        return None


def running_containers():
    rc, o = sh(["docker", "ps", "--format", "{{.Names}}"])
    return set(o.split()) if rc == 0 else set()


def flatten(d, prefix="", depth=0, acc=None):
    acc = {} if acc is None else acc
    if depth > 4 or len(acc) > 500:
        return acc
    if isinstance(d, dict):
        for k, v in d.items():
            if k in SKIP_KEYS:
                continue
            flatten(v, f"{prefix}.{k}" if prefix else str(k), depth + 1, acc)
    elif isinstance(d, (int, float)) and not isinstance(d, bool):
        acc[prefix] = float(d)
    return acc


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_flag = threading.Event()
        self.gpu, self.cpu = [], []
        self.has_gpu = sh(["which", "nvidia-smi"])[0] == 0

    def read_cpu(self):
        try:
            v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
            return sum(v), v[3] + v[4]
        except Exception:
            return None

    def run(self):
        prev = self.read_cpu()
        while not self.stop_flag.is_set():
            if self.has_gpu:
                rc, o = sh(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu",
                            "--format=csv,noheader,nounits"], timeout=5)
                if rc == 0:
                    try:
                        self.gpu.append([float(x) for x in o.strip().splitlines()[0].split(",")])
                    except Exception:
                        pass
            cur = self.read_cpu()
            if prev and cur and cur[0] > prev[0]:
                self.cpu.append(100.0 * (1 - (cur[1] - prev[1]) / (cur[0] - prev[0])))
            prev = cur
            self.stop_flag.wait(1.0)


def docker_stats():
    rc, o = sh(["docker", "stats", "--no-stream", "--format", "{{json .}}"], timeout=30)
    res = {}
    if rc == 0:
        for line in o.splitlines():
            try:
                j = json.loads(line)
                res[j["Name"]] = (float(j["CPUPerc"].strip("%")), j["MemUsage"].split("/")[0].strip())
            except Exception:
                pass
    return res


def pct(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(len(s) * p / 100))] if s else 0.0


def summarize_series(name, samples, dt):
    out(f"\n--- {name} ---")
    if not samples:
        out("  нет данных (сервис недоступен)")
        return {}
    keys = sorted(set().union(*[set(s) for s in samples]))
    lines = []
    rates = {}
    for k in keys:
        vals = [s[k] for s in samples if k in s]
        if len(vals) < 2:
            continue
        mono = all(b >= a for a, b in zip(vals, vals[1:]))
        gauge = any(w in k.lower() for w in GAUGE_WORDS)
        if mono and vals[-1] > vals[0] and not gauge:
            rate = (vals[-1] - vals[0]) / dt
            rates[k] = rate
            lines.append(f"  {k}: {vals[0]:.0f} -> {vals[-1]:.0f}  (+{rate:.2f}/с)")
        elif max(vals) != min(vals) or (gauge and max(vals) > 0):
            lines.append(f"  {k}: min {min(vals):.3g} / avg {statistics.mean(vals):.3g} / max {max(vals):.3g}")
    for l in lines[:80]:
        out(l)
    if len(lines) > 80:
        out(f"  ... ещё {len(lines) - 80} строк в live.json")
    return rates


def capture_env(outdir):
    out("=== ОКРУЖЕНИЕ ===")
    out(f"дата: {datetime.datetime.now().isoformat(timespec='seconds')}")
    for label, cmd in (("git", ["git", "log", "--oneline", "-1"]), ("ветка", ["git", "rev-parse", "--abbrev-ref", "HEAD"])):
        rc, o = sh(cmd)
        out(f"{label}: {o.strip()}")
    rc, o = sh(["lscpu"])
    for l in o.splitlines():
        if re.match(r"(Model name|CPU\(s\)|Thread|Core|Socket|CPU max MHz)", l):
            out("  " + re.sub(r"\s+", " ", l))
    rc, o = sh(["free", "-m"])
    if rc == 0 and len(o.splitlines()) > 1:
        out("  RAM: " + re.sub(r"\s+", " ", o.splitlines()[1]))
    out("  docker: " + sh(["docker", "--version"])[1].strip())
    rc, o = sh(["nvidia-smi", "-L"])
    out("  GPU: " + (o.strip() if rc == 0 else "nvidia-smi недоступен"))
    rc, o = sh(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    if rc == 0 and o.strip():
        out("  процессы на GPU сейчас:\n    " + o.strip().replace("\n", "\n    "))
    out("  контейнеры: " + ", ".join(sorted(running_containers())))
    envf = os.path.join(ROOT, ".env")
    if os.path.exists(envf):
        out("  .env (только настройки производительности):")
        for l in open(envf, encoding="utf-8", errors="replace"):
            k = l.split("=", 1)[0].strip()
            if k in ENV_WHITELIST:
                out("    " + l.strip())


def live_phase(seconds, outdir):
    out(f"\n=== 1. ЖИВАЯ СИСТЕМА ({seconds} с, боевой трафик, ничего не останавливаем) ===")
    cont = running_containers()
    if "collector" not in cont and "upipe" not in cont:
        out("collector/upipe не запущены, фаза пропущена")
        return {}
    sampler = Sampler()
    sampler.start()
    series = {"collector": [], "upipe": [], "transactioneer": []}
    bp = [(n, p) for n, p in (("bpipe", 7995), ("bpipe_2", 7996), ("bpipe_3", 7997)) if n in cont]
    for n, _ in bp:
        series[n] = []
    raw = []
    dstats = [docker_stats()]
    t0 = time.time()
    mid_done = False
    while time.time() - t0 < seconds:
        snap = {"t": time.time() - t0}
        d = http_json("http://localhost:9000/stats")
        if d:
            series["collector"].append(flatten(d)); snap["collector"] = d
        d = http_json("http://localhost:5981/health")
        if d:
            series["upipe"].append(flatten(d)); snap["upipe"] = d
        for n, p in bp:
            d = dexec_json(n, p, "health")
            if d:
                series[n].append(flatten(d)); snap[n] = d
        d = dexec_json("transactioneer", 8002, "stats")
        if d:
            series["transactioneer"].append(flatten(d)); snap["transactioneer"] = d
        raw.append(snap)
        if not mid_done and time.time() - t0 > seconds / 2:
            dstats.append(docker_stats())
            mid_done = True
        time.sleep(max(0.0, 5 - ((time.time() - t0) % 5)))
    dstats.append(docker_stats())
    sampler.stop_flag.set()
    sampler.join(3)
    dt = max(1.0, time.time() - t0)
    json.dump(raw, open(os.path.join(outdir, "live.json"), "w"), ensure_ascii=False)

    result = {}
    for name, s in series.items():
        result[name] = summarize_series(name, s, dt)

    out("\n--- CPU контейнеров (docker stats, %, 100% = одно ядро) ---")
    names = sorted({n for d in dstats for n in d})
    for n in names:
        cpus = [d[n][0] for d in dstats if n in d]
        mem = next((d[n][1] for d in reversed(dstats) if n in d), "")
        out(f"  {n:<16} max {max(cpus):>6.1f}%  mem {mem}")
    if sampler.cpu:
        out(f"\n--- CPU хоста: avg {statistics.mean(sampler.cpu):.0f}% / p95 {pct(sampler.cpu,95):.0f}% / max {max(sampler.cpu):.0f}% ---")
    if sampler.gpu:
        g = list(zip(*sampler.gpu))
        out(f"--- GPU: util avg {statistics.mean(g[0]):.0f}% / p95 {pct(g[0],95):.0f}% / max {max(g[0]):.0f}%; "
            f"VRAM max {max(g[1]):.0f}/{g[2][0]:.0f} МБ; power avg {statistics.mean(g[3]):.0f} Вт; "
            f"SM clock min/avg {min(g[4]):.0f}/{statistics.mean(g[4]):.0f} МГц; temp max {max(g[5]):.0f}°C ---")
        out(f"    доля секунд с GPU util < 50%: {sum(1 for x in g[0] if x < 50) / len(g[0]) * 100:.0f}%")
    result["_cpu"] = statistics.mean(sampler.cpu) if sampler.cpu else None
    return result


def step(title, cmd, logpath, timeout):
    out(f"\n=== {title} ===")
    rc, dt = run_logged(cmd, logpath, timeout)
    out(f"[{title}] rc={rc}, {dt:.0f} с")
    return rc


def compose_run(outdir, service, mounts, py_args):
    cmd = [RUN, "run", "--rm", "--no-deps", "-T", "-v", f"{outdir}:/out"]
    for src, dst in mounts:
        cmd += ["-v", f"{os.path.join(ROOT, src)}:{dst}:ro"]
    return cmd + [service, "python"] + py_args


def read_text(path, limit=200):
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
        return "\n".join(lines[-limit:])
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="не спрашивать подтверждение остановки сервисов")
    ap.add_argument("--live-only", action="store_true", help="только замер живой системы, без остановки сервисов")
    ap.add_argument("--live-seconds", type=int, default=90)
    ap.add_argument("--quick", action="store_true", help="короткие прогоны")
    ap.add_argument("--skip-upipe", action="store_true")
    ap.add_argument("--skip-collector", action="store_true")
    ap.add_argument("--skip-bpipe", action="store_true")
    ap.add_argument("--skip-concurrent", action="store_true")
    ap.add_argument("--texts", help="файл с реальными текстами (по одному на строку), положи в bench_out/")
    args = ap.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(ROOT, "bench_out", ts)
    os.makedirs(outdir, exist_ok=True)
    os.chmod(outdir, 0o777)

    capture_env(outdir)
    live = live_phase(args.live_seconds, outdir)

    stopped = []
    if not args.live_only:
        if not args.yes:
            print("\nДальше остановлю collector, upipe, bpipe* на время замеров (потом подниму обратно). Продолжить? [y/N] ", end="")
            if input().strip().lower() != "y":
                args.live_only = True

    if not args.live_only:
        cont = running_containers()
        stopped = [n for n in ("collector", "upipe", "bpipe", "bpipe_2", "bpipe_3") if n in cont]
        out(f"\nОстанавливаю: {stopped}")
        sh(["docker", "stop"] + stopped, timeout=120)
        rc, o = sh(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
        if rc == 0 and o.strip():
            out("! на GPU остались процессы (мешают замеру bpipe):\n  " + o.strip().replace("\n", "\n  "))
        try:
            texts_arg = ["--texts", f"/out/{os.path.basename(args.texts)}"] if args.texts else []
            q = args.quick
            if not args.skip_upipe:
                a = ["bench_upipe.py", "--chunks", "1", "--no-yake-variants", "--out", "/out/upipe.json",
                     "--workers", "1,6,12" if q else "1,2,4,6,8,12", "--queues", "200,2000", "--e2e-n", "600" if q else "1200"] + texts_arg
                step("2. upipe: стадии, пул, e2e", compose_run(outdir, "upipe", [("upipe/bench_upipe.py", "/app/bench_upipe.py")], a),
                     os.path.join(outdir, "upipe.log"), 1800)
            if not args.skip_collector:
                a = ["bench_lang.py", "--out", "/out/collector_lang.json"] + texts_arg
                step("3. collector: скорость langdetect", compose_run(outdir, "collector", [("collector/bench_lang.py", "/app/bench_lang.py")], a),
                     os.path.join(outdir, "collector_lang.log"), 900)
            if not args.skip_bpipe:
                a = ["bench_tag.py", "--out", "/out/bpipe_sweep.json", "--calls", "4" if q else "8", "--warmup", "2" if q else "3"] + texts_arg
                mounts = [("bpipe/bench_tag.py", "/app/bench_tag.py")]
                step("4. bpipe: GPU, перебор размера батча/hf/maxlen", compose_run(outdir, "bpipe", mounts, a),
                     os.path.join(outdir, "bpipe_sweep.log"), 3600)
                best = None
                try:
                    rows = json.load(open(os.path.join(outdir, "bpipe_sweep.json")))
                    best = max(rows, key=lambda r: r["docs_per_s"])
                except Exception:
                    pass
                if best and not args.skip_concurrent:
                    for k in (1, 2):
                        a = ["bench_tag.py", "--concurrent", str(k), "--n", str(best["n"]), "--hf", str(best["hf"]),
                             "--maxlen", str(best["maxlen"]), "--duration", "20" if q else "30", "--warmup", "2"] + texts_arg
                        step(f"5. bpipe: {k} реплик(и) одновременно на одной GPU (n={best['n']})",
                             compose_run(outdir, "bpipe", mounts, a), os.path.join(outdir, f"bpipe_conc{k}.log"), 1800)
        finally:
            out("\nПоднимаю сервисы обратно...")
            order = [n for n in ("bpipe", "bpipe_2", "bpipe_3", "upipe", "collector") if n in stopped]
            rc, _ = sh(["docker", "start"] + order, timeout=120)
            out(f"docker start {order} rc={rc}")

    out("\n=== ИТОГ ПО ЛОГАМ ===")
    for name in ("upipe.log", "collector_lang.log", "bpipe_sweep.log", "bpipe_conc1.log", "bpipe_conc2.log"):
        txt = read_text(os.path.join(outdir, name))
        if txt:
            out(f"\n##### {name}\n{txt}")

    try:
        cl = json.load(open(os.path.join(outdir, "collector_lang.json")))
        rate = live.get("collector", {}).get("collector.received")
        if rate:
            load = rate / cl["items_per_s_full"] * 100
            out(f"\n>>> collector: принимает ~{rate:.0f} items/s, langdetect на одно ядро даёт {cl['items_per_s_full']:.0f}/с -> "
                f"langdetect занимает ~{load:.0f}% потока collector")
    except Exception:
        pass

    rp = os.path.join(outdir, "report.txt")
    open(rp, "w", encoding="utf-8").write("\n".join(report_lines) + "\n")
    out(f"\nГОТОВО. Пришли файл: {rp}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nпрервано; если сервисы остановлены, подними их: docker start bpipe upipe collector")
