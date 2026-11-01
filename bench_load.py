#!/usr/bin/env python3
import argparse
import datetime
import http.client
import json
import math
import os
import queue
import random
import re
import statistics
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import bench_all as ba  # noqa: E402

out = ba.out
SINK_PORT = 18002
MOCK_NAME = "transactioneer"
REAL_BACKUP = "transactioneer_real_bench"
DOMAINS = ["reddit.com", "x.com", "bsky.app", "youtube.com", "news.ycombinator.com", "threads.net"]
WORDS = (
    "the be to of and a in that have i it for not on with he as you do at this but his by from they we say her she or an "
    "will my one all would there their what so up out if about who get which go me when make can like time no just him "
    "know take people into year your good some could them see other than then now look only come its over think also "
    "back after use two how our work first well way even new want because any these give day most us great love really "
    "photo camera lens light game team win season price market stock coin bitcoin health doctor law court vote policy "
    "phone laptop code python model data cloud music movie show actor song review honestly agree wrong right thanks sorry "
    "maybe never always literally actually basically pretty quite totally probably definitely government economy energy "
    "climate school city travel food coffee weather election company startup research science space football league"
).split()
OPENERS = ["Honestly,", "I think", "Just saw that", "Not sure why but", "Breaking:", "Hot take:", "Today", "Can we talk about how", "Unpopular opinion:", ""]


def iso_now():
    n = datetime.datetime.now(datetime.timezone.utc)
    return n.strftime("%Y-%m-%dT%H:%M:%S.") + f"{n.microsecond // 1000:03d}Z"


def make_pool(n, seed=7, texts_file=None):
    if texts_file:
        lines = [l.strip() for l in open(texts_file, encoding="utf-8") if len(l.strip()) >= 20]
        return [lines[i % len(lines)] for i in range(n)]
    rnd = random.Random(seed)
    pool = []
    for _ in range(n):
        k = int(min(120, max(5, rnd.lognormvariate(math.log(24), 0.8))))
        words = [rnd.choice(WORDS) for _ in range(k)]
        for j in range(rnd.randint(0, 3)):
            words.insert(rnd.randint(0, len(words)), rnd.choice(["is", "was", "are", "really", "very"]))
        txt = (rnd.choice(OPENERS) + " " + " ".join(words)).strip()
        pool.append(txt[0].upper() + txt[1:] + rnd.choice([".", "!", "?", "..."]))
    return pool


class Sink(threading.Thread):
    def __init__(self, port):
        super().__init__(daemon=True)
        self.port = port
        self.arr = {}
        self.lock = threading.Lock()
        self.stats = {}
        self.bad_samples = []
        self.stop_flag = threading.Event()

    def run(self):
        while not self.stop_flag.is_set():
            d = ba.http_json(f"http://127.0.0.1:{self.port}/bench/drain", timeout=5)
            if d:
                with self.lock:
                    for eid, t in d["arrivals"]:
                        if eid is not None:
                            self.arr[eid] = t
                    self.stats = d.get("stats", {})
                    if d.get("bad_samples"):
                        self.bad_samples = d["bad_samples"]
            self.stop_flag.wait(1.0)

    def snapshot(self):
        with self.lock:
            return dict(self.arr), dict(self.stats), list(self.bad_samples)


class Poller(threading.Thread):
    def __init__(self, bpipes, collector_url, upipe_url):
        super().__init__(daemon=True)
        self.bpipes = bpipes
        self.collector_url = collector_url
        self.upipe_url = upipe_url
        self.samples = []
        self.dstats = []
        self.stop_flag = threading.Event()
        self.lock = threading.Lock()

    def one(self):
        s = {"t": time.time()}
        d = ba.http_json(self.collector_url + "/stats")
        if d:
            s["collector"] = ba.flatten(d)
        d = ba.http_json(self.upipe_url + "/health")
        if d:
            s["upipe"] = ba.flatten(d)
        for n, p in self.bpipes:
            d = ba.dexec_json(n, p, "health")
            if d:
                s[n] = ba.flatten(d)
        return s

    def run(self):
        last_ds = 0.0
        while not self.stop_flag.is_set():
            s = self.one()
            with self.lock:
                self.samples.append(s)
            if time.time() - last_ds > 15:
                ds = ba.docker_stats()
                with self.lock:
                    self.dstats.append((time.time(), ds))
                last_ds = time.time()
            self.stop_flag.wait(2.0)

    def window(self, t0, t1):
        with self.lock:
            return [s for s in self.samples if t0 <= s["t"] <= t1], [d for t, d in self.dstats if t0 <= t <= t1]

    def nearest(self, t):
        with self.lock:
            if not self.samples:
                return {}
            return min(self.samples, key=lambda s: abs(s["t"] - t))


def pct(v, p):
    return ba.pct(v, p)


class Generator:
    def __init__(self, pool, host, port, threads, batch, tag):
        self.pool, self.host, self.port, self.batch, self.tag = pool, host, port, batch, tag
        self.q = queue.Queue()
        self.counter = 0
        self.sent = {}
        self.lock = threading.Lock()
        self.http_err = 0
        self.statuses = {}
        self.sent_ok = 0
        self.stop_flag = threading.Event()
        self.rnd = random.Random(11)
        self.workers = [threading.Thread(target=self.work, daemon=True) for _ in range(threads)]
        for w in self.workers:
            w.start()

    def work(self):
        c = http.client.HTTPConnection(self.host, self.port, timeout=60)
        while not self.stop_flag.is_set():
            try:
                items = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            t = time.time()
            try:
                c.request("POST", "/store_items", json.dumps(items).encode(), {"Content-Type": "application/json"})
                r = c.getresponse()
                body = r.read()
                if r.status != 200:
                    raise RuntimeError(r.status)
                res = json.loads(body).get("results", {})
                with self.lock:
                    for it in items:
                        self.sent[it["external_id"]] = t
                    self.sent_ok += len(items)
                    for k, v in res.items():
                        self.statuses[k] = self.statuses.get(k, 0) + v
            except Exception:
                with self.lock:
                    self.http_err += len(items)
                c.close()
                c = http.client.HTTPConnection(self.host, self.port, timeout=60)

    def make_batch(self):
        items = []
        now = iso_now()
        for _ in range(self.batch):
            i = self.counter
            self.counter += 1
            dom = DOMAINS[i % len(DOMAINS)]
            base = self.pool[self.rnd.randrange(len(self.pool))]
            items.append({
                "content": f"{base} (ref {self.tag}{i})",
                "created_at": now,
                "domain": dom,
                "url": f"https://{dom}/bench/{self.tag}/{i}",
                "external_id": f"{self.tag}_{i}",
                "author": f"user{i % 997}",
                "title": "",
                "summary": "{}",
            })
        return items

    def run_rate(self, rate, seconds):
        t0 = time.time()
        interval = self.batch / rate
        nxt = t0
        scheduled = 0
        while time.time() - t0 < seconds:
            self.q.put(self.make_batch())
            scheduled += self.batch
            nxt += interval
            d = nxt - time.time()
            if d > 0:
                time.sleep(d)
        return t0, time.time(), scheduled

    def backlog(self):
        return self.q.qsize()


def run_step(rate, seconds, gen, sink, poller, args, limits, label, record=True):
    i0 = gen.counter
    err0 = gen.http_err
    st0 = dict(gen.statuses)
    t_start = time.time()
    s0 = poller.nearest(t_start)
    sampler = ba.Sampler()
    sampler.start()
    t0, t_end, scheduled = gen.run_rate(rate, seconds)
    t_wait = time.time()
    last_n, last_move = -1, time.time()
    while time.time() - t_wait < args.drain:
        ids = [f"{gen.tag}_{i}" for i in range(i0, gen.counter)]
        with gen.lock:
            n_sent = sum(1 for k in ids if k in gen.sent)
        arr, _, _ = sink.snapshot()
        got = sum(1 for k in ids if k in arr)
        if gen.q.qsize() == 0 and got >= n_sent:
            break
        if got != last_n:
            last_n, last_move = got, time.time()
        elif time.time() - last_move > 10:
            break
        time.sleep(1.0)
    t_fin = time.time()
    sampler.stop_flag.set()
    sampler.join(3)
    time.sleep(2.5)
    arr, sstats, bad_samples = sink.snapshot()
    with gen.lock:
        sent = dict(gen.sent)
        statuses = dict(gen.statuses)
        http_err = gen.http_err
    mine = {f"{gen.tag}_{i}": sent[f"{gen.tag}_{i}"] for i in range(i0, gen.counter) if f"{gen.tag}_{i}" in sent}
    http_err -= err0
    statuses = {k: v - st0.get(k, 0) for k, v in statuses.items() if v - st0.get(k, 0)}
    arrived = {k: arr[k] for k in mine if k in arr}
    window_start = t0 + 5
    in_win = [t for t in arrived.values() if window_start <= t <= t_end]
    win_rate = len(in_win) / max(1.0, (t_end - window_start))
    lat = [(arr[k] - mine[k]) for k in arrived]
    samples, dstats = poller.window(t_start, t_fin)
    s1 = poller.nearest(t_fin)

    def delta(src, key):
        a, b = s0.get(src, {}).get(key), s1.get(src, {}).get(key)
        return (b - a) if a is not None and b is not None else None

    def maxkey(src, key):
        v = [s[src][key] for s in samples if src in s and key in s[src]]
        return max(v) if v else None

    res = {
        "label": label, "offered": rate, "scheduled": scheduled,
        "sent_ok": len(mine), "http_err": http_err, "statuses": statuses,
        "arrived": len(arrived), "arrived_frac": len(arrived) / max(1, len(mine)),
        "sink_rate": win_rate, "p50_s": pct(lat, 50) if lat else None, "p95_s": pct(lat, 95) if lat else None,
        "max_s": max(lat) if lat else None, "gen_backlog_batches": gen.q.qsize(),
        "upipe_queue_max": maxkey("upipe", "queue"),
        "upipe_dropped": delta("upipe", "stats.dropped"), "upipe_errors": delta("upipe", "stats.errors"),
        "upipe_filtered_lang": delta("upipe", "stats.filtered_lang"),
        "collector_dropped": delta("collector", "collector.dropped"),
        "bad_items": sstats.get("bad_items", 0), "bad_samples": bad_samples,
    }
    bq = {}
    for n, _ in poller.bpipes:
        bq[n] = {"queue_max": maxkey(n, "queue"), "inflight_max": maxkey(n, "inflight"),
                 "dropped": delta(n, "stats.dropped"), "errors": delta(n, "stats.errors"),
                 "items_sent": delta(n, "stats.items_sent")}
    res["bpipe"] = bq
    g = list(zip(*sampler.gpu)) if sampler.gpu else None
    res["gpu_util_avg"] = statistics.mean(g[0]) if g else None
    res["gpu_vram_max"] = max(g[1]) if g else None
    res["cpu_host_avg"] = statistics.mean(sampler.cpu) if sampler.cpu else None
    cont = {}
    for d in dstats:
        for n, (c, m) in d.items():
            cont[n] = max(cont.get(n, 0), c)
    res["container_cpu_max"] = cont

    res["sustained"] = bool(
        res["arrived_frac"] >= 0.98 and win_rate >= 0.93 * rate and (res["p95_s"] or 1e9) <= args.max_p95
        and not (res["upipe_dropped"] or 0) and not (res["collector_dropped"] or 0) and not http_err
    )
    res["bottleneck"] = diagnose(res, limits)
    if record:
        out(f"  [{label}] offered {rate}/с: sink {win_rate:.0f}/с, дошло {len(arrived)}/{len(mine)} ({res['arrived_frac']*100:.0f}%), "
            f"p50 {res['p50_s'] if res['p50_s'] is None else round(res['p50_s'],2)}с p95 {res['p95_s'] if res['p95_s'] is None else round(res['p95_s'],2)}с, "
            f"{'ОК' if res['sustained'] else 'НЕ ВЫДЕРЖАЛ'}")
        gu = res["gpu_util_avg"]
        out(f"      upipe queue max {res['upipe_queue_max']}, upipe drop {res['upipe_dropped']}, err {res['upipe_errors']}, "
            f"filtered_lang {res['upipe_filtered_lang']}; collector drop {res['collector_dropped']}; статусы collector {statuses}")
        out("      bpipe: " + "; ".join(f"{n}: q_max {v['queue_max']} infl_max {v['inflight_max']} drop {v['dropped']} err {v['errors']}"
                                      for n, v in bq.items()))
        out(f"      GPU util {gu if gu is None else round(gu)}%, VRAM max {res['gpu_vram_max']}, CPU хоста {res['cpu_host_avg'] if res['cpu_host_avg'] is None else round(res['cpu_host_avg'])}%; "
            f"контейнеры CPU max: " + ", ".join(f"{n} {c:.0f}%" for n, c in sorted(cont.items()) if c > 20))
        if not res["sustained"]:
            out(f"      вероятное узкое место: {res['bottleneck']}")
        if res["bad_items"]:
            out(f"      ! некорректные поля в commit: {res['bad_items']} ({bad_samples[:3]})")
    return res


def diagnose(r, limits):
    up_lim = limits["upipe_queue"]
    bp_lim = limits["bpipe_queue"]
    cc = r["container_cpu_max"].get("collector", 0)
    bp_q = max([v["queue_max"] or 0 for v in r["bpipe"].values()] or [0])
    bp_drop = sum((v["dropped"] or 0) for v in r["bpipe"].values())
    gpu = r["gpu_util_avg"]
    reasons = []
    if r["http_err"]:
        reasons.append("collector не принимает (HTTP-ошибки)")
    if cc > 90 and (r["upipe_queue_max"] or 0) < 0.3 * up_lim and bp_q < 0.3 * bp_lim:
        reasons.append(f"collector упёрся в CPU ({cc:.0f}%), очереди ниже пусты")
    if (r["upipe_queue_max"] or 0) >= 0.5 * up_lim or (r["upipe_dropped"] or 0) > 0:
        if bp_q >= 0.5 * bp_lim or bp_drop > 0:
            reasons.append("очередь upipe полна из-за bpipe (обратное давление)")
        else:
            reasons.append("upipe (воркеры/CPU): очередь растёт, bpipe не заполнен")
    if bp_q >= 0.5 * bp_lim or bp_drop > 0:
        if gpu is not None and gpu >= 85:
            reasons.append(f"bpipe упёрся в GPU (util {gpu:.0f}%)")
        elif gpu is not None:
            reasons.append(f"bpipe: очередь полна, но GPU util {gpu:.0f}% -> CPU/Python/синхронизация внутри bpipe")
        else:
            reasons.append("bpipe: очередь полна")
    if not reasons and r["arrived_frac"] < 0.98:
        reasons.append("потери/задержка не объяснены очередями (смотри логи контейнеров)")
    return "; ".join(reasons) or "не определено"


def parse_cfg(c):
    m = re.fullmatch(r"r(\d)w(\d+)(?:b(\d+))?(?:h(\d+))?", c.strip())
    if not m:
        raise SystemExit(f"неверный формат конфига '{c}', пример: r3w4b32h16")
    return {"name": c.strip(), "replicas": int(m.group(1)), "workers": int(m.group(2)),
            "batch": int(m.group(3)) if m.group(3) else None, "hf": int(m.group(4)) if m.group(4) else None}


def compose(args_list, env=None, timeout=900):
    e = dict(os.environ)
    e.update(env or {})
    p = subprocess.run(["docker", "compose"] + args_list, cwd=ROOT, env=e, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        out(f"  compose {' '.join(args_list)} rc={p.returncode}: {(p.stderr or p.stdout)[-400:]}")
    return p.returncode


def wait_healthy(names, timeout=1200):
    t0 = time.time()
    pending = set(names)
    while pending and time.time() - t0 < timeout:
        for n in list(pending):
            rc, o = ba.sh(["docker", "inspect", "-f", "{{.State.Health.Status}}", n])
            if rc == 0 and o.strip() == "healthy":
                pending.discard(n)
        if pending:
            time.sleep(5)
    return not pending


def apply_config(cfg, base_env):
    r = cfg["replicas"]
    bnames = ["bpipe", "bpipe_2", "bpipe_3"][:r]
    ports = [7995, 7996, 7997][:r]
    urls = ",".join(f"http://{n}:{p}/" for n, p in zip(bnames, ports))
    env = {
        "BPIPE_URLS": urls, "COMPOSE_PROFILES": ",".join(["bpipe2"] * (r >= 2) + ["bpipe3"] * (r >= 3)),
        "UPIPE_WORKERS": str(cfg["workers"]), "URL_DEDUP_MODE": "off", "FOREIGN_FORWARD_URL": "",
        "GPU_TRANSLATOR_URL": "", "TRANSLATE": "false",
    }
    if cfg["batch"]:
        env["FIXED_BATCH_SIZE"] = str(cfg["batch"])
    if cfg["hf"]:
        env["TAG_HF_BATCH_SIZE"] = str(cfg["hf"])
    allsvc = ["collector", "upipe", "bpipe", "bpipe_2", "bpipe_3"]
    ba.sh(["docker", "stop"] + [n for n in allsvc if n in ba.running_containers()], timeout=180)
    out(f"  поднимаю bpipe x{r}, upipe workers={cfg['workers']}" + (f", batch={cfg['batch']}" if cfg["batch"] else "") + (f", hf={cfg['hf']}" if cfg["hf"] else ""))
    if compose(["up", "-d", "--force-recreate", "--no-deps"] + bnames, env) != 0:
        return None
    if not wait_healthy(bnames):
        out("  bpipe не стал healthy за 20 минут")
        return None
    if compose(["up", "-d", "--force-recreate", "--no-deps", "upipe"], env) != 0 or not wait_healthy(["upipe"], 300):
        return None
    if compose(["up", "-d", "--force-recreate", "--no-deps", "collector"], env) != 0 or not wait_healthy(["collector"], 300):
        return None
    return list(zip(bnames, ports))


def inspect_val(name, fmt):
    rc, o = ba.sh(["docker", "inspect", "-f", fmt, name])
    return o.strip() if rc == 0 else ""


def install_mock():
    image = inspect_val("collector", "{{.Config.Image}}")
    network = inspect_val("collector", "{{range $k,$v := .NetworkSettings.Networks}}{{$k}} {{end}}").split()
    if not image or not network:
        raise SystemExit("не нашёл контейнер collector (его нужно хотя бы раз создать: ./run.sh up -d)")
    state = inspect_val(MOCK_NAME, "{{.State.Running}}")
    was_running = state == "true"
    if state:
        ba.sh(["docker", "stop", MOCK_NAME], timeout=120)
        ba.sh(["docker", "rename", MOCK_NAME, REAL_BACKUP])
    mock_py = os.path.join(ROOT, "bench_mock_tx.py")
    rc, o = ba.sh(["docker", "run", "-d", "--name", MOCK_NAME, "--network", network[0], "--network-alias", MOCK_NAME,
                   "-p", f"127.0.0.1:{SINK_PORT}:8002", "-v", f"{mock_py}:/app/bench_mock_tx.py:ro",
                   image, "python", "/app/bench_mock_tx.py"], timeout=120)
    if rc != 0:
        ba.sh(["docker", "rename", REAL_BACKUP, MOCK_NAME])
        if was_running:
            ba.sh(["docker", "start", MOCK_NAME])
        raise SystemExit(f"не удалось запустить заглушку transactioneer: {o}")
    return bool(state), was_running


def remove_mock(had_real, real_was_running):
    ba.sh(["docker", "rm", "-f", MOCK_NAME], timeout=60)
    if had_real:
        ba.sh(["docker", "rename", REAL_BACKUP, MOCK_NAME])
        if real_was_running:
            ba.sh(["docker", "start", MOCK_NAME], timeout=60)


def summarize(results):
    out("\n=== ИТОГ ПО КОНФИГУРАЦИЯМ ===")
    out(f"{'конфиг':<14}{'макс. устойчивый items/s':>26}{'пик sink items/s':>18}{'GPU util на пике':>18}  узкое место при срыве")
    best = None
    for name, steps in results.items():
        ok = [s for s in steps if s["sustained"]]
        top = max(ok, key=lambda s: s["offered"]) if ok else None
        peak = max(steps, key=lambda s: s["sink_rate"]) if steps else None
        fail = next((s for s in steps if not s["sustained"]), None)
        out(f"{name:<14}{(top['offered'] if top else 0):>26}{(round(peak['sink_rate']) if peak else 0):>18}"
            f"{(round(peak['gpu_util_avg']) if peak and peak['gpu_util_avg'] is not None else '-'):>17}%  {fail['bottleneck'] if fail else 'не сорвался в тестируемом диапазоне'}")
        if top and (best is None or top["offered"] > best[1]):
            best = (name, top["offered"])
    if best:
        near = [n for n, steps in results.items()
                if any(s["sustained"] and s["offered"] >= 0.95 * best[1] for s in steps)]
        out(f"\nЛучший устойчивый уровень: {best[1]} items/s. Конфигурации в пределах 5%: {', '.join(near)}")
        out("Рекомендация: выбрать из них самую лёгкую (меньше реплик/воркеров).")
    else:
        out("\nНи одна конфигурация не выдержала самый низкий шаг, см. подробности выше.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--configs", default="r1w4,r2w4,r3w4,r3w8,r3w4b64",
                    help="формат r<реплик bpipe>w<воркеров upipe>[b<FIXED_BATCH_SIZE>][h<TAG_HF_BATCH_SIZE>]")
    ap.add_argument("--rates", default="100,200,300,400,600,800", help="ступени нагрузки, items/s")
    ap.add_argument("--step-seconds", type=int, default=40)
    ap.add_argument("--drain", type=float, default=40.0)
    ap.add_argument("--max-p95", type=float, default=20.0, help="допустимый p95 задержки collector->sink, с")
    ap.add_argument("--batch", type=int, default=20, help="размер пачки в /store_items")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--texts")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--external-stack", action="store_true", help="не трогать docker: гонять нагрузку на уже поднятый collector:9000 и sink:18002")
    args = ap.parse_args()
    if args.quick:
        args.configs = "r1w4,r3w4"
        args.rates = "100,200,400,600"
        args.step_seconds = 25
    cfgs = [parse_cfg(c) for c in args.configs.split(",")]
    rates = [int(x) for x in args.rates.split(",")]
    envs = ba.read_env()
    limits = {
        "upipe_queue": int(envs.get("UPIPE_QUEUE_LIMIT") or 200),
        "bpipe_queue": int(envs.get("BPIPE_QUEUE_DEFAULT_MAX") or 200) + int(envs.get("BPIPE_QUEUE_TRANSLATED_MAX") or 100),
    }
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(ROOT, "bench_out", f"load_{ts}")
    os.makedirs(outdir, exist_ok=True)
    os.chmod(outdir, 0o777)

    ba.capture_env(outdir)
    out(f"\n=== НАГРУЗОЧНЫЙ ТЕСТ ПАЙПЛАЙНА (синтетические тексты; реальный transactioneer заменён заглушкой, в Exorde ничего не уходит) ===")
    out(f"конфигурации: {[c['name'] for c in cfgs]}; ступени: {rates} items/s по {args.step_seconds} с; лимиты: {limits}")

    if not args.external_stack and not args.yes:
        print("\nОстановлю боевые collector/upipe/bpipe/transactioneer, подниму тестовые конфигурации и заглушку вместо transactioneer.\n"
              "В конце всё вернётся в исходное состояние. Продолжить? [y/N] ", end="")
        if input().strip().lower() != "y":
            return

    cont0 = ba.running_containers()
    prod_running = [n for n in ("collector", "upipe", "bpipe", "bpipe_2", "bpipe_3") if n in cont0]
    mock_state = None
    results = {}
    pool = make_pool(5000, texts_file=args.texts)
    sink = Sink(SINK_PORT)
    try:
        if not args.external_stack:
            print("Если скрипт убить жёстко (kill -9), вернуть боевую систему так: "
                  "docker rm -f transactioneer; docker rename transactioneer_real_bench transactioneer; ./run.sh up -d --force-recreate")
            mock_state = install_mock()
            time.sleep(2)
        sink.start()
        for cfg in cfgs:
            out(f"\n##### конфигурация {cfg['name']} #####")
            if args.external_stack:
                bpipes = []
            else:
                bpipes = apply_config(cfg, envs)
                if not bpipes:
                    out("  конфигурация не поднялась, пропускаю")
                    continue
            poller = Poller(bpipes, "http://127.0.0.1:9000", "http://127.0.0.1:5981")
            poller.start()
            gen = Generator(pool, "127.0.0.1", 9000, args.threads, args.batch, uuid_tag())
            steps = []
            try:
                out("  прогрев...")
                run_step(30, 15, gen, sink, poller, args, limits, "прогрев", record=False)
                for rate in rates:
                    r = run_step(rate, args.step_seconds, gen, sink, poller, args, limits, f"{cfg['name']}@{rate}")
                    steps.append(r)
                    if not r["sustained"]:
                        out("  срыв: выше не поднимаемся")
                        break
            finally:
                gen.stop_flag.set()
                poller.stop_flag.set()
            results[cfg["name"]] = steps
            json.dump(results, open(os.path.join(outdir, "load.json"), "w"), ensure_ascii=False, indent=1, default=str)
        summarize(results)
    finally:
        sink.stop_flag.set()
        if not args.external_stack:
            out("\nВозвращаю боевую конфигурацию...")
            allsvc = ["collector", "upipe", "bpipe", "bpipe_2", "bpipe_3"]
            ba.sh(["docker", "stop"] + [n for n in allsvc if n in ba.running_containers()], timeout=180)
            if mock_state is not None:
                remove_mock(*mock_state)
            if prod_running:
                n = int(envs.get("BPIPE_REPLICAS") or 1)
                bn = ["bpipe", "bpipe_2", "bpipe_3"][:n]
                urls = ",".join(f"http://{b}:{p}/" for b, p in zip(bn, [7995, 7996, 7997]))
                env = {"BPIPE_URLS": urls, "COMPOSE_PROFILES": ",".join(["bpipe2"] * (n >= 2) + ["bpipe3"] * (n >= 3))}
                names = [x for x in ["bpipe", "bpipe_2", "bpipe_3"][:n] + ["upipe", "collector"]]
                compose(["up", "-d", "--force-recreate", "--no-deps"] + names, env)
                out(f"восстановлено: {names}")
            else:
                out("боевые сервисы не были запущены, оставляю остановленными")
        rp = os.path.join(outdir, "report.txt")
        open(rp, "w", encoding="utf-8").write("\n".join(ba.report_lines) + "\n")
        out(f"\nГОТОВО. Пришли файл: {rp}")


def uuid_tag():
    return "b" + format(random.getrandbits(32), "08x")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nпрервано")
