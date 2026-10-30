import argparse
import http.client
import importlib.util
import json
import math
import multiprocessing as mp
import os
import pickle
import queue
import random
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

WORDS = (
    "the be to of and a in that have i it for not on with he as you do at this but his by from they we say her she or an "
    "will my one all would there their what so up out if about who get which go me when make can like time no just him "
    "know take people into year your good some could them see other than then now look only come its over think also "
    "back after use two how our work first well way even new want because any these give day most us great love really "
    "photo camera lens angle light shot game team win season price market stock coin bitcoin health doctor law court "
    "vote policy phone laptop code python model data cloud music movie show actor song review honestly agree wrong right "
    "thanks sorry maybe never always ever literally actually basically pretty quite totally probably definitely".split()
)

STAGE_HINTS = {
    "keywords_yake_n1": "YAKE n=1: уменьшить top, кэшировать по хэшу контента, вынести в отдельный пул",
    "keywords_yake_n2": "YAKE n=2: dedupFunc='seqm' самый медленный, см. таблицу вариантов dedup; уменьшить windowsSize/top",
    "keywords_postfilter": "filter_strings/remove_invalid: O(n^2) проверки `s not in out`, заменить на set/dict",
    "token_count": "tiktoken: считать один раз, не дублировать; проверять длину по символам до encode",
    "build_item": "pydantic-подобная валидация Item: собирать один раз (сейчас строится и в main, и в воркере)",
    "preprocess": "почти no-op, можно убрать",
    "zero_shot": "стаб, не влияет",
}


def pct(vals, p):
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * p / 100))]


def fmt_iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def make_texts(n, seed=1):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        k = int(min(200, max(3, rnd.lognormvariate(math.log(22), 0.9))))
        out.append(" ".join(rnd.choice(WORDS) for _ in range(k)).capitalize() + ".")
    return out


def load_texts(path, n):
    lines = [l.strip() for l in open(path, encoding="utf-8") if len(l.strip()) >= 20]
    if not lines:
        raise SystemExit(f"в {path} нет строк длиннее 20 символов")
    return [lines[i % len(lines)] for i in range(n)]


def make_raws(n, texts):
    tag = uuid.uuid4().hex[:8]
    now = fmt_iso(datetime.now(timezone.utc))
    out = []
    for i in range(n):
        raw = {
            "content": texts[i % len(texts)],
            "created_at": now,
            "domain": "bench.local",
            "url": f"https://bench.local/{tag}/{i}",
            "external_id": f"{tag}_{i}",
            "author": f"a{i % 50}",
            "title": "",
            "lang_ok": True,
        }
        if i % 3 == 0:
            raw["links_suffix"] = f"\n--https://example.com/x{i}"
        out.append(raw)
    return out


def run_chunk(chunk):
    import worker
    n = 0
    for raw in chunk:
        worker.run(raw, ("en", raw["content"]))
        n += 1
    return n


def profile(raws, warm):
    import worker
    import extract_keywords as ek
    from evaluate_token_count import evaluate_token_count
    from preprocess import preprocess
    from zero_shot import zero_shot
    from exorde_data import Translation, Language, Translated

    worker.init()
    lab = worker._lab
    for raw in raws[:warm]:
        worker.run(raw, ("en", raw["content"]))

    st = {k: [] for k in (
        "build_item", "preprocess", "token_count", "keywords_yake_n1",
        "keywords_yake_n2", "keywords_postfilter", "zero_shot",
    )}
    pc = time.perf_counter_ns
    for raw in raws:
        t = pc(); item = worker.build_item(raw); st["build_item"].append(pc() - t)
        t = pc(); item = preprocess(item, False); st["preprocess"].append(pc() - t)
        text = str(item.content)
        t = pc(); evaluate_token_count(text); st["token_count"].append(pc() - t)
        tr = Translation(language=Language("en"), translation=Translated(text))
        t = pc(); r1 = ek.kw_extractor1.extract_keywords(text); st["keywords_yake_n1"].append(pc() - t)
        t = pc(); r2 = ek.kw_extractor2.extract_keywords(text); st["keywords_yake_n2"].append(pc() - t)
        t = pc()
        kx = [e[0] for e in set(r1)] + [e[0] for e in set(r2)]
        kx = ek.process_keywords(ek.remove_invalid(ek.filter_strings(kx)))
        list(set(kx))
        st["keywords_postfilter"].append(pc() - t)
        t = pc(); zero_shot(tr, lab, max_depth=2); st["zero_shot"].append(pc() - t)

    total, errors, sample = [], 0, None
    for raw in raws:
        t = pc()
        try:
            res = worker.run(raw, ("en", raw["content"]))
            sample = sample or res
        except Exception:
            errors += 1
        total.append(pc() - t)

    try:
        import orjson
        dumps = orjson.dumps
    except ImportError:
        dumps = lambda o: json.dumps(o).encode()
    raw_bytes = [json.dumps(r).encode() for r in raws]
    t = pc()
    for b in raw_bytes:
        json.loads(b)
    parse_ns = (pc() - t) / len(raws)
    t = pc()
    for _ in raws:
        dumps(sample)
    dump_ns = (pc() - t) / len(raws)
    pickled = (len(pickle.dumps(raws[0])), len(pickle.dumps(sample)) if sample else 0)

    return st, total, errors, parse_ns, dump_ns, pickled


def print_profile(st, total, errors, parse_ns, dump_ns, pickled, build_mean_ns):
    mean_total = statistics.mean(total)
    rows = []
    for name, v in st.items():
        rows.append((name, statistics.mean(v), pct(v, 50), pct(v, 95)))
    rows.sort(key=lambda r: -r[1])
    ssum = sum(r[1] for r in rows)
    print("\n=== 1. Профиль стадий (один процесс, прогретый) ===")
    print(f"{'стадия':<22}{'mean мс':>9}{'p50 мс':>9}{'p95 мс':>9}{'доля':>8}")
    for name, m, p50, p95 in rows:
        print(f"{name:<22}{m/1e6:>9.3f}{p50/1e6:>9.3f}{p95/1e6:>9.3f}{m/ssum*100:>7.1f}%")
    print(f"{'worker.run целиком':<22}{mean_total/1e6:>9.3f}{pct(total,50)/1e6:>9.3f}{pct(total,95)/1e6:>9.3f}")
    print(f"ошибок в run: {errors}; потолок 1 процесса: {1e9/mean_total:.0f} items/s")
    main_ns = parse_ns + build_mean_ns + dump_ns
    print(f"\nГлавный процесс (последовательно, 1 поток): json.loads {parse_ns/1e3:.0f}мкс + build_item {build_mean_ns/1e3:.0f}мкс "
          f"+ dumps {dump_ns/1e3:.0f}мкс = {main_ns/1e3:.0f}мкс -> потолок {1e9/main_ns:.0f} items/s (без aiohttp)")
    print(f"Pickle в пул: вход {pickled[0]} Б, выход {pickled[1]} Б")
    return rows, mean_total, main_ns


def ipc_overhead(raws, inproc_mean_ns, n=200):
    import worker
    ex = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"), initializer=worker.init)
    ex.submit(worker.ping).result()
    for raw in raws[:20]:
        ex.submit(worker.run, raw, ("en", raw["content"])).result()
    lat = []
    for raw in raws[:n]:
        t = time.perf_counter_ns()
        ex.submit(worker.run, raw, ("en", raw["content"])).result()
        lat.append(time.perf_counter_ns() - t)
    ex.shutdown()
    m = statistics.mean(lat)
    print(f"\nIPC: round-trip через пул {m/1e6:.3f} мс vs в процессе {inproc_mean_ns/1e6:.3f} мс -> накладные ~{max(0, m-inproc_mean_ns)/1e6:.3f} мс/item")
    return max(0, m - inproc_mean_ns)


def yake_variants(texts):
    try:
        import yake
        import extract_keywords as ek
    except Exception as e:
        print(f"\nYAKE-варианты пропущены: {e}")
        return []
    sample = texts[:300]
    variants = [("seqm (текущий)", "seqm"), ("leve", "leve"), ("jaro_winkler", "jaro_winkler")]
    base_sets = None
    out = []
    print("\n=== 2. Варианты dedupFunc в YAKE (только keywords) ===")
    print(f"{'вариант':<18}{'мс/item':>9}{'ускорение':>11}{'совпад. keywords':>18}")
    base_time = None
    for name, dd in variants:
        try:
            k1 = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.9, dedupFunc=dd, windowsSize=1, top=20)
            k2 = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, dedupFunc=dd, windowsSize=7, top=10)
            sets = []
            t = time.perf_counter()
            for s in sample:
                kx = [e[0] for e in set(k1.extract_keywords(s))] + [e[0] for e in set(k2.extract_keywords(s))]
                kx = ek.process_keywords(ek.remove_invalid(ek.filter_strings(kx)))
                sets.append(set(kx))
            dt = (time.perf_counter() - t) / len(sample)
        except Exception as e:
            print(f"{name:<18}ошибка: {e}")
            continue
        if base_sets is None:
            base_sets, base_time = sets, dt
        jac = statistics.mean(
            (len(a & b) / len(a | b)) if (a | b) else 1.0 for a, b in zip(base_sets, sets)
        )
        print(f"{name:<18}{dt*1e3:>9.3f}{base_time/dt:>10.2f}x{jac*100:>17.1f}%")
        out.append({"variant": name, "ms": dt * 1e3, "speedup": base_time / dt, "similarity": jac})
    return out


def pool_run(raws, w, chunk):
    import worker
    ex = ProcessPoolExecutor(max_workers=w, mp_context=mp.get_context("spawn"), initializer=worker.init)
    wait([ex.submit(worker.ping) for _ in range(w)])
    for raw in raws[:w * 5]:
        ex.submit(worker.run, raw, ("en", raw["content"])).result()
    if chunk == 1:
        tasks = [(worker.run, (r, ("en", r["content"]))) for r in raws]
    else:
        tasks = [(run_chunk, (raws[i:i + chunk],)) for i in range(0, len(raws), chunk)]
    pending, started, lat, errors = {}, {}, [], 0
    it = iter(tasks)
    t0 = time.perf_counter()
    done_flag = False
    while pending or not done_flag:
        while len(pending) < w and not done_flag:
            try:
                fn, a = next(it)
            except StopIteration:
                done_flag = True
                break
            f = ex.submit(fn, *a)
            pending[f] = time.perf_counter()
        if not pending:
            break
        done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
        now = time.perf_counter()
        for f in done:
            lat.append((now - pending.pop(f)) * 1e3)
            if f.exception():
                errors += 1
    elapsed = time.perf_counter() - t0
    ex.shutdown()
    return len(raws) / elapsed, lat, errors


def pool_sweep(raws, workers_list, chunks):
    print("\n=== 3. Пул процессов (как в upipe: in-flight = число воркеров) ===")
    print(f"{'workers':>8}{'chunk':>7}{'items/s':>10}{'p50 мс':>9}{'p95 мс':>9}{'err':>5}")
    res = []
    for w in workers_list:
        for ch in chunks:
            ips, lat, err = pool_run(raws, w, ch)
            print(f"{w:>8}{ch:>7}{ips:>10.0f}{pct(lat,50):>9.2f}{pct(lat,95):>9.2f}{err:>5}")
            res.append({"workers": w, "chunk": ch, "items_per_s": ips, "p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95), "errors": err})
    return res


ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,3})?Z$")


class Sink:
    def __init__(self, delay_ms):
        self.lock = threading.Lock()
        self.arrivals = {}
        self.bad = 0
        self.bad_reason = ""
        self.delay = delay_ms / 1000.0
        self.expect = {}
        self.first_send = None
        self.event = threading.Event()
        self.target = 0

    def add(self, body, t):
        reason = ""
        try:
            p = json.loads(body)
            it = p["item"]
            eid = it["external_id"]
            if not ISO_RE.match(it.get("created_at", "")):
                reason = f"created_at={it.get('created_at')!r}"
            elif it.get("domain") != "bench.local":
                reason = "domain"
            elif not it.get("content"):
                reason = "empty content"
            elif "top_keywords" not in p or "classification" not in p or "translation" not in p:
                reason = "нет top_keywords/classification/translation"
            else:
                suf = self.expect.get(eid, "")
                if suf and not it["content"].endswith(suf):
                    reason = "потерян links_suffix"
        except Exception as e:
            eid, reason = None, f"parse: {e}"
        with self.lock:
            if reason:
                self.bad += 1
                self.bad_reason = self.bad_reason or reason
            if eid is not None:
                self.arrivals[eid] = t
            if len(self.arrivals) >= self.target:
                self.event.set()


def make_handler(sink):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n)
            sink.add(body, time.perf_counter())
            if sink.delay:
                time.sleep(sink.delay)
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *a):
            pass
    return H


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def http_get(port, path, timeout=2):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("GET", path)
    r = c.getresponse()
    body = r.read()
    c.close()
    return r.status, body


def e2e_one(raws, w, q, conc, bpipes, delay_ms, upipe_py, timeout):
    sink = Sink(delay_ms)
    sink.target = len(raws)
    sink.expect = {r["external_id"]: r["links_suffix"] for r in raws if r.get("links_suffix")}
    servers, urls = [], []
    for _ in range(bpipes):
        port = free_port()
        srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(sink))
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        urls.append(f"http://127.0.0.1:{port}/")

    uport = free_port()
    env = dict(os.environ, UPIPE_PORT=str(uport), BPIPE_URLS=",".join(urls), UPIPE_WORKERS=str(w),
               UPIPE_QUEUE_LIMIT=str(q), TRANSLATE="false", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    logf = tempfile.NamedTemporaryFile("w+", suffix=".log", delete=False)
    proc = subprocess.Popen([sys.executable, upipe_py], cwd=os.path.dirname(upipe_py), env=env,
                            stdout=logf, stderr=subprocess.STDOUT)
    result = None
    try:
        deadline = time.time() + 180
        ready = False
        while time.time() < deadline and proc.poll() is None:
            try:
                if http_get(uport, "/health")[0] == 200:
                    ready = True
                    break
            except Exception:
                time.sleep(0.5)
        if not ready:
            logf.flush()
            tail = open(logf.name).read()[-600:]
            return {"workers": w, "queue": q, "error": f"upipe не поднялся: {tail}"}

        work = queue.Queue()
        for r in raws:
            work.put(r)
        sent, retries, send_err = {}, [0], [0]
        lock = threading.Lock()

        def client():
            c = http.client.HTTPConnection("127.0.0.1", uport, timeout=30)
            while True:
                try:
                    r = work.get_nowait()
                except queue.Empty:
                    break
                body = json.dumps(r).encode()
                while True:
                    try:
                        c.request("POST", "/", body, {"Content-Type": "application/json"})
                        resp = c.getresponse()
                        resp.read()
                    except Exception:
                        with lock:
                            send_err[0] += 1
                        c.close()
                        c = http.client.HTTPConnection("127.0.0.1", uport, timeout=30)
                        break
                    if resp.status == 503:
                        with lock:
                            retries[0] += 1
                        time.sleep(0.005)
                        continue
                    with lock:
                        sent[r["external_id"]] = time.perf_counter()
                    break
            c.close()

        t0 = time.perf_counter()
        sink.first_send = t0
        threads = [threading.Thread(target=client) for _ in range(conc)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        last_n, last_move = 0, time.time()
        while not sink.event.wait(1.0):
            n = len(sink.arrivals)
            if n != last_n:
                last_n, last_move = n, time.time()
            elif time.time() - last_move > timeout:
                break
        with sink.lock:
            arr = dict(sink.arrivals)
        elapsed = (max(arr.values()) - t0) if arr else 0.0
        lat = [(arr[k] - sent[k]) * 1e3 for k in arr if k in sent]
        try:
            stats = json.loads(http_get(uport, "/health")[1]).get("stats", {})
        except Exception:
            stats = {}
        result = {
            "workers": w, "queue": q, "arrived": len(arr), "total": len(raws),
            "items_per_s": len(arr) / elapsed if elapsed else 0.0,
            "p50_ms": pct(lat, 50), "p95_ms": pct(lat, 95),
            "retries_503": retries[0], "send_errors": send_err[0],
            "upipe_errors": stats.get("errors", 0), "bad_fields": sink.bad, "bad_reason": sink.bad_reason,
        }
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            proc.kill()
        for s in servers:
            s.shutdown()
        logf.close()
        try:
            os.unlink(logf.name)
        except OSError:
            pass
    return result


def e2e_sweep(texts, args, workers_list):
    upipe_py = os.path.join(HERE, "upipe.py")
    ok = importlib.util.find_spec("aiohttp") is not None and importlib.util.find_spec("orjson") is not None
    if not ok or not os.path.exists(upipe_py):
        print("\n=== 4. E2E пропущен: нужны aiohttp и orjson (запускай внутри контейнера upipe) ===")
        return []
    print("\n=== 4. E2E: реальный upipe.py + mock bpipe ===")
    print(f"{'workers':>8}{'queue':>7}{'items/s':>9}{'p50 мс':>9}{'p95 мс':>9}{'503':>7}{'err':>5}{'bad':>5}{'дошло':>12}")
    res = []
    for w in workers_list:
        for q in args.queues:
            raws = make_raws(args.e2e_n, texts)
            r = e2e_one(raws, w, q, args.conc, args.bpipes, args.bpipe_delay_ms, upipe_py, args.timeout)
            res.append(r)
            if "error" in r:
                print(f"{w:>8}{q:>7}  ОШИБКА: {r['error']}")
                continue
            print(f"{r['workers']:>8}{r['queue']:>7}{r['items_per_s']:>9.0f}{r['p50_ms']:>9.1f}{r['p95_ms']:>9.1f}"
                  f"{r['retries_503']:>7}{r['upipe_errors']:>5}{r['bad_fields']:>5}{r['arrived']:>7}/{r['total']}")
            if r["bad_fields"]:
                print(f"         ! некорректные поля в bpipe-payload: {r['bad_reason']}")
    return res


def pick_best(rows, key="items_per_s"):
    good = [r for r in rows if "error" not in r and not r.get("errors") and not r.get("upipe_errors") and not r.get("bad_fields")
            and r.get("arrived", r.get("total", 1)) == r.get("total", r.get("arrived", 1))]
    if not good:
        return None
    top = max(r[key] for r in good)
    cands = [r for r in good if r[key] >= 0.95 * top]
    return min(cands, key=lambda r: (r["workers"], r.get("chunk", 1), r.get("p95_ms", 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--e2e-n", type=int, default=1500)
    ap.add_argument("--texts", help="файл с реальными текстами, по одному на строку")
    ap.add_argument("--workers", help="список через запятую, по умолчанию от CPU")
    ap.add_argument("--chunks", default="1,8,32")
    ap.add_argument("--queues", default="200,2000")
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--bpipes", type=int, default=1)
    ap.add_argument("--bpipe-delay-ms", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--no-e2e", action="store_true")
    ap.add_argument("--no-yake-variants", action="store_true")
    ap.add_argument("--out", default="bench_upipe_result.json")
    args = ap.parse_args()
    args.queues = [int(x) for x in args.queues.split(",")]
    chunks = [int(x) for x in args.chunks.split(",")]
    cpu = os.cpu_count() or 1
    if args.workers:
        workers_list = [int(x) for x in args.workers.split(",")]
    else:
        workers_list = sorted({1, 2, max(1, cpu // 2), cpu, cpu * 2})

    pool_texts = load_texts(args.texts, 2000) if args.texts else make_texts(2000)
    raws = make_raws(args.n, pool_texts)
    avg_len = statistics.mean(len(r["content"]) for r in raws)
    print(f"CPU: {cpu}; items: {args.n}; средняя длина текста: {avg_len:.0f} симв.; workers: {workers_list}")
    print("Важно: останови боевой upipe/bpipe на время замера, иначе CPU делится и цифры занижены.")

    out = {"cpu": cpu, "n": args.n, "avg_len": avg_len}

    st, total, errors, parse_ns, dump_ns, pickled = profile(raws, warm=100)
    rows, mean_total, main_ns = print_profile(
        st, total, errors, parse_ns, dump_ns, pickled, statistics.mean(st["build_item"]))
    ipc_ns = ipc_overhead(raws, mean_total)
    out["profile"] = {r[0]: {"mean_ms": r[1] / 1e6, "p95_ms": r[3] / 1e6} for r in rows}
    out["profile"]["run_total_ms"] = mean_total / 1e6
    out["main_loop_ceiling_items_per_s"] = 1e9 / main_ns
    out["ipc_ms"] = ipc_ns / 1e6

    if not args.no_yake_variants:
        out["yake_variants"] = yake_variants(pool_texts)

    pool_res = pool_sweep(raws, workers_list, chunks)
    out["pool"] = pool_res

    e2e_res = [] if args.no_e2e else e2e_sweep(pool_texts, args, workers_list)
    out["e2e"] = e2e_res

    print("\n=== ВЫВОДЫ ===")
    top = rows[0]
    print(f"- Самая тяжёлая стадия: {top[0]} ({top[1]/1e6:.3f} мс, {top[1]/sum(r[1] for r in rows)*100:.0f}% суммы стадий)")
    print(f"  -> {STAGE_HINTS.get(top[0], '')}")
    ordered = [r for r in rows if r[1] / sum(x[1] for x in rows) > 0.15]
    for r in ordered[1:]:
        print(f"- Также заметно: {r[0]} ({r[1]/1e6:.3f} мс) -> {STAGE_HINTS.get(r[0], '')}")
    resid = mean_total - sum(r[1] for r in rows)
    if resid > 0.2 * mean_total:
        print(f"- Вне стадий (сборка ответа/обвязка) {resid/1e6:.3f} мс/item ({resid/mean_total*100:.0f}%)")
    best_pool = pick_best(pool_res)
    if best_pool:
        print(f"- Пул: лучше всего workers={best_pool['workers']}, chunk={best_pool['chunk']} "
              f"({best_pool['items_per_s']:.0f} items/s, p95 {best_pool['p95_ms']:.1f} мс; наименьшая конфигурация в пределах 5% от максимума)")
        base = next((r for r in pool_res if r["chunk"] == 1 and r["workers"] == best_pool["workers"]), None)
        if base and best_pool["chunk"] > 1:
            print(f"  -> батчинг даёт x{best_pool['items_per_s']/base['items_per_s']:.2f} к chunk=1: IPC на item заметен, "
                  f"имеет смысл отправлять в пул пачками")
        elif ipc_ns > 0.3 * mean_total:
            print("  -> IPC на item >30% от вычислений, но батчинг в замере не помог")
    print(f"- Потолок главного процесса ~{out['main_loop_ceiling_items_per_s']:.0f} items/s: "
          f"{'упрёмся в него раньше пула' if best_pool and best_pool['items_per_s'] > 0.7 * out['main_loop_ceiling_items_per_s'] else 'пул медленнее, узкое место в воркерах'}")
    best_e2e = pick_best([r for r in e2e_res if "error" not in r])
    if best_e2e:
        print(f"- E2E: лучше всего UPIPE_WORKERS={best_e2e['workers']}, UPIPE_QUEUE_LIMIT={best_e2e['queue']} "
              f"({best_e2e['items_per_s']:.0f} items/s, p95 {best_e2e['p95_ms']:.0f} мс, 503={best_e2e['retries_503']})")
        small_q = [r for r in e2e_res if "error" not in r and r["workers"] == best_e2e["workers"] and r["queue"] != best_e2e["queue"]]
        for r in small_q:
            if r["retries_503"] != best_e2e["retries_503"]:
                print(f"  -> при queue={r['queue']}: 503={r['retries_503']} (в боевом режиме это дропы)")
    elif e2e_res:
        print("- E2E: ни одна конфигурация не прошла без ошибок/потерь, см. таблицу")
    bad = sum(r.get("bad_fields", 0) for r in e2e_res)
    if sum(r.get("arrived", 0) for r in e2e_res):
        print(f"- Поля в payload к bpipe: {'все корректны (created_at ...Z, links_suffix сохранён, keywords/classification на месте)' if not bad else f'НАЙДЕНЫ ошибки: {bad}'}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=str)
    print(f"\nJSON: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
