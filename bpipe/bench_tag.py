import argparse
import json
import logging
import math
import os
import random
import statistics
import subprocess
import sys
import time

WORDS = (
    "the be to of and a in that have i it for not on with he as you do at this but his by from they we say her she or an "
    "will my one all would there their what so up out if about who get which go me when make can like time no just him "
    "know take people into year your good some could them see other than then now look only come its over think also "
    "back after use two how our work first well way even new want because any these give day most us great love really "
    "photo camera lens angle light shot game team win season price market stock coin bitcoin health doctor law court "
    "vote policy phone laptop code python model data cloud music movie show actor song review honestly agree wrong right "
    "thanks sorry maybe never always ever literally actually basically pretty quite totally probably definitely".split()
)


def make_texts(n, seed=1):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        k = int(min(300, max(3, rnd.lognormvariate(math.log(22), 0.9))))
        out.append(" ".join(rnd.choice(WORDS) for _ in range(k)).capitalize() + ".")
    return out


def load_texts(path, n):
    lines = [l.strip() for l in open(path, encoding="utf-8") if len(l.strip()) >= 20]
    if not lines:
        raise SystemExit(f"в {path} нет строк длиннее 20 символов")
    return [lines[i % len(lines)] for i in range(n)]


def get_pool(args):
    size = max(args.n, 128) * 8
    return load_texts(args.texts, size) if args.texts else make_texts(size)


def load_models():
    import torch
    from lab_initialization import lab_initialization
    from tag import tag

    logging.getLogger().setLevel(logging.WARNING)
    lab = lab_initialization()
    logging.getLogger().setLevel(logging.WARNING)
    return torch, lab, tag


def run_once(torch, lab, tag, pool, n, offset):
    docs = [pool[(offset + i) % len(pool)] for i in range(n)]
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = tag(docs, lab)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    if len(out) != n:
        raise RuntimeError(f"tag вернул {len(out)} из {n} (OOM или ошибка модели)")
    return dt


def set_env(hf, maxlen):
    os.environ["TAG_HF_BATCH_SIZE"] = str(hf)
    os.environ["TAG_MAX_LENGTH"] = str(maxlen)


def sweep(args):
    torch, lab, tag = load_models()
    pool = get_pool(args)
    base = (args.n, args.hf, args.maxlen)
    configs = [base]
    for n in (16, 32, 48, 64, 96, 128):
        configs.append((n, args.hf, args.maxlen))
    for hf in (16, 32, 64):
        configs.append((64, hf, args.maxlen))
    for ml in (512, 256, 128):
        configs.append((args.n, args.hf, ml))
    seen, uniq = set(), []
    for c in configs:
        if c not in seen:
            seen.add(c)
            uniq.append(c)

    rows = []
    offset = 0
    print(f"skip_deprecated={os.getenv('TAG_SKIP_DEPRECATED', 'true')} calls={args.calls} warmup={args.warmup}")
    print(f"{'n':>4} {'hf_bs':>6} {'maxlen':>7} {'p50,s':>7} {'p95,s':>7} {'docs/s':>8}")
    for n, hf, ml in uniq:
        set_env(hf, ml)
        times = []
        try:
            for i in range(args.warmup + args.calls):
                dt = run_once(torch, lab, tag, pool, n, offset)
                offset += n
                if i >= args.warmup:
                    times.append(dt)
        except Exception as e:
            print(f"{n:>4} {hf:>6} {ml:>7}  ОШИБКА: {e}")
            continue
        p50 = statistics.median(times)
        p95 = sorted(times)[max(0, int(len(times) * 0.95) - 1)]
        rate = n / p50
        rows.append({"n": n, "hf": hf, "maxlen": ml, "p50": round(p50, 3), "p95": round(p95, 3), "docs_per_s": round(rate, 1)})
        print(f"{n:>4} {hf:>6} {ml:>7} {p50:>7.3f} {p95:>7.3f} {rate:>8.1f}")
    if rows:
        best = max(rows, key=lambda r: r["docs_per_s"])
        print(f"\nлучшее: n={best['n']} hf_bs={best['hf']} maxlen={best['maxlen']} -> {best['docs_per_s']} docs/s")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)


def worker(args):
    torch, lab, tag = load_models()
    pool = get_pool(args)
    set_env(args.hf, args.maxlen)
    for i in range(args.warmup):
        run_once(torch, lab, tag, pool, args.n, i * args.n)
    print("READY", flush=True)
    sys.stdin.readline()
    docs = 0
    offset = 0
    t0 = time.perf_counter()
    end = t0 + args.duration
    while time.perf_counter() < end:
        run_once(torch, lab, tag, pool, args.n, offset)
        offset += args.n
        docs += args.n
    elapsed = time.perf_counter() - t0
    print("RESULT " + json.dumps({"docs": docs, "elapsed": elapsed}), flush=True)


def concurrent(args):
    k = args.concurrent
    cmd = [
        sys.executable, os.path.abspath(__file__), "--worker",
        "--n", str(args.n), "--hf", str(args.hf), "--maxlen", str(args.maxlen),
        "--duration", str(args.duration), "--warmup", str(args.warmup),
    ]
    if args.texts:
        cmd += ["--texts", args.texts]
    procs = []
    for i in range(k):
        log = open(f"/tmp/bench_worker_{i}.log", "w")
        procs.append(subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True))
    for i, p in enumerate(procs):
        while True:
            line = p.stdout.readline()
            if not line:
                raise SystemExit(f"воркер {i} завершился до READY, см. /tmp/bench_worker_{i}.log")
            if line.startswith("READY"):
                break
    for p in procs:
        p.stdin.write("GO\n")
        p.stdin.flush()
    results = []
    for i, p in enumerate(procs):
        for line in p.stdout:
            if line.startswith("RESULT "):
                results.append(json.loads(line[7:]))
                break
        else:
            raise SystemExit(f"воркер {i} не вернул результат, см. /tmp/bench_worker_{i}.log")
        p.wait()
    total = sum(r["docs"] for r in results)
    span = max(r["elapsed"] for r in results)
    per = [round(r["docs"] / r["elapsed"], 1) for r in results]
    print(f"реплик={k} n={args.n} hf_bs={args.hf} maxlen={args.maxlen} skip_deprecated={os.getenv('TAG_SKIP_DEPRECATED', 'true')}")
    print(f"по репликам, docs/s: {per}")
    print(f"суммарно: {total / span:.1f} docs/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--hf", type=int, default=64)
    ap.add_argument("--maxlen", type=int, default=512)
    ap.add_argument("--calls", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--concurrent", type=int, default=0)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.worker:
        worker(args)
    elif args.concurrent > 0:
        concurrent(args)
    else:
        sweep(args)


if __name__ == "__main__":
    main()
