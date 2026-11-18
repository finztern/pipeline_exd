#!/usr/bin/env python3
import json
import subprocess
import sys
import time
import urllib.request

COLLECTOR = "http://localhost:9000"
BPIPES = [("bpipe", 7995), ("bpipe_2", 7996), ("bpipe_3", 7997)]
SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read())
    except Exception:
        return {}


def bpipe_stats():
    tot = {}
    for n, p in BPIPES:
        try:
            o = subprocess.run(["docker", "exec", n, "curl", "-s", "-m", "5", f"http://localhost:{p}/health"],
                               capture_output=True, text=True, timeout=10).stdout
            for k, v in json.loads(o).get("stats", {}).items():
                tot[k] = tot.get(k, 0) + v
        except Exception:
            pass
    return tot


def snap():
    c = get(COLLECTOR + "/stats")
    return {
        "c": c.get("collector", {}),
        "u": c.get("upipe", {}).get("stats", {}),
        "tx": c.get("exorde", {}).get("totals", {}),
        "dom": c.get("domains", {}).get("totals", {}),
        "b": bpipe_stats(),
    }


def d(a, b, k):
    return b.get(k, 0) - a.get(k, 0)


def pct(x, base):
    return f"{x / base * 100:5.1f}%" if base else "    —"


a = snap()
t0 = time.time()
time.sleep(SECONDS)
b = snap()
dt = time.time() - t0

ca, cb, ua, ub, ba, bb = a["c"], b["c"], a["u"], b["u"], a["b"], b["b"]
rec = d(ca, cb, "received")
fwd = d(ca, cb, "forwarded")
gpu = d(ca, cb, "to_gpu")

print(f"окно {dt:.0f}с | вход collector: {rec} ({rec / dt:.1f}/с)\n")
print("── collector: где отсеялось ──")
rows = [
    ("filtered_dup (external_id)", d(ca, cb, "filtered_dup")),
    ("filtered_url_local (url уже слали)", d(ca, cb, "filtered_url_local")),
    ("filtered_url_dup (url-check API)", d(ca, cb, "filtered_url_dup")),
    ("filtered_old", d(ca, cb, "filtered_old")),
    ("filtered_time (created_at битый/будущее)", d(ca, cb, "filtered_time")),
    ("filtered_lang (не en)", d(ca, cb, "filtered_lang")),
    ("dropped (upipe 503/недоступен)", d(ca, cb, "dropped")),
]
known = sum(v for _, v in rows)
rest = rec - fwd - gpu - known
rows.append(("skipped_short (<MIN_TEXT_LEN) / прочее", rest))
for name, v in rows:
    print(f"  {name:<44}{v:>7} {pct(v, rec)}")
print(f"  {'→ прошло в upipe':<44}{fwd:>7} {pct(fwd, rec)}")
print(f"  {'→ ушло на GPU-переводчик':<44}{gpu:>7} {pct(gpu, rec)}")
print(f"  rescued_en (спасено эвристикой): {d(ca, cb, 'rescued_en')}")

print("\n── upipe ──")
ur = d(ua, ub, "received")
print(f"  принято {ur} | обработано {d(ua, ub, 'processed')} | в bpipe {d(ua, ub, 'forwarded')}")
print(f"  filtered_lang (NonEnglishError) {d(ua, ub, 'filtered_lang')} | errors (токен-лимит/пусто/bpipe 503) {d(ua, ub, 'errors')} | dropped (очередь) {d(ua, ub, 'dropped')}")

print("\n── bpipe ──")
print(f"  принято {d(ba, bb, 'received')} | отправлено в tx {d(ba, bb, 'items_sent')} | "
      f"dropped {d(ba, bb, 'dropped')} (translated {d(ba, bb, 'dropped_translated')}, default {d(ba, bb, 'dropped_default')}, buffer {d(ba, bb, 'dropped_buffer')}) | errors {d(ba, bb, 'errors')}")

print("\n── Exorde ──")
print(f"  принято {d(a['tx'], b['tx'], 'accepted')} | отклонено {d(a['tx'], b['tx'], 'rejected')} | дубликаты {d(a['tx'], b['tx'], 'duplicates')}")

print("\n── по доменам (collector, за окно): вход → прошло ──")
da, db = a["dom"], b["dom"]
for dom in sorted(db.get("r", {}), key=lambda x: -(db["r"][x] - da.get("r", {}).get(x, 0)))[:15]:
    r = db["r"][dom] - da.get("r", {}).get(dom, 0)
    f = db.get("f", {}).get(dom, 0) - da.get("f", {}).get(dom, 0)
    if r:
        print(f"  {dom:<28}{r:>6} → {f:<6} {pct(f, r)}")
