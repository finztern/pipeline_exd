#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

CMD="${1:-apply}"

envget() { { grep -E "^$1=" .env 2>/dev/null || true; } | tail -1 | cut -d= -f2- | tr -d '[:space:]'; }
envset() {
  grep -v -E "^$1=" .env > .env.tmp || true
  echo "$1=$2" >> .env.tmp
  mv .env.tmp .env
}
PORT="$(envget GPU_TRANSLATOR_HOST_PORT)"; PORT="${PORT:-8004}"
BASE="http://127.0.0.1:$PORT"

wait_ready() {
  echo "жду загрузку моделей (до 10 мин)..."
  for _ in $(seq 1 120); do
    out="$(curl -s -m 5 "$BASE/health" || true)"
    if [ -n "$out" ] && python3 -c "import sys,json;d=json.loads(sys.argv[1]);sys.exit(0 if d.get('ready') else 1)" "$out" 2>/dev/null; then
      python3 -c "import sys,json;d=json.loads(sys.argv[1]);print('ready: языков',len(d['langs']),'| моделей',d['models'],'| init_error',d['init_error']);print(d['langs'])" "$out"
      return 0
    fi
    sleep 5
  done
  echo "translator_gpu не стал ready за 10 мин: ./run.sh logs --tail 80 translator_gpu" >&2
  return 1
}

stats() {
  curl -s -m 5 "$BASE/stats" | python3 -c "
import sys,json
d=json.load(sys.stdin);s=d['stats']
print('compute',d['compute'],'| uptime',d['uptime_s'],'| ms/batch',round(d['gpu_busy_s']/max(1,s['batches'])*1000),'| avg_batch',d['avg_batch'],'| busy%',d['gpu_busy_pct'],'| rate',d['rate_per_sec'],'| queue',d['queue'],'| busy_reject',s['rejected_busy'],'| unsupported',s['unsupported'],'| err',s['errors'],'| identical',s['identical'])"
  docker stats --no-stream translator_gpu bpipe bpipe_2 upipe collector 2>/dev/null || docker stats --no-stream translator_gpu bpipe upipe collector
}

case "$CMD" in
  apply)
    [ -f .env ] || cp .env.example .env
    cp .env ".env.bak.$(date +%Y%m%d_%H%M%S)"
    git pull --ff-only
    envset GPU_COMPUTE_TYPE int8
    envset OPUS_LANGS none
    envset OPUS_FALLBACK Helsinki-NLP/opus-mt-mul-en
    envset NLLB_DISABLE true
    envset GPU_BATCH_SIZE 16
    envset GPU_MIN_BATCH 8
    envset GPU_MAX_WAIT_SECONDS 0.5
    envset GPU_INFLIGHT 3
    ./run.sh up -d --build --no-deps translator_gpu
    wait_ready
    echo "готово. через ~10 мин: ./tune_translator.sh stats"
    ;;
  stats)
    stats
    ;;
  sys)
    python3 - <<'EOF'
import json, re, time, urllib.request, subprocess
def get():
    return json.load(urllib.request.urlopen("http://127.0.0.1:9000/stats", timeout=10))
def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, f"{p}.{k}" if p else k)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield p, o
cnt = re.compile(r"(collector\.(received|forwarded|to_gpu|dropped|errors|filtered_lang|filtered_dup|low_conf_routed)|exorde\.totals\.(accepted|duplicates|rejected))$")
gau = re.compile(r"(bpipe\.(fill|inflight|queue)|upipe\.queue|translator\.(queue|fill|rate_per_sec))$")
a = dict(flat(get())); t0 = time.time()
print("замер 60 с...", flush=True)
time.sleep(60)
b = dict(flat(get())); dt = time.time() - t0
print("--- счётчики, /с ---")
for k in sorted(b):
    if cnt.search(k) and k in a:
        print(f"{k:45s} {(b[k]-a[k])/dt:8.2f}")
print("--- текущие значения ---")
for k in sorted(b):
    if gau.search(k):
        print(f"{k:45s} {b[k]}")
EOF
    uptime
    ;;
  bpipe)
    python3 - <<'EOF'
import json, re, subprocess, time
REP = [("bpipe", 7995), ("bpipe_2", 7996), ("bpipe_3", 7997)]
def get(c, p):
    try:
        out = subprocess.run(["docker", "exec", c, "curl", "-s", "-m", "5", f"http://localhost:{p}/health"],
                             capture_output=True, text=True, timeout=15).stdout
        return json.loads(out)
    except Exception:
        return None
def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from flat(v, f"{p}.{k}" if p else k)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        yield p, o
cnt = re.compile(r"stats\.(received|received_translated|received_default|items_sent|batches_processed|dropped|dropped_default)$")
gau = re.compile(r"^(queue|inflight|queues\.(default|translated)\.(queue|fill))$")
a = {c: get(c, p) for c, p in REP}; t0 = time.time()
print("замер 30 с...", flush=True)
time.sleep(30)
dt = time.time() - t0
for c, p in REP:
    x, y = a[c], get(c, p)
    if not x or not y:
        continue
    fx, fy = dict(flat(x)), dict(flat(y))
    print(f"--- {c} ---")
    for k in sorted(fy):
        if cnt.search(k) and k in fx:
            print(f"  {k:36s} {(fy[k]-fx[k])/dt:8.2f} /с")
    for k in sorted(fy):
        if gau.search(k):
            print(f"  {k:36s} {fy[k]}")
EOF
    docker stats --no-stream bpipe bpipe_2 translator_gpu 2>/dev/null
    ;;
  set)
    shift
    [ "$#" -ge 1 ] || { echo "использование: $0 set KEY=VAL [KEY=VAL...]" >&2; exit 1; }
    cp .env ".env.bak.$(date +%Y%m%d_%H%M%S)"
    for kv in "$@"; do
      case "$kv" in *=*) envset "${kv%%=*}" "${kv#*=}" ;; *) echo "нужен формат KEY=VAL: $kv" >&2; exit 1 ;; esac
    done
    ./run.sh up -d --no-deps --force-recreate translator_gpu
    wait_ready
    echo "готово. через ~10 мин: ./tune_translator.sh stats"
    ;;
  rollback)
    B="$(ls -1t .env.bak.* 2>/dev/null | tail -1 || true)"
    [ -n "$B" ] || { echo "нет .env.bak.*" >&2; exit 1; }
    cp "$B" .env
    echo "восстановлен $B (самый старый бэкап, до тюнинга)"
    ./run.sh up -d --build --no-deps translator_gpu
    wait_ready
    ;;
  *)
    echo "использование: $0 [apply|stats|sys|bpipe|set KEY=VAL...|rollback]" >&2
    exit 1
    ;;
esac
