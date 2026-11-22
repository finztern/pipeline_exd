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
print('compute',d['compute'],'| uptime',d['uptime_s'],'| ms/batch',round(d['gpu_busy_s']/max(1,s['batches'])*1000),'| avg_batch',d['avg_batch'],'| busy%',d['gpu_busy_pct'],'| rate',d['rate_per_sec'],'| busy_reject',s['rejected_busy'],'| unsupported',s['unsupported'],'| err',s['errors'],'| identical',s['identical'])"
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
  rollback)
    B="$(ls -1t .env.bak.* 2>/dev/null | tail -1 || true)"
    [ -n "$B" ] || { echo "нет .env.bak.*" >&2; exit 1; }
    cp "$B" .env
    echo "восстановлен $B (самый старый бэкап, до тюнинга)"
    ./run.sh up -d --build --no-deps translator_gpu
    wait_ready
    ;;
  *)
    echo "использование: $0 [apply|stats|rollback]" >&2
    exit 1
    ;;
esac
