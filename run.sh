#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

for f in vpn.txt proxy.txt; do
  [ -d "$f" ] && { echo "$f — каталог (docker создал при отсутствии файла); удали: rm -r $f" >&2; exit 1; }
  [ -e "$f" ] || : > "$f"
done

N=1
if [ -f .env ]; then
  V=$(grep -E '^BPIPE_REPLICAS=' .env | tail -1 | cut -d= -f2- | tr -d '[:space:]')
  [ -n "${V:-}" ] && N="$V"
fi

case "$N" in
  ''|*[!0-9]*) echo "BPIPE_REPLICAS в .env должен быть числом (1-3), получено: '$N'" >&2; exit 1 ;;
esac
if [ "$N" -lt 1 ] || [ "$N" -gt 3 ]; then
  echo "BPIPE_REPLICAS поддерживается только 1-3, получено: $N" >&2
  exit 1
fi

urls="http://bpipe:7995/"
profiles=""
if [ "$N" -ge 2 ]; then
  urls="$urls,http://bpipe_2:7996/"
  profiles="bpipe2"
fi
if [ "$N" -ge 3 ]; then
  urls="$urls,http://bpipe_3:7997/"
  profiles="${profiles:+$profiles,}bpipe3"
fi

export BPIPE_URLS="$urls"
export COMPOSE_PROFILES="$profiles"

echo "[run.sh] BPIPE_REPLICAS=$N -> COMPOSE_PROFILES=${COMPOSE_PROFILES:-<none>} BPIPE_URLS=$BPIPE_URLS"
exec docker compose "$@"
