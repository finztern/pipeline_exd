#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-2400}"

log()  { echo -e "\n\033[1;36m▶ $*\033[0m"; }
ok()   { echo -e "\033[32m✅ $*\033[0m"; }
warn() { echo -e "\033[33m⚠️  $*\033[0m"; }
die()  { echo -e "\033[31m❌ $*\033[0m" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] && SUDO="" || SUDO="sudo"
[ -z "$SUDO" ] || sudo -v || die "нужен sudo"

[ -r /etc/os-release ] || die "неизвестная ОС"
. /etc/os-release
case "${ID:-}" in
  debian|ubuntu) ;;
  *) die "поддерживаются Debian/Ubuntu, найдено: ${ID:-?}" ;;
esac
CODENAME="${VERSION_CODENAME:-}"
[ -n "$CODENAME" ] || die "не определён VERSION_CODENAME"

docker_ok() { docker info >/dev/null 2>&1; }
dk() { if docker_ok; then docker "$@"; else $SUDO docker "$@"; fi; }
run_sh() {
  if docker_ok; then ./run.sh "$@"; else sg docker -c "./run.sh $*"; fi
}

set_env() {
  python3 - "$1" "$2" <<'PY'
import sys, re
k, v = sys.argv[1], sys.argv[2]
p = ".env"
lines = open(p, encoding="utf-8").read().splitlines()
out, done = [], False
for l in lines:
    if re.match(rf"^\s*{re.escape(k)}=", l):
        out.append(f"{k}={v}")
        done = True
    else:
        out.append(l)
if not done:
    out.append(f"{k}={v}")
open(p, "w", encoding="utf-8").write("\n".join(out) + "\n")
PY
}

get_env() { grep -E "^$1=" .env | tail -1 | cut -d= -f2- || true; }

log "Базовые пакеты"
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -y
$SUDO apt-get install -y --no-install-recommends ca-certificates curl gnupg git python3 pciutils lsb-release

if ! command -v docker >/dev/null 2>&1; then
  log "Docker + Compose"
  $SUDO install -m 0755 -d /etc/apt/keyrings
  curl -fsSL "https://download.docker.com/linux/${ID}/gpg" | $SUDO gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/${ID} ${CODENAME} stable" \
    | $SUDO tee /etc/apt/sources.list.d/docker.list >/dev/null
  $SUDO apt-get update -y
  $SUDO apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
$SUDO systemctl enable --now docker
docker compose version >/dev/null 2>&1 || $SUDO docker compose version >/dev/null 2>&1 || die "docker compose plugin не установлен"
if [ "$(id -u)" -ne 0 ] && ! id -nG "$USER" | grep -qw docker; then
  $SUDO usermod -aG docker "$USER"
  warn "$USER добавлен в группу docker (в новых сессиях работает без sudo)"
fi
ok "Docker готов"

log "NVIDIA GPU"
lspci | grep -qi nvidia || die "NVIDIA GPU не найдена: bpipe требует GPU"

if ! nvidia-smi >/dev/null 2>&1; then
  log "NVIDIA драйвер"
  if [ "$ID" = "debian" ]; then
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then
      $SUDO sed -i -E '/^Components:/{/non-free-firmware/! s/$/ contrib non-free non-free-firmware/}' /etc/apt/sources.list.d/debian.sources
    elif [ -f /etc/apt/sources.list ]; then
      $SUDO sed -i -E '/^deb /{/non-free/! s/ main/ main contrib non-free non-free-firmware/}' /etc/apt/sources.list
    fi
    $SUDO apt-get update -y
    $SUDO apt-get install -y "linux-headers-$(dpkg --print-architecture)" nvidia-driver firmware-misc-nonfree
  else
    $SUDO apt-get install -y ubuntu-drivers-common
    $SUDO ubuntu-drivers install
  fi
  warn "Драйвер установлен. Перезагрузи систему (sudo reboot) и запусти ./install.sh ещё раз."
  exit 0
fi
ok "$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"

if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
  log "NVIDIA Container Toolkit"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | $SUDO gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  $SUDO apt-get update -y
  $SUDO apt-get install -y nvidia-container-toolkit
fi
$SUDO nvidia-ctk runtime configure --runtime=docker
$SUDO systemctl restart docker

dk run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1 \
  || die "Docker не видит GPU (docker run --gpus all ... nvidia-smi)"
ok "GPU доступна в Docker"

log "Конфигурация .env"
[ -f .env ] || { [ -f .env.example ] || die ".env.example не найден"; cp .env.example .env; }

addr="${MAIN_ADDRESS:-$(get_env MAIN_ADDRESS)}"
if ! [[ "$addr" =~ ^0x[0-9a-fA-F]{40}$ ]]; then
  [ -t 0 ] || die "MAIN_ADDRESS не задан: передай MAIN_ADDRESS=0x... ./install.sh"
  while true; do
    read -r -p "Exorde MAIN_ADDRESS (0x...): " addr
    [[ "$addr" =~ ^0x[0-9a-fA-F]{40}$ ]] && break
    warn "формат: 0x + 40 hex-символов"
  done
fi
set_env MAIN_ADDRESS "$addr"

vless="${VLESS_URL:-$(get_env VLESS_URL)}"
if [ -z "$vless" ] && [ -t 0 ] && [ ! -s vpn.txt ]; then
  read -r -p "VLESS_URL (Enter — пропустить, выгрузка напрямую/через proxy.txt): " vless || true
fi
[ -z "$vless" ] || set_env VLESS_URL "$vless"

mkdir -p models_cache argos_data ct2_models
touch vpn.txt proxy.txt
ok ".env готов"

log "Сборка и запуск (первый раз 10–20 минут: образы + модели HuggingFace)"
run_sh up -d --build

log "Ожидание healthy"
services=(vpn transactioneer bpipe upipe collector)
n="$(get_env BPIPE_REPLICAS)"; n="${n:-1}"
[ "$n" -ge 2 ] 2>/dev/null && services+=(bpipe_2)
[ "$n" -ge 3 ] 2>/dev/null && services+=(bpipe_3)

t0=$(date +%s)
while true; do
  pending=()
  for s in "${services[@]}"; do
    st="$(dk inspect -f '{{.State.Health.Status}}' "$s" 2>/dev/null || echo missing)"
    [ "$st" = "healthy" ] || pending+=("$s:$st")
  done
  [ ${#pending[@]} -eq 0 ] && break
  el=$(( $(date +%s) - t0 ))
  [ "$el" -lt "$HEALTH_TIMEOUT" ] || { dk ps; die "не стали healthy за ${HEALTH_TIMEOUT}с: ${pending[*]} (логи: docker compose logs <сервис>)"; }
  printf "\r  %4ds  ждём: %s        " "$el" "${pending[*]}"
  sleep 10
done
echo
ok "Все сервисы healthy"

dk ps --format 'table {{.Names}}\t{{.Status}}'
echo
echo "Проверка:  curl -s -X POST localhost:9000/store_item -H 'Content-Type: application/json' -d '{\"content\":\"This is a normal test tweet about sunny weather in California today.\",\"external_id\":\"inst-1\",\"created_at\":\"'\$(date -u +%Y-%m-%dT%H:%M:%S.000Z)'\",\"domain\":\"twitter.com\",\"url\":\"https://twitter.com/t/1\"}'"
echo "Дашборд:   ./dashboard.sh"
echo "Автотюнинг (при живом потоке): python3 autotune.py --dry"
