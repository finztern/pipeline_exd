#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
command -v python3 >/dev/null || { echo "нужен python3: sudo apt install -y python3" >&2; exit 1; }
exec python3 dashboard.py "$@"
