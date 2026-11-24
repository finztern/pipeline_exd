#!/bin/bash
set -e

echo "🔒 VPN контейнер (sing-box: VLESS/REALITY + Hysteria2)"
python3 /vpn/build_config.py

echo "🚀 Запускаем sing-box..."
exec sing-box run -c /vpn/config.json
