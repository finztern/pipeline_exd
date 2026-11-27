#!/bin/bash
set -e

if [ -n "${SUBSCRIPTION_URL:-}" ]; then
    echo "🔒 VPN контейнер: управляемый режим (подписка, автовыбор узлов по API, запас, failover)"
    exec python3 /vpn/manager.py
fi

echo "🔒 VPN контейнер (sing-box: VLESS/REALITY + Hysteria2), статический режим по vpn.txt"
python3 /vpn/build_config.py

echo "🚀 Запускаем sing-box..."
exec sing-box run -c /vpn/config.json
