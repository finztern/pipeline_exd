#!/bin/sh
# Управляемый режим: менеджер сообщает, что sing-box жив, активный узел выбран и API через него отвечает.
if [ -n "${SUBSCRIPTION_URL:-}" ]; then
    exec curl -sf http://127.0.0.1:${STATUS_PORT:-9091}/health -o /dev/null
fi
exec curl -sf -x socks5h://127.0.0.1:1080 https://www.gstatic.com/generate_204 -o /dev/null
