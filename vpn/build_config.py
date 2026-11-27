#!/usr/bin/env python3
"""Статический режим (без подписки): vpn.txt -> конфиг sing-box. Разбор ссылок общий с managed-режимом (links.py)."""
import json
import os
import sys
from urllib.parse import urlparse, unquote

from links import SCHEMES, parse_link

VPN_FILE = os.environ.get("VPN_FILE", "/vpn/vpn.txt")
VLESS_URL = os.environ.get("VLESS_URL", "").strip()
SOCKS_PORT = int(os.environ.get("SOCKS_PORT", "1080"))
HTTP_PORT = int(os.environ.get("HTTP_PORT", "1081"))
BASE_PORT = int(os.environ.get("LINK_BASE_PORT", "10800"))
LOG_LEVEL = os.environ.get("XRAY_LOG_LEVEL", "warning")
OBS_URL = os.environ.get("OBSERVATORY_URL", "https://www.gstatic.com/generate_204")
URLTEST_INTERVAL = os.environ.get("URLTEST_INTERVAL", "10m")


def read_links():
    links = []
    try:
        with open(VPN_FILE, encoding="utf-8") as f:
            for line in f.read().splitlines():
                line = line.strip()
                if line.startswith(SCHEMES):
                    links.append(line)
    except OSError:
        pass
    if not links and VLESS_URL.startswith(SCHEMES):
        links = [VLESS_URL]
    return links


links = read_links()
inbounds = [
    {"type": "socks", "tag": "socks-in", "listen": "0.0.0.0", "listen_port": SOCKS_PORT},
    {"type": "http", "tag": "http-in", "listen": "0.0.0.0", "listen_port": HTTP_PORT},
]
outbounds = []
rules = []
tags = []

for i, link in enumerate(links):
    try:
        ob, kind = parse_link(link)
    except Exception as e:
        print(f"⚠️  ссылка #{i} пропущена: {e}", file=sys.stderr)
        continue
    tag = f"vpn-{i}"
    ob["tag"] = tag
    outbounds.append(ob)
    tags.append(tag)
    inbounds.append({"type": "socks", "tag": f"in-{i}", "listen": "0.0.0.0", "listen_port": BASE_PORT + i})
    rules.append({"inbound": [f"in-{i}"], "outbound": tag})
    remark = unquote(urlparse(link).fragment) or urlparse(link).hostname
    print(f"✅ {tag} :{BASE_PORT + i} {remark} ({kind})")

if tags:
    outbounds.append({
        "type": "urltest", "tag": "auto", "outbounds": tags,
        "url": OBS_URL, "interval": URLTEST_INTERVAL, "tolerance": 50,
    })
    final = "auto"
else:
    print("⚠️  нет рабочих ссылок (vpn.txt / VLESS_URL) — общий прокси идёт напрямую", file=sys.stderr)
    final = "direct"

outbounds.append({"type": "direct", "tag": "direct"})

config = {
    "log": {"level": LOG_LEVEL},
    "dns": {"servers": [{"type": "local", "tag": "local"}]},
    "inbounds": inbounds,
    "outbounds": outbounds,
    "route": {"rules": rules, "final": final, "default_domain_resolver": "local"},
}

with open("/vpn/config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"✅ Конфиг sing-box собран: узлов={len(tags)} из {len(links)} ссылок")
