#!/usr/bin/env python3
import json
import os
import sys
from urllib.parse import urlparse, parse_qs, unquote

VPN_FILE = os.environ.get("VPN_FILE", "/vpn/vpn.txt")
VLESS_URL = os.environ.get("VLESS_URL", "").strip()
SOCKS_PORT = int(os.environ.get("SOCKS_PORT", "1080"))
HTTP_PORT = int(os.environ.get("HTTP_PORT", "1081"))
BASE_PORT = int(os.environ.get("LINK_BASE_PORT", "10800"))
LOG_LEVEL = os.environ.get("XRAY_LOG_LEVEL", "warning")
OBS_URL = os.environ.get("OBSERVATORY_URL", "https://www.gstatic.com/generate_204")
DISABLE_PQV = os.environ.get("DISABLE_PQV", "false").strip().lower() == "true"


def read_links():
    links = []
    try:
        with open(VPN_FILE, encoding="utf-8") as f:
            for line in f.read().splitlines():
                line = line.strip()
                if line.startswith("vless://"):
                    links.append(line)
    except OSError:
        pass
    if not links and VLESS_URL.startswith("vless://"):
        links = [VLESS_URL]
    return links


def parse_vless(url):
    p = urlparse(url)
    q = {k: v[0] for k, v in parse_qs(p.query).items()}
    uuid, server, port = unquote(p.username or ""), p.hostname, p.port
    if not (uuid and server and port):
        raise ValueError("нет uuid/host/port")

    security = q.get("security", "none")
    network = q.get("type", "tcp")
    host = q.get("host", "")
    path = q.get("path", "/")
    sni = q.get("sni") or host or server
    fp = q.get("fp", "chrome")

    user = {"id": uuid, "encryption": q.get("encryption") or "none"}
    if q.get("flow") and network == "tcp":
        user["flow"] = q["flow"]

    ss = {"network": network, "security": security}
    if network == "ws":
        ss["wsSettings"] = {"path": path, "headers": {"Host": host} if host else {}}
    elif network == "httpupgrade":
        ss["httpupgradeSettings"] = {"path": path, "host": host}
    elif network in ("xhttp", "splithttp"):
        ss["network"] = "xhttp"
        ss["xhttpSettings"] = {"path": path, "host": host, "mode": q.get("mode", "auto")}
    elif network == "grpc":
        ss["grpcSettings"] = {"serviceName": q.get("serviceName", ""), "multiMode": q.get("mode") == "multi"}
    elif network != "tcp":
        raise ValueError(f"type={network!r} не поддерживается")

    if security == "reality":
        pbk = q.get("pbk")
        if not pbk:
            raise ValueError("reality без pbk")
        rs = {
            "show": False,
            "fingerprint": fp,
            "serverName": sni,
            "publicKey": pbk,
            "shortId": q.get("sid", ""),
            "spiderX": unquote(q.get("spx", "/")),
        }
        if q.get("pqv") and not DISABLE_PQV:
            rs["mldsa65Verify"] = q["pqv"]
        ss["realitySettings"] = rs
    elif security == "tls":
        ts = {
            "serverName": sni,
            "fingerprint": fp,
            "allowInsecure": q.get("allowInsecure", q.get("insecure", "0")) in ("1", "true"),
        }
        if q.get("alpn"):
            ts["alpn"] = q["alpn"].split(",")
        ss["tlsSettings"] = ts
    elif security != "none":
        raise ValueError(f"security={security!r} не поддерживается")

    remark = unquote(p.fragment) if p.fragment else server
    out = {
        "protocol": "vless",
        "settings": {"vnext": [{"address": server, "port": port, "users": [user]}]},
        "streamSettings": ss,
    }
    return out, remark, f"{network}/{security}"


links = read_links()
inbounds = [
    {"listen": "0.0.0.0", "port": SOCKS_PORT, "protocol": "socks",
     "settings": {"auth": "noauth", "udp": True}, "tag": "socks-in"},
    {"listen": "0.0.0.0", "port": HTTP_PORT, "protocol": "http", "settings": {}, "tag": "http-in"},
]
outbounds = []
rules = []

for i, link in enumerate(links):
    try:
        ob, remark, kind = parse_vless(link)
    except Exception as e:
        print(f"⚠️  ссылка #{i} пропущена: {e}", file=sys.stderr)
        continue
    ob["tag"] = f"vpn-{i}"
    outbounds.append(ob)
    inbounds.append({"listen": "0.0.0.0", "port": BASE_PORT + i, "protocol": "socks",
                     "settings": {"auth": "noauth", "udp": False}, "tag": f"in-{i}"})
    rules.append({"type": "field", "inboundTag": [f"in-{i}"], "outboundTag": f"vpn-{i}"})
    print(f"✅ vpn-{i} :{BASE_PORT + i} {remark} ({kind})")

if not outbounds:
    print("⚠️  нет рабочих vless-ссылок (vpn.txt / VLESS_URL) — общий прокси идёт напрямую", file=sys.stderr)
    rules.append({"type": "field", "inboundTag": ["socks-in", "http-in"], "outboundTag": "direct"})
    routing_extra = {}
else:
    rules.append({"type": "field", "inboundTag": ["socks-in", "http-in"], "balancerTag": "auto"})
    routing_extra = {"balancers": [{"tag": "auto", "selector": ["vpn-"], "strategy": {"type": "leastPing"}}]}

outbounds += [{"protocol": "freedom", "tag": "direct"}, {"protocol": "blackhole", "tag": "block"}]

config = {
    "log": {"loglevel": LOG_LEVEL},
    "inbounds": inbounds,
    "outbounds": outbounds,
    "routing": {"domainStrategy": "IPIfNonMatch", "rules": rules, **routing_extra},
}
if len(outbounds) > 2:
    config["observatory"] = {"subjectSelector": ["vpn-"], "probeUrl": OBS_URL, "probeInterval": "30s"}

with open("/vpn/config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"✅ Конфиг Xray собран: узлов={len(outbounds) - 2} из {len(links)} ссылок")
