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
SCHEMES = ("vless://", "hysteria2://", "hy2://")


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


def truthy(v):
    return str(v).lower() in ("1", "true", "yes")


def parse_vless(url):
    p = urlparse(url)
    q = {k: v[0] for k, v in parse_qs(p.query).items()}
    uuid, server, port = unquote(p.username or ""), p.hostname, p.port
    if not (uuid and server and port):
        raise ValueError("нет uuid/host/port")

    security = q.get("security", "none")
    network = q.get("type", "tcp")
    host = q.get("host", "")
    path = unquote(q.get("path", "/"))
    sni = q.get("sni") or host or server

    ob = {"type": "vless", "server": server, "server_port": port, "uuid": uuid}
    if q.get("flow") and network == "tcp":
        ob["flow"] = q["flow"]

    if network == "ws":
        ob["transport"] = {"type": "ws", "path": path, **({"headers": {"Host": host}} if host else {})}
    elif network == "httpupgrade":
        ob["transport"] = {"type": "httpupgrade", "path": path, **({"host": host} if host else {})}
    elif network == "grpc":
        ob["transport"] = {"type": "grpc", "service_name": q.get("serviceName", "")}
    elif network != "tcp":
        raise ValueError(f"type={network!r} не поддерживается sing-box")

    if security == "reality":
        if not q.get("pbk"):
            raise ValueError("reality без pbk")
        ob["tls"] = {
            "enabled": True,
            "server_name": sni,
            "utls": {"enabled": True, "fingerprint": q.get("fp", "chrome")},
            "reality": {"enabled": True, "public_key": q["pbk"], "short_id": q.get("sid", "")},
        }
    elif security == "tls":
        tls = {
            "enabled": True,
            "server_name": sni,
            "insecure": truthy(q.get("allowInsecure", q.get("insecure", "0"))),
            "utls": {"enabled": True, "fingerprint": q.get("fp", "chrome")},
        }
        if q.get("alpn"):
            tls["alpn"] = q["alpn"].split(",")
        ob["tls"] = tls
    elif security != "none":
        raise ValueError(f"security={security!r} не поддерживается")

    return ob, f"{network}/{security}"


def parse_hy2(url):
    p = urlparse(url)
    q = {k: v[0] for k, v in parse_qs(p.query).items()}
    server, port = p.hostname, p.port
    user, pwd = unquote(p.username or ""), unquote(p.password or "")
    password = f"{user}:{pwd}" if pwd else user
    if not (password and server and port):
        raise ValueError("нет password/host/port")

    tls = {
        "enabled": True,
        "server_name": q.get("sni") or server,
        "insecure": truthy(q.get("insecure", "0")),
        "alpn": ["h3"],
    }
    ob = {"type": "hysteria2", "server": server, "server_port": port, "password": password, "tls": tls}
    if q.get("obfs"):
        if q["obfs"] != "salamander":
            raise ValueError(f"obfs={q['obfs']!r} не поддерживается")
        ob["obfs"] = {"type": "salamander", "password": q.get("obfs-password", "")}
    return ob, "udp/hysteria2"


def parse_link(link):
    if link.startswith("vless://"):
        return parse_vless(link)
    return parse_hy2(link)


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
        "url": OBS_URL, "interval": "30s", "tolerance": 50,
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
