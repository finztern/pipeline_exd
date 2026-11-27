"""Разбор ссылок подписки (vless / hysteria2 / socks5) в outbound'ы sing-box."""
import hashlib
import os
from urllib.parse import urlparse, parse_qs, unquote

SCHEMES = ("vless://", "hysteria2://", "hy2://", "socks5://")

# Некоторые провайдеры (gRPC+TLS с параметром pcs=) закрепляют сертификат по SHA-256 всей цепочки.
# sing-box такой пин не поддерживает (только sha256 публичного ключа), поэтому такие узлы без
# отключения проверки цепочки просто не поднимутся. VPN_PCS_INSECURE=false оставит строгую проверку.
PCS_INSECURE = os.environ.get("VPN_PCS_INSECURE", "true").lower() in ("1", "true", "yes")


def truthy(v) -> bool:
    return str(v).lower() in ("1", "true", "yes")


def parse_vless(url: str):
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

    kind = f"{network}/{security}"
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
        if q.get("pcs") and PCS_INSECURE and not tls["insecure"]:
            tls["insecure"] = True
            kind += "+pcs"
        if q.get("alpn"):
            tls["alpn"] = q["alpn"].split(",")
        ob["tls"] = tls
    elif security != "none":
        raise ValueError(f"security={security!r} не поддерживается")

    return ob, kind


def parse_hy2(url: str):
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


def parse_socks5(url: str):
    """Простой SOCKS5-узел без шифрования (для внутренних/тестовых нужд)."""
    p = urlparse(url)
    if not (p.hostname and p.port):
        raise ValueError("нет host/port")
    ob = {"type": "socks", "server": p.hostname, "server_port": p.port, "version": "5"}
    if p.username:
        ob["username"] = unquote(p.username)
        ob["password"] = unquote(p.password or "")
    return ob, "socks5"


def parse_link(link: str):
    """Возвращает (outbound без tag, краткое описание транспорта)."""
    if link.startswith("vless://"):
        return parse_vless(link)
    if link.startswith("socks5://"):
        return parse_socks5(link)
    return parse_hy2(link)


def link_key(link: str) -> str:
    """Стабильный ключ узла: не зависит от названия (фрагмента #...) и порядка в подписке."""
    return hashlib.sha1(link.split("#", 1)[0].encode("utf-8")).hexdigest()[:12]


def link_name(link: str) -> str:
    p = urlparse(link)
    return unquote(p.fragment).strip() or p.hostname or "node"
