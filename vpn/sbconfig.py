"""Сборка конфига sing-box для управляемого режима (подписка + менеджер)."""
import sys

from links import parse_link, link_key, link_name


class Node:
    __slots__ = ("key", "tag", "link", "name", "kind", "outbound", "port")

    def __init__(self, link, outbound, kind):
        self.key = link_key(link)
        self.tag = f"n-{self.key}"
        self.link = link
        self.name = link_name(link)
        self.kind = kind
        self.outbound = outbound
        self.port = 0  # локальный порт «пробного» inbound, назначается в assign_ports()

    def brief(self) -> dict:
        return {"key": self.key, "name": self.name, "kind": self.kind}


def make_nodes(links, log=None):
    """Ссылки -> (список Node без дублей, список текстов пропущенных ссылок)."""
    nodes, skipped, seen = [], [], set()
    for i, link in enumerate(links):
        try:
            ob, kind = parse_link(link)
        except Exception as e:
            skipped.append(f"#{i} {link_name(link)[:40]}: {e}")
            continue
        node = Node(link, ob, kind)
        if node.key in seen:
            continue
        seen.add(node.key)
        nodes.append(node)
    nodes.sort(key=lambda n: n.key)  # порядок в подписке не влияет на порты/теги
    return nodes, skipped


def assign_ports(nodes, base_port: int) -> None:
    for i, n in enumerate(nodes):
        n.port = base_port + i


def build_managed_config(nodes, *, socks_port, http_port, base_port, log_level,
                         clash_port, clash_secret, initial_key=None) -> dict:
    if not nodes:
        raise ValueError("нет узлов для конфига")
    assign_ports(nodes, base_port)
    inbounds = [
        {"type": "socks", "tag": "socks-in", "listen": "0.0.0.0", "listen_port": socks_port},
        {"type": "http", "tag": "http-in", "listen": "0.0.0.0", "listen_port": http_port},
    ]
    outbounds, rules = [], []
    for n in nodes:
        ob = dict(n.outbound)
        ob["tag"] = n.tag
        outbounds.append(ob)
        # «пробный» inbound на каждый узел слушает только localhost: им пользуется менеджер для замеров
        inbounds.append({"type": "socks", "tag": f"in-{n.tag}", "listen": "127.0.0.1", "listen_port": n.port})
        rules.append({"inbound": [f"in-{n.tag}"], "outbound": n.tag})
    by_key = {n.key: n for n in nodes}
    default = (by_key.get(initial_key) or nodes[0]).tag
    # В selector только узлы: прямого выхода нет, значит утечки трафика мимо VPN быть не может
    # interrupt_exist_connections: при смене узла рвём соединения через прежний (иначе загрузка через мёртвый узел
    # висела бы до таймаута клиента); клиент повторит запрос уже через новый узел
    outbounds.append({"type": "selector", "tag": "proxy", "outbounds": [n.tag for n in nodes], "default": default,
                      "interrupt_exist_connections": True})
    return {
        "log": {"level": log_level},
        "dns": {"servers": [{"type": "local", "tag": "local"}]},
        "inbounds": inbounds,
        "outbounds": outbounds,
        "route": {"rules": rules, "final": "proxy", "default_domain_resolver": "local"},
        "experimental": {"clash_api": {"external_controller": f"127.0.0.1:{clash_port}", "secret": clash_secret}},
    }


def warn(msg: str) -> None:
    print(msg, file=sys.stderr)
