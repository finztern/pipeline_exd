import json
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()
ARRIVALS = []
STATS = {"batches_received": 0, "items_received": 0, "bad_items": 0, "bad_batches": 0}
BAD_SAMPLES = []
ISO_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$")
STARTED = time.time()


def find_key(o, key, depth=0):
    if depth > 3:
        return None
    if isinstance(o, dict):
        if key in o:
            return o[key]
        for v in o.values():
            r = find_key(v, key, depth + 1)
            if r is not None:
                return r
    return None


def check(it):
    inner = it.get("item") if isinstance(it.get("item"), dict) else it
    ca, col = inner.get("created_at"), it.get("collected_at")
    if not isinstance(ca, str) or not ISO_Z.match(ca):
        return f"item.created_at={ca!r}"
    if not isinstance(col, str) or not ISO_Z.match(col):
        return f"collected_at={col!r}"
    if ca[:19] > col[:19]:
        return f"created_at {ca} позже collected_at {col}"
    if not inner.get("content"):
        return "пустой content"
    an = it.get("analysis")
    if not isinstance(an, dict) or "top_keywords" not in an or "classification" not in an:
        return "нет analysis.top_keywords/classification"
    return ""


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, body):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        t = time.time()
        if self.path.rstrip("/") != "/commit":
            return self._send(404, {"error": "not found"})
        try:
            items = json.loads(raw)
            assert isinstance(items, list)
        except Exception:
            with LOCK:
                STATS["bad_batches"] += 1
            return self._send(400, {"error": "expected array"})
        recs, bad = [], 0
        for it in items:
            eid = find_key(it, "external_id") if isinstance(it, dict) else None
            reason = check(it) if isinstance(it, dict) else "не dict"
            if reason:
                bad += 1
                if len(BAD_SAMPLES) < 5:
                    BAD_SAMPLES.append(reason)
            recs.append((eid, t))
        with LOCK:
            STATS["batches_received"] += 1
            STATS["items_received"] += len(items)
            STATS["bad_items"] += bad
            ARRIVALS.extend(recs)
        self._send(200, b"received")

    def do_GET(self):
        p = self.path.split("?")[0].rstrip("/")
        if p == "/bench/drain":
            with LOCK:
                out = list(ARRIVALS)
                ARRIVALS.clear()
            return self._send(200, {"arrivals": out, "stats": STATS, "bad_samples": BAD_SAMPLES})
        base = {"status": "ok", "uptime_s": int(time.time() - STARTED), "stats": STATS, "active_tasks": 0,
                "totals": {}, "rate": {}, "domains": {}, "route": "mock", "main_address": "MOCK"}
        self._send(200, base)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("MOCK_PORT", "8002"))), H).serve_forever()
