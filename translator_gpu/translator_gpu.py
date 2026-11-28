import asyncio
import json
import logging
import os
import faulthandler
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor

import aiohttp
import ctranslate2
from aiohttp import web
from transformers import AutoTokenizer
from transformers.models.m2m_100 import modeling_m2m_100 as _m2m

# ctranslate2 4.4.0 читает encoder/decoder.embed_scale, а в новых transformers он лежит в embed_tokens
if hasattr(_m2m, "M2M100ScaledWordEmbedding"):
    for _cls in (_m2m.M2M100Encoder, _m2m.M2M100Decoder):
        if not hasattr(_cls, "embed_scale"):
            _cls.embed_scale = property(lambda self: self.embed_tokens.embed_scale)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [translator_gpu] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PORT        = int(os.getenv("GPU_TRANSLATOR_PORT", "8004"))
UPIPE_URL   = os.getenv("UPIPE_URL", "").strip()
QUEUE_SIZE  = int(os.getenv("GPU_QUEUE_SIZE", "2000"))
BATCH_SIZE  = max(1, int(os.getenv("GPU_BATCH_SIZE", "16")))
MAX_WAIT    = max(0.1, float(os.getenv("GPU_MAX_WAIT_SECONDS", "5")))
MIN_BATCH   = max(1, min(BATCH_SIZE, int(os.getenv("GPU_MIN_BATCH", "1"))))
INFLIGHT    = max(1, int(os.getenv("GPU_INFLIGHT", "2")))
# Максимум текстов, которые воркер держит в buckets (вне asyncio.Queue).
# Без лимита drain() перекладывал бы всю очередь в buckets, очередь не заполнялась,
# 503 не отдавался и хвост рос в RAM без ограничений.
BUCKET_CAP  = max(BATCH_SIZE * 2, int(os.getenv("GPU_BUCKET_CAP", "400")))
BEAM        = max(1, int(os.getenv("GPU_BEAM_SIZE", "1")))
MAX_TOKENS  = int(os.getenv("GPU_MAX_TOKENS", "150"))
DECODE_RATIO = float(os.getenv("GPU_DECODE_RATIO", "2.0"))
DECODE_PAD  = int(os.getenv("GPU_DECODE_PAD", "16"))
SEND_RETRIES = max(1, int(os.getenv("GPU_SEND_RETRIES", "4")))
NLLB_ONLY   = {x.strip().lower() for x in os.getenv("NLLB_ONLY", "").split(",") if x.strip()}
COMPUTE     = os.getenv("GPU_COMPUTE_TYPE", "int8")
DEVICE      = os.getenv("GPU_DEVICE", "cuda")
BLOCKING_SYNC = os.getenv("GPU_BLOCKING_SYNC", "1").strip().lower() in ("1", "true", "yes")
MODELS_DIR  = os.getenv("CT2_MODELS_DIR", "/models")
NLLB_NAME   = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
NLLB_OFF    = os.getenv("NLLB_DISABLE", "").strip().lower() in ("1", "true", "yes")
TARGET      = "eng_Latn"
OPUS_LANGS  = (
    "af,ar,az,bg,bn,ca,cs,cy,da,de,el,es,et,eu,fi,fr,gl,hi,hu,hy,id,is,it,ja,ka,ko,"
    "lv,mk,ml,mr,mt,nl,pl,sk,sq,sv,th,tl,tr,uk,ur,vi,zh"
)
ROMANCE     = "Helsinki-NLP/opus-mt-ROMANCE-en"


GROUPS = {
    "lt": "bat", "hr": "sla", "sl": "sla", "sr": "sla", "bs": "sla", "no": "gmq", "nb": "gmq", "nn": "gmq",
    "gu": "inc", "pa": "inc", "ne": "inc", "kn": "dra", "ta": "dra", "te": "dra", "he": "afa",
    "so": "cus", "sw": "bnt", "el": "grk",
}
DEDICATED = set()
OPUS_FALLBACK = os.getenv("OPUS_FALLBACK", "Helsinki-NLP/opus-mt-mul-en").strip()
SAMPLE_PATH = os.getenv("GPU_SAMPLE_PATH", "").strip()
SAMPLE_MAX  = int(os.getenv("GPU_SAMPLE_MAX", "5000"))


def _opus_map() -> dict:
    raw = os.getenv("OPUS_MODELS", "").strip()
    if raw:
        return dict(p.strip().split(":", 1) for p in raw.split(",") if ":" in p)
    explicit = os.getenv("OPUS_LANGS", "").strip()
    if explicit.lower() in ("none", "off", "-"):
        return {}
    langs =[l.strip() for l in (explicit or OPUS_LANGS).split(",") if l.strip()]
    if not explicit:
        langs += [l for l in [*GROUPS, "pt", "ro"] if l not in langs]
    m = {}
    for l in langs:
        if l in GROUPS and l not in DEDICATED:
            m[l] = f"Helsinki-NLP/opus-mt-{GROUPS[l]}-en"
        elif l in ("pt", "ro"):
            m[l] = ROMANCE
        else:
            m[l] = f"Helsinki-NLP/opus-mt-{l}-en"
    return m


OPUS_MODELS = _opus_map()


def _cuda_blocking_sync():
    if DEVICE != "cuda" or not BLOCKING_SYNC:
        return
    import ctypes
    for lib in ("libcudart.so.12", "libcudart.so"):
        try:
            rc = ctypes.CDLL(lib).cudaSetDeviceFlags(0x04)
            log.info(f"cudaSetDeviceFlags(BlockingSync) {lib} rc={rc}")
            return
        except OSError:
            continue
    log.warning("libcudart не найден, BlockingSync не включён")


_cuda_blocking_sync()

FLORES = {
    "af": "afr_Latn", "ar": "arb_Arab", "bg": "bul_Cyrl", "bn": "ben_Beng", "ca": "cat_Latn",
    "cs": "ces_Latn", "cy": "cym_Latn", "da": "dan_Latn", "de": "deu_Latn", "el": "ell_Grek",
    "es": "spa_Latn", "et": "est_Latn", "fa": "pes_Arab", "fi": "fin_Latn", "fr": "fra_Latn",
    "gu": "guj_Gujr", "he": "heb_Hebr", "hi": "hin_Deva", "hr": "hrv_Latn", "hu": "hun_Latn",
    "id": "ind_Latn", "it": "ita_Latn", "ja": "jpn_Jpan", "kn": "kan_Knda", "ko": "kor_Hang",
    "lt": "lit_Latn", "lv": "lvs_Latn", "mk": "mkd_Cyrl", "ml": "mal_Mlym", "mr": "mar_Deva",
    "ne": "npi_Deva", "nl": "nld_Latn", "no": "nob_Latn", "nb": "nob_Latn", "pa": "pan_Guru",
    "pl": "pol_Latn", "pt": "por_Latn", "ro": "ron_Latn", "ru": "rus_Cyrl", "sk": "slk_Latn",
    "sl": "slv_Latn", "so": "som_Latn", "sq": "als_Latn", "sv": "swe_Latn", "sw": "swh_Latn",
    "ta": "tam_Taml", "te": "tel_Telu", "th": "tha_Thai", "tl": "tgl_Latn", "tr": "tur_Latn",
    "uk": "ukr_Cyrl", "ur": "urd_Arab", "vi": "vie_Latn", "zh-cn": "zho_Hans", "zh": "zho_Hans",
    "zh-tw": "zho_Hant", "zt": "zho_Hant",
}

_queue: asyncio.Queue | None = None
_pool = ThreadPoolExecutor(max_workers=INFLIGHT, thread_name_prefix="ct2")
_bg: set = set()
_gpu_busy = 0.0
_session: aiohttp.ClientSession | None = None
_ready = False
_init_error = ""
_started = time.monotonic()
_backends: dict = {}
_sample_n = 0
_lang_counts: Counter = Counter()
_done_ts: deque = deque()
RATE_WINDOWS = {"10m": 600, "30m": 1800, "60m": 3600}
_bucket_items = 0  # сколько текстов сейчас лежит в buckets воркера
_stats = {
    "received": 0, "translated": 0, "returned": 0, "identical": 0,
    "unsupported": 0, "rejected_busy": 0, "errors": 0, "send_errors": 0,
    "batches": 0, "batched_items": 0, "too_long": 0, "mul_routed": 0, "mul_identical_passed": 0,
    "send_retries": 0,
}


CONVERT_TIMEOUT = int(os.getenv("CONVERT_TIMEOUT_SECONDS", "600"))
LOAD_TIMEOUT = int(os.getenv("LOAD_TIMEOUT_SECONDS", str(2 * CONVERT_TIMEOUT + 300)))


def _convert(name: str, path: str):
    exe = shutil.which("ct2-transformers-converter")
    cmd = ([exe] if exe else [sys.executable, "-m", "ctranslate2.converters.transformers"]) + [
        "--model", name, "--output_dir", path, "--quantization", "int8", "--force",
    ]
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(name, "config.json")
    except Exception as e:
        if type(e).__name__ in ("RepositoryNotFoundError", "GatedRepoError", "RevisionNotFoundError", "EntryNotFoundError"):
            raise RuntimeError(f"модель не найдена на HuggingFace: {name}")
    for attempt in (1, 2):
        t0 = time.time()
        try:
            r = subprocess.run(cmd, timeout=CONVERT_TIMEOUT)
            if r.returncode == 0 and os.path.isfile(os.path.join(path, "model.bin")):
                log.info(f"Конвертация {name} готова за {time.time() - t0:.0f}с")
                return
            log.warning(f"Конвертация {name}: код {r.returncode} (попытка {attempt}/2)")
        except subprocess.TimeoutExpired:
            log.warning(f"Конвертация {name}: таймаут {CONVERT_TIMEOUT}с (попытка {attempt}/2)")
    raise RuntimeError("конвертация не удалась")


class Backend:
    def __init__(self, name: str, nllb: bool, compute: str = "", inter: int = 0):
        self.nllb = nllb
        path = os.path.join(MODELS_DIR, name.replace("/", "__"))
        if not os.path.isfile(os.path.join(path, "model.bin")):
            log.info(f"Конвертация {name} -> CT2 int8")
            _convert(name, path)
        self.tr = ctranslate2.Translator(
            path, device=DEVICE, compute_type=compute or COMPUTE, inter_threads=inter or INFLIGHT,
        )
        log.info(f"{name}: CT2 модель загружена, токенизатор...")
        try:
            self.tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
        except Exception:
            self.tok = AutoTokenizer.from_pretrained(name)
        log.info(f"{name}: токенизатор готов")

    def _encode(self, text: str, src: str) -> list:
        if self.nllb:
            ids = self.tok.encode(text, add_special_tokens=False, truncation=True, max_length=MAX_TOKENS)
            toks = [src] + self.tok.convert_ids_to_tokens(ids) + [self.tok.eos_token]
        else:
            ids = self.tok.encode(text, truncation=True, max_length=MAX_TOKENS + 1)
            toks = self.tok.convert_ids_to_tokens(ids)
        return toks if len(toks) <= MAX_TOKENS else None

    @staticmethod
    def _chunks(order: list, toks: list):
        cur = []
        for i in order:
            if len(cur) >= BATCH_SIZE:
                yield cur
                cur = []
            cur.append(i)
        if cur:
            yield cur

    def translate(self, texts: list, srcs: list) -> list:
        toks = [self._encode(t, s) for t, s in zip(texts, srcs)]
        order = sorted((i for i in range(len(toks)) if toks[i] is not None), key=lambda i: len(toks[i]))
        out = [None if t is None else "" for t in toks]
        for chunk in self._chunks(order, toks):
            batch = [toks[i] for i in chunk]
            limit = min(MAX_TOKENS, int(len(batch[-1]) * DECODE_RATIO) + DECODE_PAD)
            kwargs = dict(
                beam_size=BEAM, max_batch_size=len(batch), max_decoding_length=limit, return_scores=False,
            )
            if self.nllb:
                kwargs["target_prefix"] = [[TARGET]] * len(batch)
            results = self.tr.translate_batch(batch, **kwargs)
            hyps = [r.hypotheses[0][1:] if self.nllb else r.hypotheses[0] for r in results]
            decoded = self.tok.batch_decode(
                [self.tok.convert_tokens_to_ids(h) for h in hyps], skip_special_tokens=True,
            )
            for i, t in zip(chunk, decoded):
                out[i] = t
        return out


def _candidates(lang: str) -> list:
    lang = lang.lower()
    if lang == "mul":
        return [(OPUS_FALLBACK, False, "")] if OPUS_FALLBACK else []
    base = lang.split("-")[0]
    code = FLORES.get(lang) or FLORES.get(base)
    out = []
    if lang not in NLLB_ONLY and base not in NLLB_ONLY:
        if base in OPUS_MODELS:
            out.append((OPUS_MODELS[base], False, ""))
        if OPUS_FALLBACK and (code or base in OPUS_MODELS):
            out.append((OPUS_FALLBACK, False, ""))
    if code and not NLLB_OFF:
        out.append((NLLB_NAME, True, code))
    return out


def _all_langs() -> set:
    return set(OPUS_MODELS) if NLLB_OFF and not OPUS_FALLBACK else set(FLORES) | set(OPUS_MODELS)


def _load():
    names = set(OPUS_MODELS.values())
    if OPUS_FALLBACK:
        names.add(OPUS_FALLBACK)
    total = len(names)
    for i, name in enumerate(sorted(names), 1):
        box = {}

        def run(n=name, b=box):
            try:
                b["be"] = Backend(n, False)
            except Exception as e:
                b["err"] = e

        th = threading.Thread(target=run, daemon=True)
        t0 = time.time()
        th.start()
        ready = os.path.isfile(os.path.join(MODELS_DIR, name.replace("/", "__"), "model.bin"))
        th.join(120 if ready else LOAD_TIMEOUT)
        if th.is_alive():
            log.error(f"[{i}/{total}] {name}: загрузка зависла, пропускаю; стеки потоков:")
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        elif "err" in box:
            e = box["err"]
            log.warning(f"[{i}/{total}] Opus {name} не загрузился ({type(e).__name__}: {e})")
        else:
            _backends[name] = box["be"]
            log.info(f"[{i}/{total}] {name} загружен за {time.time() - t0:.0f}с")
    need_nllb = False
    for lang in _all_langs():
        for name, nllb, _ in _candidates(lang):
            if _backends.get(name):
                break
            if nllb:
                need_nllb = True
    if need_nllb:
        _backends[NLLB_NAME] = Backend(NLLB_NAME, True)
        log.info("NLLB загружен как запасной/принудительный")


def _route(lang: str):
    for name, _, code in _candidates(lang):
        be = _backends.get(name)
        if be:
            return be, code
    return None


async def _send_back(item: dict, text: str):
    payload = {k: v for k, v in item.items() if k not in ("foreign_author", "translate_route")}
    payload["pretranslated"] = text
    data = json.dumps(payload, ensure_ascii=False).encode()
    for attempt in range(SEND_RETRIES):
        try:
            async with _session.post(
                UPIPE_URL,
                data=data,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                await resp.read()
                if 200 <= resp.status < 300:
                    _stats["returned"] += 1
                    return
                if resp.status != 503:
                    log.warning(f"upipe вернул {resp.status} на {UPIPE_URL}")
                    break
        except Exception as e:
            log.warning(f"send back error -> {UPIPE_URL}: {type(e).__name__}: {e}")
        if attempt + 1 < SEND_RETRIES:
            _stats["send_retries"] += 1
            await asyncio.sleep(min(0.3 * 2 ** attempt, 3.0))
    _stats["send_errors"] += 1


def _record_done(n: int):
    now = time.monotonic()
    _done_ts.append((now, n))
    while _done_ts and _done_ts[0][0] < now - 3600:
        _done_ts.popleft()


def _rates() -> dict:
    now = time.monotonic()
    up = max(1.0, now - _started)
    out = {}
    for name, w in RATE_WINDOWS.items():
        total = sum(n for t, n in _done_ts if t >= now - w)
        out[name] = round(total / min(w, up), 3)
    return out


async def _process_batch(loop, backend: Backend, items: list):
    global _gpu_busy
    texts = [j[0]["content"] for j in items]
    srcs = [j[2] for j in items]
    t0 = time.monotonic()
    try:
        outs = await loop.run_in_executor(_pool, backend.translate, texts, srcs)
    except Exception as e:
        _stats["errors"] += len(items)
        log.error(f"Ошибка перевода батча: {e}", exc_info=True)
        return
    dt = time.monotonic() - t0
    _gpu_busy += dt
    _stats["batches"] += 1
    _stats["batched_items"] += len(items)

    sent = 0
    for (item, _, _), out in zip(items, outs):
        if out is None:
            _stats["too_long"] += 1
            continue
        out = out.strip()
        same = out == item["content"].strip()
        mul = item.get("translate_route") == "mul"
        if not out or (same and not mul):
            _stats["identical"] += 1
            continue
        if same:
            _stats["mul_identical_passed"] += 1
        _stats["translated"] += 1
        _lang_counts[str(item.get("detected_lang", "")).lower()] += 1
        task = asyncio.create_task(_send_back(item, out))
        _bg.add(task)
        task.add_done_callback(_bg.discard)
        sent += 1
    _record_done(sent)
    if dt > 0.5 or _stats["batches"] % 200 == 0:
        log.info(f"batch={len(items)} ok={sent} {dt:.2f}s ({len(items) / max(dt, 1e-3):.1f}/s) q={_queue.qsize()} backlog={_queue.qsize() + _bucket_items} batches={_stats['batches']}")


async def _worker():
    global _bucket_items
    loop = asyncio.get_running_loop()
    slots = asyncio.Semaphore(INFLIGHT)
    buckets: dict = {}

    def add(job):
        b = buckets.get(id(job[1]))
        if b is None:
            b = buckets[id(job[1])] = {"backend": job[1], "items": [], "t0": loop.time()}
        b["items"].append(job)

    def pending() -> int:
        return sum(len(b["items"]) for b in buckets.values())

    def drain():
        global _bucket_items
        n = pending()
        while n < BUCKET_CAP:
            try:
                add(_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
            n += 1
        _bucket_items = n

    while True:
        held = False
        try:
            if not buckets:
                add(await _queue.get())
            await slots.acquire()
            held = True
            drain()
            while True:
                oldest = min(buckets.values(), key=lambda b: b["t0"])
                age = loop.time() - oldest["t0"]
                biggest = max(buckets.values(), key=lambda b: len(b["items"]))
                if age >= MAX_WAIT:
                    pick = oldest
                    break
                if len(biggest["items"]) >= MIN_BATCH:
                    pick = biggest
                    break
                await asyncio.sleep(min(0.02, MAX_WAIT - age))
                drain()
            del buckets[id(pick["backend"])]
            items = pick["items"]
            if len(items) > BATCH_SIZE:
                buckets[id(pick["backend"])] = {
                    "backend": pick["backend"], "items": items[BATCH_SIZE:], "t0": pick["t0"],
                }
                items = items[:BATCH_SIZE]
            _bucket_items = pending()
            task = loop.create_task(_process_batch(loop, pick["backend"], items))
            task.add_done_callback(lambda _t: slots.release())
            held = False
        except Exception as e:
            if held:
                slots.release()
            log.error(f"Воркер: {e}", exc_info=True)
            await asyncio.sleep(1)


def _save_sample(lang: str, text: str):
    global _sample_n
    if not SAMPLE_PATH or _sample_n >= SAMPLE_MAX:
        return
    try:
        with open(SAMPLE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"detected_lang": lang, "content": text}, ensure_ascii=False) + "\n")
        _sample_n += 1
    except OSError:
        pass


async def handle_translate(request: web.Request) -> web.Response:
    if not _ready:
        return web.json_response({"error": "not_ready"}, status=503)
    try:
        item = await request.json()
        lang = str(item["detected_lang"])
        if not isinstance(item.get("content"), str) or not item["content"].strip():
            raise ValueError
    except Exception:
        return web.json_response({"error": "bad_request"}, status=400)

    route = _route(str(item.get("translate_route") or lang))
    if route is None:
        _stats["unsupported"] += 1
        return web.json_response({"error": "unsupported_lang"}, status=422)
    if _queue.full():
        _stats["rejected_busy"] += 1
        return web.json_response({"error": "queue_full"}, status=503)

    _stats["received"] += 1
    if item.get("translate_route") == "mul":
        _stats["mul_routed"] += 1
    _save_sample(lang, item["content"])
    _queue.put_nowait((item, route[0], route[1]))
    return web.Response(status=202, text="accepted")


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok" if _ready else "starting",
        "ready": _ready,
        "queue": _queue.qsize() if _queue else 0,
        "backlog": (_queue.qsize() if _queue else 0) + _bucket_items,
        "max": QUEUE_SIZE,
        "opus": sorted(l for l, n in OPUS_MODELS.items() if _backends.get(n)),
        "langs": sorted(l for l in _all_langs() if _route(l)) if _ready else [],
        "models": len([b for b in _backends.values() if b]),
        "init_error": _init_error or None,
    }, status=200 if _ready else 503)


async def handle_stats(request: web.Request) -> web.Response:
    return web.json_response({
        "languages_count": len(_lang_counts),
        "uptime_s": int(time.monotonic() - _started),
        "queue": _queue.qsize() if _queue else 0,
        "bucket_items": _bucket_items,
        "bucket_cap": BUCKET_CAP,
        "backlog": (_queue.qsize() if _queue else 0) + _bucket_items,
        "batch_size": BATCH_SIZE,
        "max_tokens": MAX_TOKENS,
        "inflight": INFLIGHT,
        "beam": BEAM,
        "compute": COMPUTE,
        "avg_batch": round(_stats["batched_items"] / max(1, _stats["batches"]), 1),
        "gpu_busy_s": round(_gpu_busy, 2),
        "gpu_busy_pct": round(100 * _gpu_busy / max(1.0, time.monotonic() - _started), 1),
        "rate_per_sec": _rates(),
        "stats": _stats,
        "languages": dict(_lang_counts.most_common()),
    })


def _log_vram():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()[0]
        used, total = [int(x) for x in out.split(",")]
        msg = f"VRAM: {used}/{total} МБ ({100 * used // total}%), моделей: {len([b for b in _backends.values() if b])}"
        (log.warning if used > 0.9 * total else log.info)(msg)
    except Exception:
        pass


async def _init():
    global _ready, _init_error, _session, _queue, _sample_n
    loop = asyncio.get_running_loop()
    _queue = asyncio.Queue(maxsize=QUEUE_SIZE)
    if SAMPLE_PATH and os.path.isfile(SAMPLE_PATH):
        with open(SAMPLE_PATH, encoding="utf-8") as f:
            _sample_n = sum(1 for _ in f)
    _session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=20, keepalive_timeout=60))
    try:
        await loop.run_in_executor(None, _load)
        await loop.run_in_executor(None, _log_vram)
    except Exception as e:
        _init_error = str(e)
        log.error(f"Инициализация упала: {e}", exc_info=True)
        return
    asyncio.create_task(_worker())
    _ready = True
    log.info(
        f"Готов | models={len([b for b in _backends.values() if b])} nllb={'да' if _backends.get(NLLB_NAME) else 'нет'} device={DEVICE}/{COMPUTE} "
        f"beam={BEAM} batch={BATCH_SIZE} max_tokens={MAX_TOKENS} inflight={INFLIGHT} -> {UPIPE_URL}"
    )


async def on_startup(app: web.Application):
    if not UPIPE_URL:
        log.error("UPIPE_URL не задан — результаты некуда отправлять")
    app["init"] = asyncio.create_task(_init())


async def on_shutdown(app: web.Application):
    if _session:
        await _session.close()


app = web.Application(client_max_size=10 * 1024 * 1024)
app.router.add_post("/", handle_translate)
app.router.add_get("/health", handle_health)
app.router.add_get("/stats", handle_stats)
app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=PORT, print=None)
