import asyncio
import json
import logging
import os
import sys
import time
from collections import Counter
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
BEAM        = int(os.getenv("GPU_BEAM_SIZE", "2"))
MAX_TOKENS  = int(os.getenv("GPU_MAX_TOKENS", "512"))
COMPUTE     = os.getenv("GPU_COMPUTE_TYPE", "int8")
DEVICE      = os.getenv("GPU_DEVICE", "cuda")
MODELS_DIR  = os.getenv("CT2_MODELS_DIR", "/models")
NLLB_NAME   = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
TARGET      = "eng_Latn"
OPUS_MODELS = dict(
    p.strip().split(":", 1)
    for p in os.getenv(
        "OPUS_MODELS",
        "de:Helsinki-NLP/opus-mt-de-en,fr:Helsinki-NLP/opus-mt-fr-en,"
        "es:Helsinki-NLP/opus-mt-es-en,it:Helsinki-NLP/opus-mt-it-en,"
        "pt:Helsinki-NLP/opus-mt-ROMANCE-en",
    ).split(",")
    if ":" in p
)

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
_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ct2")
_session: aiohttp.ClientSession | None = None
_ready = False
_init_error = ""
_started = time.monotonic()
_opus_by_lang: dict = {}
_nllb = None
_lang_counts: Counter = Counter()
_stats = {
    "received": 0, "translated": 0, "returned": 0, "identical": 0,
    "unsupported": 0, "rejected_busy": 0, "errors": 0, "send_errors": 0,
}


class Backend:
    def __init__(self, name: str, nllb: bool):
        self.nllb = nllb
        path = os.path.join(MODELS_DIR, name.replace("/", "__"))
        if not os.path.isfile(os.path.join(path, "model.bin")):
            log.info(f"Конвертация {name} -> CT2 int8")
            ctranslate2.converters.TransformersConverter(name).convert(path, quantization="int8", force=True)
        self.tr = ctranslate2.Translator(path, device=DEVICE, compute_type=COMPUTE)
        self.tok = AutoTokenizer.from_pretrained(name)

    def _encode(self, text: str) -> list:
        ids = self.tok.encode(text, truncation=True, max_length=MAX_TOKENS)
        return self.tok.convert_ids_to_tokens(ids)

    def translate(self, texts: list, srcs: list) -> list:
        tokens = []
        for text, src in zip(texts, srcs):
            if self.nllb:
                self.tok.src_lang = src
            tokens.append(self._encode(text))
        kwargs = dict(beam_size=BEAM, max_batch_size=BATCH_SIZE, max_decoding_length=MAX_TOKENS)
        if self.nllb:
            kwargs["target_prefix"] = [[TARGET]] * len(tokens)
        results = self.tr.translate_batch(tokens, **kwargs)
        out = []
        for r in results:
            hyp = r.hypotheses[0][1:] if self.nllb else r.hypotheses[0]
            out.append(self.tok.decode(self.tok.convert_tokens_to_ids(hyp), skip_special_tokens=True))
        return out


def _load():
    global _nllb
    cache = {}
    for lang, name in OPUS_MODELS.items():
        if name not in cache:
            cache[name] = Backend(name, False)
        _opus_by_lang[lang] = cache[name]
    _nllb = Backend(NLLB_NAME, True)


def _route(lang: str):
    lang = lang.lower()
    base = lang.split("-")[0]
    if base in _opus_by_lang and lang not in ("zh-cn", "zh-tw"):
        return _opus_by_lang[base], ""
    code = FLORES.get(lang) or FLORES.get(base)
    return (_nllb, code) if code else None


def _run_group(backend: Backend, texts: list, srcs: list) -> list:
    return backend.translate(texts, srcs)


async def _send_back(item: dict, text: str):
    payload = {k: v for k, v in item.items() if k != "foreign_author"}
    payload["pretranslated"] = text
    try:
        async with _session.post(
            UPIPE_URL,
            data=json.dumps(payload, ensure_ascii=False).encode(),
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            await resp.read()
            if 200 <= resp.status < 300:
                _stats["returned"] += 1
                return
            _stats["send_errors"] += 1
            log.warning(f"upipe вернул {resp.status} на {UPIPE_URL}")
    except Exception as e:
        _stats["send_errors"] += 1
        log.warning(f"send back error -> {UPIPE_URL}: {type(e).__name__}: {e}")


async def _worker():
    loop = asyncio.get_running_loop()
    while True:
        try:
            jobs = [await _queue.get()]
            while len(jobs) < BATCH_SIZE * 2:
                try:
                    jobs.append(_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            groups: dict = {}
            for job in jobs:
                groups.setdefault(id(job[1]), (job[1], []))[1].append(job)

            for backend, items in groups.values():
                items.sort(key=lambda j: len(j[0]["content"]))
                texts = [j[0]["content"] for j in items]
                srcs = [j[2] for j in items]
                try:
                    outs = await loop.run_in_executor(_pool, _run_group, backend, texts, srcs)
                except Exception as e:
                    _stats["errors"] += len(items)
                    log.error(f"Ошибка перевода батча: {e}", exc_info=True)
                    continue

                sends = []
                for (item, _, _), out in zip(items, outs):
                    out = out.strip()
                    if not out or out == item["content"].strip():
                        _stats["identical"] += 1
                        continue
                    _stats["translated"] += 1
                    _lang_counts[str(item.get("detected_lang", "")).lower()] += 1
                    sends.append(_send_back(item, out))
                await asyncio.gather(*sends)
                log.info(f"batch={len(items)} переведено={len(sends)} | {_stats}")
        except Exception as e:
            log.error(f"Воркер: {e}", exc_info=True)
            await asyncio.sleep(1)


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

    route = _route(lang)
    if route is None:
        _stats["unsupported"] += 1
        return web.json_response({"error": "unsupported_lang"}, status=422)
    if _queue.full():
        _stats["rejected_busy"] += 1
        return web.json_response({"error": "queue_full"}, status=503)

    _stats["received"] += 1
    _queue.put_nowait((item, route[0], route[1]))
    return web.Response(status=202, text="accepted")


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok" if _ready else "starting",
        "ready": _ready,
        "queue": _queue.qsize() if _queue else 0,
        "max": QUEUE_SIZE,
        "opus": sorted(_opus_by_lang),
        "init_error": _init_error or None,
    }, status=200 if _ready else 503)


async def handle_stats(request: web.Request) -> web.Response:
    return web.json_response({
        "uptime_s": int(time.monotonic() - _started),
        "queue": _queue.qsize() if _queue else 0,
        "stats": _stats,
        "languages": dict(_lang_counts),
    })


async def _init():
    global _ready, _init_error, _session, _queue
    loop = asyncio.get_running_loop()
    _queue = asyncio.Queue(maxsize=QUEUE_SIZE)
    _session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=20, keepalive_timeout=60))
    try:
        await loop.run_in_executor(None, _load)
    except Exception as e:
        _init_error = str(e)
        log.error(f"Инициализация упала: {e}", exc_info=True)
        return
    asyncio.create_task(_worker())
    _ready = True
    log.info(f"Готов | opus={sorted(_opus_by_lang)} nllb={NLLB_NAME} device={DEVICE}/{COMPUTE} -> {UPIPE_URL}")


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
