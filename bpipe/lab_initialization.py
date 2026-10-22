import os
import gc
import json
import logging
import torch
from transformers import AutoTokenizer, pipeline

import onnxruntime
try:
    onnxruntime.preload_dlls()
    logging.info("[ORT] preload_dlls() выполнен")
except Exception as e:
    logging.warning(f"[ORT] preload_dlls() недоступен/не сработал ({e}) — полагаемся на LD_LIBRARY_PATH из Dockerfile")

try:
    logging.info(f"[ORT] Version: {onnxruntime.__version__}")
    logging.info(f"[ORT] Available providers ДО загрузки моделей: {onnxruntime.get_available_providers()}")
except Exception as e:
    logging.warning(f"[ORT] диагностика упала: {e}")

from transformers import AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from finvader import finvader
from huggingface_hub import hf_hub_download


CLASSIFICATION_BACKEND = os.getenv("CLASSIFICATION_BACKEND", "minilm_cosine").strip().lower()
if CLASSIFICATION_BACKEND not in ("zs", "minilm_cosine"):
    raise ValueError(f"CLASSIFICATION_BACKEND должен быть zs|minilm_cosine, получено: {CLASSIFICATION_BACKEND!r}")

SKIP_DEPRECATED = os.getenv("TAG_SKIP_DEPRECATED", "true").lower() == "true"

CLASSIFICATION_LABELS = [
    "Business", "People", "Investing", "Health", "Law", "Economy", "Technology", "Sports",
    "Cryptocurrency", "Environment", "Politics", "Finance", "Science", "Entertainment", "Social",
]


def clear_gpu_memory():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        logging.debug("GPU memory cleared")
    except Exception as e:
        logging.warning(f"Error clearing GPU memory: {e}")


def _load_text_pipeline(model_id, dtype, device):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if not tokenizer.model_max_length or tokenizer.model_max_length > 100_000:
        tokenizer.model_max_length = 512
    model = AutoModelForSequenceClassification.from_pretrained(model_id, torch_dtype=dtype)
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        top_k=None,
        max_length=512,
        padding=True,
        truncation=True,
        device=device,
    )


def initialize_models(device):
    logging.info("[TAGGING] Initializing models...")

    USE_PROXY = os.getenv("USE_PROXY", "false").lower() == "true"
    PROXY_URL = os.getenv("PROXY_URL", "")
    if USE_PROXY and PROXY_URL:
        os.environ["HTTP_PROXY"] = PROXY_URL
        os.environ["HTTPS_PROXY"] = PROXY_URL
        logging.info(f"[TAGGING] Proxy: {PROXY_URL}")

    ort_provider = "CUDAExecutionProvider" if device >= 0 else "CPUExecutionProvider"
    logging.info(f"[TAGGING] ORT provider (для sentence-transformer): {ort_provider}")

    models = {}

    logging.info("[TAGGING] Loading: sentence-transformers/all-MiniLM-L6-v2 (ONNX)")
    models["sentence_transformer"] = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        backend="onnx",
        model_kwargs={"provider": ort_provider},
    )
    clear_gpu_memory()

    torch_device = device
    _dtype = torch.float16 if device >= 0 else None

    if CLASSIFICATION_BACKEND == "zs":
        logging.info("[TAGGING] Loading: MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33")
        _zs_id = "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33"
        _zs_tokenizer = AutoTokenizer.from_pretrained(_zs_id)
        if not _zs_tokenizer.model_max_length or _zs_tokenizer.model_max_length > 100_000:
            _zs_tokenizer.model_max_length = 512
        else:
            _zs_tokenizer.model_max_length = min(_zs_tokenizer.model_max_length, 512)
        _zs_model = AutoModelForSequenceClassification.from_pretrained(_zs_id, torch_dtype=_dtype)
        models["zs_pipe"] = pipeline(
            "zero-shot-classification",
            model=_zs_model,
            tokenizer=_zs_tokenizer,
            device=torch_device,
        )
        clear_gpu_memory()
    else:
        logging.info("[TAGGING] CLASSIFICATION_BACKEND=minilm_cosine: deberta zs_pipe не грузится")

    text_classification_models = [
        ("Emotion",  "SamLowe/roberta-base-go_emotions"),
        ("Irony",    "cardiffnlp/twitter-roberta-base-irony"),
        ("TextType", "marieke93/MiniLM-evidence-types"),
    ]

    if SKIP_DEPRECATED:
        text_classification_models = [m for m in text_classification_models if m[0] == "Emotion"]
        logging.info("[TAGGING] TAG_SKIP_DEPRECATED=true: Irony и TextType не грузятся")

    SNAPSHOT_REVISIONS = {
        "cardiffnlp/twitter-roberta-base-irony": "refs/pr/3",
        "marieke93/MiniLM-evidence-types":       "refs/pr/1",
    }

    for col_name, model_name in text_classification_models:
        logging.info(f"[TAGGING] Loading: {model_name}")

        source_id = model_name
        if model_name in SNAPSHOT_REVISIONS:
            from huggingface_hub import snapshot_download
            source_id = snapshot_download(
                repo_id=model_name,
                revision=SNAPSHOT_REVISIONS[model_name],
            )

        models[col_name] = _load_text_pipeline(source_id, _dtype, torch_device)
        clear_gpu_memory()

    logging.info("[TAGGING] Loading: vaderSentiment")
    models["sentiment_analyzer"] = SentimentIntensityAnalyzer()
    logging.info("[TAGGING] Loading: finvader")
    models["finvader_analyzer"] = finvader

    try:
        emoji_lexicon = hf_hub_download(
            repo_id="ExordeLabs/SentimentDetection",
            filename="emoji_unic_lexicon.json",
        )
        loughran_dict = hf_hub_download(
            repo_id="ExordeLabs/SentimentDetection",
            filename="loughran_dict.json",
        )
        with open(emoji_lexicon) as f:
            unic_emoji_dict = json.load(f)
        with open(loughran_dict) as f:
            Loughran_dict = json.load(f)
        models["sentiment_analyzer"].lexicon.update(Loughran_dict)
        models["sentiment_analyzer"].lexicon.update(unic_emoji_dict)
        logging.info("[TAGGING] Loughran + emoji lexicons loaded.")
    except Exception as e:
        logging.warning(f"[TAGGING] Lexicon load failed (продолжаем без них): {e}")

    logging.info("[TAGGING] Loading: mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")
    models["fdb_pipe"] = _load_text_pipeline(
        "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", _dtype, torch_device
    )
    clear_gpu_memory()

    logging.info("[TAGGING] Loading: lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    models["gdb_pipe"] = _load_text_pipeline(
        "lxyuan/distilbert-base-multilingual-cased-sentiments-student", _dtype, torch_device
    )
    clear_gpu_memory()

    logging.info("[TAGGING] Все модели загружены.")

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved  = torch.cuda.memory_reserved()  / 1024**3
        logging.info(
            f"[TAGGING] GPU после загрузки — allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB"
        )

    return models


def lab_initialization():
    device = 0 if torch.cuda.is_available() else -1
    logging.info(f"[LAB INIT] Device: {'GPU' if device != -1 else 'CPU'}")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
        logging.info(
            f"[LAB INIT] {torch.cuda.get_device_name(0)} — "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        )

    mappings = {
        "Gender": {0: "Female", 1: "Male"},
        "Age":    {0: "<20", 1: "20<30", 2: "30<40", 3: ">=40"},
    }

    models = initialize_models(device)

    labels = {name: "" for name in CLASSIFICATION_LABELS}
    logging.info(f"[LAB INIT] Classification labels: {len(labels)}, backend={CLASSIFICATION_BACKEND}")

    lab_configuration = {
        "labeldict":           labels,
        "device":              device,
        "mappings":            mappings,
        "max_depth":           2,
        "remove_stopwords":    False,
        "installed_languages": [],
        "models":              models,
        "classification_backend": CLASSIFICATION_BACKEND,
    }

    logging.info("[LAB INIT] Done.")
    return lab_configuration
