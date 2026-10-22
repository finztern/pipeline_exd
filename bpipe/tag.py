import logging
import os
import time
import cupy as cp
import numpy as np
import torch
import gc
from concurrent.futures import ThreadPoolExecutor
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from finvader import finvader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from exorde_compat import (
    Classification, LanguageScore, Sentiment, Embedding, TextType,
    Emotion, Irony, Age, Gender, Analysis,
)

logging.basicConfig(level=logging.INFO)

_cpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tag_cpu")

SENTIMENT_WEIGHTS = (
    (0.6, (0.30, 0.10, 0.60)),
    (0.4, (0.40, 0.20, 0.40)),
    (0.1, (0.60, 0.25, 0.15)),
    (0.0, (0.60, 0.40, 0.00)),
)


def clear_gpu_memory():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()
        logging.debug("GPU memory cleared")
    except Exception as e:
        logging.warning(f"Error clearing GPU memory: {e}")

def soft_clear_gpu_cache():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        logging.warning(f"Error in soft_clear_gpu_cache: {e}")

def get_gpu_memory_info():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return allocated, reserved
    return 0, 0

_label_cache = {"key": None, "emb": None}
_cls_batches = 0
CLASSIFICATION_LOG_EVERY = max(1, int(os.getenv("CLASSIFICATION_LOG_EVERY", "20")))


def _normalize(m):
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


def _label_embeddings(model, labels):
    key = tuple(labels)
    if _label_cache["key"] != key:
        emb = model.encode(labels, convert_to_numpy=True, device='cuda', show_progress_bar=False)
        _label_cache["emb"] = _normalize(np.asarray(emb, dtype=np.float32))
        _label_cache["key"] = key
        logging.info(f"[CLS] label embeddings пересчитаны: labels={len(labels)}")
    return _label_cache["emb"]


def classify_cosine(model, text_embeddings, labels):
    label_emb = _label_embeddings(model, labels)
    sims = _normalize(np.asarray(text_embeddings, dtype=np.float32)) @ label_emb.T
    best = sims.argmax(axis=1)
    scores = np.clip((sims[np.arange(len(best)), best] + 1.0) / 2.0, 0.0, 1.0)
    return [{"labels": [labels[int(b)]], "scores": [float(sc)]} for b, sc in zip(best, scores)]


def _pos_minus_neg(prediction):
    d = {str(e["label"]).lower(): float(e["score"]) for e in prediction}
    return round(d.get("positive", 0.0) - d.get("negative", 0.0), 3)


def ensemble_sentiment(gdb, vader, fin):
    a = abs(fin)
    for threshold, (wg, wv, wf) in SENTIMENT_WEIGHTS:
        if a >= threshold:
            return round(max(-1.0, min(1.0, wg * gdb + wv * vader + wf * fin)), 2)
    return 0.0


def tag(documents: list[str], lab_configuration):
    assert documents is not None and len(documents) > 0
    logging.info(f"Starting Tagging Batch pipeline for {len(documents)} documents...")

    t_start = time.perf_counter()
    t_lap = t_start
    timings = {}

    def lap(name):
        nonlocal t_lap
        now = time.perf_counter()
        timings[name] = now - t_lap
        t_lap = now

    allocated_start, reserved_start = get_gpu_memory_info()
    logging.info(f"GPU Memory at start - Allocated: {allocated_start:.2f}GB, Reserved: {reserved_start:.2f}GB")

    n_docs = len(documents)
    order = sorted(range(n_docs), key=lambda i: len(documents[i]))
    documents = [documents[i] for i in order]

    models = lab_configuration["models"]

    model = models['sentence_transformer']
    backend = lab_configuration.get("classification_backend", "zs")
    zs_pipe = models['zs_pipe'] if backend == "zs" else None
    classification_labels = list(lab_configuration["labeldict"].keys())
    sentiment_analyzer = models['sentiment_analyzer']
    fdb_pipe = models['fdb_pipe']
    gdb_pipe = models['gdb_pipe']

    text_classification_models = {
        "Emotion": models['Emotion'],
        "Irony": models['Irony'],
        "TextType": models['TextType']
    }

    batch_size = min(len(documents), int(os.getenv("TAG_HF_BATCH_SIZE", "64")))
    logging.info(f"Using batch_size: {batch_size} for {len(documents)} documents, labels={len(classification_labels)}")

    zs_batch_size = int(os.getenv("TAG_ZS_BATCH_SIZE", "120"))

    HF_SAFETY_KWARGS = {"truncation": True, "max_length": 512}

    def _cpu_sentiment():
        v = [sentiment_analyzer.polarity_scores(t)["compound"] for t in documents]
        f = [finvader(t, use_sentibignomics=True, use_henry=True, indicator='compound') for t in documents]
        return v, f

    try:
        cpu_future = _cpu_pool.submit(_cpu_sentiment)
        try:
            if getattr(model, "max_seq_length", None) and model.max_seq_length > 512:
                model.max_seq_length = 512
        except Exception:
            pass
        embedding_vectors = model.encode(
            documents,
            convert_to_tensor=True,
            device='cuda',
            batch_size=batch_size,
            show_progress_bar=False
        )
        embedding_vectors = embedding_vectors.cpu().numpy()
        lap("emb")

        if backend == "minilm_cosine":
            classification_results = classify_cosine(model, embedding_vectors, classification_labels)
        else:
            classification_results = zs_pipe(documents, candidate_labels=classification_labels, batch_size=zs_batch_size)
        lap("cls")
        classification_ms = timings.pop("cls") * 1000

        text_type_results = text_classification_models['TextType'](documents, batch_size=batch_size, **HF_SAFETY_KWARGS)
        lap("ttype")

        emotion_results = text_classification_models['Emotion'](documents, batch_size=batch_size, **HF_SAFETY_KWARGS)
        lap("emo")

        irony_results = text_classification_models['Irony'](documents, batch_size=batch_size, **HF_SAFETY_KWARGS)
        lap("irony")

        fdb_predictions = fdb_pipe(documents, batch_size=batch_size, **HF_SAFETY_KWARGS)
        lap("fdb")

        gdb_predictions = gdb_pipe(documents, batch_size=batch_size, **HF_SAFETY_KWARGS)
        lap("gdb")

        vader_scores, finvader_scores = cpu_future.result()
        lap("cpu_wait")

    except torch.cuda.OutOfMemoryError as e:
        logging.error(f"CUDA OOM Error during model inference: {e}")
        clear_gpu_memory()
        return []
    except Exception as e:
        logging.error(f"Error during model inference: {e}")
        clear_gpu_memory()
        return []

    _out = []

    for idx, text in enumerate(documents):
        try:
            embedding_vector = embedding_vectors[idx]
            embedding = Embedding(list(embedding_vector.astype(float)))

            classification_result = classification_results[idx]
            top_label = classification_result["labels"][0]
            top_score = round(classification_result["scores"][0], 4)
            classification = Classification(label=top_label, score=top_score)

            text_type_result = [(y["label"], float(y["score"])) for y in text_type_results[idx]]
            types = {item[0]: item[1] for item in text_type_result}
            text_type = TextType(
                assumption=types.get("Assumption", 0.0),
                anecdote=types.get("Anecdote", 0.0),
                none=types.get("None", 0.0),
                definition=types.get("Definition", 0.0),
                testimony=types.get("Testimony", 0.0),
                other=types.get("Other", 0.0),
                study=types.get("Statistics/Study", 0.0),
            )

            emotion_result = [(y["label"], float(y["score"])) for y in emotion_results[idx]]
            emotions = {item[0]: item[1] for item in emotion_result}
            emotions = {k: round(v, 4) for k, v in emotions.items()}
            emotion = Emotion(
                love=emotions.get("love", 0.0),
                admiration=emotions.get("admiration", 0.0),
                joy=emotions.get("joy", 0.0),
                approval=emotions.get("approval", 0.0),
                caring=emotions.get("caring", 0.0),
                excitement=emotions.get("excitement", 0.0),
                gratitude=emotions.get("gratitude", 0.0),
                desire=emotions.get("desire", 0.0),
                anger=emotions.get("anger", 0.0),
                optimism=emotions.get("optimism", 0.0),
                disapproval=emotions.get("disapproval", 0.0),
                grief=emotions.get("grief", 0.0),
                annoyance=emotions.get("annoyance", 0.0),
                pride=emotions.get("pride", 0.0),
                curiosity=emotions.get("curiosity", 0.0),
                neutral=emotions.get("neutral", 0.0),
                disgust=emotions.get("disgust", 0.0),
                disappointment=emotions.get("disappointment", 0.0),
                realization=emotions.get("realization", 0.0),
                fear=emotions.get("fear", 0.0),
                relief=emotions.get("relief", 0.0),
                confusion=emotions.get("confusion", 0.0),
                remorse=emotions.get("remorse", 0.0),
                embarrassment=emotions.get("embarrassment", 0.0),
                surprise=emotions.get("surprise", 0.0),
                sadness=emotions.get("sadness", 0.0),
                nervousness=emotions.get("nervousness", 0.0),
            )

            irony_result = [(y["label"], float(y["score"])) for y in irony_results[idx]]
            ironies = {item[0]: item[1] for item in irony_result}
            irony = Irony(
                irony=ironies.get("irony", 0.0),
                non_irony=ironies.get("non_irony", 0.0)
            )

            vader_sent_score = round(vader_scores[idx], 2)
            fin_vader_sent_score = round(finvader_scores[idx], 2)

            fdb_prediction = fdb_predictions[idx]
            fdb_sentiment_dict = {e["label"]: round(e["score"], 3) for e in fdb_prediction}
            fdb_sent_score = round(fdb_sentiment_dict.get("positive", 0.0) - fdb_sentiment_dict.get("negative", 0.0), 3)

            gdb_sent_score = _pos_minus_neg(gdb_predictions[idx])

            compounded_fin_sentiment = round((0.70 * fdb_sent_score + 0.30 * fin_vader_sent_score), 2)

            sentiment_score = ensemble_sentiment(gdb_sent_score, vader_sent_score, compounded_fin_sentiment)

            sentiment = Sentiment(sentiment_score)

            gender = Gender(male=0.5, female=0.5)

            age = Age(
                below_twenty=0.25,
                twenty_thirty=0.25,
                thirty_forty=0.25,
                forty_more=0.25
            )

            language_score = LanguageScore(1.0)

            analysis = Analysis(
                classification=classification,
                language_score=language_score,
                sentiment=sentiment,
                embedding=embedding,
                gender=gender,
                text_type=text_type,
                emotion=emotion,
                irony=irony,
                age=age,
            )

            _out.append(analysis)

        except Exception as e:
            logging.error(f"Error processing document {idx}: {e}")
            _out.append(create_fallback_analysis())

    lap("build")
    soft_clear_gpu_cache()
    lap("clear")
    _sorted_out = _out
    _out = [None] * n_docs
    for pos, orig in enumerate(order):
        _out[orig] = _sorted_out[pos]
    allocated_end, _ = get_gpu_memory_info()
    logging.info(
        "⏱ tag n=%d total=%.2fs | %s | classification_ms=%.1f (%s) | gpu_alloc_end=%.2fGB",
        len(_out), time.perf_counter() - t_start,
        " ".join(f"{k}={v:.2f}" for k, v in timings.items()),
        classification_ms, backend, allocated_end,
    )
    global _cls_batches
    _cls_batches += 1
    if _cls_batches % CLASSIFICATION_LOG_EVERY == 1:
        logging.info(
            "[CLS] backend=%s labels=%d batch_ms=%.1f classification_ms=%.1f",
            backend, len(classification_labels), (time.perf_counter() - t_start) * 1000, classification_ms,
        )

    return _out

def create_fallback_analysis():
    return Analysis(
        classification=Classification(label="Social", score=0.5),
        language_score=LanguageScore(1.0),
        sentiment=Sentiment(0.0),
        embedding=Embedding([0.0] * 384),
        gender=Gender(male=0.5, female=0.5),
        text_type=TextType(assumption=0.0, anecdote=0.0, none=1.0, definition=0.0, testimony=0.0, other=0.0, study=0.0),
        emotion=Emotion(love=0.0, admiration=0.0, joy=0.0, approval=0.0, caring=0.0, excitement=0.0, gratitude=0.0, desire=0.0, anger=0.0, optimism=0.0, disapproval=0.0, grief=0.0, annoyance=0.0, pride=0.0, curiosity=0.0, neutral=1.0, disgust=0.0, disappointment=0.0, realization=0.0, fear=0.0, relief=0.0, confusion=0.0, remorse=0.0, embarrassment=0.0, surprise=0.0, sadness=0.0, nervousness=0.0),
        irony=Irony(irony=0.0, non_irony=1.0),
        age=Age(below_twenty=0.25, twenty_thirty=0.25, thirty_forty=0.25, forty_more=0.25),
    )
