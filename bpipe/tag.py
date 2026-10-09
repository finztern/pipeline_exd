import logging
import os
import time
import cupy as cp
import torch
import gc
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from finvader import finvader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from exorde_compat import (
    Classification, LanguageScore, Sentiment, Embedding, TextType,
    Emotion, Irony, Age, Gender, Analysis,
)

logging.basicConfig(level=logging.INFO)

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

    models = lab_configuration["models"]

    model = models['sentence_transformer']
    zs_pipe = models['zs_pipe']
    classification_labels = list(lab_configuration["labeldict"].keys())
    sentiment_analyzer = models['sentiment_analyzer']
    fdb_pipe = models['fdb_pipe']

    text_classification_models = {
        "Emotion": models['Emotion'],
        "Irony": models['Irony'],
        "TextType": models['TextType']
    }

    batch_size = min(len(documents), int(os.getenv("TAG_HF_BATCH_SIZE", "32")))
    logging.info(f"Using batch_size: {batch_size} for {len(documents)} documents, labels={len(classification_labels)}")

    HF_SAFETY_KWARGS = {"truncation": True, "max_length": 512}

    try:
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
        soft_clear_gpu_cache()
        lap("emb")

        # truncation/max_length kwargs в ZeroShot-пайплайне ломаются в части версий transformers
        classification_results = zs_pipe(documents, candidate_labels=classification_labels, batch_size=batch_size)
        soft_clear_gpu_cache()
        lap("zs")

        text_type_results = text_classification_models['TextType'](documents, batch_size=batch_size, **HF_SAFETY_KWARGS)
        soft_clear_gpu_cache()
        lap("ttype")

        emotion_results = text_classification_models['Emotion'](documents, batch_size=batch_size, **HF_SAFETY_KWARGS)
        soft_clear_gpu_cache()
        lap("emo")

        irony_results = text_classification_models['Irony'](documents, batch_size=batch_size, **HF_SAFETY_KWARGS)
        soft_clear_gpu_cache()
        lap("irony")

        fdb_predictions = fdb_pipe(documents, batch_size=batch_size, **HF_SAFETY_KWARGS)
        soft_clear_gpu_cache()
        lap("fdb")

        vader_scores = [sentiment_analyzer.polarity_scores(text)["compound"] for text in documents]
        lap("vader")
        finvader_scores = [finvader(text, use_sentibignomics=True, use_henry=True, indicator='compound') for text in documents]
        lap("finvader")

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

            compounded_fin_sentiment = round((0.70 * fdb_sent_score + 0.30 * fin_vader_sent_score), 2)

            if abs(compounded_fin_sentiment) >= 0.6:
                sentiment_score = round((0.15 * vader_sent_score + 0.85 * compounded_fin_sentiment), 2)
            elif abs(compounded_fin_sentiment) >= 0.4:
                sentiment_score = round((0.35 * vader_sent_score + 0.65 * compounded_fin_sentiment), 2)
            elif abs(compounded_fin_sentiment) >= 0.1:
                sentiment_score = round((0.60 * vader_sent_score + 0.40 * compounded_fin_sentiment), 2)
            else:
                sentiment_score = round((0.50 * vader_sent_score + 0.50 * fin_vader_sent_score), 2)

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
    clear_gpu_memory()
    lap("clear")
    allocated_end, _ = get_gpu_memory_info()
    logging.info(
        "⏱ tag n=%d total=%.2fs | %s | gpu_alloc_end=%.2fGB",
        len(_out), time.perf_counter() - t_start,
        " ".join(f"{k}={v:.2f}" for k, v in timings.items()),
        allocated_end,
    )

    return _out

def create_fallback_analysis():
    return Analysis(
        classification=Classification(label="other", score=0.5),
        language_score=LanguageScore(1.0),
        sentiment=Sentiment(0.0),
        embedding=Embedding([0.0] * 384),
        gender=Gender(male=0.5, female=0.5),
        text_type=TextType(assumption=0.0, anecdote=0.0, none=1.0, definition=0.0, testimony=0.0, other=0.0, study=0.0),
        emotion=Emotion(love=0.0, admiration=0.0, joy=0.0, approval=0.0, caring=0.0, excitement=0.0, gratitude=0.0, desire=0.0, anger=0.0, optimism=0.0, disapproval=0.0, grief=0.0, annoyance=0.0, pride=0.0, curiosity=0.0, neutral=1.0, disgust=0.0, disappointment=0.0, realization=0.0, fear=0.0, relief=0.0, confusion=0.0, remorse=0.0, embarrassment=0.0, surprise=0.0, sadness=0.0, nervousness=0.0),
        irony=Irony(irony=0.0, non_irony=1.0),
        age=Age(below_twenty=0.25, twenty_thirty=0.25, thirty_forty=0.25, forty_more=0.25),
    )