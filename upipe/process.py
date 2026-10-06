import time, logging
from preprocess import preprocess
from translate import translate
from extract_keywords import extract_keywords
from zero_shot import zero_shot
from evaluate_token_count import evaluate_token_count, MAX_MODEL_TOKENS
from exorde_data import Translation, Classification, Keywords, Processed, Item, Translated

def process(item: Item, lab_configuration, max_depth_classification, translation: Translation | None = None) -> Processed:
    t0 = time.perf_counter()
    try:
        item = preprocess(item, False)
        if translation is None:
            translation = translate(item, lab_configuration["installed_languages"])
        if translation.translation == "":
            raise ValueError("No content to work with")

        n_tokens = evaluate_token_count(translation.translation)
        if n_tokens > MAX_MODEL_TOKENS:
            raise ValueError(f"Токен-лимит превышен ({n_tokens} > {MAX_MODEL_TOKENS}), item отброшен")

        top_keywords: Keywords = extract_keywords(translation)
        classification: Classification = zero_shot(translation, lab_configuration, max_depth=max_depth_classification)
        logging.info(f"⏱ Обработка: {(time.perf_counter()-t0)*1000:.1f}ms")
        return Processed(item=item, translation=translation, top_keywords=top_keywords, classification=classification)
    except Exception as err:
        logging.info(f"⏱ Обработка (ошибка): {(time.perf_counter()-t0)*1000:.1f}ms")
        raise err
