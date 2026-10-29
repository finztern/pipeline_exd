import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exorde_data import (
    Item, CreatedAt, Content, Domain, Url, Title, ExternalId, Author, ExternalParentId,
    Translation, Language, Translated,
)

MAX_DEPTH = int(os.getenv("MAX_DEPTH_CLASSIFICATION", "2"))

_lab = None
_process = None


def build_item(raw: dict) -> Item:
    return Item(
        created_at=CreatedAt(raw["created_at"]),
        title=Title(raw.get("title", "")),
        content=Content(raw["content"]),
        domain=Domain(raw["domain"]),
        url=Url(raw["url"]),
        external_id=ExternalId(raw.get("external_id", "")),
        external_parent_id=ExternalParentId(raw.get("external_parent_id") or ""),
        author=Author(raw.get("author", "")),
    )


def init():
    global _lab, _process
    from lab_initialization import lab_initialization
    from process import process
    from evaluate_token_count import evaluate_token_count
    _lab = lab_initialization()
    _process = process
    evaluate_token_count("warm up")


def ping() -> bool:
    time.sleep(0.3)
    return True


def run(raw: dict, tr) -> dict:
    item = build_item(raw)
    translation = None
    if tr:
        translation = Translation(language=Language(tr[0]), translation=Translated(tr[1]))
    processed = _process(item, _lab, MAX_DEPTH, translation)
    return {
        "item": {
            "created_at":         str(processed.item.get("created_at", "")),
            "title":              str(processed.item.get("title", "")),
            "content":            str(processed.item.get("content", "")),
            "domain":             str(processed.item.get("domain", "")),
            "url":                str(processed.item.get("url", "")),
            "external_id":        str(processed.item.get("external_id", "")),
            "external_parent_id": str(processed.item.get("external_parent_id", "")),
            "author":             str(processed.item.get("author", "")),
            "username":           str(processed.item.get("username", "")) if processed.item.get("username") else "",
        },
        "translation": {
            "language":    str(processed.translation.language),
            "translation": str(processed.translation.translation),
        },
        "top_keywords": list(processed.top_keywords),
        "classification": {
            "label": str(processed.classification.label),
            "score": float(processed.classification.score),
        },
    }
