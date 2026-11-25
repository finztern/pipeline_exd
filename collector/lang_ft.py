"""Определение языка через fastText (lid.176.ftz, ~1 МБ, лежит внутри пакета fast-langdetect, работает офлайн).

Возвращает (lang, prob) в той же схеме кодов, которую ждёт остальной collector (как у langdetect):
zh-cn / zh-tw вместо zh, no вместо nn/nb и т.д.
Скорость: ~20-50 мкс на текст против ~0.5-1 мс у langdetect.

ВАЖНО: уверенность fastText откалибрована иначе, чем у langdetect (на коротких неоднозначных
текстах она заметно ниже, а не «уверенно неверная»), поэтому для неё свои пороги
(LANG_FT_CONFIDENCE / LANG_FT_LOW_CONF_MIN в collector.py).
"""
import os
import re

from fast_langdetect import detect as _ft_detect

MAX_CHARS = max(50, int(os.getenv("LANG_FT_MAX_CHARS", "1000")))
_WS = re.compile(r"\s+")

# fastText -> коды, которые использует collector (схема langdetect)
_CODE_MAP = {"nn": "no", "nb": "no", "iw": "he", "fil": "tl", "jw": "jv"}

# Пары «традиционный / упрощённый» иероглиф (только однозначные, без 里/裡, 后/後, 么/麼).
# fastText отдаёт только "zh", а в пайплайне есть отдельный язык zt (традиционный).
_TRAD = "這們來說對學體電國際發時會長與開關問題點資應當還個無為經現實讓從東車見話認議識讀請謝寫樣價錢隨雖嗎歲愛氣語蘭藝畫專業務員團報導"
_SIMP = "这们来说对学体电国际发时会长与开关问题点资应当还个无为经现实让从东车见话认议识读请谢写样价钱随虽吗岁爱气语兰艺画专业务员团报导"
assert len(_TRAD) == len(_SIMP), "таблицы _TRAD/_SIMP должны быть одной длины"
# оставляем только символы, которые реально различаются
_PAIRS = [(t, s) for t, s in zip(_TRAD, _SIMP) if t != s]
_TRAD_SET = {t for t, _ in _PAIRS}
_SIMP_SET = {s for _, s in _PAIRS}


def _zh_variant(text: str) -> str:
    t = sum(1 for c in text if c in _TRAD_SET)
    s = sum(1 for c in text if c in _SIMP_SET)
    return "zh-tw" if t > s else "zh-cn"


def detect(text: str) -> tuple[str | None, float]:
    t = _WS.sub(" ", text[:MAX_CHARS]).strip()
    if not t:
        return None, 0.0
    res = _ft_detect(t, model="lite", k=1)
    if isinstance(res, list):
        if not res:
            return None, 0.0
        res = res[0]
    lang = str(res.get("lang", "")).lower()
    if not lang:
        return None, 0.0
    lang = _CODE_MAP.get(lang, lang)
    if lang == "zh":
        lang = _zh_variant(t)
    return lang, float(res.get("score", 0.0))
