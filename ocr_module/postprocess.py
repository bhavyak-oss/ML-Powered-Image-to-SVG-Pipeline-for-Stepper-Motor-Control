"""
postprocess.py
--------------
OCR ke baad spelling correction. vocab.txt me apne document ke sahi words
(ek line me ek word/phrase) likho; OCR ka galat text sabse kareebi word se
replace ho jayega. Jo word vocab ke kareeb nahi hota wo waisa hi rehta hai.
"""
from __future__ import annotations
import difflib
import os
import re
from typing import List

_VOCAB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocab.txt")


def load_vocab(path: str = _VOCAB_PATH) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _match_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original.islower():
        return replacement.lower()
    return replacement


def correct_text(text: str, vocab: List[str], cutoff: float = 0.65) -> str:
    if not vocab or not text.strip():
        return text
    low = {v.lower(): v for v in vocab}

    # 1) poora text ek vocab entry se match karta hai? (e.g. "SELF-MARAGENERT")
    whole = text.strip().lower().replace(" ", "")
    hit = difflib.get_close_matches(whole, list(low), n=1, cutoff=cutoff)
    if hit:
        return _match_case(text.strip(), low[hit[0]])

    # 2) warna har word alag se
    def fix(m: "re.Match[str]") -> str:
        tok = m.group(0)
        if tok.lower() in low:
            return tok
        h = difflib.get_close_matches(tok.lower(), list(low), n=1, cutoff=cutoff)
        return _match_case(tok, low[h[0]]) if h else tok

    return re.sub(r"[A-Za-z][A-Za-z\-]*", fix, text)