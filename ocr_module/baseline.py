from __future__ import annotations
from functools import lru_cache
from typing import List, Dict, Any
from rapidocr_onnxruntime import RapidOCR

@lru_cache(maxsize=1)
def _get_engine(lang: str = "en"):
    return RapidOCR()

def run_baseline_ocr(image_path: str, lang: str = "en") -> List[Dict[str, Any]]:
    engine = _get_engine(lang)
    result, _ = engine(image_path)
    out = []
    if not result:
        return out
    for box, text, score in result:
        out.append({
            "text": text,
            "confidence": float(score),
            "box": box
        })
    return out