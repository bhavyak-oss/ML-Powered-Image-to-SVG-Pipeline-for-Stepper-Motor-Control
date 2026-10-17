"""
ocr_multilang.py
----------------
Non-English language ke liye OCR. RapidOCR ko us language ka recognition model
dekar chalata hai. Output format baseline.run_baseline_ocr jaisa hi hai:
    [{"box": quad, "text": str, "confidence": float}, ...]

NOTE: RapidOCR ke kwargs (rec_model_path, rec_keys_path) mere hisaab se sahi hain,
par maine inhe asli Hindi model ke saath chalaya nahi hai. Pehle ek chhoti Hindi
image pe check karo.
"""
from __future__ import annotations
from typing import Any, Dict, List

from lang_config import model_paths, is_ready, LANGUAGES

_ENGINES: Dict[str, Any] = {}


def _engine(lang: str):
    if lang in _ENGINES:
        return _ENGINES[lang]
    if lang not in LANGUAGES:
        raise ValueError(f"Unknown language {lang!r}. Supported: {', '.join(LANGUAGES)}")
    if not is_ready(lang):
        rec, keys = model_paths(lang)
        raise FileNotFoundError(
            f"{LANGUAGES[lang]['name']} OCR model nahi mila.\n"
            f"  Ye 2 files rakho:\n    {rec}\n    {keys}\n"
            f"  (us language ka recognition ONNX model + uski keys.txt)"
        )
    from rapidocr_onnxruntime import RapidOCR
    rec, keys = model_paths(lang)
    _ENGINES[lang] = RapidOCR(rec_model_path=rec, rec_keys_path=keys)
    return _ENGINES[lang]


def run_ocr(image_path: str, lang: str) -> List[Dict[str, Any]]:
    result, _ = _engine(lang)(image_path)
    return [{"box": box, "text": text, "confidence": float(conf)}
            for box, text, conf in (result or [])]
