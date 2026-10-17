"""
lang_config.py
--------------
Ek jagah jahan saari supported languages likhi hain. Nayi language jodni ho to
sirf LANGUAGES me ek entry add karo + models/<code>/ me recognition model rakho.

Models folder (project root ke andar):
    models/hi/rec.onnx      <- us language ka text-recognition ONNX model
    models/hi/keys.txt      <- us model ki character list (ek line me ek akshar)
English ke liye kuch rakhna nahi padta (RapidOCR ka default model chalta hai).
"""
from __future__ import annotations
import os

DEFAULT_LANG = "en"

# script = Unicode script ka naam (script_detect.py ke saath match hota hai)
LANGUAGES = {
    "en": {"name": "English",          "script": "LATIN"},
    "hi": {"name": "Hindi",            "script": "DEVANAGARI"},
    "mr": {"name": "Marathi",          "script": "DEVANAGARI"},
    "ta": {"name": "Tamil",            "script": "TAMIL"},
    "bn": {"name": "Bengali",          "script": "BENGALI"},
}

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")


def model_paths(lang: str):
    d = os.path.join(MODELS_DIR, lang)
    return os.path.join(d, "rec.onnx"), os.path.join(d, "keys.txt")


def is_ready(lang: str) -> bool:
    """English hamesha ready; baaki tab jab rec.onnx aur keys.txt maujood hon."""
    if lang == DEFAULT_LANG:
        return True
    rec, keys = model_paths(lang)
    return os.path.exists(rec) and os.path.exists(keys)


def list_languages():
    return [{"code": c, "name": v["name"], "script": v["script"], "ready": is_ready(c)}
            for c, v in LANGUAGES.items()]
