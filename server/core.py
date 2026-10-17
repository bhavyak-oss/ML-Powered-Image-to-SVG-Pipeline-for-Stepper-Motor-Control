"""
core.py
-------
Server ka asli kaam (FastAPI se alag rakha hai, taaki bina server chalaye test ho sake).

  process_image(bytes, lang)  -> OCR + SVG + review elements
  export_session(id, edits)   -> Figma me edit kiya text lagakar naya SVG
"""
from __future__ import annotations
import io
import json
import os
import re
import sys
import uuid
from typing import Any, Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OCR_DIR = os.path.join(BASE_DIR, "ocr_module")
UI_DIR = os.path.join(BASE_DIR, "ui_export_module")
SESSIONS_DIR = os.path.join(BASE_DIR, "server", "sessions")

_SID_RE = re.compile(r"^[a-f0-9]{12}$")


def init() -> None:
    """Paths set karo. ocr_module me chdir, kyunki checkpoint ka relative path wahin se resolve hota hai."""
    for p in (OCR_DIR, UI_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(OCR_DIR)
    os.makedirs(SESSIONS_DIR, exist_ok=True)


def _session_dir(sid: str) -> str:
    if not _SID_RE.match(sid or ""):
        raise ValueError("Invalid session id")
    return os.path.join(SESSIONS_DIR, sid)


def _svg_for_figma(svg: str) -> str:
    """Figma ke liye width/height 'mm' se numeric px me (96 dpi). Plotter wali file mm me hi rehti hai."""
    def conv(m):
        return f'{m.group(1)}="{float(m.group(2)) * 96 / 25.4:.2f}"'
    return re.sub(r'\b(width|height)="([\d.]+)mm"', conv, svg, count=2)


def _client_elements(doc: Dict[str, Any]):
    from script_detect import needs_trace
    return [{
        "id": e["id"], "text": e["text"], "raw_text": e.get("raw_text", e["text"]),
        "bbox": e["bounding_box_pixel"], "confidence": e["confidence"],
        "traced": needs_trace(str(e["text"])),
    } for e in doc["text_elements"]]


def _build(doc: Dict[str, Any], sid: str) -> Dict[str, Any]:
    from svg_export_multi import export_plot_ready_multi
    out = os.path.join(_session_dir(sid), "out.svg")
    _, mode, warnings = export_plot_ready_multi(doc, out, image_path=doc.get("source_image", ""))
    with open(out, "r", encoding="utf-8") as f:
        svg = f.read()
    return {
        "session_id": sid, "lang": doc.get("lang", "en"), "mode": mode,
        "image_px": doc["image_px"], "elements": _client_elements(doc),
        "svg": svg, "svg_figma": _svg_for_figma(svg), "warnings": warnings,
    }


def process_image(data: bytes, lang: str = "en", use_finetuned: bool = False) -> Dict[str, Any]:
    from PIL import Image
    from lang_config import LANGUAGES, is_ready
    from pipeline import run_pipeline

    if lang not in LANGUAGES:
        raise ValueError(f"Unknown language {lang!r}")
    ocr_fn = None
    if lang != "en":
        from ocr_multilang import run_ocr         # model missing ho to clear FileNotFoundError
        if not is_ready(lang):
            run_ocr("", lang)                      # sirf wahi error message uthane ke liye
        ocr_fn = lambda p, _l=lang: run_ocr(p, _l)

    sid = uuid.uuid4().hex[:12]
    d = _session_dir(sid)
    os.makedirs(d, exist_ok=True)
    img_path = os.path.join(d, "in.png")
    Image.open(io.BytesIO(data)).convert("RGB").save(img_path)

    doc = run_pipeline(img_path, use_finetuned=(use_finetuned and lang == "en"),
                       spell_fix=(lang == "en"), ocr_fn=ocr_fn, document_id=sid)
    doc["lang"] = lang
    doc["source_image"] = img_path
    with open(os.path.join(d, "doc.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return _build(doc, sid)


def export_session(sid: str, edits: Dict[str, str]) -> Dict[str, Any]:
    path = os.path.join(_session_dir(sid), "doc.json")
    if not os.path.exists(path):
        raise FileNotFoundError("Session nahi mili - image dobara scan karo")
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    for e in doc["text_elements"]:
        if e["id"] in edits:
            e["text"] = edits[e["id"]]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return _build(doc, sid)
