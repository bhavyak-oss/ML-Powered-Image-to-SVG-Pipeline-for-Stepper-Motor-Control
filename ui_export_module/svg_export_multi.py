"""
svg_export_multi.py
-------------------
Multilingual export. Do mode:

  reflow  - doc me sirf Latin text hai: purana export_plot_ready (clean uniform layout)
  inplace - doc me non-Latin text bhi hai: har element apni original jagah pe;
            Latin -> Hershey font, non-Latin -> original image se trace

Limitation: traced (non-Latin) element ka text Figma me edit karoge to wo plot me
nahi dikhega, kyunki wo image se trace hota hai, text se nahi.
"""
from __future__ import annotations
import copy
from typing import Any, Dict, List, Tuple

from hershey_font import text_to_strokes, text_width, DEFAULT_ADVANCE
from script_detect import needs_trace, latin_fold
from svg_export import export_plot_ready, write_plot_ready_svg
from trace_export import trace_region

PX_TO_MM = lambda px, dpi: px * 25.4 / dpi


def _inplace_segments(doc: Dict[str, Any], image_path: str, trace_mode: str,
                      warnings: List[str]):
    dpi = float(doc.get("dpi", 300.0))
    segs = []
    for el in doc["text_elements"]:
        text = str(el.get("text", "")).strip()
        bbox = el.get("bounding_box_pixel")
        if not text or not bbox:
            continue
        if needs_trace(text):
            if not image_path:
                warnings.append(f"{el.get('id')}: original image nahi mili, trace skip hua")
                continue
            segs += trace_region(image_path, bbox, dpi, mode=trace_mode)
            continue
        x_mm, y_mm = PX_TO_MM(bbox[0], dpi), PX_TO_MM(bbox[1], dpi)
        w_mm, h_mm = PX_TO_MM(bbox[2] - bbox[0], dpi), PX_TO_MM(bbox[3] - bbox[1], dpi)
        text = latin_fold(text)
        fs = min(max(h_mm * 0.75, 2.5), 15.0)
        tw = text_width(text, fs, DEFAULT_ADVANCE)
        if tw > w_mm * 1.15:
            fs *= (w_mm * 1.15) / tw
        segs += text_to_strokes(text, x_mm, y_mm + (h_mm - fs) / 2.0, fs)
    return segs


def export_plot_ready_multi(doc: Dict[str, Any], out_path: str, image_path: str = "",
                            trace_mode: str = "skeleton") -> Tuple[str, str, List[str]]:
    """Return (out_path, mode_used, warnings)."""
    warnings: List[str] = []
    elements = [e for e in doc["text_elements"] if str(e.get("text", "")).strip()]
    traced = [e for e in elements if needs_trace(str(e["text"]))]

    if not traced:
        d = copy.deepcopy(doc)
        for e in d["text_elements"]:
            e["text"] = latin_fold(str(e.get("text", "")))
        export_plot_ready(d, out_path)
        return out_path, "reflow", warnings

    segs = _inplace_segments(doc, image_path, trace_mode, warnings)
    write_plot_ready_svg(segs, doc["page_dimensions_mm"], out_path)
    warnings.append(f"{len(traced)} non-Latin line(s) image se trace hui; unka text edit plot me nahi aayega.")
    return out_path, "inplace", warnings
