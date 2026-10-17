"""
svg_export.py
--------------
Semantic JSON document -> Plot-Ready SVG (stroke-only) ya Editable SVG.

CHANGES (patch):
  * layout_elements(): lines ko group karta hai, ek jaisa font size deta hai,
    line pitch font ke hisaab se rakhta hai -> overlap nahi hota, width
    max_width_mm me fit hoti hai.
  * write_plot_ready_svg(): asli strokes ke bounding box se viewBox nikalta hai
    aur content ko (margin, margin) par shift karta hai -> kuch cut nahi hota.
  * write_editable_svg() wapas add kiya (pichhle patch me gayab ho gaya tha).
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Tuple

import svgwrite

from hershey_font import text_to_strokes, text_width, DEFAULT_ADVANCE

Point = Tuple[float, float]
Segment = Tuple[Point, Point]


def load_document_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_document_json(doc: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def layout_elements(
    doc: Dict[str, Any],
    max_width_mm: float = 180.0,
    margin_mm: float = 8.0,
    min_fs: float = 3.5,
    max_fs: float = 6.0,
    line_gap_ratio: float = 0.9,
    adv: float = DEFAULT_ADVANCE,
) -> List[Dict[str, Any]]:
    """Return [{'text','x','y','fs'}] - no overlap, uniform font, fits page width."""
    els = [dict(e, text=str(e["text"]).strip()) for e in doc["text_elements"]
           if str(e.get("text", "")).strip()]
    if not els:
        return []

    for e in els:
        e["_yc"] = e["y_mm"] + e["font_size_mm"] / 2.0
    hs = sorted(max(e["font_size_mm"], 0.1) for e in els)
    med_h = hs[len(hs) // 2]
    fs = min(max(med_h * 0.8, min_fs), max_fs)

    # 1) line grouping (y-centre ke hisaab se)
    els.sort(key=lambda e: e["_yc"])
    lines: List[List[Dict[str, Any]]] = []
    for e in els:
        if lines:
            yc_mean = sum(x["_yc"] for x in lines[-1]) / len(lines[-1])
            if abs(e["_yc"] - yc_mean) <= 0.6 * med_h:
                lines[-1].append(e)
                continue
        lines.append([e])
    for ln in lines:
        ln.sort(key=lambda e: e["x_mm"])

    gx0 = min(e["x_mm"] for e in els)
    avail = max_width_mm - 2 * margin_mm

    def place(fs_try: float):
        placed, widest = [], 0.0
        pitch = fs_try * (1.0 + line_gap_ratio)
        for i, ln in enumerate(lines):
            cursor = 0.0
            for e in ln:
                x_rel = max(e["x_mm"] - gx0, cursor)
                w = text_width(e["text"], fs_try, adv)
                placed.append({"text": e["text"], "x": margin_mm + x_rel,
                               "y": margin_mm + i * pitch, "fs": fs_try})
                cursor = x_rel + w + 2 * adv * fs_try   # do akshar ka gap
                widest = max(widest, x_rel + w)
        return placed, widest

    placed, widest = place(fs)
    while widest > avail and fs > 2.0:
        fs *= 0.95
        placed, widest = place(fs)
    if widest > avail:
        print(f"[svg_export] WARNING: content {widest:.1f}mm > available {avail:.1f}mm")
    return placed


# ---------------------------------------------------------------------------
# Plot-Ready SVG
# ---------------------------------------------------------------------------

def compile_strokes(doc: Dict[str, Any]) -> List[Segment]:
    segments: List[Segment] = []
    for p in layout_elements(doc):
        segments += text_to_strokes(p["text"], p["x"], p["y"], p["fs"])
    return segments


def write_plot_ready_svg(
    segments: List[Segment],
    page_mm: Dict[str, float],
    out_path: str,
    stroke_width_mm: float = 0.35,
    margin: float = 8.0,
) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if segments:
        xs = [p[0] for seg in segments for p in seg]
        ys = [p[1] for seg in segments for p in seg]
        dx, dy = margin - min(xs), margin - min(ys)
        w = max(xs) - min(xs) + 2 * margin
        h = max(ys) - min(ys) + 2 * margin
    else:
        dx = dy = 0.0
        w, h = page_mm["width"], page_mm["height"]

    dwg = svgwrite.Drawing(out_path, size=(f"{w:.2f}mm", f"{h:.2f}mm"),
                           viewBox=f"0 0 {w:.2f} {h:.2f}")
    if segments:
        d = " ".join(
            f"M {a[0] + dx:.2f} {a[1] + dy:.2f} L {b[0] + dx:.2f} {b[1] + dy:.2f}"
            for a, b in segments
        )
        dwg.add(dwg.path(d=d, fill="none", stroke="black", stroke_width=stroke_width_mm))
    dwg.save()
    print(f"[svg_export] page = {w:.1f} x {h:.1f} mm")
    return out_path


def export_plot_ready(doc: Dict[str, Any], out_path: str, stroke_width_mm: float = 0.35) -> str:
    segments = compile_strokes(doc)
    return write_plot_ready_svg(segments, doc["page_dimensions_mm"], out_path, stroke_width_mm)


# ---------------------------------------------------------------------------
# Editable SVG
# ---------------------------------------------------------------------------

def write_editable_svg(doc: Dict[str, Any], out_path: str, font_family: str = "Arial") -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    page = doc["page_dimensions_mm"]
    w, h = page["width"], page["height"]
    dwg = svgwrite.Drawing(out_path, size=(f"{w}mm", f"{h}mm"), viewBox=f"0 0 {w} {h}")
    for el in doc["text_elements"]:
        baseline_y = el["y_mm"] + el["font_size_mm"]
        dwg.add(dwg.text(el["text"], insert=(el["x_mm"], baseline_y),
                         font_family=font_family, font_size=f"{el['font_size_mm']}mm",
                         fill="black"))
    dwg.save()
    return out_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python svg_export.py <document.json> <output.svg>")
        sys.exit(1)

    document = load_document_json(sys.argv[1])
    print(f"Wrote Plot-Ready SVG -> {export_plot_ready(document, sys.argv[2])}")