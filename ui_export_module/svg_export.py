"""
svg_export.py
--------------
Converts a Semantic JSON document (schemas/document_schema.json, the same
format Member 3's pipeline.py produces) into either:

    - a Plot-Ready SVG  — stroke-only <path>, fill="none", for CNC/pen
      plotters (Sec 2.1 / Step 6 of the execution plan)
    - an Editable SVG   — normal <text> elements, for viewing/editing in
      Inkscape, Illustrator, Figma, etc.

Zero Streamlit dependency, so this can be unit-tested or run headless in CI
independently of app.py.
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Tuple

import svgwrite

from hershey_font import text_to_strokes

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
# Plot-Ready SVG
# ---------------------------------------------------------------------------

def compile_strokes(doc: Dict[str, Any]) -> List[Segment]:
    """Every text_element -> stroke segments (see hershey_font.text_to_strokes)."""
    segments: List[Segment] = []
    for el in doc["text_elements"]:
        segments += text_to_strokes(
            el["text"], el["x_mm"], el["y_mm"], el["font_size_mm"]
        )
    return segments


def write_plot_ready_svg(
    segments: List[Segment],
    page_mm: Dict[str, float],
    out_path: str,
    stroke_width_mm: float = 0.5,
) -> str:
    """
    Write a stroke-only SVG: fill="none", one merged <path> for the whole
    page. One merged path keeps the file small and lets the plotter driver
    optimize pen-up travel order.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    w, h = page_mm["width"], page_mm["height"]
    dwg = svgwrite.Drawing(out_path, size=(f"{w}mm", f"{h}mm"), viewBox=f"0 0 {w} {h}")

    if segments:
        d = " ".join(
            f"M {x1:.2f} {y1:.2f} L {x2:.2f} {y2:.2f}"
            for (x1, y1), (x2, y2) in segments
        )
        dwg.add(dwg.path(d=d, fill="none", stroke="black", stroke_width=stroke_width_mm))

    dwg.save()
    return out_path


def export_plot_ready(doc: Dict[str, Any], out_path: str, stroke_width_mm: float = 0.5) -> str:
    """Convenience wrapper: doc -> compile_strokes -> write_plot_ready_svg."""
    segments = compile_strokes(doc)
    return write_plot_ready_svg(segments, doc["page_dimensions_mm"], out_path, stroke_width_mm)


# ---------------------------------------------------------------------------
# Editable SVG (for design tools — screen rendering, not plotting)
# ---------------------------------------------------------------------------

def write_editable_svg(doc: Dict[str, Any], out_path: str, font_family: str = "Arial") -> str:
    """
    Write a standard editable SVG using <text> elements — the counterpart
    described in Sec 2.1 of the execution plan. This is what a designer
    would open in Inkscape/Illustrator; it is NOT meant for a plotter.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    page = doc["page_dimensions_mm"]
    w, h = page["width"], page["height"]
    dwg = svgwrite.Drawing(out_path, size=(f"{w}mm", f"{h}mm"), viewBox=f"0 0 {w} {h}")

    for el in doc["text_elements"]:
        # y_mm is the top of the text; SVG <text> anchors on the baseline,
        # so shift down by roughly one font-size to align with Plot-Ready output.
        baseline_y = el["y_mm"] + el["font_size_mm"]
        dwg.add(dwg.text(
            el["text"],
            insert=(el["x_mm"], baseline_y),
            font_family=font_family,
            font_size=f"{el['font_size_mm']}mm",
            fill="black",
        ))

    dwg.save()
    return out_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python svg_export.py <document.json> <output.svg>")
        print("       (writes a Plot-Ready SVG by default)")
        sys.exit(1)

    document = load_document_json(sys.argv[1])
    out_path = export_plot_ready(document, sys.argv[2])
    print(f"Wrote Plot-Ready SVG -> {out_path}")
