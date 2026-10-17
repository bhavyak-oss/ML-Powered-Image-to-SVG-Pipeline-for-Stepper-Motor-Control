"""
line_merge.py
-------------
Ek hi line ke tute hue detection boxes (jaise "Lanyuay" + "he") ko ek box
me jodta hai, taaki recognition poori line par chale.
"""
from __future__ import annotations
from typing import Any, Dict, List


def _bbox(box) -> List[float]:
    if isinstance(box[0], (list, tuple)):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return [min(xs), min(ys), max(xs), max(ys)]
    return [float(v) for v in box]


def merge_line_boxes(detections: List[Dict[str, Any]], gap_ratio: float = 1.5,
                     v_overlap: float = 0.35) -> List[Dict[str, Any]]:
    """Same line ke boxes jodo: y-centre kareeb ho aur x-gap chhota ho."""
    if len(detections) < 2:
        return detections
    items = []
    for d in detections:
        b = _bbox(d["box"])
        items.append({"b": b, "texts": [(b[0], d.get("text", ""))],
                      "conf": [float(d.get("confidence", 0.0))]})
    heights = sorted(it["b"][3] - it["b"][1] for it in items)
    gap_max = gap_ratio * heights[len(heights) // 2]

    items.sort(key=lambda it: it["b"][0])
    merged: List[Dict[str, Any]] = []
    for it in items:
        target = None
        for m in merged:
            mb, b = m["b"], it["b"]
            # same line = dono boxes ke y-centre kareeb hon (sirf overlap kaafi nahi,
            # kyunki handwriting ke lambe boxes dusri line se bhi overlap karte hain)
            yc_a, yc_b = (mb[1] + mb[3]) / 2.0, (b[1] + b[3]) / 2.0
            tall = max(mb[3] - mb[1], b[3] - b[1])
            same_line = abs(yc_a - yc_b) <= v_overlap * tall
            gap = b[0] - mb[2]
            if same_line and gap <= gap_max:
                target = m
                break
        if target is None:
            merged.append(it)
        else:
            tb = target["b"]
            target["b"] = [min(tb[0], it["b"][0]), min(tb[1], it["b"][1]),
                           max(tb[2], it["b"][2]), max(tb[3], it["b"][3])]
            target["texts"] += it["texts"]
            target["conf"] += it["conf"]

    out = []
    for m in merged:
        texts = [t for _, t in sorted(m["texts"], key=lambda p: p[0]) if t]
        out.append({"box": m["b"], "text": " ".join(texts),
                    "confidence": min(m["conf"]) if m["conf"] else 0.0})
    out.sort(key=lambda d: (d["box"][1], d["box"][0]))
    return out
