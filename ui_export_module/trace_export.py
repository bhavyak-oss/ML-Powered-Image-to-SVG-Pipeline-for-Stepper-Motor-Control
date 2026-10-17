"""
trace_export.py
---------------
Non-Latin text (Hindi, Tamil, ...) ke liye Hershey font nahi hai, isliye original
image ke us text-region ko seedha trace karke plotter ke liye line segments banate hain.

  mode="skeleton": har stroke ki ek centerline (plotter ke liye best; skimage chahiye)
  mode="contour" : akshar ka outline (double line aati hai; sirf OpenCV chahiye)

Output absolute-mm segments hain (x_mm = px * 25.4 / dpi), pipeline ke x_mm/y_mm
jaise hi, to text apni original jagah pe aata hai.
"""
from __future__ import annotations
from typing import List, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]
Segment = Tuple[Point, Point]

_N8 = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def _skeleton_polylines(sk: np.ndarray) -> List[List[Tuple[int, int]]]:
    """1-pixel skeleton ko pixel-chains (polylines) me badlo."""
    ys, xs = np.nonzero(sk)
    pts = set(zip(xs.tolist(), ys.tolist()))
    adj = {p: [(p[0] + dx, p[1] + dy) for dx, dy in _N8 if (p[0] + dx, p[1] + dy) in pts]
           for p in pts}
    used = set()

    def key(a, b):
        return (a, b) if a < b else (b, a)

    def walk(start, nxt):
        used.add(key(start, nxt))
        path, prev, cur = [start], start, nxt
        while True:
            path.append(cur)
            if cur == start or len(adj[cur]) != 2:
                break
            cand = [q for q in adj[cur] if q != prev and key(cur, q) not in used]
            if not cand:
                break
            q = cand[0]
            used.add(key(cur, q))
            prev, cur = cur, q
        return path

    paths = []
    for p in pts:                                  # endpoints / junctions se shuru
        if len(adj[p]) != 2:
            for q in adj[p]:
                if key(p, q) not in used:
                    paths.append(walk(p, q))
    for p in pts:                                  # bache hue closed loops
        for q in adj[p]:
            if key(p, q) not in used:
                paths.append(walk(p, q))
    return paths


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if (ink > 0).mean() > 0.5:         # ink hamesha minority hoti hai; warna invert
        ink = 255 - ink
    return ink


def trace_region(image_path: str, bbox_px, dpi: float = 300.0, mode: str = "skeleton",
                 eps_px: float = 1.0, min_len_px: float = 6.0, pad: int = 2) -> List[Segment]:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not readable: {image_path}")
    h, w = img.shape
    x0, y0, x1, y1 = [int(round(v)) for v in bbox_px]
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return []
    ink = _ink_mask(img[y0:y1, x0:x1])

    paths: List[np.ndarray] = []
    used_mode = mode
    if mode == "skeleton":
        try:
            from skimage.morphology import skeletonize
            sk = skeletonize(ink > 0)
            for pl in _skeleton_polylines(sk):
                paths.append(np.array(pl, dtype=np.int32).reshape(-1, 1, 2))
        except ImportError:
            used_mode = "contour"
    if used_mode == "contour":
        cnts, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            closed = np.vstack([c, c[:1]])
            paths.append(closed)

    k = 25.4 / dpi
    segs: List[Segment] = []
    for p in paths:
        if cv2.arcLength(p, False) < min_len_px:
            continue
        p = cv2.approxPolyDP(p, eps_px, False).reshape(-1, 2)
        for (ax, ay), (bx, by) in zip(p[:-1], p[1:]):
            segs.append((((x0 + ax) * k, (y0 + ay) * k), ((x0 + bx) * k, (y0 + by) * k)))
    return segs
