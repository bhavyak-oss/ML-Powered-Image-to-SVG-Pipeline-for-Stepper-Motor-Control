"""
trace_export.py  (v4)
---------------------
Non-Latin text (Hindi, Tamil, ...) ko original image se trace karke plotter ke
liye line segments banata hai.

v4: skeleton ke baad jo moti jagah (bulb, jaise 'ग' ka gola) bach jaati hai, uska outline
    bhi jodte hain (blob=True), warna wo sirf ek chhoti tick reh jaata tha.
v3 me sudhar (Hindi ke chhote hisse kho rahe the: matra, bindu, nukta, poonchh):
  * lakeeron ke sire (endpoints) ko bahar ki taraf aadhi stroke-width badhaya,
    kyunki skeleton sire chhote kar deta hai (khambe, patti aur poonchh chhoti pad jaati thi).
  * chhote ink-dots (bindu, nukta, full stop) ko alag pehchan kar ek chhoti lakeer banate hain.
  * region ko pehle `upscale` guna bada karte hain, phir threshold + skeleton.
  * simplify (eps) aur noise-filter (min_len) ab bade image ke hisaab se kam hain.
  * akele bindu/nukta (1 pixel ke dot) ab hatate nahi, chhoti line bana dete hain.

Output absolute-mm segments hain (x_mm = px * 25.4 / dpi).
"""
from __future__ import annotations
from typing import List, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]
Segment = Tuple[Point, Point]

_N8 = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


def _skeleton_polylines(sk: np.ndarray, dist=None) -> List[List[Tuple[int, int]]]:
    """1-pixel skeleton ko pixel-chains (polylines) me badlo. Akele dots bhi rakhta hai."""
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
        if len(adj[p]) != 0 and len(adj[p]) != 2:
            for q in adj[p]:
                if key(p, q) not in used:
                    paths.append(walk(p, q))
    for p in pts:                                  # bache hue closed loops
        for q in adj[p]:
            if key(p, q) not in used:
                paths.append(walk(p, q))
    if dist is not None:                           # sire bahar ki taraf badhao
        for path in paths:
            for end, back in ((0, 1), (-1, -2)):
                e = path[end]
                if len(adj[e]) != 1 or len(path) < 3:
                    continue
                k = min(5, len(path) - 1)
                ref = path[end + k] if end == 0 else path[-1 - k]
                dx, dy = e[0] - ref[0], e[1] - ref[1]
                n = (dx * dx + dy * dy) ** 0.5
                if n == 0:
                    continue
                r = float(dist[e[1], e[0]])
                ne = (int(round(e[0] + dx / n * r)), int(round(e[1] + dy / n * r)))
                if end == 0:
                    path.insert(0, ne)
                else:
                    path.append(ne)
    return paths


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if (ink > 0).mean() > 0.5:         # ink hamesha minority hoti hai; warna invert
        ink = 255 - ink
    return ink


def trace_region(image_path: str, bbox_px, dpi: float = 300.0, mode: str = "skeleton",
                 eps_px: float = 0.7, min_len_px: float = 2.0, pad: int = 2,
                 upscale: int = 3, dot_ratio: float = 0.2, blob: bool = True,
                 blob_area: float = 0.5) -> List[Segment]:
    """eps_px / min_len_px ORIGINAL image ke pixels me hain (upscale ka hisaab andar hota hai)."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not readable: {image_path}")
    h, w = img.shape
    x0, y0, x1, y1 = [int(round(v)) for v in bbox_px]
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    if x1 <= x0 or y1 <= y0:
        return []

    up = max(1, int(upscale))
    crop = img[y0:y1, x0:x1]
    if up > 1:
        crop = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    ink = _ink_mask(crop)

    # chhote ink-dots (bindu, nukta, full stop) alag karo
    dot_paths: List[np.ndarray] = []
    n_lab, lab, stats, cent = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    small = dot_ratio * crop.shape[0]
    for i in range(1, n_lab):
        cw, ch = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if max(cw, ch) < small:
            ink[lab == i] = 0
            cx, cy = cent[i]
            half = max(cw, ch, 1.5 * up) * 0.4
            if cw >= ch:
                a, b = (cx - half, cy), (cx + half, cy)
            else:
                a, b = (cx, cy - half), (cx, cy + half)
            dot_paths.append(np.array([a, b], dtype=np.float32))

    paths: List[np.ndarray] = []
    used_mode = mode
    if mode == "skeleton":
        try:
            from skimage.morphology import skeletonize
            sk = skeletonize(ink > 0)
            dist = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 3)
            for pl in _skeleton_polylines(sk, dist):
                paths.append(np.array(pl, dtype=np.int32).reshape(-1, 1, 2))
            if blob and sk.any():                  # moti jagah (bulb) ka outline jodo
                wid = 2.0 * float(np.median(dist[sk]))
                cover = np.zeros(ink.shape, np.uint8)
                rad = int(round(wid * 0.5 * 1.15))
                for yy, xx in zip(*np.nonzero(sk)):
                    cv2.circle(cover, (int(xx), int(yy)), rad, 255, -1)
                resid = (((ink > 0) & (cover == 0)).astype(np.uint8)) * 255
                resid = cv2.morphologyEx(resid, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
                nb, lb, st, _c = cv2.connectedComponentsWithStats(resid, 8)
                for i in range(1, nb):
                    if st[i, cv2.CC_STAT_AREA] >= blob_area * wid * wid:
                        cs, _h = cv2.findContours((lb == i).astype(np.uint8), cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_NONE)
                        for cc in cs:
                            paths.append(np.vstack([cc, cc[:1]]).astype(np.int32))
        except ImportError:
            used_mode = "contour"
    if used_mode == "contour":
        cnts, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            paths.append(np.vstack([c, c[:1]]))

    k = 25.4 / dpi
    eps, min_len = eps_px * up, min_len_px * up
    segs: List[Segment] = []
    for dp in dot_paths:                           # dots seedhe, filter ke bina
        (ax, ay), (bx, by) = dp
        segs.append((((x0 + ax / up) * k, (y0 + ay / up) * k),
                     ((x0 + bx / up) * k, (y0 + by / up) * k)))
    for p in paths:
        if cv2.arcLength(p, False) < min_len:
            continue
        p = cv2.approxPolyDP(p, eps, False).reshape(-1, 2)
        for (ax, ay), (bx, by) in zip(p[:-1], p[1:]):
            segs.append((((x0 + ax / up) * k, (y0 + ay / up) * k),
                         ((x0 + bx / up) * k, (y0 + by / up) * k)))
    return segs
