"""
trace_page.py
-------------
TRACE MODE: poori image ko seedha trace karke plotter ke liye line segments banata hai.
Koi OCR nahi, koi font nahi -> kisi bhi language / handwriting / sketch pe chalta hai.

Steps:
  1. Roshni barabar karo (background se divide) -> ink aur kagaz ki ruled lines alag.
  2. Ink mask banao; lambi horizontal lines (agar bachi hon) hata do.
  3. Chhote dots (bindu, nukta, full stop) alag; baaki ka skeleton (ek patli lakeer).
  4. Poori image ko `fit_width_mm` x `fit_height_mm` ke andar scale karo (plotter bed ke hisaab se;
     160 x 180 mm mera andaza hai, asli bed size sir se pooch kar badlo).

Output: (segments_in_mm, (page_w_mm, page_h_mm))
"""
from __future__ import annotations
from typing import List, Tuple

import cv2
import numpy as np

from trace_export import _skeleton_polylines

Point = Tuple[float, float]
Segment = Tuple[Point, Point]


def _line_kernel(length: int, angle_deg: float) -> np.ndarray:
    """1 pixel patli, `angle_deg` pe jhuki seedhi lakeer ka kernel (0 = horizontal, 90 = vertical)."""
    k = np.zeros((length, length), np.uint8)
    c = (length - 1) / 2.0
    dx = np.cos(np.radians(angle_deg)) * c
    dy = -np.sin(np.radians(angle_deg)) * c
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1, 1)
    return k


def remove_ruled_lines(ink: np.ndarray, h_frac: float = 0.12, v_frac: float = 0.35,
                       max_tilt: int = 6) -> np.ndarray:
    """Notebook ki ruled lines + margin line hatao.

    Purana tareeka sirf bilkul seedhi, poori chaudai ki line hatata tha. Photo me lines thodi tedhi
    hoti hain aur akshar unhe todte hain, isliye ab:
      * horizontal lines: -max_tilt..+max_tilt degree tak ke kai tedhe kernel (lambai = chaudai ka h_frac)
      * vertical margin line: 90 degree ke aas-paas, lambai = unchai ka v_frac
    Hatane ke baad thodi chaudai (dilate) bhi hatate hain. Jahan akshar line ko kaatta hai wahan chhota gap rahega.
    DHYAN: technical diagram ki lambi seedhi lakeeren (dimension/axis lines) bhi hat sakti hain, isliye
    ye sirf remove_ruled=True par chalta hai."""
    h, w = ink.shape
    lines = np.zeros_like(ink)
    # detection ke liye chhote gaps band karo (line wobbly/toot-toot ke hoti hai); ink khud nahi badalti
    det_h = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((1, 15), np.uint8))
    det_v = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((15, 1), np.uint8))
    lh = max(25, int(w * h_frac)) | 1
    for a in range(-max_tilt, max_tilt + 1):
        lines = cv2.bitwise_or(lines, cv2.morphologyEx(det_h, cv2.MORPH_OPEN, _line_kernel(lh, a)))
    lv = max(25, int(h * v_frac)) | 1
    for a in range(90 - max_tilt, 90 + max_tilt + 1):
        lines = cv2.bitwise_or(lines, cv2.morphologyEx(det_v, cv2.MORPH_OPEN, _line_kernel(lv, a)))
    lines = cv2.dilate(lines, np.ones((5, 5), np.uint8))
    kept = cv2.subtract(ink, lines)
    # akshar ka jo stroke line ko paar karta hai (upar aur neeche ink hai), wo wapas jodo,
    # warna line hatane se akshar me gap aa jaata hai. Sirf line wali patti ke andar ke pixel wapas aate hain.
    bridge = cv2.morphologyEx(kept, cv2.MORPH_CLOSE, np.ones((15, 1), np.uint8))
    ink = cv2.bitwise_or(kept, cv2.bitwise_and(cv2.bitwise_and(bridge, ink), lines))
    # 2nd pass: bache hue lambe-patle tukde (tuti line ke) hatao. Akshar ke tukde moti/chhote hote hain.
    n, lab, st, _c = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    long_ = max(30, int(w * 0.04))         # lamba = ~40 px (1041 px chaudi image me)
    for i in range(1, n):
        cw, ch = int(st[i, cv2.CC_STAT_WIDTH]), int(st[i, cv2.CC_STAT_HEIGHT])
        long_side, short_side = max(cw, ch), max(1, min(cw, ch))
        if long_side >= long_ and short_side <= 14 and long_side / short_side >= 8:
            ink[lab == i] = 0
    return ink


def ink_mask_page(img_bgr: np.ndarray, thr: float = 0.60, remove_ruled: bool = False) -> np.ndarray:
    """Ink = background ke mukable kaafi gehra pixel. Ruled lines halki hoti hain, isliye bahar."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape
    k = max(15, int(w / 30)) | 1
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 2)
    norm = gray / np.maximum(bg, 1.0)
    ink = ((norm < thr).astype(np.uint8)) * 255
    if remove_ruled:
        ink = remove_ruled_lines(ink)
    return ink


def trace_page(image_path: str, fit_width_mm: float = 160.0, fit_height_mm: float = 180.0, upscale: int = 3,
               thr: float = 0.60, eps_px: float = 0.7, min_len_px: float = 2.0,
               dot_px: float = 0.0, speck_px2: float = 5.0, remove_ruled: bool = False):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not readable: {image_path}")
    H, W = img.shape[:2]
    up = max(1, int(upscale))
    ink = ink_mask_page(img, thr, remove_ruled)
    if up > 1:
        ink = cv2.resize(ink, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
        ink = ((ink > 127).astype(np.uint8)) * 255

    mm = min(fit_width_mm / float(W), fit_height_mm / float(H))   # mm per ORIGINAL pixel (bed me fit)
    dot_lim = (dot_px if dot_px > 0 else max(4.0, 0.008 * H)) * up
    segs: List[Segment] = []

    # chhote dots alag, specks (noise) gayab
    n, lab, st, cent = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    for i in range(1, n):
        cw, ch, area = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT], st[i, cv2.CC_STAT_AREA]
        if max(cw, ch) < dot_lim:
            ink[lab == i] = 0
            if area >= speck_px2 * up * up:
                cx, cy = cent[i]
                half = max(cw, ch, 1.5 * up) * 0.4
                a, b = ((cx - half, cy), (cx + half, cy)) if cw >= ch else ((cx, cy - half), (cx, cy + half))
                segs.append((((a[0] / up) * mm, (a[1] / up) * mm), ((b[0] / up) * mm, (b[1] / up) * mm)))

    from skimage.morphology import skeletonize
    sk = skeletonize(ink > 0)
    dist = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 3)
    eps, min_len = eps_px * up, min_len_px * up
    for pl in _skeleton_polylines(sk, dist):
        p = np.array(pl, dtype=np.int32).reshape(-1, 1, 2)
        if cv2.arcLength(p, False) < min_len:
            continue
        p = cv2.approxPolyDP(p, eps, False).reshape(-1, 2)
        for (ax, ay), (bx, by) in zip(p[:-1], p[1:]):
            segs.append((((ax / up) * mm, (ay / up) * mm), ((bx / up) * mm, (by / up) * mm)))
    return segs, (W * mm, H * mm)