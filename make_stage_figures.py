"""
make_stage_figures.py
---------------------
Poster ke liye: pipeline ke HAR step ka intermediate output image ke roop me save karta hai
(jaise chemistry me reactant -> intermediate -> product), aur ek montage (arrows ke saath) banata hai.

Chalane ka tarika (project ke main folder me, venv active):
    python3 make_stage_figures.py HINDI-2.jpeg                # trace pipeline ke saare stages
    python3 make_stage_figures.py quote.png --ocr             # OCR pipeline ke stages bhi (RapidOCR chahiye)
    python3 make_stage_figures.py --montage-only stages_out/HINDI-2   # sirf montage dobara
    python3 make_stage_figures.py HINDI-2.jpeg --zoom 0.1,0.1,0.5,0.35  # har step me ek hi jagah ka close-up
        (--zoom x0,y0,x1,y1 = image ka hissa, 0 se 1 ke fraction me; left-top se)

Output folder: stages_out/<image-naam>/  (01_input.png, 02_..., ..., pipeline_stages.png)
Team ke preprocessing steps (deskew, threshold, k-means) ki images isi folder me
`00a_deskew.png`, `00b_kmeans.png` jaise naam se daalo, phir --montage-only chalao;
montage naam ke order me sab jod dega.
"""
from __future__ import annotations
import argparse, os, re, sys, random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.abspath(__file__))
for sub in ("ui_export_module", "ocr_module"):
    p = os.path.join(BASE, sub)
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------- helpers
def _font(size):
    for path in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _save(arr_or_img, folder, name):
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(arr_or_img)
    path = os.path.join(folder, name)
    img.convert("RGB").save(path)
    print("  saved", path)


def _render_polylines(polys, size, colors=None, width=2):
    """polys: list of [(x,y),...] in pixel coords of an image of `size`=(w,h)."""
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    for i, p in enumerate(polys):
        col = colors[i % len(colors)] if colors else "black"
        if len(p) == 1:
            x, y = p[0]; d.ellipse([x - 1, y - 1, x + 1, y + 1], fill=col)
        else:
            d.line([tuple(q) for q in p], fill=col, width=width)
    return img


# ---------------------------------------------------------------- TRACE pipeline stages
def trace_stages(image_path, out, remove_ruled=False):
    from trace_page import ink_mask_page, trace_page
    from trace_export import _skeleton_polylines
    from skimage.morphology import skeletonize

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    H, W = img.shape[:2]
    up = 3
    _save(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), out, "01_input.png")

    # 2) roshni barabar (background se divide) -> kagaz ki ruled lines / shadow bahar
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = max(15, int(W / 30)) | 1
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 2)
    norm = np.clip(gray / np.maximum(bg, 1.0), 0, 1.0)
    _save((norm * 255).astype(np.uint8), out, "02_illumination_flattened.png")

    # 3) ink mask (binary)
    ink = ink_mask_page(img, 0.60, remove_ruled)
    _save(255 - ink, out, "03_ink_mask.png")

    # 4) chhote dots (bindu, nukta, full stop) alag
    inkU = cv2.resize(ink, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
    inkU = ((inkU > 127).astype(np.uint8)) * 255
    n, lab, st, _c = cv2.connectedComponentsWithStats((inkU > 0).astype(np.uint8), 8)
    dot_lim = max(4.0, 0.008 * H) * up
    vis = np.full((inkU.shape[0], inkU.shape[1], 3), 255, np.uint8)
    for i in range(1, n):
        cw, ch = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT]
        small = max(cw, ch) < dot_lim
        vis[lab == i] = (230, 40, 40) if small else (60, 60, 60)
        if small:
            inkU[lab == i] = 0
    _save(vis, out, "04_dots_red_strokes_grey.png")

    # 5) skeleton (ek pixel chaudi centerline)
    sk = skeletonize(inkU > 0)
    skimg = np.full(sk.shape + (3,), 255, np.uint8)
    kk = max(3, int(round(inkU.shape[0] / 420 * 1.8))) | 1       # tile me chhota hone par bhi dikhe
    skimg[cv2.dilate(sk.astype(np.uint8), np.ones((kk, kk), np.uint8)) > 0] = (20, 20, 20)
    _save(skimg, out, "05_skeleton.png")

    # 6) polylines (har alag lakeer alag rang) -> simplify ke baad
    dist = cv2.distanceTransform((inkU > 0).astype(np.uint8), cv2.DIST_L2, 3)
    random.seed(3)
    lw = max(3, int(round(inkU.shape[0] / 420 * 1.8)))            # line width: montage tile ke hisaab se
    palette = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (255, 127, 14), (148, 103, 189), (140, 86, 75)]
    polys = []
    for pl in _skeleton_polylines(sk, dist):
        p = np.array(pl, dtype=np.int32).reshape(-1, 1, 2)
        if cv2.arcLength(p, False) < 2.0 * up:
            continue
        p = cv2.approxPolyDP(p, 0.7 * up, False).reshape(-1, 2)
        polys.append([tuple(q) for q in p])
    random.shuffle(palette)
    _save(_render_polylines(polys, (inkU.shape[1], inkU.shape[0]), palette, width=lw), out, "06_polylines_colored.png")

    # 7) final plot-ready (black strokes) + 8) original ke upar overlay
    segs, (pw, ph) = trace_page(image_path, remove_ruled=remove_ruled)
    scale = (W * up) / pw          # mm -> pixel of the upscaled canvas
    final = Image.new("RGB", (inkU.shape[1], inkU.shape[0]), "white")
    d = ImageDraw.Draw(final)
    for a, b in segs:
        d.line([a[0] * scale, a[1] * scale, b[0] * scale, b[1] * scale], fill="black", width=lw)
    _save(final, out, "07_plot_ready.png")
    over = Image.open(image_path).convert("RGB").resize(final.size, Image.LANCZOS)
    od = ImageDraw.Draw(over)
    for a, b in segs:
        od.line([a[0] * scale, a[1] * scale, b[0] * scale, b[1] * scale], fill=(230, 30, 30), width=max(2, lw - 1))
    _save(over, out, "08_overlay_on_original.png")
    print(f"  trace: {len(segs)} segments, page {pw:.1f} x {ph:.1f} mm")


# ---------------------------------------------------------------- OCR pipeline stages (optional)
def ocr_stages(image_path, out):
    from baseline import run_baseline_ocr
    from line_merge import merge_line_boxes
    from pipeline import run_pipeline
    from svg_export import compile_strokes

    def bbox(b):
        if isinstance(b[0], (list, tuple)):
            xs = [p[0] for p in b]; ys = [p[1] for p in b]
            return [min(xs), min(ys), max(xs), max(ys)]
        return list(b)

    base = Image.open(image_path).convert("RGB")
    raw = run_baseline_ocr(image_path)
    a = base.copy(); d = ImageDraw.Draw(a)
    for r in raw:
        d.rectangle(bbox(r["box"]), outline=(220, 30, 30), width=3)
    _save(a, out, "A1_ocr_detected_boxes.png")

    merged = merge_line_boxes(raw)
    b = base.copy(); d = ImageDraw.Draw(b)
    for r in merged:
        d.rectangle(bbox(r["box"]), outline=(30, 160, 60), width=3)
    _save(b, out, "A2_lines_merged.png")

    doc = run_pipeline(image_path, use_finetuned=False)
    # A3: image ke daayein side ek saaf panel me padha hua text (overlap nahi hota)
    side = int(base.width * 1.1)
    c = Image.new("RGB", (base.width + side, base.height), "white")
    c.paste(base, (0, 0)); d = ImageDraw.Draw(c)
    els = doc["text_elements"]
    fs = max(14, int(min((e["bounding_box_pixel"][3] - e["bounding_box_pixel"][1]) for e in els) * 0.8)) if els else 16
    f = _font(fs)
    for e in els:
        x0, y0, x1, y1 = e["bounding_box_pixel"]
        d.rectangle([x0, y0, x1, y1], outline=(30, 90, 200), width=3)
        cy = (y0 + y1) // 2
        d.line([x1, cy, base.width + 10, cy], fill=(150, 170, 220), width=2)
        d.text((base.width + 16, cy - fs // 2 - 2), e["text"], fill=(30, 60, 170), font=f)
    d.line([base.width, 0, base.width, base.height], fill=(180, 180, 180), width=2)
    _save(c, out, "A3_recognized_text.png")

    segs = compile_strokes(doc)
    if segs:
        xs = [p[0] for s in segs for p in s]; ys = [p[1] for s in segs for p in s]
        sc = 8
        img = Image.new("RGB", (int((max(xs) - min(xs) + 16) * sc), int((max(ys) - min(ys) + 16) * sc)), "white")
        d = ImageDraw.Draw(img)
        for p, q in segs:
            d.line([(p[0] - min(xs) + 8) * sc, (p[1] - min(ys) + 8) * sc,
                    (q[0] - min(xs) + 8) * sc, (q[1] - min(ys) + 8) * sc], fill="black", width=3)
        _save(img, out, "A4_hershey_strokes.png")
    print(f"  ocr: {len(doc['text_elements'])} text lines, {len(segs)} segments")


# ---------------------------------------------------------------- montage
def _title(fname):
    stem = os.path.splitext(fname)[0]
    stem = re.sub(r"^[0-9A-Za-z]{1,3}_", "", stem)       # '01_' / 'A1_' / '00a_' hatao
    return stem.replace("_", " ").capitalize()


def build_montage(folder, cols=4, tile_h=420, zoom=None):
    """tile_h badhao (jaise 1400) to montage high-resolution banta hai; label, gap, arrow sab saath scale hote hain."""
    files = sorted(f for f in os.listdir(folder)
                   if f.lower().endswith(".png") and not f.startswith("pipeline_stages"))
    if not files:
        print("koi png nahi mili:", folder); return
    s = tile_h / 420.0
    ft = _font(max(14, int(26 * s)))
    tiles = []
    for i, f in enumerate(files, 1):
        im = Image.open(os.path.join(folder, f)).convert("RGB")
        if zoom:                                              # sabhi stages me ek hi hissa kaato
            zx0, zy0, zx1, zy1 = zoom
            im = im.crop((int(zx0 * im.width), int(zy0 * im.height), int(zx1 * im.width), int(zy1 * im.height)))
        w = max(1, int(im.width * tile_h / im.height))
        tiles.append((im.resize((w, tile_h), Image.LANCZOS), f"{i}. {_title(f)}"))
    tw = max(t.width for t, _ in tiles)
    gap, label_h, pad = int(70 * s), int(48 * s), int(24 * s)
    lw, ah = max(2, int(5 * s)), int(12 * s)                  # arrow line width, arrowhead half-width
    rows = (len(tiles) + cols - 1) // cols
    W = cols * tw + (cols - 1) * gap + 2 * pad
    H = rows * (tile_h + label_h + gap) + pad
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    for i, (t, label) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (tw + gap) + (tw - t.width) // 2
        y = pad + r * (tile_h + label_h + gap)
        d.rectangle([x - 2, y + label_h - 2, x + t.width + 1, y + label_h + tile_h + 1],
                    outline=(150, 150, 150), width=max(2, int(2 * s)))
        sheet.paste(t, (x, y + label_h))
        d.text((x, y + int(8 * s)), label, fill=(20, 20, 20), font=ft)
        if i < len(tiles) - 1:                                 # arrow -> agle step ki taraf
            m, hd = int(8 * s), int(24 * s)
            if c < cols - 1:
                ax = x + t.width + m; ay = y + label_h + tile_h // 2
                d.line([ax, ay, ax + gap - hd, ay], fill=(60, 60, 60), width=lw)
                d.polygon([(ax + gap - hd, ay - ah), (ax + gap - hd, ay + ah), (ax + gap - m // 2, ay)], fill=(60, 60, 60))
            else:                                              # row ke end me neeche ka arrow
                ax = pad + (W - 2 * pad) // 2; ay = y + label_h + tile_h + m
                d.line([ax, ay, ax, ay + gap - hd], fill=(60, 60, 60), width=lw)
                d.polygon([(ax - ah, ay + gap - hd), (ax + ah, ay + gap - hd), (ax, ay + gap - m // 2)], fill=(60, 60, 60))
    path = os.path.join(folder, "pipeline_stages_zoom.png" if zoom else "pipeline_stages.png")
    sheet.save(path, dpi=(300, 300))
    print("montage:", path, sheet.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--ocr", action="store_true", help="OCR pipeline ke stages bhi banao")
    ap.add_argument("--montage-only", metavar="FOLDER")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--zoom", metavar="x0,y0,x1,y1", help="close-up: image ka hissa (0-1 fraction)")
    ap.add_argument("--tile-h", type=int, default=420,
                    help="har tile ki unchai (pixel). Poster ke liye 1400 ya 1800 do (default 420)")
    ap.add_argument("--remove-ruled", action="store_true",
                    help="notebook ki gehri ruled lines hatao (diagram me mat lagao)")
    a = ap.parse_args()
    zoom = tuple(float(v) for v in a.zoom.split(",")) if a.zoom else None
    if zoom and (len(zoom) != 4 or not all(0 <= v <= 1 for v in zoom) or zoom[0] >= zoom[2] or zoom[1] >= zoom[3]):
        ap.error("--zoom ko x0,y0,x1,y1 (0 se 1 ke beech, x0<x1, y0<y1) me do")
    if a.montage_only:
        build_montage(a.montage_only, a.cols, a.tile_h, zoom=zoom); return
    if not a.image:
        ap.error("image ka path do")
    img_abs = os.path.abspath(a.image)
    name = os.path.splitext(os.path.basename(a.image))[0]
    out = os.path.join(BASE, "stages_out", name)
    os.makedirs(out, exist_ok=True)
    print("TRACE stages ->", out)
    trace_stages(img_abs, out, a.remove_ruled)
    if a.ocr:
        os.chdir(os.path.join(BASE, "ocr_module"))
        print("OCR stages ->", out)
        try:
            ocr_stages(img_abs, out)
        except Exception as e:
            import traceback
            print("  OCR stages skip hue:", type(e).__name__, e)
            traceback.print_exc()
            print("  (python:", sys.executable, ")")
        os.chdir(BASE)
    build_montage(out, a.cols, a.tile_h)
    if zoom:
        build_montage(out, a.cols, a.tile_h, zoom=zoom)


if __name__ == "__main__":
    main()