import re, sys

def patch(path, edits):
    s = open(path, encoding="utf-8").read()
    for old, new in edits:
        if new in s:          # already applied
            continue
        assert s.count(old) == 1, f"{path}: pattern not found exactly once:\n{old[:80]}"
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)
    print("patched", path)

# ---------------- pipeline.py ----------------
PRE = '''
def _preprocess_for_ocr(image_path: str, scale: int = 2) -> str:
    """Autocontrast + upscale: halki/krim background wali printed images ke liye."""
    with Image.open(image_path) as img:
        g = ImageOps.autocontrast(img.convert("L"), cutoff=2)
        g = g.resize((g.width * scale, g.height * scale), Image.LANCZOS)
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        g.save(tmp.name)
        return tmp.name


def _scale_box(box, factor: float):
    if isinstance(box[0], (list, tuple)):
        return [[p[0] * factor, p[1] * factor] for p in box]
    return [v * factor for v in box]


def _ocr_score(dets) -> float:
    return sum(len(str(d.get("text", "")).strip()) * float(d.get("confidence", 0.0)) for d in dets)


def _detect_best(image_path: str, scale: int = 2):
    """Original aur preprocessed (autocontrast + upscale) dono par OCR chalao,
    jisme zyada confident text mile wahi lo. Boxes original pixel coords me wapas."""
    plain = run_baseline_ocr(image_path)
    up = run_baseline_ocr(_preprocess_for_ocr(image_path, scale))
    for d in up:
        d["box"] = _scale_box(d["box"], 1.0 / scale)
    use_up = _ocr_score(up) > _ocr_score(plain)
    print(f"[pipeline] OCR: plain={_ocr_score(plain):.1f} preprocessed={_ocr_score(up):.1f} -> "
          f"{'preprocessed' if use_up else 'plain'}")
    return up if use_up else plain

'''
patch("ocr_module/pipeline.py", [
    ("import json\nfrom typing", "import json\nimport tempfile\nfrom typing"),
    ("from PIL import Image\n", "from PIL import Image, ImageOps\n"),
    ("def _load_model(use_finetuned", PRE + "\ndef _load_model(use_finetuned"),
    ("    detections = run_baseline_ocr(image_path)\n    if merge_lines:",
     "    detections = _detect_best(image_path)\n    if merge_lines:"),
    ("        text = correct_text(raw_text, vocab) if vocab else raw_text\n",
     "        # vocab sirf kam-confidence text par; accha OCR text na chhedo\n"
     "        text = (correct_text(raw_text, vocab)\n"
     "                if (vocab and float(pred[\"confidence\"]) < 0.9) else raw_text.strip())\n"),
])

# ---------------- hershey_font.py ----------------
patch("ui_export_module/hershey_font.py", [
    ('    "/": [[_BL, _TR]],\n}',
     '    "/": [[_BL, _TR]],\n'
     '    \'"\': [[(0.8, 7.0), (0.8, 5.5)], [(2.2, 7.0), (2.2, 5.5)]],\n'
     '    "?": [[(0.0, 6.0), (0.0, 7.0), (3.0, 7.0), (3.0, 4.0), (1.5, 4.0), (1.5, 2.5)], [(1.4, 0.0), (1.6, 0.0)]],\n'
     '    "!": [[(1.5, 7.0), (1.5, 2.0)], [(1.4, 0.0), (1.6, 0.0)]],\n'
     '    "(": [[(2.2, 7.0), (1.0, 5.5), (1.0, 1.5), (2.2, 0.0)]],\n'
     '    ")": [[(0.8, 7.0), (2.0, 5.5), (2.0, 1.5), (0.8, 0.0)]],\n}'),
    ("_UNSUPPORTED_WARNED: set = set()\n",
     "_UNSUPPORTED_WARNED: set = set()\n"
     "_ALIASES = {'\\u201c': '\"', '\\u201d': '\"', '\\u2018': \"'\", '\\u2019': \"'\",\n"
     "            '\\u2013': '-', '\\u2014': '-'}\n"),
    ("    glyph = STROKE_FONT.get(char.upper())",
     "    char = _ALIASES.get(char, char)\n    glyph = STROKE_FONT.get(char.upper())"),
])

# ---------------- svg_export.py ----------------
patch("ui_export_module/svg_export.py", [
    ('    els = [dict(e) for e in doc["text_elements"] if str(e.get("text", "")).strip()]\n',
     '    els = [dict(e, text=str(e["text"]).strip()) for e in doc["text_elements"]\n'
     '           if str(e.get("text", "")).strip()]\n'),
])
