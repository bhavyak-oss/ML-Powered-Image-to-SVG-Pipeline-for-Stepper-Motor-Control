"""pipeline.py me 3 chhote badlav: ocr_fn (language ke liye), image_px aur dpi output me.
Do baar chalane se kuch nahi bigadta."""
path = "ocr_module/pipeline.py"
s = open(path, encoding="utf-8").read()

edits = [
    ("def _detect_best(image_path: str, scale: int = 2):",
     "def _detect_best(image_path: str, scale: int = 2, ocr_fn=None):"),
    ("    plain = run_baseline_ocr(image_path)\n    up = run_baseline_ocr(_preprocess_for_ocr(image_path, scale))",
     "    fn = ocr_fn or run_baseline_ocr\n    plain = fn(image_path)\n    up = fn(_preprocess_for_ocr(image_path, scale))"),
    ("    merge_lines: bool = True,\n) -> Dict[str, Any]:",
     "    merge_lines: bool = True,\n    ocr_fn=None,\n) -> Dict[str, Any]:"),
    ("        height_mm = round(_px_to_mm(img.height, dpi), 1)\n",
     "        height_mm = round(_px_to_mm(img.height, dpi), 1)\n        px_w, px_h = img.width, img.height\n"),
    ("    detections = _detect_best(image_path)\n",
     "    detections = _detect_best(image_path, ocr_fn=ocr_fn)\n"),
    ('        "text_elements": text_elements,\n    }',
     '        "text_elements": text_elements,\n        "image_px": {"width": px_w, "height": px_h},\n        "dpi": dpi,\n    }'),
]
for old, new in edits:
    if new in s:
        continue
    assert s.count(old) == 1, f"pattern not found exactly once:\n{old[:70]}"
    s = s.replace(old, new)
open(path, "w", encoding="utf-8").write(s)
print("patched", path)
