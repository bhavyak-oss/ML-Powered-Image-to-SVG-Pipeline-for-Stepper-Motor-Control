import sys, os

base_dir = os.path.dirname(os.path.abspath(__file__))
ocr_dir = os.path.join(base_dir, "ocr_module")
ui_dir = os.path.join(base_dir, "ui_export_module")
sys.path.insert(0, ocr_dir)
sys.path.insert(1, ui_dir)

IMAGE = "T5.png"          # <-- apni image ka naam
img_path = os.path.join(base_dir, IMAGE)
os.chdir(ocr_dir)

from PIL import Image, ImageDraw
from baseline import run_baseline_ocr
from line_merge import merge_line_boxes
from pipeline import run_pipeline
from svg_export import layout_elements


def bbox(box):
    if isinstance(box[0], (list, tuple)):
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        return [min(xs), min(ys), max(xs), max(ys)]
    return list(box)


raw = run_baseline_ocr(img_path)
print(f"\nSTAGE 1 - OCR detection: {len(raw)} boxes")
for d in raw:
    print(f"  {d['text']!r:40} conf={d['confidence']:.2f} box={[round(v) for v in bbox(d['box'])]}")

merged = merge_line_boxes(raw)
print(f"\nSTAGE 2 - line merge: {len(merged)} boxes")
for d in merged:
    print(f"  {d['text']!r:40} box={[round(v) for v in bbox(d['box'])]}")

doc = run_pipeline(img_path, use_finetuned=False)
print(f"\nSTAGE 3 - pipeline text: {len(doc['text_elements'])} elements")
for e in doc["text_elements"]:
    print(f"  {e['raw_text']!r} -> {e['text']!r}")

placed = layout_elements(doc)
print(f"\nSTAGE 4 - layout: {len(placed)} lines placed")
for p in placed:
    print(f"  y={p['y']:.1f} fs={p['fs']:.1f} {p['text']!r}")

img = Image.open(img_path).convert("RGB")
dr = ImageDraw.Draw(img)
for d in raw:
    dr.rectangle(bbox(d["box"]), outline="red", width=3)
for d in merged:
    dr.rectangle(bbox(d["box"]), outline="green", width=2)
out = os.path.join(base_dir, "outputs", "debug_boxes.png")
img.save(out)
print("\nBoxes picture:", out)