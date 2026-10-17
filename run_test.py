import sys
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
ocr_dir = os.path.join(base_dir, "ocr_module")
ui_dir = os.path.join(base_dir, "ui_export_module")

# ocr_module pehle (flat imports: baseline, finetune, line_merge, postprocess)
# preprocessing_module ko path me mat daalna (usme bhi pipeline.py hai)
sys.path.insert(0, ocr_dir)
sys.path.insert(1, ui_dir)

from pipeline import run_pipeline
from svg_export import export_plot_ready

USE_FINETUNED = False   # True karke dekho agar fine-tuned checkpoint bana hai
IMAGE = "T5.png"

os.makedirs(os.path.join(base_dir, "outputs"), exist_ok=True)
img_path = os.path.join(base_dir, IMAGE)
output_svg = os.path.join(base_dir, "outputs", "final_output.svg")

os.chdir(ocr_dir)  # finetune checkpoint ka relative path yahin se resolve hota hai
doc = run_pipeline(img_path, use_finetuned=USE_FINETUNED)

print("Detected elements (raw -> corrected):")
for el in doc["text_elements"]:
    print(f"{el['raw_text']!r} -> {el['text']!r} | x={el['x_mm']} y={el['y_mm']} h={el['font_size_mm']}")

export_plot_ready(doc, output_svg)
print("SVG generated successfully! Check: outputs/final_output.svg")