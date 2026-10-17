"""
pipeline.py
-----------
The single entry point Member 4 needs to import. Runs OCR (baseline or
fine-tuned) on an image and converts the raw predictions into the
Intermediate Document JSON Schema (schemas/document_schema.json) that
Member 4's review UI and SVG exporter consume directly.

    from pipeline import run_pipeline
    doc = run_pipeline("test_images/sample_dark.jpg", use_finetuned=True)

CHANGES (patch): tute hue boxes ek line me merge (line_merge.py) aur
OCR ke baad vocab-based spelling correction (postprocess.py).
"""

from __future__ import annotations
import os
import json
import tempfile
from typing import List, Dict, Any, Optional

from PIL import Image, ImageOps

from baseline import run_baseline_ocr
from finetune import load_finetuned_model, run_finetuned_ocr, DEFAULT_OUTPUT_DIR
from line_merge import merge_line_boxes
from postprocess import load_vocab, correct_text

# Assumed scan/capture resolution used to convert pixel measurements into
# physical millimeters. Override per-document if the real DPI is known.
DEFAULT_DPI = 300.0


def _px_to_mm(pixels: float, dpi: float = DEFAULT_DPI) -> float:
    return (pixels / dpi) * 25.4


def _quad_to_bbox(quad: List[List[float]]) -> List[float]:
    """Collapse a 4-point quadrilateral (from PaddleOCR) into [xmin,ymin,xmax,ymax]."""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return [min(xs), min(ys), max(xs), max(ys)]


def _predictions_to_text_elements(
    predictions: List[Dict[str, Any]], dpi: float, spell_fix: bool = True
) -> List[Dict[str, Any]]:
    vocab = load_vocab() if spell_fix else []
    elements = []
    for i, pred in enumerate(predictions):
        bbox_px = _quad_to_bbox(pred["box"]) if isinstance(pred["box"][0], list) else pred["box"]
        x_min, y_min, x_max, y_max = bbox_px
        raw_text = pred["text"]
        # vocab sirf kam-confidence text par; accha OCR text na chhedo
        text = (correct_text(raw_text, vocab)
                if (vocab and float(pred["confidence"]) < 0.9) else raw_text.strip())
        elements.append({
            "id": f"txt_{i + 1:02d}",
            "text": text,
            "raw_text": raw_text,
            "x_mm": round(_px_to_mm(x_min, dpi), 2),
            "y_mm": round(_px_to_mm(y_min, dpi), 2),
            "font_size_mm": round(_px_to_mm(y_max - y_min, dpi), 2),
            "confidence": round(float(pred["confidence"]), 3),
            "bounding_box_pixel": [round(v, 1) for v in bbox_px],
        })
    return elements



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


def _detect_best(image_path: str, scale: int = 2, ocr_fn=None):
    """Original aur preprocessed (autocontrast + upscale) dono par OCR chalao,
    jisme zyada confident text mile wahi lo. Boxes original pixel coords me wapas."""
    fn = ocr_fn or run_baseline_ocr
    plain = fn(image_path)
    up = fn(_preprocess_for_ocr(image_path, scale))
    for d in up:
        d["box"] = _scale_box(d["box"], 1.0 / scale)
    use_up = _ocr_score(up) > _ocr_score(plain)
    print(f"[pipeline] OCR: plain={_ocr_score(plain):.1f} preprocessed={_ocr_score(up):.1f} -> "
          f"{'preprocessed' if use_up else 'plain'}")
    return up if use_up else plain


def _load_model(use_finetuned: bool, checkpoint_dir: str):
    if not use_finetuned:
        return None  # baseline.run_baseline_ocr manages its own cached engine
    return load_finetuned_model(checkpoint_dir)


def run_pipeline(
    image_path: str,
    use_finetuned: bool = True,
    checkpoint_dir: str = DEFAULT_OUTPUT_DIR,
    dpi: float = DEFAULT_DPI,
    document_id: Optional[str] = None,
    spell_fix: bool = True,
    merge_lines: bool = True,
    ocr_fn=None,
) -> Dict[str, Any]:
    """
    Run OCR on a single image and return a dict matching
    schemas/document_schema.json.

    use_finetuned : True  -> detection baseline se, recognition fine-tuned TrOCR se
                    False -> poora PaddleOCR baseline
    spell_fix     : vocab.txt se spelling correction
    merge_lines   : ek line ke tute boxes jodna
    """
    doc_id = document_id or os.path.splitext(os.path.basename(image_path))[0]

    with Image.open(image_path) as img:
        width_mm = round(_px_to_mm(img.width, dpi), 1)
        height_mm = round(_px_to_mm(img.height, dpi), 1)
        px_w, px_h = img.width, img.height

    detections = _detect_best(image_path, ocr_fn=ocr_fn)
    if merge_lines:
        detections = merge_line_boxes(detections)

    if use_finetuned:
        processor, model = _load_model(use_finetuned, checkpoint_dir)
        for det in detections:
            crop_path = _crop_to_temp(image_path, det["box"])
            det["text"] = run_finetuned_ocr(crop_path, processor, model)
    predictions = detections

    text_elements = _predictions_to_text_elements(predictions, dpi, spell_fix)

    return {
        "document_id": doc_id,
        "page_dimensions_mm": {"width": width_mm, "height": height_mm},
        "text_elements": text_elements,
        "image_px": {"width": px_w, "height": px_h},
        "dpi": dpi,
    }


def _crop_to_temp(image_path: str, quad: List[List[float]]) -> str:
    """Crop one detected text region to a temp file for recognition-only inference."""
    import tempfile

    bbox = _quad_to_bbox(quad) if isinstance(quad[0], list) else quad
    with Image.open(image_path) as img:
        crop = img.crop(tuple(bbox))
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        crop.save(tmp.name)
        return tmp.name


def save_document_json(doc: Dict[str, Any], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <image_path> [--baseline]")
        sys.exit(1)

    image_path = sys.argv[1]
    use_finetuned = "--baseline" not in sys.argv

    document = run_pipeline(image_path, use_finetuned=use_finetuned)

    out_path = os.path.join("outputs", f"{document['document_id']}.json")
    save_document_json(document, out_path)
    print(f"Wrote {out_path}")
    print(json.dumps(document, indent=2, ensure_ascii=False))