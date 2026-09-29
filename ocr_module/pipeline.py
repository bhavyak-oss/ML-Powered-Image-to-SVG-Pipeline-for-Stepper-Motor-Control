"""
pipeline.py
-----------
The single entry point Member 4 needs to import. Runs OCR (baseline or
fine-tuned) on an image and converts the raw predictions into the
Intermediate Document JSON Schema (schemas/document_schema.json) that
Member 4's review UI and SVG exporter consume directly.

    from pipeline import run_pipeline
    doc = run_pipeline("test_images/sample_dark.jpg", use_finetuned=True)
"""

from __future__ import annotations
import os
import json
from typing import List, Dict, Any, Optional

from PIL import Image

from baseline import run_baseline_ocr
from finetune import load_finetuned_model, run_finetuned_ocr, DEFAULT_OUTPUT_DIR

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
    predictions: List[Dict[str, Any]], dpi: float
) -> List[Dict[str, Any]]:
    elements = []
    for i, pred in enumerate(predictions):
        bbox_px = _quad_to_bbox(pred["box"]) if isinstance(pred["box"][0], list) else pred["box"]
        x_min, y_min, x_max, y_max = bbox_px
        elements.append({
            "id": f"txt_{i + 1:02d}",
            "text": pred["text"],
            "x_mm": round(_px_to_mm(x_min, dpi), 2),
            "y_mm": round(_px_to_mm(y_min, dpi), 2),
            "font_size_mm": round(_px_to_mm(y_max - y_min, dpi), 2),
            "confidence": round(float(pred["confidence"]), 3),
            "bounding_box_pixel": [round(v, 1) for v in bbox_px],
        })
    return elements


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
) -> Dict[str, Any]:
    """
    Run OCR on a single image and return a dict matching
    schemas/document_schema.json.

    Parameters
    ----------
    image_path : str
        Path to a preprocessed image (deskewed / binarized) from Member 2.
    use_finetuned : bool
        If True, loads the fine-tuned TrOCR checkpoint. If False, falls
        back to the pretrained PaddleOCR baseline. Fine-tuned inference
        currently returns one transcription for the whole image rather than
        per-region boxes; for region-level output during development, keep
        use_finetuned=False and cross-check text against the finetuned model
        separately until per-region fine-tuned inference is wired up.
    dpi : float
        Capture/scan resolution used to convert pixel coordinates to mm.
    document_id : str, optional
        Defaults to the image's filename (without extension).
    """
    doc_id = document_id or os.path.splitext(os.path.basename(image_path))[0]

    with Image.open(image_path) as img:
        width_mm = round(_px_to_mm(img.width, dpi), 1)
        height_mm = round(_px_to_mm(img.height, dpi), 1)

    if use_finetuned:
        processor, model = _load_model(use_finetuned, checkpoint_dir)
        # Region detection still comes from the baseline detector; only the
        # recognition step uses the fine-tuned decoder.
        detections = run_baseline_ocr(image_path)
        for det in detections:
            crop_path = _crop_to_temp(image_path, det["box"])
            det["text"] = run_finetuned_ocr(crop_path, processor, model)
        predictions = detections
    else:
        predictions = run_baseline_ocr(image_path)

    text_elements = _predictions_to_text_elements(predictions, dpi)

    return {
        "document_id": doc_id,
        "page_dimensions_mm": {"width": width_mm, "height": height_mm},
        "text_elements": text_elements,
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
