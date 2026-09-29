"""
baseline.py
-----------
Runs a pretrained OCR engine (PaddleOCR / PP-OCRv4 by default) over an image
and returns raw text-detection + recognition results in a simple, engine-
agnostic format. This is the ONLY file that should ever import paddleocr
directly — every other module in this package works with the plain dict
format returned here, so swapping engines later only means editing this file.

Output format (one dict per detected text region):
    {
        "text":       str,                       # recognized string
        "confidence": float,                      # 0.0 - 1.0
        "box":        [[x,y], [x,y], [x,y], [x,y]]  # quadrilateral, pixel coords
    }
"""

from __future__ import annotations
from functools import lru_cache
from typing import List, Dict, Any

from paddleocr import PaddleOCR


@lru_cache(maxsize=1)
def _get_engine(lang: str = "en") -> PaddleOCR:
    """
    Lazily construct and cache a single PaddleOCR instance.
    PaddleOCR is expensive to initialize (loads 3 sub-models: detection,
    angle classification, recognition), so we only want to pay that cost once
    per process, not once per image.
    """
    return PaddleOCR(lang=lang, use_angle_cls=True, show_log=False)


def run_baseline_ocr(image_path: str, lang: str = "en") -> List[Dict[str, Any]]:
    """
    Run the pretrained baseline OCR engine on a single image.

    Parameters
    ----------
    image_path : str
        Path to a preprocessed (deskewed, binarized) image from Member 2's
        pipeline, or any raw image for quick testing.
    lang : str
        PaddleOCR language code. "en" covers the Phase 1 dataset.

    Returns
    -------
    List[dict] — see module docstring for the shape of each entry.
    """
    engine = _get_engine(lang)
    raw_result = engine.ocr(image_path, cls=True)

    # PaddleOCR wraps everything in an extra list per image; raw_result[0] is
    # None when the image has no detected text at all.
    if not raw_result or raw_result[0] is None:
        return []

    predictions = []
    for box, (text, confidence) in raw_result[0]:
        predictions.append({
            "text": text,
            "confidence": float(confidence),
            "box": box,  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        })
    return predictions


def run_baseline_ocr_batch(image_paths: List[str], lang: str = "en") -> Dict[str, List[Dict[str, Any]]]:
    """Convenience wrapper: run_baseline_ocr over many images, keyed by path."""
    return {path: run_baseline_ocr(path, lang) for path in image_paths}


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 2:
        print("Usage: python baseline.py <image_path>")
        sys.exit(1)

    results = run_baseline_ocr(sys.argv[1])
    print(json.dumps(results, indent=2, ensure_ascii=False))
