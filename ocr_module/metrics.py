"""
metrics.py
----------
Quantitative evaluation metrics: Character Error Rate (CER) and Word Error
Rate (WER), plus a driver function that scores an OCR callable against a
labeled test set, broken down by dataset category (clean / handwritten /
dark / mixed — per Step 1 of the execution plan).

This file has ZERO dependency on any specific OCR engine. It only ever
deals with plain strings, so the exact same functions score PaddleOCR,
TrOCR, or the fine-tuned model without any changes.
"""

from __future__ import annotations
import json
from collections import defaultdict
from typing import Callable, Dict, List, Any

import editdistance
import jiwer


def cer(prediction: str, ground_truth: str) -> float:
    """
    Character Error Rate = edit distance / number of ground-truth characters.
    Empty ground truth is treated as a 0.0 CER if the prediction is also
    empty, and 1.0 otherwise, to avoid a divide-by-zero.
    """
    if len(ground_truth) == 0:
        return 0.0 if len(prediction) == 0 else 1.0
    return editdistance.eval(prediction, ground_truth) / len(ground_truth)


def wer(prediction: str, ground_truth: str) -> float:
    """Word Error Rate, via the jiwer library (handles tokenization/normalization)."""
    if len(ground_truth.strip()) == 0:
        return 0.0 if len(prediction.strip()) == 0 else 1.0
    return jiwer.wer(ground_truth, prediction)


def _predictions_to_text(predictions: List[Dict[str, Any]]) -> str:
    """Join an OCR engine's per-region predictions into one transcription string."""
    return " ".join(p["text"] for p in predictions)


def load_labels(labels_path: str) -> List[Dict[str, str]]:
    """
    Load ground-truth annotations produced by LabelImg / Label Studio and
    normalized into a flat list of:
        {"image_path": str, "text": str, "category": str}
    """
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(
    ocr_fn: Callable[[str], List[Dict[str, Any]]],
    labels: List[Dict[str, str]],
) -> Dict[str, Any]:
    """
    Run `ocr_fn` (e.g. baseline.run_baseline_ocr, or a fine-tuned equivalent)
    over every labeled image and compute CER/WER, grouped by category.

    Returns
    -------
    {
        "per_image": [ {image_path, category, cer, wer, prediction, ground_truth}, ... ],
        "per_category": { category: {"mean_cer": float, "mean_wer": float, "n": int}, ... },
        "overall": {"mean_cer": float, "mean_wer": float, "n": int}
    }
    """
    per_image = []
    by_category = defaultdict(list)

    for item in labels:
        image_path = item["image_path"]
        ground_truth = item["text"]
        category = item.get("category", "uncategorized")

        predictions = ocr_fn(image_path)
        prediction_text = _predictions_to_text(predictions)

        row = {
            "image_path": image_path,
            "category": category,
            "cer": cer(prediction_text, ground_truth),
            "wer": wer(prediction_text, ground_truth),
            "prediction": prediction_text,
            "ground_truth": ground_truth,
        }
        per_image.append(row)
        by_category[category].append(row)

    def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, float]:
        n = len(rows)
        return {
            "mean_cer": sum(r["cer"] for r in rows) / n if n else 0.0,
            "mean_wer": sum(r["wer"] for r in rows) / n if n else 0.0,
            "n": n,
        }

    per_category = {cat: _summarize(rows) for cat, rows in by_category.items()}
    overall = _summarize(per_image)

    return {"per_image": per_image, "per_category": per_category, "overall": overall}


def print_report(results: Dict[str, Any], title: str = "OCR Evaluation") -> None:
    """Pretty-print a category-wise CER/WER report to the console."""
    print(f"\n=== {title} ===")
    for category, stats in sorted(results["per_category"].items()):
        print(f"  {category:<24} CER={stats['mean_cer']:.3f}  WER={stats['mean_wer']:.3f}  (n={stats['n']})")
    o = results["overall"]
    print(f"  {'OVERALL':<24} CER={o['mean_cer']:.3f}  WER={o['mean_wer']:.3f}  (n={o['n']})")


if __name__ == "__main__":
    import sys
    from baseline import run_baseline_ocr

    if len(sys.argv) != 2:
        print("Usage: python metrics.py <path_to_labels.json>")
        sys.exit(1)

    labels = load_labels(sys.argv[1])
    results = evaluate(run_baseline_ocr, labels)
    print_report(results, title="Baseline (Pretrained) OCR")
