# Member 3 — OCR & Fine-Tuning Module

Baseline OCR execution, CER/WER metric setup, data augmentation, and OCR
fine-tuning (Phase 1, Steps 2 & 5).

## Setup

```bash
pip install -r requirements.txt
```

GPU strongly recommended for `finetune.py` (Colab T4 free tier is enough).

## Files

| File | Purpose |
|---|---|
| `baseline.py` | Runs pretrained PaddleOCR and returns raw predictions. Only file that imports `paddleocr` directly. |
| `metrics.py` | CER / WER calculation, category-wise evaluation report. No OCR-engine dependency. |
| `augment.py` | Synthetic data augmentation (brightness, blur, perspective, noise) for fine-tuning. |
| `finetune.py` | Fine-tunes TrOCR (frozen encoder + trainable decoder) and saves a checkpoint. |
| `pipeline.py` | **Import this one.** Ties baseline/fine-tuned OCR together, converts pixel coordinates to mm, and outputs the Semantic JSON Member 4 consumes. |
| `schemas/document_schema.json` | JSON Schema for the output contract with Member 4. |

## Ground-truth labels format

`metrics.py` and `finetune.py` both expect a flat JSON list, e.g.
`ground_truth/labels.json`:

```json
[
  {"image_path": "test_images/clean_01.png", "text": "Invoice #4021", "category": "clean_printed"},
  {"image_path": "test_images/hw_03.png",     "text": "meet at 6pm",   "category": "handwritten"}
]
```

`category` should be one of: `clean_printed`, `handwritten`,
`adverse_lighting`, `mixed_diagrams` (Step 1's four dataset categories).

## Typical Phase 1 workflow

```bash
# 1. Baseline evaluation (before any fine-tuning)
python metrics.py ground_truth/labels.json

# 2. Fine-tune on the same (augmented) labeled set
python finetune.py ground_truth/labels.json

# 3. Re-run evaluation with the fine-tuned model to confirm CER/WER improved
python -c "
from finetune import load_finetuned_model, run_finetuned_ocr
from metrics import load_labels, evaluate, print_report

processor, model = load_finetuned_model()
ocr_fn = lambda path: [{'text': run_finetuned_ocr(path, processor, model), 'confidence': 1.0, 'box': [0,0,0,0]}]

labels = load_labels('ground_truth/labels.json')
print_report(evaluate(ocr_fn, labels), title='Fine-Tuned OCR')
"

# 4. Generate the Semantic JSON for a single document (what Member 4 receives)
python pipeline.py test_images/sample_dark.jpg
```

## Handoff contract

- **In** (from Member 2): a preprocessed image path (deskewed, binarized).
- **Out** (to Member 4): `run_pipeline(image_path)` → dict matching
  `schemas/document_schema.json` — the same schema Member 4's `app.py`
  loads directly, so no translation layer sits between OCR output and the
  review UI.
