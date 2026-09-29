"""
finetune.py
-----------
Fine-tunes TrOCR on the augmented internal dataset (Step 5 of the execution
plan). Early encoder layers are frozen; only the decoder and the last two
encoder blocks are trained, which keeps this runnable on a single free-tier
Colab GPU without overfitting the small internal dataset.

Two entry points other modules rely on:
    - finetune()          : runs the training loop, saves a checkpoint
    - load_finetuned_model(): loads a saved checkpoint for inference
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

BASE_CHECKPOINT = "microsoft/trocr-base-handwritten"
DEFAULT_OUTPUT_DIR = "checkpoints/trocr_finetuned"


class OCRDataset(Dataset):
    """
    Wraps (image, text) pairs into the tensor format TrOCR's Seq2SeqTrainer
    expects. Accepts either file paths or already-loaded numpy arrays for
    the images, so it works directly with augment.py's output.
    """

    def __init__(self, images: List, labels: List[str], processor: TrOCRProcessor, max_length: int = 64):
        assert len(images) == len(labels), "images and labels must be the same length"
        self.images = images
        self.labels = labels
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.images)

    def _load_image(self, entry) -> Image.Image:
        if isinstance(entry, np.ndarray):
            return Image.fromarray(entry).convert("RGB")
        return Image.open(entry).convert("RGB")

    def __getitem__(self, idx: int):
        image = self._load_image(self.images[idx])
        text = self.labels[idx]

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(
            text, padding="max_length", max_length=self.max_length, truncation=True
        ).input_ids
        # Replace pad tokens with -100 so the loss function ignores them.
        labels = [l if l != self.processor.tokenizer.pad_token_id else -100 for l in labels]

        return {"pixel_values": pixel_values, "labels": torch.tensor(labels)}


def _build_model(freeze_encoder: bool = True, unfreeze_last_n_layers: int = 2):
    processor = TrOCRProcessor.from_pretrained(BASE_CHECKPOINT)
    model = VisionEncoderDecoderModel.from_pretrained(BASE_CHECKPOINT)

    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False
        # Unfreeze only the last N encoder blocks so the model can still
        # adapt low-level visual features (stroke width, ink texture) without
        # retraining the whole vision backbone.
        for param in model.encoder.encoder.layer[-unfreeze_last_n_layers:].parameters():
            param.requires_grad = True

    # Required generation config for TrOCR.
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    return processor, model


def finetune(
    train_images: List,
    train_labels: List[str],
    val_images: List,
    val_labels: List[str],
    output_dir: str = DEFAULT_OUTPUT_DIR,
    num_train_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
) -> str:
    """
    Fine-tunes TrOCR on the given (augmented) training set and validates on
    a held-out set. Saves the resulting checkpoint to `output_dir`.

    Returns the output_dir path (so callers can immediately load it).
    """
    processor, model = _build_model()

    train_dataset = OCRDataset(train_images, train_labels, processor)
    val_dataset = OCRDataset(val_images, val_labels, processor)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    return output_dir


def load_finetuned_model(checkpoint_dir: str = DEFAULT_OUTPUT_DIR) -> Tuple[TrOCRProcessor, VisionEncoderDecoderModel]:
    """Load a previously fine-tuned checkpoint for inference."""
    processor = TrOCRProcessor.from_pretrained(checkpoint_dir)
    model = VisionEncoderDecoderModel.from_pretrained(checkpoint_dir)
    model.eval()
    return processor, model


def run_finetuned_ocr(image_path: str, processor: TrOCRProcessor, model: VisionEncoderDecoderModel) -> str:
    """
    Single-image inference helper for the fine-tuned model, matching the
    plain-string interface metrics.py expects.
    """
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_length=64)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 2:
        print("Usage: python finetune.py <training_manifest.json>")
        print('Manifest format: [{"image_path": "...", "text": "..."}, ...]')
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        manifest = json.load(f)

    split = int(len(manifest) * 0.85)
    train_items, val_items = manifest[:split], manifest[split:]

    output_dir = finetune(
        train_images=[m["image_path"] for m in train_items],
        train_labels=[m["text"] for m in train_items],
        val_images=[m["image_path"] for m in val_items],
        val_labels=[m["text"] for m in val_items],
    )
    print(f"Fine-tuned model saved to: {output_dir}")
