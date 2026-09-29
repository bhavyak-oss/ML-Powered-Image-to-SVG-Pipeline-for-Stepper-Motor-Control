"""
augment.py
----------
Synthetic data augmentation strategy for OCR fine-tuning (Step 5 of the
execution plan). Each transform targets one specific failure mode called
out in Step 1's dataset categories:

    - RandomBrightnessContrast / RandomShadow  -> adverse lighting & exposure
    - MotionBlur                                -> handwriting / camera shake
    - Perspective                               -> phone-camera capture angle
    - GaussNoise                                -> low-light sensor noise / ink bleed

Use this to multiply a small internal dataset into a much larger set of
"hard" training examples before fine-tuning, without collecting more photos.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import albumentations as A


augment_pipeline = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.6),
    A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.3),
    A.MotionBlur(blur_limit=5, p=0.3),
    A.Perspective(scale=(0.02, 0.08), p=0.4),
    A.GaussNoise(var_limit=(10.0, 60.0), p=0.3),  # simulates sensor noise / ink bleed
])


def generate_augmented_samples(image: np.ndarray, n: int = 5) -> List[np.ndarray]:
    """
    Produce `n` synthetic variants of a single clean training image.

    Parameters
    ----------
    image : np.ndarray
        HxWx3 uint8 array (RGB).
    n : int
        Number of augmented variants to generate.
    """
    return [augment_pipeline(image=image)["image"] for _ in range(n)]


def build_augmented_dataset(
    clean_images: List[np.ndarray],
    labels: List[str],
    n_per_image: int = 5,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Expand a (images, labels) pair into a larger augmented set. The original
    clean images and labels are included alongside the synthetic variants,
    so fine-tuning still sees "easy" examples too.
    """
    aug_images: List[np.ndarray] = list(clean_images)
    aug_labels: List[str] = list(labels)

    for image, label in zip(clean_images, labels):
        aug_images += generate_augmented_samples(image, n_per_image)
        aug_labels += [label] * n_per_image

    return aug_images, aug_labels


if __name__ == "__main__":
    import sys
    from PIL import Image

    if len(sys.argv) != 3:
        print("Usage: python augment.py <input_image> <output_dir>")
        sys.exit(1)

    import os
    src_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    img = np.array(Image.open(src_path).convert("RGB"))
    variants = generate_augmented_samples(img, n=5)

    base = os.path.splitext(os.path.basename(src_path))[0]
    for i, variant in enumerate(variants):
        Image.fromarray(variant).save(os.path.join(out_dir, f"{base}_aug{i}.png"))

    print(f"Wrote {len(variants)} augmented variants to {out_dir}")
