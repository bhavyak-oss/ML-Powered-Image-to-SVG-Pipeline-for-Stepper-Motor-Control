"""
pipeline.py — the only file Member 3 needs to import.

Workflow:
  load_image() -> deskew() -> kmeans_segment() -> binarize() -> return

Contract with Member 3 (OCR):
  preprocess(image_path) returns a clean, deskewed, binary numpy array
  (uint8, values 0/255) that Member 3's OCR call can take directly as input.
"""

import cv2

from deskew import deskew
from kmeans_segment import kmeans_segment
from threshold import binarize


def preprocess(image_path, threshold_method="sauvola", k=3):
    """
    Runs the full preprocessing pipeline on a single image file.

    Args:
        image_path: path to the input image (jpg/png)
        threshold_method: "otsu" or "sauvola" (passed to binarize)
        k: number of K-Means clusters (default 3)

    Returns:
        Binary numpy array (uint8, values 0/255) — ready for OCR.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image load nahi hui, path check karo: {image_path}")

    image = deskew(image)
    segmented, _ = kmeans_segment(image, k=k)
    gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    binary = binarize(gray, method=threshold_method)
    return binary  # ready for OCR


if __name__ == "__main__":
    # Quick manual test - ek sample image par chala ke dekho
    import os

    sample = "test_images/sample_dark.jpg"
    if os.path.exists(sample):
        result = preprocess(sample, threshold_method="sauvola")
        os.makedirs("outputs", exist_ok=True)
        cv2.imwrite("outputs/clean.png", result)
        print(f"Done — outputs/clean.png likh diya. Shape: {result.shape}")
    else:
        print(f"'{sample}' nahi mila. test_images/ folder mein sample images daalo pehle.")
