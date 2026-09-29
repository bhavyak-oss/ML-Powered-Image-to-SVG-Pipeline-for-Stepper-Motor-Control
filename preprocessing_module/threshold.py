"""
threshold.py — FILE 3

Key idea: expose method as a parameter — lets you A/B test Otsu vs.
Sauvola on the same image without duplicating code.
"""

import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def binarize(gray_image, method="sauvola"):
    """
    Converts a grayscale image to pure black/white (binary) using either
    Otsu's global thresholding or Sauvola's local adaptive thresholding.

    Args:
        gray_image: single-channel (grayscale) numpy array
        method: "otsu" or "sauvola" (default). Sauvola is better for
                uneven lighting / shadows / glare (per-region threshold),
                Otsu is faster and works well on clean, evenly-lit scans.

    Returns:
        Binary numpy array (uint8), values are 0 or 255.
    """
    if method == "otsu":
        _, binary = cv2.threshold(
            gray_image, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:  # sauvola — better for uneven lighting
        t = threshold_sauvola(gray_image, window_size=25)
        binary = (gray_image > t).astype(np.uint8) * 255

    return binary
