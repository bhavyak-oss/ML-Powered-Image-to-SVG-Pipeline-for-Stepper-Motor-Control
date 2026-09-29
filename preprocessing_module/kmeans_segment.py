"""
kmeans_segment.py — FILE 2

Key idea: cluster pixels in LAB colour space (not grayscale) so ink and
shadow — which differ in colour, not just brightness — end up in
different clusters.
"""

import cv2
import numpy as np


def kmeans_segment(image, k=3):
    """
    Segments the image into k clusters using K-Means in LAB color space.
    LAB separates lightness (L) from color (A/B), which helps distinguish
    ink from shadows/glare that a pure grayscale approach would miss.

    Args:
        image: BGR numpy array
        k: number of clusters (default 3 — e.g. background, ink, shadow)

    Returns:
        (segmented_image, labels)
        segmented_image: BGR numpy array, same shape as input, each pixel
                          replaced by its cluster's center color
        labels: 2D numpy array (H, W) with the cluster index per pixel
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    pixels = lab.reshape((-1, 3)).astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0
    )
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    segmented_lab = centers[labels.flatten()].reshape(lab.shape).astype(np.uint8)
    segmented_bgr = cv2.cvtColor(segmented_lab, cv2.COLOR_LAB2BGR)

    return segmented_bgr, labels.reshape(image.shape[:2])
