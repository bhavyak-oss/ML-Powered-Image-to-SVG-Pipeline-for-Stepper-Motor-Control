"""
deskew.py — FILE 1

Key idea: find the dominant line angle with Hough Transform, then rotate
the whole image back by that angle.
"""

import cv2
import numpy as np


def deskew(image):
    """
    Detects the dominant skew angle in a scanned/photographed document
    using edge detection + Hough Line Transform, then rotates the image
    to correct it.

    Args:
        image: BGR numpy array (as read by cv2.imread)

    Returns:
        Deskewed BGR numpy array, same shape as input.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 100,
        minLineLength=100, maxLineGap=10
    )

    # Agar koi line detect na ho (blank/very clean image), toh rotate mat karo
    if lines is None or len(lines) == 0:
        return image

    angles = [
        np.degrees(np.arctan2(y2 - y1, x2 - x1))
        for x1, y1, x2, y2 in lines[:, 0]
    ]
    median_angle = np.median(angles)

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
