import cv2
import numpy as np
from skimage.filters import threshold_sauvola

def binarize(gray_image, method="sauvola"):
    if method == "otsu":
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        t = threshold_sauvola(gray_image, window_size=25)
        binary = (gray_image > t).astype(np.uint8) * 255
    return binary