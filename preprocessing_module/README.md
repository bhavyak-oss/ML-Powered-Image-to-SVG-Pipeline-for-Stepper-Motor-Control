# preprocessing_module (Member 2)

Setup, code aur workflow — raw scanned/photographed document image se lekar
clean, deskewed, binary image (OCR-ready) tak.

## Setup

```bash
pip install -r requirements.txt
```

Ya individually:
```bash
pip install opencv-python numpy scikit-image matplotlib
```

**Recommended:** Google Colab — free GPU/CPU, local install ki zaroorat
nahi, notebooks teammates ke saath share karna easy hai.

## Folder Structure

```
preprocessing_module/
├── deskew.py              - Step 4a: rotation correction (Hough Transform)
├── kmeans_segment.py      - Step 4b: K-Means clustering (LAB color space)
├── threshold.py           - Step 4c: Otsu / Sauvola adaptive thresholding
├── pipeline.py            - Sab kuch jodta hai - Member 3 isi ko import karega
├── organize_dataset.py    - Step 1: dataset ko 4 categories mein organize karna
├── test_images/           - Sample/test images (categories ke sub-folders)
├── outputs/                - Processed output images yahan save hoti hain
├── schemas/
│   └── output_schema.md   - Member 3 ke saath data contract
└── requirements.txt
```

## Kyun alag files mein split kiya?

- Har teammate/step independently test ho sakta hai
- `pipeline.py` short rehta hai — bas teeno functions ko order mein call
  karta hai
- Otsu se Sauvola (ya vice versa) switch karna easy hai, baaki files
  touch kiye bina

## Usage

```python
from pipeline import preprocess
import cv2

binary_image = preprocess(
    "test_images/sample_dark.jpg",
    threshold_method="sauvola"
)
cv2.imwrite("outputs/clean.png", binary_image)
```

Ya seedha terminal se test karne ke liye:
```bash
python pipeline.py
```

## Pipeline Steps (in order)

1. **`load_image()`** — `cv2.imread()` se BGR numpy array
2. **`deskew(image)`** — Canny edges + Hough Transform se tilt angle nikal
   ke rotate karta hai
3. **`kmeans_segment(image)`** — LAB color space mein K-Means, taaki ink
   aur shadow (jo color mein differ karte hain, sirf brightness mein nahi)
   alag clusters mein aa jayen
4. **`binarize(gray, method)`** — final black/white image, Otsu ya
   Sauvola se
5. Result: clean binary image, Member 3 ke OCR call ke liye ready

## Contract with Member 3 (OCR)

`preprocess(image_path)` → clean, deskewed, binary numpy array (`uint8`,
values `0`/`255`) jo Member 3 ka OCR call directly input ke taur par le
sakta hai. Poora schema `schemas/output_schema.md` mein hai.

## Testing Checklist (handoff se pehle)

- [ ] Pipeline ko 4 dataset categories par chalao: clean printed,
      handwritten, dark/glare, mixed diagrams (`organize_dataset.py` use
      karo folders banane ke liye)
- [ ] Har category ke liye before/after visually compare karo — side-by-side
      images team ke liye save karo
- [ ] Har step ka time measure karo — jo bhi slow lage UI ke liye flag karo
- [ ] Ek dark/uneven-lighting image par Otsu vs Sauvola test karo, confirm
      karo ki Sauvola behtar result deta hai
- [ ] Output format Member 3 ke saath confirm karo, uske baad hi OCR code
      aage badhao
