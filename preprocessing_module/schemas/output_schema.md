# Preprocessing Output Schema

Ye document define karta hai ki `preprocess()` function kya return karta
hai — taaki Member 3 (OCR module) bina confusion ke isko consume kar sake.

## Function Signature

```python
from pipeline import preprocess

binary_image = preprocess(
    image_path: str,
    threshold_method: str = "sauvola",  # "sauvola" ya "otsu"
    k: int = 3                           # K-Means clusters
)
```

## Return Value

| Property     | Value                                    |
|--------------|-------------------------------------------|
| Type         | `numpy.ndarray`                          |
| dtype        | `uint8`                                  |
| Shape        | `(H, W)` — single channel (grayscale-like) |
| Value range  | Binary — sirf `0` ya `255`               |
| Resolution   | Original image ki resolution (resize nahi kiya) |

## Example — File se save karna

```python
import cv2
cv2.imwrite("outputs/clean.png", binary_image)
```

## Open Decisions (Member 3 ke saath confirm karna hai)

- [ ] Return numpy array hi rahega, ya hamesha `outputs/` mein PNG save
      bhi karna hai?
- [x] Binary (0/255) confirmed — grayscale nahi
- [ ] Original resolution rakhni hai, ya OCR ke liye fixed size par resize
      karna hai?

## Dataset Categories (Step 1 output)

`organize_dataset.py` se ye 4 category folders `test_images/` ke andar
banti hain — pipeline ko in sab par test karna hai before handoff:

1. `clean_printed`
2. `handwritten`
3. `dark_glare`
4. `mixed_diagrams`
