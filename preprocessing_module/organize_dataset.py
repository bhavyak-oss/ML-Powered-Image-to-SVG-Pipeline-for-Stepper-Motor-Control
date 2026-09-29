"""
organize_dataset.py — STEP 1: Dataset Collection & Categorization

Slide 9 (Testing Checklist) ke hisaab se pipeline ko 4 categories par
test karna hai:
  1. clean_printed
  2. handwritten
  3. dark_glare
  4. mixed_diagrams

Ye script un 4 category folders ko banata hai aur ek raw images folder
se manually-sorted images ko sahi jagah copy karne mein madad karta hai.

Usage:
    python organize_dataset.py
    (interactive - har category ke liye kitni images hain wo batayega)
"""

import os
import shutil

CATEGORIES = ["clean_printed", "handwritten", "dark_glare", "mixed_diagrams"]
DATASET_ROOT = "test_images"


def create_category_folders(root=DATASET_ROOT):
    """4 category folders bana deta hai agar exist nahi karte."""
    os.makedirs(root, exist_ok=True)
    for category in CATEGORIES:
        path = os.path.join(root, category)
        os.makedirs(path, exist_ok=True)
        print(f"Ready: {path}/")


def categorize_summary(root=DATASET_ROOT):
    """Har category mein kitni images hain, uska summary dikhata hai."""
    print("\n--- Dataset Summary ---")
    total = 0
    for category in CATEGORIES:
        path = os.path.join(root, category)
        if os.path.exists(path):
            count = len([
                f for f in os.listdir(path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
        else:
            count = 0
        print(f"  {category:20s}: {count} images")
        total += count
    print(f"  {'TOTAL':20s}: {total} images\n")

    if total == 0:
        print("Abhi koi image nahi hai. Manually images ko test_images/<category>/ mein daal do.")


def copy_image_to_category(image_path, category, root=DATASET_ROOT):
    """Ek image ko sahi category folder mein copy karta hai."""
    if category not in CATEGORIES:
        raise ValueError(f"Category '{category}' valid nahi hai. Options: {CATEGORIES}")

    dest_folder = os.path.join(root, category)
    os.makedirs(dest_folder, exist_ok=True)
    dest_path = os.path.join(dest_folder, os.path.basename(image_path))
    shutil.copy2(image_path, dest_path)
    print(f"Copied: {image_path} -> {dest_path}")


if __name__ == "__main__":
    create_category_folders()
    categorize_summary()
