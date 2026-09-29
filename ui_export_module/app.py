"""
app.py
------
Human-in-the-Loop Correction Interface (Step 3 of the execution plan).

Loads the Semantic JSON document produced by Member 3's pipeline.py,
renders the original image with the OCR bounding boxes overlaid, lets the
reviewer fix misread text / adjust box coordinates / delete artifacts, and
exports both an Editable SVG and a Plot-Ready SVG from the corrected data.

Run with:
    streamlit run app.py

Recommended: run this on your own machine (Mac/Windows/Linux), NOT Google
Colab — Streamlit needs a persistent local server, and Colab's tunneling
makes the live editing UI unreliable.
"""

from __future__ import annotations
import json
import os
import copy

import streamlit as st
from PIL import Image, ImageDraw

from svg_export import export_plot_ready, write_editable_svg, save_document_json

DEFAULT_JSON_PATH = "sample_data/doc_sample_001.json"
DEFAULT_IMAGE_DIR = "sample_data"
OUTPUT_DIR = "outputs"

st.set_page_config(page_title="OCR Review — Human-in-the-Loop", layout="wide")
st.title("Human-in-the-Loop Correction Interface")


# ---------------------------------------------------------------------------
# Load document
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Document")
    json_path = st.text_input("Semantic JSON path", value=DEFAULT_JSON_PATH)
    image_path_override = st.text_input(
        "Image path (leave blank to auto-guess from document_id)", value=""
    )

if not os.path.exists(json_path):
    st.warning(f"No file found at '{json_path}'. Point the sidebar at a document "
               f"produced by Member 3's `pipeline.py` (or a sample you've placed "
               f"in `{DEFAULT_JSON_PATH}`).")
    st.stop()

with open(json_path, "r", encoding="utf-8") as f:
    original_doc = json.load(f)

image_path = image_path_override or os.path.join(
    DEFAULT_IMAGE_DIR, f"{original_doc['document_id']}.png"
)

if not os.path.exists(image_path):
    st.warning(f"No image found at '{image_path}'. Set the correct path in the sidebar.")
    st.stop()

source_image = Image.open(image_path).convert("RGB")


# ---------------------------------------------------------------------------
# Review UI — image with overlay + per-element editable fields
# ---------------------------------------------------------------------------

col_image, col_fields = st.columns([2, 1])

corrected_elements = []
any_deleted = False

with col_fields:
    st.subheader("Detected text")
    for el in original_doc["text_elements"]:
        with st.expander(f"{el['id']}  ·  conf {el['confidence']:.2f}", expanded=True):
            text = st.text_input("Text", value=el["text"], key=f"text_{el['id']}")

            x_min, y_min, x_max, y_max = el["bounding_box_pixel"]
            c1, c2, c3, c4 = st.columns(4)
            x_min = c1.number_input("x_min", value=float(x_min), key=f"xmin_{el['id']}")
            y_min = c2.number_input("y_min", value=float(y_min), key=f"ymin_{el['id']}")
            x_max = c3.number_input("x_max", value=float(x_max), key=f"xmax_{el['id']}")
            y_max = c4.number_input("y_max", value=float(y_max), key=f"ymax_{el['id']}")

            delete = st.checkbox("Delete this element", key=f"del_{el['id']}")

        if delete:
            any_deleted = True
            continue

        updated = copy.deepcopy(el)
        updated["text"] = text
        updated["bounding_box_pixel"] = [x_min, y_min, x_max, y_max]
        corrected_elements.append(updated)

corrected_doc = copy.deepcopy(original_doc)
corrected_doc["text_elements"] = corrected_elements

with col_image:
    st.subheader("Original image + current boxes")
    overlay = source_image.copy()
    draw = ImageDraw.Draw(overlay)
    for el in corrected_elements:
        x_min, y_min, x_max, y_max = el["bounding_box_pixel"]
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
        draw.text((x_min, max(0, y_min - 12)), el["id"], fill="red")
    st.image(overlay, use_container_width=True)

    if any_deleted:
        st.caption("Deleted elements are hidden here and excluded from every export below.")


# ---------------------------------------------------------------------------
# Export controls
# ---------------------------------------------------------------------------

st.divider()
st.subheader("Export")

os.makedirs(OUTPUT_DIR, exist_ok=True)
doc_id = corrected_doc["document_id"]

col_a, col_b, col_c = st.columns(3)

with col_a:
    if st.button("Save corrected JSON"):
        out_path = os.path.join(OUTPUT_DIR, f"{doc_id}_corrected.json")
        save_document_json(corrected_doc, out_path)
        st.success(f"Saved {out_path}")
        with open(out_path, "rb") as f:
            st.download_button("Download JSON", f, file_name=os.path.basename(out_path))

with col_b:
    if st.button("Export Editable SVG"):
        out_path = os.path.join(OUTPUT_DIR, f"{doc_id}_editable.svg")
        write_editable_svg(corrected_doc, out_path)
        st.success(f"Saved {out_path}")
        with open(out_path, "rb") as f:
            st.download_button("Download Editable SVG", f, file_name=os.path.basename(out_path))

with col_c:
    if st.button("Export Plot-Ready SVG"):
        out_path = os.path.join(OUTPUT_DIR, f"{doc_id}_plot_ready.svg")
        export_plot_ready(corrected_doc, out_path)
        st.success(f"Saved {out_path}")
        with open(out_path, "rb") as f:
            st.download_button("Download Plot-Ready SVG", f, file_name=os.path.basename(out_path))
