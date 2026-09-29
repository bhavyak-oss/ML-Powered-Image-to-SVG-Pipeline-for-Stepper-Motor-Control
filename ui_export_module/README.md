# Member 4 — UI, Architecture & Output Module

Human-in-the-loop OCR review interface, single-stroke vector font mapping,
and Plot-Ready / Editable SVG export (Phase 1, Step 3 & Step 6).

## Where to run this

**Your own machine (Mac/Windows/Linux) — not Google Colab.** Streamlit needs
a persistent local server; Colab's tunneling makes the live editing UI
unreliable. This module does no GPU work, so a normal laptop is fine.

## Setup

```bash
pip install -r requirements.txt
```

## Files

| File | Purpose |
|---|---|
| `hershey_font.py` | Text → single-stroke line segments. Ships with a built-in stick font (zero extra dependencies); can switch to the real `hershey-fonts` package via `set_backend("hershey")`. |
| `svg_export.py` | `compile_strokes()` / `export_plot_ready()` for the Plot-Ready SVG, `write_editable_svg()` for a standard `<text>`-based SVG. No Streamlit dependency — testable headless. |
| `app.py` | **Run this one.** Streamlit review UI: image + bounding boxes, editable text/coordinates, delete-artifact checkboxes, and export buttons. |
| `schemas/document_schema.json` | The input contract — identical to Member 3's output schema. |
| `sample_data/` | A tiny sample document + blank placeholder image so `app.py` runs immediately, before you have real OCR output to test with. |

## Running the review UI

```bash
streamlit run app.py
```

This opens a browser tab. By default it loads `sample_data/doc_sample_001.json`
and looks for a matching image at `sample_data/doc_sample_001.png`. Once
Member 3's `pipeline.py` gives you a real document JSON + source image, point
the sidebar fields at those paths instead.

## Using it headless (no UI) — e.g. for testing

```bash
python svg_export.py sample_data/doc_sample_001.json outputs/test_plot_ready.svg
```

```python
from svg_export import load_document_json, export_plot_ready, write_editable_svg

doc = load_document_json("sample_data/doc_sample_001.json")
export_plot_ready(doc, "outputs/out_plot_ready.svg")
write_editable_svg(doc, "outputs/out_editable.svg")
```

## Handoff contract

- **In** (from Member 3): a JSON file matching `schemas/document_schema.json`
  plus the source image it was OCR'd from.
- **Out**: two SVGs per reviewed document —
  - `*_plot_ready.svg` — stroke-only (`fill="none"`), physical mm dimensions,
    ready for a CNC/pen plotter.
  - `*_editable.svg` — normal `<text>` elements, for opening in
    Inkscape/Illustrator/Figma.
  - Plus the corrected JSON itself, in case a downstream step needs the
    human-verified text/coordinates rather than the SVG.

## Upgrading the font

The built-in stick font in `hershey_font.py` covers A–Z, 0–9, and a handful
of punctuation marks — enough to demo the full pipeline without installing
anything extra. For nicer letterforms:

```bash
pip install hershey-fonts
```
```python
import hershey_font
hershey_font.set_backend("hershey")
```
Every other function (`text_to_strokes`, `compile_strokes`, the Streamlit
app) keeps working unchanged.
