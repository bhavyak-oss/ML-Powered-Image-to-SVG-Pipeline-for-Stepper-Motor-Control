# Phase 2 - Figma plugin + multilingual

## Files (project root ke andar same jagah copy karo)

| File | Kaam |
|---|---|
| `server/core.py` | Server ka asli kaam: OCR + SVG banana (FastAPI ke bina test ho sakta hai) |
| `server/server.py` | FastAPI routes: /health /languages /process /export |
| `server/requirements-server.txt` | Server ke extra packages |
| `figma_plugin/manifest.json, code.js, ui.html` | Figma plugin |
| `ocr_module/lang_config.py` | Supported languages + model paths (nayi language yahin jodo) |
| `ocr_module/script_detect.py` | Text Latin hai ya nahi |
| `ocr_module/ocr_multilang.py` | Non-English OCR (RapidOCR + us language ka model) |
| `ui_export_module/trace_export.py` | Non-Latin text ko image se trace karna (skeleton / contour) |
| `ui_export_module/svg_export_multi.py` | Mixed text export (reflow / inplace) |
| `apply_phase2.py` | `ocr_module/pipeline.py` me 3 chhote badlav (idempotent) |

## Setup

```bash
# project root me, venv active
unzip -o phase2_files.zip
python3 apply_phase2.py
pip install -r server/requirements-server.txt
cd server
uvicorn server:app --port 8000
```
Check: browser me http://localhost:8000/docs khulna chahiye.

## Figma plugin lagana

1. Figma **desktop** app: Plugins > Development > New plugin... > "With UI & browser APIs", folder save karo.
2. Us folder me Figma ka banaya `manifest.json` rakho, aur `code.js` + `ui.html` ko is zip wale se replace karo.
3. Figma wale `manifest.json` me `main`, `ui`, `documentAccess`, `networkAccess` (is zip wali manifest se dekho) add karo.
4. Plugins > Development > Image to Plot SVG. Ek frame/image select karo > Scan selection.

## Flow
Scan -> server OCR -> canvas pe "OCR review" frame (editable text layers) -> text sudharo -> Export plot SVG -> canvas pe vector + Download .svg

## Hindi / doosri language
`models/hi/rec.onnx` aur `models/hi/keys.txt` rakho (us language ka recognition ONNX model + character list). Jab tak ye nahi hain, plugin me wo language "(model missing)" dikhegi.
