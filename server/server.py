"""
server.py - Figma plugin ke liye local backend.

Chalane ka tarika (project ke `server` folder me):
    pip install -r requirements-server.txt
    uvicorn server:app --port 8000
Check: browser me http://localhost:8000/docs
"""
import threading

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict

import core

core.init()

app = FastAPI(title="Image to Plot SVG")
# Figma plugin ka iframe 'null' origin se aata hai, isliye CORS khula (sirf local dev ke liye)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_LOCK = threading.Lock()   # OCR engine / chdir ek time pe ek hi request


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/languages")
def languages():
    from lang_config import list_languages
    return {"languages": list_languages()}


def _guard(fn, *a, **k):
    with _LOCK:
        try:
            return fn(*a, **k)
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:                      # noqa
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/process")
def process(file: UploadFile = File(...), lang: str = Form("en"), use_finetuned: bool = Form(False)):
    data = file.file.read()
    return _guard(core.process_image, data, lang, use_finetuned)


class ExportReq(BaseModel):
    session_id: str
    edits: Dict[str, str] = {}


@app.post("/export")
def export(req: ExportReq):
    return _guard(core.export_session, req.session_id, req.edits)
