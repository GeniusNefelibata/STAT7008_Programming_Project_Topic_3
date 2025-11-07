# app/services/ocr.py
"""
Unified OCR service.

Choose engine by env:
  OCR_ENGINE=easyocr|tesseract|paddle    (default: easyocr)
  OCR_LANGS  language tags, comma-separated
      - easyocr: e.g. "en,ch_sim" (your old default)
      - tesseract: e.g. "eng,chi_sim" (mapped automatically from ch_sim→chi_sim)
      - paddle: "en" or "ch" etc. (we map ch_sim→ch)

Extra envs:
  TESSERACT_EXE  absolute path to tesseract.exe on Windows (optional)
Limit:
  Max text length is truncated to 50k chars to protect DB.
"""

from __future__ import annotations
import os
import re
from typing import List, Optional

# -------- configuration --------
_OCR_ENGINE = (os.getenv("OCR_ENGINE") or "easyocr").strip().lower()
_OCR_LANGS  = (os.getenv("OCR_LANGS")  or "en,ch_sim").strip()

# Windows users may need to point to the executable:
_TESSERACT_EXE = os.getenv("TESSERACT_EXE")  # e.g. r"C:\Program Files\Tesseract-OCR\tesseract.exe"

_MAX_LEN = 50_000


def _clean(text: str) -> str:
    """Normalize whitespace, strip nuls, clip to max length."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    # collapse whitespace while preserving line breaks roughly
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if len(text) > _MAX_LEN:
        text = text[:_MAX_LEN]
    return text


# ===================== EASYOCR =====================
_easy_reader = None

def _easyocr_langs(spec: str) -> List[str]:
    # pass through (easyocr expects codes like 'en', 'ch_sim')
    langs = [x.strip() for x in spec.split(",") if x.strip()]
    return langs or ["en"]

def _run_easyocr(path: str) -> str:
    global _easy_reader
    try:
        import easyocr
    except Exception:
        return ""

    if _easy_reader is None:
        langs = _easyocr_langs(_OCR_LANGS)
        # gpu=False keeps it CPU-only and avoids CUDA requirements
        _easy_reader = easyocr.Reader(langs, gpu=False)

    try:
        # detail=0 returns text lines list directly
        lines = _easy_reader.readtext(path, detail=0)
        txt = "\n".join(map(str, lines))
        return _clean(txt)
    except Exception:
        return ""


# ===================== TESSERACT =====================
def _map_langs_tesseract(spec: str) -> str:
    # map ch_sim to chi_sim; keep others
    mapping = {"ch_sim": "chi_sim", "zh_cn": "chi_sim", "en": "eng"}
    out = []
    for tok in [x.strip() for x in spec.split(",") if x.strip()]:
        out.append(mapping.get(tok, tok))
    if not out:
        out = ["eng"]
    return "+".join(out)  # tesseract uses plus-joined codes, e.g., "eng+chi_sim"

def _run_tesseract(path: str) -> str:
    try:
        import pytesseract
        from PIL import Image
    except Exception:
        return ""

    try:
        if _TESSERACT_EXE:
            pytesseract.pytesseract.tesseract_cmd = _TESSERACT_EXE
        langs = _map_langs_tesseract(_OCR_LANGS)
        with Image.open(path) as img:
            txt = pytesseract.image_to_string(img, lang=langs)
        return _clean(txt)
    except Exception:
        return ""


# ===================== PADDLEOCR =====================
_paddle_ocr = None

def _map_lang_paddle(spec: str) -> str:
    # PaddleOCR supports 'en', 'ch', 'chinese_cht', etc.
    # We map ch_sim -> 'ch', else first token.
    toks = [x.strip() for x in spec.split(",") if x.strip()]
    if not toks:
        return "en"
    if "ch_sim" in toks or "zh_cn" in toks:
        return "ch"
    return toks[0]

def _run_paddle(path: str) -> str:
    global _paddle_ocr
    try:
        from paddleocr import PaddleOCR
    except Exception:
        return ""
    try:
        if _paddle_ocr is None:
            _paddle_ocr = PaddleOCR(
                use_angle_cls=True,
                lang=_map_lang_paddle(_OCR_LANGS),
            )
        result = _paddle_ocr.ocr(path, cls=True)
        lines: List[str] = []
        for det in result:
            for _, (txt, prob) in det:
                # keep reasonably confident tokens
                if prob >= 0.5:
                    lines.append(str(txt))
        return _clean("\n".join(lines))
    except Exception:
        return ""


# ===================== Public API =====================
def run_ocr(image_path: str) -> str:
    """
    Run OCR with the selected backend and return normalized text.
    Returns empty string on failure.
    """
    engine = _OCR_ENGINE
    if engine == "tesseract":
        txt = _run_tesseract(image_path)
        if txt:
            return txt
        # fallback to easyocr if tesseract fails silently
        return _run_easyocr(image_path)
    elif engine == "paddle":
        txt = _run_paddle(image_path)
        if txt:
            return txt
        # fallback
        return _run_easyocr(image_path)
    else:
        # default easyocr
        txt = _run_easyocr(image_path)
        if txt:
            return txt
        # fallback to tesseract
        return _run_tesseract(image_path)


# Backward-compatible name (your old code called extract_text)
def extract_text(image_path: str) -> str:
    """Alias of run_ocr(image_path)."""
    return run_ocr(image_path)
