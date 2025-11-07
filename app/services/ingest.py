# app/services/ingest.py
from flask import current_app
from .. import db
from ..models import Image as ImageModel, OcrText
from .ocr import run_ocr

def _upsert_ocr(image_id: int, text: str) -> bool:
    if not text:
        return False
    row = db.session.get(OcrText, image_id)
    if row:
        row.text = text
    else:
        row = OcrText(image_id=image_id, text=text)
        db.session.add(row)
    return True

def _add_vector(image_id: int, vec) -> bool:
    fs = current_app.extensions.get("faiss_store")
    if fs is None:
        return False
    # Your FaissStore likely accepts just vectors; if it supports ID mapping, adapt here.
    # If it does not track IDs, you still can add the vector (order must match internal id map).
    fs.add([vec])          # adapt to your FaissStore.add signature
    fs.save()
    return True

def ingest_after_save(img: ImageModel, image_path: str):
    """Call this after Image row is created and file is saved."""
    vm = current_app.extensions.get("vec_model")
    # 1) vector -> faiss
    try:
        if vm is not None:
            vec = vm.embed_image(image_path)  # L2-normalized
            _add_vector(img.id, vec)
    except Exception as e:
        current_app.logger.warning(f"[ingest] vector fail id={img.id}: {e}")
    # 2) OCR -> db
    try:
        txt = run_ocr(image_path)
        if txt:
            _upsert_ocr(img.id, txt)
    except Exception as e:
        current_app.logger.warning(f"[ingest] ocr fail id={img.id}: {e}")
    db.session.commit()
