# app/api/maintenance.py
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required
from .. import db
from ..models import Image as ImageModel, OcrText
from ..services.ocr import run_ocr

bp = Blueprint("maintenance", __name__)

@bp.get("/api/maintenance/_counts")
@jwt_required(optional=True)
def counts():
    images_total = db.session.query(ImageModel).count()
    ocr_covered = db.session.query(OcrText).count()
    fs = current_app.extensions.get("faiss_store")
    faiss_ntotal = fs.ntotal() if fs else 0
    return jsonify(
        images_total=images_total,
        ocr_covered=ocr_covered,
        faiss_ntotal=faiss_ntotal,
        ocr_coverage=(ocr_covered / images_total) if images_total else 0.0
    )

@bp.post("/api/maintenance/reindex_ocr")
@jwt_required(optional=True)
def reindex_ocr():
    """Backfill OCR text for a range of image IDs."""
    data = request.get_json(silent=True) or {}
    start_id = int(data.get("start_id", 1))
    end_id = int(data.get("end_id", 1 << 30))
    q = ImageModel.query.filter(ImageModel.id.between(start_id, end_id))
    n_ok = 0
    for img in q.yield_per(100):
        try:
            txt = run_ocr(img.path)
            if txt:
                row = db.session.get(OcrText, img.id)
                if row:
                    row.text = txt
                else:
                    db.session.add(OcrText(image_id=img.id, text=txt))
                n_ok += 1
        except Exception:
            pass
    db.session.commit()
    return jsonify(ok=True, updated=n_ok, range=[start_id, end_id])
