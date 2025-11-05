# app/api/images.py
import os
from flask import Blueprint, jsonify, send_file, abort, request, current_app
from ..models import Image as ImageModel, AuditLog
from flask_jwt_extended import jwt_required, get_jwt_identity
from .. import db
from sqlalchemy import or_

bp = Blueprint("images", __name__)

def _safe_remove(path: str):
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass

def _to01(score):
    """Map cosine/IP score [-1,1] to [0,1] for UI."""
    if score is None:
        return None
    v = (float(score) + 1.0) / 2.0
    if v < 0: v = 0.0
    if v > 1: v = 1.0
    return v

@bp.get("/api/images")
@jwt_required(optional=True)
def list_images():
    """
    List/browse images with pagination & category filter.

    Query params:
      - order: 'asc' | 'desc' (default: desc)
      - limit: int (<=200, default 24/50 按需)
      - offset: int (>=0, default 0)
      - category: exact match; 'uncategorized' means NULL or ''
    Response:
      { "items": [...], "total": <int> }
    """
    order = (request.args.get("order") or "desc").lower()
    limit = min(int(request.args.get("limit", 50)), 200)
    offset = max(int(request.args.get("offset", 0)), 0)
    category = (request.args.get("category") or "").strip()

    base = ImageModel.query

    if category:
        if category == "uncategorized":
            base = base.filter(or_(ImageModel.category.is_(None), ImageModel.category == ""))  # NULL or ""
        else:
            base = base.filter(ImageModel.category == category)

    total = base.count()

    q = base.order_by(ImageModel.id.asc() if order == "asc" else ImageModel.id.desc())
    rows = q.offset(offset).limit(limit).all()

    items = [{
        "id": i.id,
        "sha256": i.sha256,
        "width": i.width,
        "height": i.height,
        "thumb": i.thumb_path,
        "mime": i.mime,
        "size_bytes": i.size_bytes,
        "category": i.category or ""
    } for i in rows]

    return jsonify(items=items, total=total)

@bp.get("/api/images/<int:image_id>")
@jwt_required(optional=True)
def image_detail(image_id):
    i = ImageModel.query.get_or_404(image_id)
    return jsonify(
        id=i.id, sha256=i.sha256, path=i.path, thumb=i.thumb_path,
        width=i.width, height=i.height, mime=i.mime, size_bytes=i.size_bytes,
        category=i.category, created_at=i.created_at.isoformat() if i.created_at else None
    )

# Preview: original image
@bp.get("/api/images/<int:image_id>/view")
@jwt_required(optional=True)
def view_image(image_id):
    i = ImageModel.query.get_or_404(image_id)
    if not os.path.exists(i.path):
        abort(404)
    return send_file(i.path, as_attachment=False)

# Preview: thumbnail (fallback to original)
@bp.get("/api/images/<int:image_id>/thumb")
@jwt_required(optional=True)
def view_thumb(image_id):
    i = ImageModel.query.get_or_404(image_id)
    path = i.thumb_path or i.path
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=False)

@bp.get("/api/images/<int:image_id>/download")
@jwt_required(optional=True)
def download(image_id):
    i = ImageModel.query.get_or_404(image_id)
    log = AuditLog(
        user_id=get_jwt_identity(),
        action="download", target_type="image", target_id=i.id,
        ip=None, ua=None
    )
    db.session.add(log); db.session.commit()
    if not os.path.exists(i.path):
        abort(404)
    return send_file(i.path, as_attachment=True)

# NEW: similar-by-id
@bp.get("/api/images/<int:image_id>/similar")
@jwt_required(optional=True)
def similar_by_id(image_id: int):
    """
    Find similar images to a given image id.
    GET /api/images/<id>/similar?k=18&include_self=0
    - returns {"seed": {...}, "results": [...]}
    - if include_self=1, seed will also appear as the first item in results
    """
    k = int(request.args.get("k") or 18)
    include_self = (request.args.get("include_self") or "0") in ("1", "true", "yes")

    seed = ImageModel.query.get_or_404(image_id)

    vm = current_app.extensions.get("vec_model")
    fs = current_app.extensions.get("faiss_store")
    if vm is None or fs is None:
        return jsonify(error="vector search unavailable"), 503

    if not seed.path or not os.path.exists(seed.path):
        return jsonify(error="image file missing"), 404

    try:
        qv = vm.embed_image(seed.path)
        hits = fs.search(qv, k=k)
    except Exception as e:
        return jsonify(error=f"search failed: {e}"), 500

    # 组织 seed 信息（方便前端单独展示/高亮）
    seed_payload = {
        "image_id": seed.id,
        "score": 1.0,
        "score01": 1.0,
    }

    results = []
    for iid, score in hits:
        iid = int(iid)
        if not include_self and iid == image_id:
            continue
        results.append({
            "image_id": iid,
            "score": float(score),
            "score01": _to01(score),
        })

    # 如果需要把自己也放进 results 的第一位
    if include_self:
        # 去重后把种子插到最前
        results = [r for r in results if r["image_id"] != image_id]
        results.insert(0, {"image_id": image_id, "score": 1.0, "score01": 1.0})

    return jsonify(seed=seed_payload, results=results)


# Delete image + files
@bp.delete("/api/images/<int:image_id>")
@jwt_required(optional=True)
def delete_image(image_id):
    """Delete DB record + files, and audit."""
    i = ImageModel.query.get_or_404(image_id)
    user_id = get_jwt_identity()

    _safe_remove(i.path)
    _safe_remove(i.thumb_path)

    log = AuditLog(
        user_id=user_id,
        action="delete", target_type="image", target_id=i.id,
        ip=None, ua=None
    )
    db.session.add(log)

    db.session.delete(i)
    db.session.commit()
    return jsonify(ok=True, deleted=image_id)
