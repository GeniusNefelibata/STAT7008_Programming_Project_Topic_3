# app/api/images.py
import os
from flask import Blueprint, jsonify, send_file, abort, request, current_app
from ..models import Image as ImageModel, AuditLog
from flask_jwt_extended import jwt_required, get_jwt_identity
from .. import db
from sqlalchemy import or_
from sqlalchemy.orm import load_only  # 新增：用于一次性取回候选的类别，避免 N 次查询

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
"""@bp.get("/api/images/<int:image_id>/view")
@jwt_required(optional=True)
def view_image(image_id):
    i = ImageModel.query.get_or_404(image_id)
    if not os.path.exists(i.path):
        abort(404)
    return send_file(i.path, as_attachment=False)"""

# Preview: thumbnail (fallback to original)
"""@bp.get("/api/images/<int:image_id>/thumb")
@jwt_required(optional=True)
def view_thumb(image_id):
    i = ImageModel.query.get_or_404(image_id)
    path = i.thumb_path or i.path
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=False)"""
@bp.get("/api/images/<int:image_id>/view")
@jwt_required(optional=True)
def view_image(image_id):
    i = ImageModel.query.get_or_404(image_id)
    path = _resolve_image_path(i)
    if not path:
        abort(404)
    mime = (getattr(i, "mime", None)
            or mimetypes.guess_type(path)[0]
            or "image/jpeg")
    return send_file(path, mimetype=mime, as_attachment=False, conditional=True)

@bp.get("/api/images/<int:image_id>/thumb")
@jwt_required(optional=True)
def view_thumb(image_id):
    i = ImageModel.query.get_or_404(image_id)
    path = _resolve_thumb_path(i) or _resolve_image_path(i)
    if not path:
        abort(404)
    # 缩略图统一按 jpeg 下发
    return send_file(path, mimetype="image/jpeg", as_attachment=False, conditional=True)

#临时更改
"""@bp.get("/api/images/<int:image_id>/download")
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
    return send_file(i.path, as_attachment=True)"""
import os, mimetypes
from flask import current_app, send_file, abort
# -------- 路径与缩略图 “自愈” 辅助 --------
def _norm(p: str) -> str:
    return p.replace("\\", "/")

def _data_dir() -> str:
    return current_app.config.get(
        "DATA_DIR",
        os.path.abspath(os.path.join(current_app.root_path, "..", "data"))
    )

def _img_store_path(sha: str) -> str:
    return os.path.join(_data_dir(), "images", sha[:2], sha[2:4], sha)

def _thumb_store_path(sha: str) -> str:
    return os.path.join(_data_dir(), "thumbs", sha[:2], sha[2:4], f"{sha}.jpg")

def _ensure_dir(p: str):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def _ensure_thumb(src_path: str, thumb_path: str, size: int = 512):
    if os.path.exists(thumb_path):
        return
    _ensure_dir(thumb_path)
    with PILImage.open(src_path) as im:
        im.thumbnail((size, size))
        im.convert("RGB").save(thumb_path, "JPEG", quality=85)

def _resolve_image_path(img) -> str | None:
    # 1) 先用数据库记录
    if img.path and os.path.exists(img.path):
        return img.path
    # 2) 退化：按 sha256 推导仓储路径
    sha = getattr(img, "sha256", None)
    if sha:
        alt = _img_store_path(sha)
        if os.path.exists(alt):
            if img.path != _norm(alt):
                img.path = _norm(alt)
                from .. import db
                db.session.commit()
            return alt
    return None

def _resolve_thumb_path(img) -> str | None:
    # 若 DB 中的缩略图存在就用
    if getattr(img, "thumb_path", None) and os.path.exists(img.thumb_path):
        return img.thumb_path
    # 否则尝试从原图生成
    src = _resolve_image_path(img)
    if not src:
        return None
    thumb = _thumb_store_path(img.sha256)
    try:
        _ensure_thumb(src, thumb, 512)
    except Exception:
        traceback.print_exc()
        return None
    if getattr(img, "thumb_path", None) != _norm(thumb):
        img.thumb_path = _norm(thumb)
        from .. import db
        db.session.commit()
    return thumb

# 下载用的扩展名推断与文件名
_EXT_MAP = {
    "image/jpeg": ".jpg", "image/jpg": ".jpg",
    "image/png": ".png",  "image/webp": ".webp",
    "image/bmp": ".bmp",  "image/tiff": ".tiff",
}
def _infer_ext(img) -> str:
    if getattr(img, "mime", None):
        mt = img.mime.lower()
        ext = _EXT_MAP.get(mt) or (mimetypes.guess_extension(mt) or "")
        if ext in (".jpe", ".jpeg"):
            ext = ".jpg"
        if ext:
            return ext
    if getattr(img, "path", None):
        _, ext = os.path.splitext(img.path)
        if ext:
            return ext.lower()
    return ".jpg"

def _download_filename(img) -> str:
    prefix = (getattr(img, "category", None) or "image").replace("/", "_")
    return f"{prefix}-{img.id}{_infer_ext(img)}"

@bp.get("/api/images/<int:image_id>/download")
@jwt_required(optional=True)
def download(image_id):
    i = ImageModel.query.get_or_404(image_id)

    # 审计日志（保留你原来的逻辑）
    log = AuditLog(
        user_id=get_jwt_identity(),
        action="download", target_type="image", target_id=i.id,
        ip=None, ua=None
    )
    db.session.add(log)
    db.session.commit()

    # 先用数据库里的路径
    path = i.path if i.path else None

    # 如果不存在，按 sha256 推导一次仓储路径（data/images/aa/bb/sha）
    if not path or not os.path.exists(path):
        sha = getattr(i, "sha256", None)
        if sha:
            data_dir = current_app.config.get(
                "DATA_DIR",
                os.path.abspath(os.path.join(current_app.root_path, "..", "data"))
            )
            alt = os.path.join(data_dir, "images", sha[:2], sha[2:4], sha)
            if os.path.exists(alt):
                path = alt

    if not path or not os.path.exists(path):
        abort(404)

    # 统一内容类型，并指定下载文件名（带扩展名）
    mime = (getattr(i, "mime", None)
            or mimetypes.guess_type(path)[0]
            or "application/octet-stream")

    return send_file(
        path,
        mimetype=mime,
        as_attachment=True,
        download_name=_download_filename(i),  # <- 关键：带扩展名
        conditional=True
    )

# NEW: similar-by-id（增强：默认限制同类 same_category=1）
@bp.get("/api/images/<int:image_id>/similar")
@jwt_required(optional=True)
def similar_by_id(image_id: int):
    """
    Find similar images to a given image id.

    GET /api/images/<id>/similar?k=18&include_self=0&same_category=1
      - k: 数量，默认 18
      - include_self: 是否包含自己，默认 0
      - same_category: 是否限制在同一 category（默认 1）
    Response: { items: [{id, score}], seed: {...} }
    """
    k = int(request.args.get("k") or 18)
    k = max(1, min(200, k))
    include_self = (request.args.get("include_self") or "0") in ("1", "true", "yes")
    same_category = (request.args.get("same_category") or "1") in ("1", "true", "yes")

    seed = ImageModel.query.get_or_404(image_id)
    seed_cat = (seed.category or "").strip()  # 可能为空字符串

    vm = current_app.extensions.get("vec_model")
    fs = current_app.extensions.get("faiss_store")
    if vm is None or fs is None:
        return jsonify(error="vector search unavailable"), 503

    if not seed.path or not os.path.exists(seed.path):
        return jsonify(error="image file missing"), 404

    try:
        qv = vm.embed_image(seed.path)

        # 为了过滤同类后还能凑足 k 条，这里先多取一些候选
        topN = max(k * 5, k + 32)
        raw_hits = fs.search(qv, k=topN)  # [(id, score)]，score 越大越相似/越好
    except Exception as e:
        return jsonify(error=f"search failed: {e}"), 500

    # 构造 seed 信息
    seed_payload = {
        "image_id": seed.id,
        "score": 1.0,
        "score01": 1.0,
        "category": seed_cat or ""
    }

    # 先按 include_self 处理去重
    cand_pairs = [(int(iid), float(score)) for iid, score in raw_hits]
    if not include_self:
        cand_pairs = [(iid, s) for (iid, s) in cand_pairs if iid != image_id]

    # 如果需要按同类过滤：一次性把候选 id 的类别查出来
    if same_category:
        if seed_cat == "":
            # “无类别”视为同类（即候选也必须无类别/空字符串/NULL）
            imgs = (
                ImageModel.query.options(load_only(ImageModel.id, ImageModel.category))
                .filter(ImageModel.id.in_([iid for iid, _ in cand_pairs]))
                .all()
            )
            by_id = {x.id: (x.category or "") for x in imgs}
            cand_pairs = [(iid, s) for (iid, s) in cand_pairs if (by_id.get(iid, "") == "")]
        else:
            imgs = (
                ImageModel.query.options(load_only(ImageModel.id, ImageModel.category))
                .filter(ImageModel.id.in_([iid for iid, _ in cand_pairs]))
                .all()
            )
            by_id = {x.id: (x.category or "") for x in imgs}
            cand_pairs = [(iid, s) for (iid, s) in cand_pairs if by_id.get(iid, "") == seed_cat]

    # 截断到 k 条，保持 FS 返回的相似度顺序（已经是降序）
    cand_pairs = cand_pairs[:k]

    # 构造返回
    items = [{
        "id": iid,
        "score": _to01(score) if _to01(score) is not None else 0.0
    } for iid, score in cand_pairs]

    # include_self 情况：把自己插到最前面（items 仍保持 id/score 字段）
    if include_self:
        items = [{"id": image_id, "score": 1.0}] + [it for it in items if it["id"] != image_id]

    return jsonify(items=items, seed=seed_payload)

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
