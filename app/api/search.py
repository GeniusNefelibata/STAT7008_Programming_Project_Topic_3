# app/api/search.py
from __future__ import annotations
import os
import tempfile
from typing import Iterable, List, Tuple, Optional

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required
from sqlalchemy import select
from .. import db
from ..models import Image as ImageModel, OcrText

bp = Blueprint("search", __name__)

def _to01(score: Optional[float]) -> Optional[float]:
    """Map cosine/IP score in [-1,1] to [0,1] for UI."""
    if score is None:
        return None
    try:
        v = (float(score) + 1.0) / 2.0
    except Exception:
        return None
    return max(0.0, min(1.0, v))

def _norm_hits(hits: Iterable) -> List[Tuple[int, Optional[float]]]:
    """
    Normalize FAISS hits into [(id, score?)].
    Accepts: [(id, score)], [id], numpy arrays, etc.
    """
    out: List[Tuple[int, Optional[float]]] = []
    for h in hits:
        # tuple/list (id, score)
        if isinstance(h, (tuple, list)) and len(h) >= 2:
            try:
                out.append((int(h[0]), float(h[1])))
            except Exception:
                # if score cannot be parsed, keep id only
                out.append((int(h[0]), None))
            continue
        # single id
        try:
            out.append((int(h), None))
        except Exception:
            # unknown shape, skip
            continue
    return out

def _get_vm_and_index():
    vm = current_app.extensions.get("vec_model")
    fs = current_app.extensions.get("faiss_store")
    return vm, fs

@bp.get("/api/search")
@jwt_required(optional=True)
def search_text():
    """Text → Image (vector search).  GET /api/search?q=dog&k=12"""
    q = (request.args.get("q") or "").strip()
    k = int(request.args.get("k") or 12)
    if not q:
        return jsonify(error="empty query"), 400

    vm, fs = _get_vm_and_index()
    if vm is None or fs is None:
        return jsonify(error="vector search unavailable"), 503

    qv = vm.embed_text(q)  # normalized [dim]
    raw_hits = fs.search(qv, k=k)  # tolerant to various return formats
    hits = _norm_hits(raw_hits)
    results = [{"image_id": i, "score": s, "score01": _to01(s)} for i, s in hits]
    return jsonify(results=results)

@bp.post("/api/search_by_image")
@jwt_required(optional=True)
def search_by_image():
    """Image → Image (vector search).  POST multipart file under key 'file'."""
    f = request.files.get("file")
    k = int(request.args.get("k") or 12)
    if not f:
        return jsonify(error="no file"), 400

    vm, fs = _get_vm_and_index()
    if vm is None or fs is None:
        return jsonify(error="vector search unavailable"), 503

    fd, path = tempfile.mkstemp(prefix="qimg_", suffix=".bin",
                                dir=current_app.config["UPLOAD_DIR"])
    os.close(fd)
    try:
        f.save(path)
        qv = vm.embed_image(path)
        raw_hits = fs.search(qv, k=k)
        hits = _norm_hits(raw_hits)
        results = [{"image_id": i, "score": s, "score01": _to01(s)} for i, s in hits]
        return jsonify(results=results)
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

@bp.get("/api/search_ocr")
@jwt_required(optional=True)
def search_ocr():
    """OCR text search (ILIKE). GET /api/search_ocr?q=total&k=24"""
    q = (request.args.get("q") or "").strip()
    k = int(request.args.get("k") or 24)
    if not q:
        return jsonify(error="empty query"), 400

    pattern = f"%{q}%"
    stmt = (
        select(OcrText.image_id)
        .join(ImageModel, ImageModel.id == OcrText.image_id)
        .where(OcrText.text.ilike(pattern))
        .limit(k)
    )
    rows = db.session.execute(stmt).all()
    results = [{"image_id": int(iid), "score": None, "score01": None} for (iid,) in rows]
    return jsonify(results=results)

# ---------------- Diagnostics & Utilities ---------------- #

@bp.get("/api/search/_deepcheck")
def deepcheck():
    """
    Return current config + FAISS status + model status + an optional probe.
    Helpful when search endpoints do not return results.
    """
    app = current_app
    vm, fs = _get_vm_and_index()

    # config snapshot
    out = {
        "cfg": {
            "EMBED_DEVICE": app.config.get("EMBED_DEVICE"),
            "EMBED_MODEL": app.config.get("EMBED_MODEL"),
            "INDEX_PATH": app.config.get("FAISS_INDEX_PATH"),
        }
    }

    # faiss status (best effort)
    faiss_info = {"ok": False}
    try:
        if fs is not None:
            # Try to expose basic stats if available
            attrs = {}
            for key in ("dim", "metric", "ntotal", "path", "index_path"):
                if hasattr(fs, key):
                    attrs[key] = getattr(fs, key)
            faiss_info.update(attrs)
            # consider ok if it can accept a dummy query OR ntotal > 0
            ok = False
            try:
                # some stores accept vector length 1 and will error; so guard
                if getattr(fs, "ntotal", 0) > 0:
                    ok = True
            except Exception:
                ok = False
            faiss_info["ok"] = bool(ok)
    except Exception:
        pass
    out["faiss"] = faiss_info

    # model status
    model_info = {"ok": False, "name": None, "dim": None}
    try:
        if vm is not None:
            model_info["name"] = getattr(vm, "name", None) or app.config.get("EMBED_MODEL")
            model_info["dim"] = getattr(vm, "dim", None)
            model_info["ok"] = True
    except Exception:
        model_info["ok"] = False
    out["model"] = model_info

    # quick probe: pick one image id and try to search its neighbors
    try:
        row = db.session.execute(select(ImageModel.id).limit(1)).first()
        if row and fs is not None and vm is not None:
            iid = int(row[0])
            # if your VecModel can embed by image path quickly:
            img = db.session.get(ImageModel, iid)
            qv = vm.embed_image(img.path)
            raw_hits = fs.search(qv, k=5)
            hits = _norm_hits(raw_hits)
            out["probe"] = {"seed_id": iid, "hits": hits}
        else:
            out["probe"] = None
    except Exception as e:
        out["probe"] = {"err": repr(e)}

    return jsonify(out)

@bp.get("/api/search/_reload")
def reload_index():
    """
    Force reload FAISS index in-place.
    If your FaissStore exposes an 'open'/'load' or 'rebuild' method, we try them in order.
    Otherwise, we replace the extension with a fresh instance.
    """
    app = current_app
    path = app.config.get("FAISS_INDEX_PATH")
    if not path:
        return jsonify(ok=False, error="FAISS_INDEX_PATH not set"), 500

    vm, fs = _get_vm_and_index()

    # Determine dimension
    dim = None
    try:
        dim = getattr(vm, "dim", None)
        if not dim:
            # fallback to service constant if present
            from ..services import embeddings as EMB  # type: ignore
            dim = getattr(EMB, "DIM", None)
    except Exception:
        pass
    if not dim:
        dim = 512

    # Try reopen on existing store
    try:
        if fs is not None:
            if hasattr(fs, "open"):
                fs.open(path)  # type: ignore
            elif hasattr(fs, "load"):
                fs.load(path)  # type: ignore
            elif hasattr(fs, "reload"):
                fs.reload()  # type: ignore
            # else: nothing to call; we will replace instance below
    except Exception:
        fs = None

    # If no usable store, replace it
    if fs is None:
        from ..faiss_store import FaissStore
        app.extensions["faiss_store"] = FaissStore(dim=int(dim), index_path=path)

    return jsonify(ok=True, path=path, dim=int(dim))
