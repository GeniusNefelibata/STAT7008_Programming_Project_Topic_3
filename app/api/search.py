# app/api/search.py
from __future__ import annotations
import os
import tempfile
from typing import Iterable, List, Tuple, Optional

import numpy as np

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required
from sqlalchemy import select, func
from .. import db
from ..models import Image as ImageModel, OcrText, Embedding

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
    健康检查（鲁棒版）：把可能是 method/np 类型的字段转换成可 JSON 序列化的值，避免 500。
    """
    import numpy as np

    def jval(v):
        """将 v 转成 jsonify 友好的类型：可调用则尝试调用；np 标量转原生；其余转 str。"""
        try:
            if callable(v):
                v = v()
        except Exception:
            # 调用失败就保留原值
            pass
        # numpy 标量 -> python 标量
        if isinstance(v, np.generic):
            return v.item()
        # 允许的原生类型直接返回
        if isinstance(v, (type(None), bool, int, float, str)):
            return v
        # 其它（例如 Path、enum、对象实例、方法等）做字符串化
        return str(v)

    out = {
        "cfg": {
            "EMBED_DEVICE": current_app.config.get("EMBED_DEVICE"),
            "EMBED_MODEL": current_app.config.get("EMBED_MODEL"),
            "INDEX_PATH": current_app.config.get("FAISS_INDEX_PATH"),
        },
        "faiss": {"ok": False},
        "model": {"ok": False},
        "probe": None,
    }

    # model 信息
    try:
        vm = current_app.extensions.get("vec_model")
        out["model"]["ok"] = vm is not None
        out["model"]["name"] = current_app.config.get("EMBED_MODEL")
        out["model"]["dim"]  = jval(getattr(vm, "dim", None))
    except Exception as e:
        out["model"]["err"] = str(e)

    # faiss 信息
    try:
        fs = current_app.extensions.get("faiss_store")
        if fs is not None:
            out["faiss"]["dim"]        = jval(getattr(fs, "dim", None))
            out["faiss"]["ntotal"]     = jval(getattr(fs, "ntotal", None) or getattr(fs, "n_total", None))
            out["faiss"]["metric"]     = jval(getattr(fs, "metric", None) or getattr(fs, "METRIC", None))
            out["faiss"]["index_path"] = jval(getattr(fs, "index_path", None) or getattr(fs, "path", None)
                                              or current_app.config.get("FAISS_INDEX_PATH"))
            out["faiss"]["ok"] = True
    except Exception as e:
        out["faiss"]["err"] = str(e)

    # 探针：随便拿一条 embedding 做一次 search
    try:
        fs = current_app.extensions.get("faiss_store")
        if fs is not None:
            row = db.session.execute(select(Embedding).limit(1)).scalar_one_or_none()
            if row is not None:
                vec = np.frombuffer(row.vector_blob, dtype="float32")
                hits = []
                if hasattr(fs, "search"):
                    try:
                        raw = fs.search(vec, k=5) or []
                        for h in raw:
                            if isinstance(h, (list, tuple)) and len(h) >= 2:
                                hits.append([int(h[0]), float(h[1])])
                            else:
                                # 只有 id 的情况
                                hits.append([int(h), None])
                    except Exception as e:
                        out["probe"] = {"seed_id": int(row.image_id), "err": str(e)}
                out["probe"] = out.get("probe") or {"seed_id": int(row.image_id), "hits": hits}
            else:
                out["probe"] = {"seed_id": None, "hits": []}
    except Exception as e:
        out["probe"] = {"err": str(e)}

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
