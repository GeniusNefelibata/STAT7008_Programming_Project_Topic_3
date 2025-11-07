# app/api/uploads.py
from __future__ import annotations

import os
import hashlib
import tempfile
from mimetypes import guess_type

import numpy as np
from PIL import Image as PILImage
from flask import Blueprint, request, current_app, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

from .. import db
from ..models import Image as ImageModel, AuditLog, Embedding, OcrText

# 入库阶段用到的服务
from ..services import embeddings as EMB
from ..services import ocr as OCR

bp = Blueprint("uploads", __name__)


# ------------- helpers -------------
def _sha256_file(fpath: str) -> str:
    h = hashlib.sha256()
    with open(fpath, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _gen_thumb(src_path: str, sha: str) -> tuple[int | None, int | None, str | None]:
    """返回 (width, height, thumb_path)；失败则返回 (None, None, None)"""
    try:
        with PILImage.open(src_path) as im:
            w, h = im.size
            im.thumbnail((512, 512))
            tdir = os.path.join(current_app.config["THUMB_DIR"], sha[:2], sha[2:4])
            os.makedirs(tdir, exist_ok=True)
            tpath = os.path.join(tdir, f"{sha}.jpg")
            im.convert("RGB").save(tpath, "JPEG", quality=85)
            return w, h, tpath
    except Exception:
        return None, None, None


def _upsert_embedding_and_index(image_id: int, img_path: str) -> None:
    """
    若 Embedding 缺失则写入；然后追加到 FAISS 索引，并立刻持久化到磁盘。
    """
    vec = None
    try:
        # Embedding upsert
        has_vec = db.session.get(Embedding, image_id)
        if has_vec is None:
            vec = EMB.encode_image(img_path)  # ndarray float32, 已归一化
            model_name = current_app.config.get("EMBED_MODEL", "clip-ViT-B-32")
            emb = Embedding(
                image_id=image_id,
                model_name=model_name,
                dim=int(len(vec)),
                vector_blob=vec.astype("float32").tobytes(),
            )
            db.session.add(emb)
            db.session.commit()
        else:
            # 若 DB 已有向量，但索引可能缺，必要时还原向量
            try:
                vec = np.frombuffer(has_vec.vector_blob, dtype="float32")
            except Exception:
                vec = EMB.encode_image(img_path)
    except Exception:
        db.session.rollback()
        # 尝试从 DB 还原；失败就放弃索引阶段
        try:
            rec = db.session.get(Embedding, image_id)
            if rec is not None:
                vec = np.frombuffer(rec.vector_blob, dtype="float32")
        except Exception:
            vec = None

    # 写入 FAISS 索引（允许重复 add）
    try:
        fs = current_app.extensions.get("faiss_store")
        if fs is not None and vec is not None:
            ids = np.array([int(image_id)], dtype=np.int64)
            fs.add(ids, vec.reshape(1, -1).astype("float32"))
            # ✨ 关键：立刻持久化索引（兼容 save / write_index 两种命名）
            try:
                if hasattr(fs, "save"):
                    fs.save()
                elif hasattr(fs, "write_index"):
                    fs.write_index()
            except Exception:
                pass
    except Exception:
        # 索引失败不阻断主流程
        pass


def _upsert_ocr(image_id: int, img_path: str) -> None:
    """若 OCR 文本缺失则识别并入库。失败静默跳过。"""
    try:
        has_ocr = db.session.get(OcrText, image_id)
        if has_ocr is None:
            txt = OCR.extract_text(img_path) or ""
            db.session.add(OcrText(image_id=image_id, text=txt))
            db.session.commit()
    except Exception:
        db.session.rollback()


def _audit(user_id: int | None, image_id: int, duplicate: bool) -> None:
    try:
        db.session.add(
            AuditLog(
                user_id=user_id,
                action="upload",
                target_type="image",
                target_id=image_id,
                ip=request.remote_addr,
                ua=request.headers.get("User-Agent"),
                extra_json=('{"duplicate": true}' if duplicate else '{"duplicate": false}'),
            )
        )
        db.session.commit()
    except Exception:
        db.session.rollback()


# ------------- route -------------
@bp.post("/api/upload")
@jwt_required(optional=True)
def upload():
    """
    多文件上传流程：
      1) 临时文件建在 UPLOAD_DIR（同盘，避免 WinError 17）
      2) sha256 内容寻址保存 & 去重
      3) 生成缩略图
      4) 写入 Image + 审计
      5) Upsert：Embedding + OCR，并追加到 FAISS 索引（且立即持久化）
    """
    files = request.files.getlist("file")
    if not files:
        return jsonify(error="no files"), 400

    user_id = get_jwt_identity()
    saved = []

    for file in files:
        if not file or not file.filename:
            saved.append({"error": "empty filename"})
            continue

        # 1) 临时文件：放到 UPLOAD_DIR（同盘移动不跨盘）
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix="ing_", suffix="_upload", dir=current_app.config["UPLOAD_DIR"]
        )
        os.close(tmp_fd)
        try:
            file.save(tmp_path)

            # 2) 内容寻址 & 去重
            sha = _sha256_file(tmp_path)
            subdir = os.path.join(current_app.config["UPLOAD_DIR"], sha[:2], sha[2:4])
            os.makedirs(subdir, exist_ok=True)
            dst_path = os.path.join(subdir, sha)

            existed: ImageModel | None = ImageModel.query.filter_by(sha256=sha).first()
            if existed and os.path.exists(existed.path):
                # 去重：补齐 Embedding / OCR / 索引
                _upsert_embedding_and_index(existed.id, existed.path)
                _upsert_ocr(existed.id, existed.path)
                _audit(user_id, existed.id, duplicate=True)
                saved.append({"image_id": existed.id, "duplicate": True})
                continue

            # 若 DB 有记录但磁盘路径缺失，则覆盖修复
            if existed and not os.path.exists(existed.path):
                os.replace(tmp_path, dst_path)
                target_path = dst_path

                # 生成缩略图（旧记录可能没有）
                w, h, tpath = _gen_thumb(target_path, sha)
                try:
                    existed.path = target_path
                    existed.thumb_path = existed.thumb_path or tpath
                    existed.width = existed.width or w
                    existed.height = existed.height or h
                    existed.size_bytes = os.path.getsize(target_path)
                    existed.mime = existed.mime or (
                        guess_type(file.filename)[0] or "application/octet-stream"
                    )
                    db.session.commit()
                except Exception:
                    db.session.rollback()

                _upsert_embedding_and_index(existed.id, existed.path)
                _upsert_ocr(existed.id, existed.path)
                _audit(user_id, existed.id, duplicate=False)
                saved.append({"image_id": existed.id, "duplicate": False})
                continue

            # 3) 正常新图：同盘移动
            os.replace(tmp_path, dst_path)
            target_path = dst_path

            # 4) 缩略图 + 基本属性
            width, height, thumb_path = _gen_thumb(target_path, sha)
            size_bytes = os.path.getsize(target_path)
            mime = guess_type(file.filename)[0] or "application/octet-stream"

            # 写入 Image
            image = ImageModel(
                user_id=user_id,
                sha256=sha,
                path=target_path,
                thumb_path=thumb_path,
                width=width,
                height=height,
                size_bytes=size_bytes,
                mime=mime,
            )
            try:
                db.session.add(image)
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                saved.append({"error": f"db.insert image failed: {e}"})
                # 清理文件以免脏数据
                try:
                    if os.path.exists(target_path):
                        os.remove(target_path)
                    if thumb_path and os.path.exists(thumb_path):
                        os.remove(thumb_path)
                except Exception:
                    pass
                continue

            # 审计
            _audit(user_id, image.id, duplicate=False)

            # 5) 向量 + 索引 + OCR（失败不阻断）
            _upsert_embedding_and_index(image.id, image.path)
            _upsert_ocr(image.id, image.path)

            saved.append({"image_id": image.id, "duplicate": False})

        except Exception as e:
            db.session.rollback()
            saved.append({"error": str(e)})
        finally:
            # 清理残留 tmp
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    return jsonify(saved=saved)
