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
from ..services import autotag as AT  # 零样本自动打标签

bp = Blueprint("uploads", __name__)

# ---------------- helpers ----------------
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
        has_vec = db.session.get(Embedding, image_id)
        if has_vec is None:
            vec = EMB.encode_image(img_path)  # float32、单位向量
            model_name = current_app.config.get("EMBED_MODEL", "clip-ViT-B-32")
            db.session.add(Embedding(
                image_id=image_id,
                model_name=model_name,
                dim=int(len(vec)),
                vector_blob=vec.astype("float32").tobytes(),
            ))
            db.session.commit()
        else:
            try:
                vec = np.frombuffer(has_vec.vector_blob, dtype="float32")
            except Exception:
                vec = EMB.encode_image(img_path)
    except Exception:
        db.session.rollback()
        try:
            rec = db.session.get(Embedding, image_id)
            if rec is not None:
                vec = np.frombuffer(rec.vector_blob, dtype="float32")
        except Exception:
            vec = None

    # 加入 FAISS（允许重复 add）
    try:
        fs = current_app.extensions.get("faiss_store")
        if fs is not None and vec is not None:
            ids = np.array([int(image_id)], dtype=np.int64)
            fs.add(ids, vec.reshape(1, -1).astype("float32"))
            # 立即持久化
            try:
                if hasattr(fs, "save"):
                    fs.save()
                elif hasattr(fs, "write_index"):
                    fs.write_index()
            except Exception:
                pass
    except Exception:
        pass  # 索引失败不阻断主流程


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


def _audit(user_id: int | None, image_id: int, duplicate: bool, extra: str | None = None) -> None:
    """
    记录审计日志；extra 传入形如 '"auto_tag":"cat"' 的 key-value 片段（不含最外层花括号）。
    """
    try:
        base = '{"duplicate": true}' if duplicate else '{"duplicate": false}'
        if extra:
            # base[:-1] 去掉末尾 '}'，然后追加 , <extra> 再补回 '}'。
            # 注意：f-string 内要输出字面量 '}' 必须写成 '}}'
            extra_json = base[:-1] + f", {extra}}}"
        else:
            extra_json = base

        db.session.add(AuditLog(
            user_id=user_id,
            action="upload",
            target_type="image",
            target_id=image_id,
            ip=request.remote_addr,
            ua=request.headers.get("User-Agent"),
            extra_json=extra_json,
        ))
        db.session.commit()
    except Exception:
        db.session.rollback()


def _auto_tag(image_obj: ImageModel, force: bool = False) -> str | None:
    """
    上传即打标签：
      - category 为空时写入主标签；force=True 时覆盖；
      - 若存在 ImageTag 表，写入多标签+分数；
      - 返回实际写入的主标签（无变更则 None）。
    """
    try:
        res = AT.predict_labels(image_obj.path, labels=None, top_k=3, threshold=0.30)
        # 统一到规范名（如果 autotag 提供了 to_canonical）
        new_cat = AT.to_canonical(res.primary) if hasattr(AT, "to_canonical") else (res.primary or "others")
        if not new_cat:
            new_cat = "others"

        if not force and image_obj.category:
            return None

        image_obj.category = new_cat
        db.session.commit()

        # 可选：多标签（如果定义了 ImageTag）
        try:
            from ..models import ImageTag  # 没这表会异常，直接忽略
            for lab in res.labels:
                sc = float(res.scores.get(lab, 0.0))
                db.session.add(ImageTag(image_id=image_obj.id, tag=lab, score=sc))
            db.session.commit()
        except Exception:
            db.session.rollback()

        return new_cat
    except Exception:
        db.session.rollback()
        return None


# ---------------- route ----------------
@bp.post("/api/upload")
@jwt_required(optional=True)
def upload():
    """
    多文件上传流程：
      1) 同盘临时文件
      2) sha256 内容寻址 & 去重
      3) 缩略图
      4) 写入 Image + 审计
      5) 向量入库 + FAISS 追加并持久化
      6) OCR 入库
      7) 自动打标签（默认仅填空；?force_tag=1 可覆盖）
    """
    files = request.files.getlist("file")
    if not files:
        return jsonify(error="no files"), 400

    user_id = get_jwt_identity()
    force_tag = (request.args.get("force_tag") or "0").lower() in ("1", "true", "yes")
    saved = []

    for file in files:
        if not file or not file.filename:
            saved.append({"error": "empty filename"})
            continue

        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix="ing_", suffix="_upload", dir=current_app.config["UPLOAD_DIR"]
        )
        os.close(tmp_fd)

        try:
            file.save(tmp_path)

            # 内容寻址
            sha = _sha256_file(tmp_path)
            subdir = os.path.join(current_app.config["UPLOAD_DIR"], sha[:2], sha[2:4])
            os.makedirs(subdir, exist_ok=True)
            dst_path = os.path.join(subdir, sha)

            existed: ImageModel | None = ImageModel.query.filter_by(sha256=sha).first()

            # ===== 已存在且文件在磁盘上：做补全并返回 =====
            if existed and os.path.exists(existed.path):
                _upsert_embedding_and_index(existed.id, existed.path)
                _upsert_ocr(existed.id, existed.path)
                tag_written = _auto_tag(existed, force=force_tag)
                _audit(user_id, existed.id, duplicate=True,
                       extra=(f'"auto_tag":"{tag_written}"' if tag_written else None))
                saved.append({"image_id": existed.id, "duplicate": True, "auto_tag": tag_written})
                continue

            # ===== DB 里有记录但文件丢失：修复路径并补全 =====
            if existed and not os.path.exists(existed.path):
                os.replace(tmp_path, dst_path)
                target_path = dst_path

                w, h, tpath = _gen_thumb(target_path, sha)
                try:
                    existed.path = target_path
                    existed.thumb_path = existed.thumb_path or tpath
                    existed.width = existed.width or w
                    existed.height = existed.height or h
                    existed.size_bytes = os.path.getsize(target_path)
                    existed.mime = existed.mime or (guess_type(file.filename)[0] or "application/octet-stream")
                    db.session.commit()
                except Exception:
                    db.session.rollback()

                _upsert_embedding_and_index(existed.id, existed.path)
                _upsert_ocr(existed.id, existed.path)
                tag_written = _auto_tag(existed, force=force_tag)
                _audit(user_id, existed.id, duplicate=False,
                       extra=(f'"auto_tag":"{tag_written}"' if tag_written else None))
                saved.append({"image_id": existed.id, "duplicate": False, "auto_tag": tag_written})
                continue

            # ===== 全新图片 =====
            os.replace(tmp_path, dst_path)
            target_path = dst_path

            width, height, thumb_path = _gen_thumb(target_path, sha)
            size_bytes = os.path.getsize(target_path)
            mime = guess_type(file.filename)[0] or "application/octet-stream"

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
                try:
                    if os.path.exists(target_path):
                        os.remove(target_path)
                    if thumb_path and os.path.exists(thumb_path):
                        os.remove(thumb_path)
                except Exception:
                    pass
                continue

            _audit(user_id, image.id, duplicate=False)

            _upsert_embedding_and_index(image.id, image.path)
            _upsert_ocr(image.id, image.path)
            tag_written = _auto_tag(image, force=force_tag)

            saved.append({"image_id": image.id, "duplicate": False, "auto_tag": tag_written})

        except Exception as e:
            db.session.rollback()
            saved.append({"error": str(e)})
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    return jsonify(saved=saved)
