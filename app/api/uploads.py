# app/api/uploads.py
from flask import Blueprint, request, current_app, jsonify
import os, hashlib, tempfile
from mimetypes import guess_type
from werkzeug.utils import secure_filename
from PIL import Image as PILImage
import numpy as np

from .. import db
from ..models import Image as ImageModel, AuditLog, Embedding, OcrText
from flask_jwt_extended import jwt_required, get_jwt_identity

# 入库阶段用到的服务（你已有）
from ..services import embeddings as EMB
from ..services import ocr as OCR

# ✅ 一定要先定义 blueprint，再写路由
bp = Blueprint("uploads", __name__)

def sha256_file(fpath: str) -> str:
    h = hashlib.sha256()
    with open(fpath, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

@bp.post("/api/upload")
@jwt_required(optional=True)
def upload():
    """
    多文件上传流程：
      1) 临时文件建在 UPLOAD_DIR（同盘，避免 WinError 17）
      2) sha256 内容寻址保存 & 去重
      3) 生成缩略图
      4) 写入 Image + Audit
      5) 入库阶段仅计算一次：Embedding + OCR 并持久化
      6) 将向量加入 FAISS 持久化索引（上传即可检索）
    """
    files = request.files.getlist("file")
    if not files:
        return jsonify(error="no files"), 400

    saved = []
    user_id = get_jwt_identity()

    # 模型与索引（应用启动时在 app.extensions 里注册过）
    vm = current_app.extensions.get('vec_model')
    fs = current_app.extensions.get('faiss_store')

    for file in files:
        if not file or not file.filename:
            saved.append({"error": "empty filename"})
            continue

        # 临时文件放到上传目录（与目标同盘）
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix="ing_", suffix="_upload",
            dir=current_app.config["UPLOAD_DIR"]
        )
        os.close(tmp_fd)

        try:
            file.save(tmp_path)

            sha = sha256_file(tmp_path)
            subdir = os.path.join(current_app.config["UPLOAD_DIR"], sha[:2], sha[2:4])
            os.makedirs(subdir, exist_ok=True)
            dst = os.path.join(subdir, sha)

            # 去重：已存在则补齐缺失的向量/OCR/索引 后直接返回
            existed = ImageModel.query.filter_by(sha256=sha).first()
            if existed and os.path.exists(dst):
                # —— 补 embeddings 表
                try:
                    has_vec = db.session.get(Embedding, existed.id)
                    if has_vec is None:
                        vec = EMB.encode_image(existed.path)
                        db.session.add(Embedding(
                            image_id=existed.id, model_name="clip-ViT-B-32",
                            dim=len(vec), vector_blob=vec.tobytes()
                        ))
                        db.session.commit()
                        # 写入 FAISS 索引
                        if fs is not None:
                            ids = np.array([int(existed.id)], dtype=np.int64)
                            fs.add(ids, vec.reshape(1, -1).astype("float32"))
                except Exception:
                    db.session.rollback()

                # —— 补 OCR 表
                try:
                    has_ocr = db.session.get(OcrText, existed.id)
                    if has_ocr is None:
                        txt = OCR.extract_text(existed.path) or ""
                        db.session.add(OcrText(image_id=existed.id, text=txt))
                        db.session.commit()
                except Exception:
                    db.session.rollback()

                # —— 审计
                db.session.add(AuditLog(
                    user_id=user_id, action="upload",
                    target_type="image", target_id=existed.id,
                    ip=request.remote_addr, ua=request.headers.get("User-Agent"),
                    extra_json='{"duplicate": true}'
                ))
                db.session.commit()

                # —— 为防止索引里还没有该 id：再尝试一次加入索引（即便是重复 add，FAISS 也能工作）
                try:
                    if fs is not None and vm is not None:
                        vec = EMB.encode_image(existed.path)  # 也可 vm.embed_image
                        ids = np.array([int(existed.id)], dtype=np.int64)
                        fs.add(ids, vec.reshape(1, -1).astype("float32"))
                except Exception:
                    pass

                saved.append({"image_id": existed.id, "duplicate": True})
                # 清理临时文件并下一张
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
                continue

            # 同盘移动，不会触发 WinError 17
            os.replace(tmp_path, dst)
            target_path = dst

            # 缩略图
            width = height = None
            thumb_path = None
            try:
                with PILImage.open(target_path) as im:
                    width, height = im.size
                    im.thumbnail((512, 512))
                    thumb_dir = os.path.join(current_app.config["THUMB_DIR"], sha[:2], sha[2:4])
                    os.makedirs(thumb_dir, exist_ok=True)
                    thumb_path = os.path.join(thumb_dir, f"{sha}.jpg")
                    im.convert("RGB").save(thumb_path, "JPEG", quality=85)
            except Exception:
                thumb_path = None

            size_bytes = os.path.getsize(target_path)
            # 用原始文件名做 mime 猜测，避免用哈希路径识别失败
            mime = guess_type(secure_filename(file.filename))[0] or "application/octet-stream"

            # 写入 Image
            image = ImageModel(
                user_id=user_id, sha256=sha, path=target_path, thumb_path=thumb_path,
                width=width, height=height, size_bytes=size_bytes, mime=mime
            )
            db.session.add(image)
            db.session.commit()

            # 审计
            db.session.add(AuditLog(
                user_id=user_id, action="upload",
                target_type="image", target_id=image.id,
                ip=request.remote_addr, ua=request.headers.get("User-Agent"),
                extra_json='{"duplicate": false}'
            ))
            db.session.commit()

            # ===== 入库阶段：向量 + OCR =====
            # 向量入库 + 加入 FAISS 索引
            try:
                # 你已有的编码函数（内部已归一化）
                vec = EMB.encode_image(target_path)
                db.session.add(Embedding(
                    image_id=image.id, model_name="clip-ViT-B-32",
                    dim=len(vec), vector_blob=vec.tobytes()
                ))
                db.session.commit()

                # —— 写入 FAISS 持久化索引（上传即检索可用）
                if fs is not None:
                    ids = np.array([int(image.id)], dtype=np.int64)
                    fs.add(ids, vec.reshape(1, -1).astype("float32"))
            except Exception:
                db.session.rollback()

            # OCR 入库
            try:
                txt = OCR.extract_text(target_path) or ""
                db.session.add(OcrText(image_id=image.id, text=txt))
                db.session.commit()
            except Exception:
                db.session.rollback()
            # =================================

            saved.append({"image_id": image.id, "duplicate": False})

        except Exception as e:
            db.session.rollback()
            saved.append({"error": str(e)})

        finally:
            # 正常 replace 后 tmp_path 不在；若仍在则清理
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    return jsonify(saved=saved)
