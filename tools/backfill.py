# tools/backfill.py
import os
import time
import argparse
import numpy as np
from sqlalchemy import select

# ---- 保证能 import app 包 ----
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import create_app, db
from app.models import Image, Embedding, OcrText
from app.services import embeddings as EMB
from app.services import ocr as OCR


def infer_dim(vec_model) -> int:
    """稳健推断 embedding 维度：模型 -> 服务常量 -> 兜底 512"""
    dim = getattr(vec_model, "dim", None)
    if not dim:
        dim = getattr(EMB, "DIM", None)
    if not dim:
        dim = 512
    return int(dim)


def add_to_faiss(faiss_store, pairs):
    """
    批量写入 FAISS。
    pairs: list[(id:int, vec:np.ndarray float32 [dim])]
    """
    if not pairs or faiss_store is None:
        return 0
    ids = np.array([pid for pid, _ in pairs], dtype=np.int64).reshape(-1)
    vecs = np.stack([v for _, v in pairs], axis=0).astype("float32", copy=False)
    faiss_store.add(ids, vecs)
    return ids.size


def main():
    parser = argparse.ArgumentParser(
        description="Backfill embeddings/OCR in batches and (optionally) rebuild FAISS index."
    )
    # 范围与分批
    parser.add_argument("--offset", type=int, default=0,
                        help="从第 offset 条记录开始（基于 id 升序的逻辑偏移）。")
    parser.add_argument("--limit", type=int, default=None,
                        help="最多处理 limit 条记录（默认全部）。")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="单批处理条数（默认 200）。")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="每批之间休眠秒数（默认 0）。")
    parser.add_argument("--category", type=str, default=None,
                        help="仅处理指定类别（如 receipts）。")

    # 开关
    parser.add_argument("--no-emb", action="store_true", help="跳过向量计算，仅做 OCR/FAISS。")
    parser.add_argument("--no-ocr", action="store_true", help="跳过 OCR。")
    parser.add_argument("--rebuild", action="store_true", help="先删除并重建 FAISS 索引文件。")

    # 设备覆盖（可选）
    parser.add_argument("--device", type=str, default=None,
                        help="覆盖 EMB/VecModel 设备（如 cuda / cpu）。")

    args = parser.parse_args()

    if args.device:
        os.environ["EMBED_DEVICE"] = args.device  # 供 app.vec / services.embeddings 使用

    app = create_app()

    with app.app_context():
        # 组件
        vec_model = app.extensions.get("vec_model")
        faiss_store = app.extensions.get("faiss_store")
        embed_dim = infer_dim(vec_model)

        # 重建索引（可选）
        if args.rebuild:
            index_path = app.config.get("FAISS_INDEX_PATH")
            try:
                if index_path and os.path.exists(index_path):
                    os.remove(index_path)
                    print(f"[rebuild] removed old index: {index_path}")
            except Exception as e:
                print(f"[rebuild] failed to remove index: {e}")
            from app.faiss_store import FaissStore
            faiss_store = FaissStore(dim=embed_dim, index_path=index_path)
            app.extensions["faiss_store"] = faiss_store
            print("[rebuild] created fresh FAISS index")

        if faiss_store is None:
            from app.faiss_store import FaissStore
            index_path = app.config.get("FAISS_INDEX_PATH")
            faiss_store = FaissStore(dim=embed_dim, index_path=index_path)
            app.extensions["faiss_store"] = faiss_store

        # 基础查询
        stmt = select(Image).order_by(Image.id.asc())
        if args.category:
            stmt = stmt.where(Image.category == args.category)
        if args.offset:
            stmt = stmt.offset(args.offset)
        if args.limit:
            stmt = stmt.limit(args.limit)

        rows = db.session.execute(stmt).scalars().all()
        total = len(rows)
        print(f"[plan] total rows to process: {total} "
              f"(offset={args.offset} limit={args.limit} category={args.category})")

        done_vec = done_ocr = added_idx = 0
        processed = 0

        # 分批处理
        for i in range(0, total, args.batch_size):
            chunk = rows[i: i + args.batch_size]
            faiss_pairs = []

            for img in chunk:
                processed += 1
                vec = None

                # 1) 向量
                if not args.no_emb:
                    emb_row = db.session.get(Embedding, img.id)
                    if emb_row is None:
                        try:
                            vec = EMB.encode_image(img.path)  # float32 [dim]，内部已 L2 归一
                            db.session.add(Embedding(
                                image_id=img.id,
                                model_name="clip-ViT-B-32",
                                dim=len(vec),
                                vector_blob=vec.tobytes()
                            ))
                            db.session.commit()
                            done_vec += 1
                        except Exception as e:
                            db.session.rollback()
                            print(f"[embedding] skip id={img.id}: {e}")
                            vec = None
                    else:
                        try:
                            arr = np.frombuffer(emb_row.vector_blob, dtype=np.float32)
                            if emb_row.dim and arr.size != emb_row.dim:
                                arr = arr[:emb_row.dim]
                            vec = arr
                        except Exception as e:
                            print(f"[embedding] corrupt row id={img.id}: {e}")
                            vec = None

                # 2) OCR
                if not args.no_ocr:
                    if db.session.get(OcrText, img.id) is None:
                        try:
                            txt = OCR.extract_text(img.path) or ""
                            db.session.add(OcrText(image_id=img.id, text=txt))
                            db.session.commit()
                            done_ocr += 1
                        except Exception as e:
                            db.session.rollback()
                            print(f"[ocr] skip id={img.id}: {e}")

                # 3) FAISS 暂存
                if vec is not None:
                    faiss_pairs.append((int(img.id), vec))

            # 批量落入 FAISS
            try:
                added = add_to_faiss(faiss_store, faiss_pairs)
                added_idx += added
            except Exception as e:
                print(f"[faiss] add batch failed: {e}")

            print(f"[batch] processed={processed}/{total} "
                  f"(+faiss {added_idx}, +emb {done_vec}, +ocr {done_ocr})")

            if args.sleep > 0:
                time.sleep(args.sleep)

        stats = faiss_store.stats() if faiss_store is not None else {}
        print(f"[done] embeddings_added={done_vec}, "
              f"ocr_added={done_ocr}, faiss_added={added_idx}, index_stats={stats}")


if __name__ == "__main__":
    main()
