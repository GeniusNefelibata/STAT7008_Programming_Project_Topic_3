# scripts/backfill_and_reindex.py
import os, sys, inspect
# ===== A) 强制 CPU 与静默处理 =====
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("EMBED_DEVICE", "cpu")

# ===== B) 把项目根插到 sys.path 前面，确保导入到本地 app 包 =====
ROOT = os.path.dirname(os.path.dirname(__file__))  # .../image_drive_starter
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sqlalchemy import text
import numpy as np

# 打印我们实际 import 到的 app 包位置，便于确认
import app as app_pkg
print("[IMPORT] app loaded from:", getattr(app_pkg, "__file__", app_pkg))

from app import create_app, db
from app.models import Image as ImageModel, Embedding
from app.services import embeddings as EMB
from app.faiss_store import FaissStore

app = create_app(light=True)
ctx = app.app_context()
ctx.push()

DB_URI = app.config["SQLALCHEMY_DATABASE_URI"]
INDEX_PATH = app.config["FAISS_INDEX_PATH"]
print(f"[DB] {DB_URI}")

# 1) 统计缺失 embeddings
n_images = db.session.execute(text("SELECT COUNT(*) FROM image")).scalar() or 0
n_embed  = db.session.execute(text("SELECT COUNT(*) FROM embedding")).scalar() or 0
print(f"[COUNT] images={n_images}, embeddings={n_embed}")

missing_rows = db.session.execute(text("""
    SELECT i.id, i.path
    FROM image i
    LEFT JOIN embedding e ON e.image_id = i.id
    WHERE e.image_id IS NULL
    ORDER BY i.id ASC
""")).fetchall()
print(f"[MISSING] embeddings to backfill: {len(missing_rows)}")

# 2) 回填缺失向量（CPU）
inserted = 0
for iid, path in missing_rows:
    try:
        vec = EMB.encode_image(path)  # 请确保 embeddings.py 内部用 device='cpu'
        db.session.add(Embedding(
            image_id=int(iid),
            model_name="clip-ViT-B-32",
            dim=len(vec),
            vector_blob=vec.tobytes(),
        ))
        db.session.commit()
        inserted += 1
    except Exception as e:
        db.session.rollback()
        print(f"  ! fail image #{iid}: {e}")

print(f"[BACKFILL] inserted {inserted} embeddings")

# 3) 全量重建 FAISS 索引
print(f"[FAISS] rebuild @ {INDEX_PATH}")
rows = db.session.execute(text(
    "SELECT image_id, vector_blob FROM embedding ORDER BY image_id ASC"
)).fetchall()
if not rows:
    print("[FAISS] no embeddings, abort.")
    ctx.pop()
    sys.exit(0)

ids = np.fromiter((int(r[0]) for r in rows), dtype=np.int64)
vecs = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
if vecs.dtype != np.float32:
    vecs = vecs.astype(np.float32, copy=False)

# 删除旧 index
try:
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)
except Exception:
    pass

# 新建并写入
fs = FaissStore(dim=vecs.shape[1], index_path=INDEX_PATH)

# 兼容不同 FaissStore 实现
if hasattr(fs, "add"):
    fs.add(ids, vecs)       # 你项目里我们就是这么用的
elif hasattr(fs, "build"):
    fs.build(ids, vecs)     # 有的实现叫 build
else:
    raise RuntimeError(
        f"FaissStore from {inspect.getsourcefile(FaissStore)} "
        f"has neither 'add' nor 'build'. You are importing the wrong class."
    )

# 保存并打印 ntotal
if hasattr(fs, "save"):
    fs.save()
info = fs.info() if hasattr(fs, "info") else {}
print(f"[FAISS] done. ntotal={info.get('ntotal', len(ids))}")

ctx.pop()


