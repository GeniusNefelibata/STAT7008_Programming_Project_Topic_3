# scripts/rebuild_faiss.py
import os, sys, pathlib, numpy as np, faiss

# --- ensure we import *your* local package "app", not site-packages/app ---
ROOT = pathlib.Path(__file__).resolve().parents[1]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from app import create_app, db
    from app.models import Embedding
except Exception as e:
    raise RuntimeError(f"Failed to import local 'app' package from {ROOT}: {e}")

app = create_app(light=True)  # DB only; no heavy model init

# Resolve index output path
INDEX_PATH = os.environ.get(
    "FAISS_INDEX_PATH",
    str(ROOT / "data" / "index" / "faiss.index")
)

with app.app_context():
    print(f"[DB] {app.config['SQLALCHEMY_DATABASE_URI']}")
    rows = (
        db.session.query(Embedding.image_id, Embedding.dim, Embedding.vector_blob)
        .order_by(Embedding.image_id.asc())
        .all()
    )
    if not rows:
        raise SystemExit("[FAISS] no embeddings, abort.")

    first_dim = None
    ids, vecs = [], []
    for iid, dim, blob in rows:
        v = np.frombuffer(blob, dtype="float32")
        if first_dim is None:
            first_dim = int(dim) if dim else v.shape[0]
        if v.shape[0] != first_dim:
            raise RuntimeError(f"Embedding dim mismatch for id={iid}: got {v.shape[0]}, expect {first_dim}")
        ids.append(int(iid))
        vecs.append(v)

    ids  = np.array(ids, dtype="int64")
    vecs = np.stack(vecs).astype("float32")

    print(f"[FAISS] rebuild: dim={first_dim}, ntotal={vecs.shape[0]}")
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(first_dim))  # cosine via unit-norm + inner product
    index.add_with_ids(vecs, ids)

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    print(f"[FAISS] wrote index to: {INDEX_PATH}")
