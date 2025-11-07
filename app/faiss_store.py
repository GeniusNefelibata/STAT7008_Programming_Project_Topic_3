# app/faiss_store.py
from __future__ import annotations
import os
import numpy as np

try:
    import faiss  # pip install faiss-cpu
except Exception as e:
    faiss = None

class FaissStore:
    """
    Simple persistent FAISS wrapper (Inner-Product / cosine).
    - dim: embedding dim
    - index_path: file path to store/load the index
    - autosave: if True, save() after each add()
    API:
      - add(ids: np.ndarray[int64], vecs: np.ndarray[float32])
      - search(qvec: np.ndarray[float32], k: int) -> list[(id, score)]
      - save() / write_index()        # write to disk
      - load()                         # load from disk if exists
      - ntotal() -> int
    """
    def __init__(self, dim: int, index_path: str, metric: str = "IP"):
        self.dim = int(dim)
        self.index_path = os.path.abspath(index_path)
        self.metric = metric.upper()  # "IP" or "L2"
        self._index = None  # faiss.Index
        self._idmap = None  # faiss.IndexIDMap2

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        self._load_if_exists()

    # 兼容旧名字
    def write_index(self):  # alias
        return self.save()

    def save(self):
        if self._idmap is None:
            return
        faiss.write_index(self._idmap, self.index_path)

    def load(self):
        if faiss is None:
            return
        if self.index_path and os.path.exists(self.index_path):
            idx = faiss.read_index(self.index_path)
            # 统一成 IDMap
            if not isinstance(idx, faiss.IndexIDMap):
                idx = faiss.IndexIDMap(idx)
            self.index = idx

    @property
    def ntotal(self) -> int:
        if self._idmap is None:
            return 0
        return int(self._idmap.ntotal)

    def add(self, ids: np.ndarray, vecs: np.ndarray):
        """
        追加：不会清空已有索引。
        ids: int64 shape (n,)
        vecs: float32 shape (n, dim) —— 已归一化（若用 IP 作余弦）
        """
        self._ensure_loaded()
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        vecs = np.asarray(vecs, dtype=np.float32).reshape(-1, self.dim)
        if ids.shape[0] != vecs.shape[0]:
            raise ValueError("ids and vecs must have same length")

        # 追加
        self._idmap.add_with_ids(vecs, ids)
        # 立刻持久化，避免进程崩溃丢增量
        self.save()

    def search(self, qvec: np.ndarray, k: int = 12):
        self._ensure_loaded()
        if self.ntotal == 0:
            return []
        q = np.asarray(qvec, dtype=np.float32).reshape(1, self.dim)
        D, I = self._idmap.search(q, k)
        # 返回 [(id, score)]
        return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i != -1]

    def _new_base_index(self):
        if self.metric == "IP":
            base = faiss.IndexFlatIP(self.dim)
        else:
            base = faiss.IndexFlatL2(self.dim)
        return base

    def _ensure_loaded(self):
        if self._idmap is None:
            self._idmap = faiss.IndexIDMap2(self._new_base_index())

    def _load_if_exists(self):
        if os.path.isfile(self.index_path):
            idx = faiss.read_index(self.index_path)
            # 兼容“裸 base index”或已是 IDMap 的两种情况
            if isinstance(idx, faiss.IndexIDMap2):
                self._idmap = idx
            else:
                self._idmap = faiss.IndexIDMap2(idx)
        else:
            self._idmap = faiss.IndexIDMap2(self._new_base_index())
