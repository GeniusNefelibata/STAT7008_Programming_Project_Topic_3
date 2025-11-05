# app/faiss_store.py
from __future__ import annotations
import os
from typing import List, Tuple, Optional

import numpy as np

try:
    import faiss  # pip install faiss-cpu
except Exception as e:
    faiss = None

class FaissStore:
    """
    Minimal FAISS wrapper.
    - Uses Inner Product (IP) for cosine-like search (expects unit-normalized vectors).
    - Exposes .open(path) /.load(path) to (re)load an index file.
    - Provides .search(vec, k) -> List[(id, score)]  (score in FAISS's metric space).
    """

    def __init__(self, dim: int, index_path: Optional[str] = None):
        self.dim = int(dim)
        self.index_path = index_path
        self.index = None  # type: Optional[faiss.Index]
        self.metric = "IP"  # for diagnostics only
        self.ntotal = 0

        if index_path and os.path.exists(index_path):
            self.open(index_path)

    # aliases for compatibility
    def load(self, path: str) -> None:
        self.open(path)

    def open(self, path: str) -> None:
        if faiss is None:
            raise RuntimeError("faiss not installed (pip install faiss-cpu)")
        if not os.path.exists(path):
            raise FileNotFoundError(f"FAISS index not found: {path}")

        idx = faiss.read_index(path)
        # Optional: ensure it's an IndexIDMap (so ids == your image ids)
        # If your file is already an IDMap, this is a no-op.
        if not isinstance(idx, faiss.IndexIDMap):
            # keep as-is; many pipelines already save an IDMap
            pass

        # Store & stats
        self.index = idx
        try:
            self.ntotal = int(getattr(idx, "ntotal", 0))
        except Exception:
            self.ntotal = 0
        self.index_path = path

    def is_ready(self) -> bool:
        return self.index is not None and self.ntotal > 0

    def _ensure_ready(self) -> None:
        if not self.index:
            raise RuntimeError("FAISS index is not loaded. Call .open(index_path) first.")

    @staticmethod
    def _as_row(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype="float32").reshape(-1).astype("float32")
        return v.reshape(1, -1)

    def search(self, vec, k: int = 12) -> List[Tuple[int, float]]:
        """
        vec: 1-D embedding (float32), already unit-normalized if using cosine/IP.
        returns: [(id, score)]  -- score is FAISS distance/similarity (IP here).
        """
        self._ensure_ready()
        q = self._as_row(vec)

        # Some indices need normalization if you want cosine; assume caller already did it.
        distances, ids = self.index.search(q, k)  # shapes: (1,k), (1,k)
        ids = ids[0]
        distances = distances[0]

        out: List[Tuple[int, float]] = []
        for i, d in zip(ids, distances):
            if int(i) < 0:
                continue  # FAISS uses -1 for empty
            out.append((int(i), float(d)))
        return out
