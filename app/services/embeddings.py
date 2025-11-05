# app/services/embeddings.py
from functools import lru_cache
import numpy as np
from PIL import Image

@lru_cache(maxsize=1)
def _load_model():
    from sentence_transformers import SentenceTransformer
    # 轻量稳定：CLIP ViT-B/32
    return SentenceTransformer("clip-ViT-B-32")

def encode_image(image_path: str) -> np.ndarray:
    model = _load_model()
    img = Image.open(image_path).convert("RGB")
    vec = model.encode([img], convert_to_numpy=True, normalize_embeddings=True)[0]
    return vec.astype("float32")

def encode_text(q: str) -> np.ndarray:
    model = _load_model()
    vec = model.encode([q], convert_to_numpy=True, normalize_embeddings=True)[0]
    return vec.astype("float32")
