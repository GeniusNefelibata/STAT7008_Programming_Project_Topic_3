# app/vec.py
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch

class VecModel:
    def __init__(self, model_name: str = "clip-ViT-B-32", device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        # 统一输出维度
        self.dim = self.model.get_sentence_embedding_dimension()

    @torch.no_grad()
    def embed_image(self, path: str) -> np.ndarray:
        """返回 L2 归一化后的 float32 向量（用于 cosine/IP 检索）"""
        img = Image.open(path).convert("RGB")
        vec = self.model.encode(img, convert_to_numpy=True, normalize_embeddings=True)
        return vec.astype("float32")

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        return vec.astype("float32")
