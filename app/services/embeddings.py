# app/services/embeddings.py
from __future__ import annotations
import os
from functools import lru_cache
from typing import Tuple

import numpy as np
from PIL import Image


def _env(name: str, default: str) -> str:
    v = os.environ.get(name)
    return v if v and v.strip() else default


# ---- 环境配置：默认 CPU、可通过环境变量切换 ----
DEFAULT_MODEL = "sentence-transformers/clip-ViT-B-32"  # 比 "clip-ViT-B-32" 更稳的官方命名
DEFAULT_DEVICE = _env("EMBED_DEVICE", "cpu").lower()   # cpu | cuda (如强制 CPU，设 EMBED_DEVICE=cpu)
MODEL_NAME = _env("EMBED_MODEL", DEFAULT_MODEL)

# 降噪：避免 HuggingFace 的多进程 tokenizer 警告
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """确保向量做 L2 归一化（即使上游已归一化，再保险一次）"""
    if x.ndim == 1:
        n = float(np.linalg.norm(x) + eps)
        return (x / n).astype("float32", copy=False)
    # batch 情况
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return (x / n).astype("float32", copy=False)


@lru_cache(maxsize=1)
def _load_model_cached() -> Tuple[object, str]:
    """
    返回 (model, device_used)。如果指定了 GPU 但初始化失败，会自动回退 CPU，并缓存结果。
    """
    from sentence_transformers import SentenceTransformer

    # 先按环境变量尝试
    dev = DEFAULT_DEVICE
    try:
        model = SentenceTransformer(MODEL_NAME, device=dev)
        # 轻探测一次，确保模型 ready（不会真正编译大算子）
        _ = getattr(model, "get_sentence_embedding_dimension", lambda: None)()
        return model, dev
    except Exception as e:
        # GPU 上经常因 CUDA 能力不匹配而失败：回退 CPU
        # 仅当不是 CPU 且失败时回退；CPU 失败就原样抛出
        if dev != "cpu":
            try:
                model = SentenceTransformer(MODEL_NAME, device="cpu")
                return model, "cpu"
            except Exception as e2:
                # 两次都失败，抛出第一次的详细异常更有价值
                raise RuntimeError(
                    f"Failed to init embedding model on device '{dev}', and CPU fallback also failed: {e2}"
                ) from e
        raise


def _ensure_rgb_image(image_path: str) -> Image.Image:
    try:
        img = Image.open(image_path)
    except Exception as e:
        raise ValueError(f"Cannot open image: {image_path} ({e})")
    try:
        return img.convert("RGB")
    except Exception:
        # 少见：GIF/P 模式等转换失败时，直接抛出
        raise ValueError(f"Cannot convert image to RGB: {image_path}")


def _encode_any(model, payload, normalize: bool = True) -> np.ndarray:
    """
    统一编码入口；payload 可以是 str（文本）或 PIL.Image（图像）。
    SentenceTransformers 的 CLIP 模型支持混合输入，但这里做了显式区分更稳。
    """
    # 注意：SentenceTransformer.encode 支持 list；保持 batch 形式
    if isinstance(payload, str):
        vec = model.encode([payload], convert_to_numpy=True, normalize_embeddings=False)[0]
    else:
        # 认为是 PIL.Image
        vec = model.encode([payload], convert_to_numpy=True, normalize_embeddings=False)[0]
    vec = vec.astype("float32", copy=False)
    return _l2_normalize(vec) if normalize else vec


def encode_image(image_path: str) -> np.ndarray:
    """
    图像 -> 向量 (float32, L2 归一化)
    - 强制在 CPU 或按环境变量指定设备加载；
    - 自动回退 CPU，避免 CUDA 能力不匹配导致报错。
    """
    model, _device = _load_model_cached()
    img = _ensure_rgb_image(image_path)
    return _encode_any(model, img, normalize=True)


def encode_text(q: str) -> np.ndarray:
    """
    文本 -> 向量 (float32, L2 归一化)
    """
    if not isinstance(q, str) or not q.strip():
        raise ValueError("Empty text for encoding.")
    model, _device = _load_model_cached()
    return _encode_any(model, q.strip(), normalize=True)
