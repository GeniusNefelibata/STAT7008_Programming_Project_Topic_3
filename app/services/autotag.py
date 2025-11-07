# app/services/autotag.py
from __future__ import annotations
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
from . import embeddings as EMB  # uses encode_text / encode_image from your project

# -----------------------------------------------------------------------------
# 1) Your label set (aligned to your folders)
#    你可以随时增删；类名要与 image.category 的值一致，便于前端筛选/统计
# -----------------------------------------------------------------------------
LABELS: Dict[str, List[str]] = {
    "cats": [
        "a photo of a cat",
        "a cute domestic cat, feline animal",
        "a kitten with whiskers",
        "猫 猫咪 的照片",
    ],
    "dogs": [
        "a photo of a dog",
        "a cute domestic dog, canine animal",
        "a puppy with leash",
        "狗 小狗 的照片",
    ],
    "elephants": [
        "a photo of an elephant in the wild",
        "an elephant with trunk and tusks",
        "非洲或亚洲 大象 的照片",
    ],
    "flowers": [
        "a photo of flowers in bloom",
        "a close-up of colorful flower petals",
        "盛开的 花朵 花瓣 的特写照片",
    ],
    "forests": [
        "a photo of forest with trees and foliage",
        "dense woodland scenery",
        "森林 树木 林地 的风景照片",
    ],
    "fruits": [
        "a photo of assorted fruits",
        "fresh fruits like apples, bananas, oranges",
        "水果 果盘 新鲜水果 的照片",
    ],
    "humans": [
        "a photo of a person",
        "a portrait photo of a human",
        "人物 人像 肖像 的照片",
    ],
    "landscapes": [
        "a landscape photo of mountains, lakes or fields",
        "wide outdoor natural scenery",
        "风景 风光 山水 自然景观 的照片",
    ],
    "receipts": [
        "a store receipt printed on thermal paper with items and totals",
        "a small shopping receipt",
        "超市/商店 打印的小票 收据 的照片或扫描件",
    ],
    "seas": [
        "a photo of the sea or ocean",
        "coastline, beach and waves",
        "海 海洋 海岸 沙滩 海浪 的照片",
    ],
    "streets": [
        "a street scene in a city or town",
        "urban street with buildings and cars",
        "街道 城市街景 路面 的照片",
    ],
    # 兜底类（可保留）
    "other": [
        "an image that does not match the given categories",
        "miscellaneous content",
        "其他 杂项 难以归类 的图片",
    ],
}

# -----------------------------------------------------------------------------
# 2) Scoring knobs
# -----------------------------------------------------------------------------
AGGREGATION = "max"     # "max" or "mean"  —— 同一类多 prompt 的聚合方式
TOP_K = 3
THRESHOLD = 0.55        # 多标签阈值；>= 阈值的类会出现在 multilabel 里
PRIMARY_FALLBACK = "other"

# -----------------------------------------------------------------------------
# 3) Core helpers
# -----------------------------------------------------------------------------
def _to01(x: np.ndarray) -> np.ndarray:
    return (x + 1.0) / 2.0  # cosine/IP [-1,1] → [0,1] for readability

@lru_cache(maxsize=1)
def _label_name_list() -> Tuple[List[str], List[List[str]]]:
    # 缓存 label 名与其 prompts，便于外层循环
    names = list(LABELS.keys())
    prompts = [LABELS[n] for n in names]
    return names, prompts

# -----------------------------------------------------------------------------
# 4) Public APIs
# -----------------------------------------------------------------------------
def score_image(image_path: str) -> Dict[str, float]:
    """
    返回每个类的分数（0~1）。计算方式：
    - 先 encode 图像向量；
    - 每个类的多条 prompt 分别 encode 文本向量；
    - 相似度聚合（max/mean）。
    """
    q = EMB.encode_image(image_path)                # (dim,) normalized
    names, prompts_groups = _label_name_list()

    out: Dict[str, float] = {}
    for label, prompts in zip(names, prompts_groups):
        tvecs = np.stack([EMB.encode_text(p) for p in prompts], axis=0)  # (m, dim)
        sims = tvecs @ q                                                 # [-1, 1]
        sims01 = _to01(sims)                                             # [0, 1]
        agg = float(np.max(sims01) if AGGREGATION == "max" else np.mean(sims01))
        out[label] = agg
    return out

def predict_labels(image_path: str, top_k: int = TOP_K, threshold: float = THRESHOLD):
    """
    返回：
      primary    —— 分数最高的主类（或 fallback）
      top        —— 前 top_k 的 (label, score)
      multilabel —— 所有高于阈值的类（至少包含 primary）
      scores     —— 所有类的分数字典
    """
    scores = score_image(image_path)
    ranking = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = ranking[:max(1, int(top_k))]
    primary = (top[0][0] if top else PRIMARY_FALLBACK) or PRIMARY_FALLBACK

    ml = [lab for lab, s in scores.items() if s >= float(threshold)]
    if primary not in ml:
        ml.insert(0, primary)

    return {"primary": primary, "top": top, "multilabel": ml, "scores": scores}
