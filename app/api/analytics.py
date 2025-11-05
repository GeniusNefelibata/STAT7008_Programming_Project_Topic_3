# app/api/analytics.py
from __future__ import annotations
from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required
from sqlalchemy import func, select
from .. import db
from ..models import Image as ImageModel

bp = Blueprint("analytics", __name__)

def _percentiles(values: list[int | float], ps=(0.5, 0.9, 0.99)):
    if not values:
        return {f"p{int(p*100)}": None for p in ps}
    vs = sorted(float(v) for v in values)
    n = len(vs)
    out = {}
    for p in ps:
        # 最近邻分位（简化版，避免依赖 numpy）
        idx = min(n - 1, max(0, int(round(p * (n - 1)))))
        out[f"p{int(p*100)}"] = vs[idx]
    return out

@bp.get("/api/analytics/summary")
@jwt_required(optional=True)
def summary():
    """
    统计摘要：
      - 总数 / 总体积 / 平均尺寸(像素&KB)
      - 按类别计数
      - 大小分布分位数（KB）
    """
    # 1) 总体摘要
    total = db.session.execute(select(func.count(ImageModel.id))).scalar_one()
    total_bytes = db.session.execute(select(func.coalesce(func.sum(ImageModel.size_bytes), 0))).scalar_one()
    avg_w, avg_h = db.session.execute(
        select(func.coalesce(func.avg(ImageModel.width), 0),
               func.coalesce(func.avg(ImageModel.height), 0))
    ).one()

    # 2) 按类别分布
    rows = db.session.execute(
        select(ImageModel.category, func.count(ImageModel.id))
        .group_by(ImageModel.category)
        .order_by(func.count(ImageModel.id).desc())
    ).all()
    by_category = [{"category": (c or "uncategorized"), "count": int(n)} for c, n in rows]

    # 3) 大小分位（拉出所有 size_bytes 做分位，数据量 2k 级足够轻）
    sizes = db.session.execute(select(ImageModel.size_bytes)).scalars().all()
    kb = [ (s or 0) / 1024.0 for s in sizes ]
    pct = _percentiles(kb, ps=(0.5, 0.9, 0.99))

    data = {
        "totals": {
            "images": int(total),
            "bytes": int(total_bytes),
            "mb": round(total_bytes / (1024 * 1024), 2),
        },
        "dimensions": {
            "avg_width": round(float(avg_w or 0), 1),
            "avg_height": round(float(avg_h or 0), 1),
        },
        "sizes_kb": {
            "avg": round(sum(kb) / len(kb), 1) if kb else None,
            **{k: (round(v, 1) if v is not None else None) for k, v in pct.items()}
        },
        "by_category": by_category,
    }
    return jsonify(data)

@bp.get("/api/analytics/category_breakdown")
@jwt_required(optional=True)
def category_breakdown():
    """
    各类别更多细节：计数 + 取前几张样例（id）
    """
    rows = db.session.execute(
        select(ImageModel.category, func.count(ImageModel.id))
        .group_by(ImageModel.category)
        .order_by(func.count(ImageModel.id).desc())
    ).all()

    out = []
    for c, n in rows:
        ids = db.session.execute(
            select(ImageModel.id)
            .where(ImageModel.category == c)
            .order_by(ImageModel.id.desc())
            .limit(8)
        ).scalars().all()
        out.append({
            "category": c or "uncategorized",
            "count": int(n),
            "sample_ids": [int(x) for x in ids],
        })
    return jsonify(out)
