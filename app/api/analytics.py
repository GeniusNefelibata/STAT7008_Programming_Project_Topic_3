# app/api/analytics.py
from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple

from flask import Blueprint, request, jsonify, Response
from sqlalchemy import func, select, and_

from .. import db
from ..models import Image

bp = Blueprint("analytics", __name__, url_prefix="/api/analytics")


# ----------------------------- helpers ---------------------------------
def _parse_int(qname: str, default: int) -> int:
    try:
        return int(request.args.get(qname, default))
    except Exception:
        return default


def _as_bool(name: str) -> bool:
    v = (request.args.get(name) or "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _localize_dt(dt_utc: datetime, tz_offset_min: int) -> datetime:
    """
    将 UTC 时间按分钟偏移转换到“本地时间”（不做真时区，避免平台差异）。
    tz_offset_min: 例如 +480(东八区) / 0(UTC)
    """
    return (dt_utc or datetime.utcnow()).replace(tzinfo=timezone.utc) + timedelta(minutes=tz_offset_min)


def _daterange(start_date: datetime, days: int) -> List[datetime]:
    return [start_date + timedelta(days=i) for i in range(days)]


def _fmt_date_dmy(d: datetime) -> str:
    # 形如 2025/11/4（不补零）
    return f"{d.year}/{d.month}/{d.day}"


def _iso_year_week(d: datetime) -> Tuple[int, int]:
    y, w, _ = d.isocalendar()
    return int(y), int(w)


# ----------------------------- endpoints --------------------------------

@bp.get("/category_breakdown")
def category_breakdown():
    """
    返回 [{category, count}]；与前端 Gallery 的筛选联动。
    """
    q = db.session.query(Image.category, func.count(Image.id)) \
        .group_by(Image.category) \
        .order_by(func.count(Image.id).desc())
    rows = [{"category": c or "uncategorized", "count": int(n or 0)} for c, n in q.all()]
    return jsonify(rows)


# ---- Summary core ------------------------------------------------------

def _compute_summary(days: int, tz_offset_min: int, compact_weeks: bool, with_cum: bool) -> Dict:
    """
    计算统计摘要：
      - totals
      - by_day（可选加入累计列 cum）
      - by_week（可选 compact 去除 0 周）
      - size_bins/mp_bins/mime/category
    注意：所有日期粒度都以“本地时间偏移”为准（仅用于分桶显示）。
    """
    # --- totals ---
    total_count = db.session.query(func.count(Image.id)).scalar() or 0
    first_dt_utc = db.session.query(func.min(Image.created_at)).scalar()
    last_dt_utc = db.session.query(func.max(Image.created_at)).scalar()

    # time window：过去 N 天，右开区间今日不含（仅用于“展示范围起点”）
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    start_local = _localize_dt(now_utc, tz_offset_min) - timedelta(days=days)
    bucket_days = _daterange(start_local.replace(hour=0, minute=0, second=0, microsecond=0), days)

    # 将窗口内的 created_at 拉到 Python，用“本地日期”计数
    begin_utc = (start_local - timedelta(minutes=tz_offset_min)).replace(tzinfo=timezone.utc)
    rows = db.session.execute(
        select(Image.created_at)
        .where(Image.created_at.isnot(None))
        .where(Image.created_at >= begin_utc - timedelta(days=1))
    ).all()

    day_counter: Dict[str, int] = {}
    for (created_at_utc,) in rows:
        if not created_at_utc:
            continue
        d_local = _localize_dt(created_at_utc, tz_offset_min)
        if d_local.date() < start_local.date():
            continue
        key = _fmt_date_dmy(d_local)
        day_counter[key] = day_counter.get(key, 0) + 1

    # by_day：填满窗口；可选累计
    by_day = []
    running = 0
    for d in bucket_days:
        k = _fmt_date_dmy(d)
        c = int(day_counter.get(k, 0))
        running += c
        if with_cum:
            by_day.append({"date": k, "count": c, "cum": running})
        else:
            by_day.append({"date": k, "count": c})

    # by_week：用“本地日期”的 ISO 周聚合
    week_counter: Dict[str, int] = {}
    for d in bucket_days:
        y, w = _iso_year_week(d)
        wk = f"{y}-W{w:02d}"
        week_counter[wk] = week_counter.get(wk, 0) + int(day_counter.get(_fmt_date_dmy(d), 0))
    by_week_all = [{"week": k, "count": int(v)} for k, v in sorted(week_counter.items())]
    by_week = by_week_all if not compact_weeks else [r for r in by_week_all if r["count"] > 0]

    # size bins —— 用 ORM 表达式 + coalesce，避免 raw SQL
    sb = func.coalesce(Image.size_bytes, 0)
    size_filters = [
        ("<256KB", sb < 262_144),
        ("256KB–1MB", and_(sb >= 262_144, sb < 1_048_576)),
        ("1–2MB", and_(sb >= 1_048_576, sb < 2_097_152)),
        ("2–5MB", and_(sb >= 2_097_152, sb < 5_242_880)),
        ("5–10MB", and_(sb >= 5_242_880, sb < 10_485_760)),
        ("≥10MB", sb >= 10_485_760),
    ]
    size_rows = []
    for label, flt in size_filters:
        cnt = db.session.query(func.count(Image.id)).filter(flt).scalar() or 0
        size_rows.append({"bin": label, "count": int(cnt)})

    # megapixels bins —— width/height 可能为 NULL，coalesce 成 0
    mp = func.coalesce(Image.width, 0) * func.coalesce(Image.height, 0)
    mp_filters = [
        ("<0.5MP", mp < 500_000),
        ("0.5–1MP", and_(mp >= 500_000, mp < 1_000_000)),
        ("1–2MP", and_(mp >= 1_000_000, mp < 2_000_000)),
        ("2–4MP", and_(mp >= 2_000_000, mp < 4_000_000)),
        ("4–8MP", and_(mp >= 4_000_000, mp < 8_000_000)),
        ("≥8MP", mp >= 8_000_000),
    ]
    mp_rows = []
    for label, flt in mp_filters:
        cnt = db.session.query(func.count(Image.id)).filter(flt).scalar() or 0
        mp_rows.append({"bin": label, "count": int(cnt)})

    # mime
    mime_rows = []
    for m, n in db.session.query(Image.mime, func.count(Image.id)).group_by(Image.mime).all():
        mime_rows.append({"mime": m or "unknown", "count": int(n or 0)})
    mime_rows.sort(key=lambda r: (-r["count"], r["mime"]))

    # category
    cat_rows = []
    for c, n in db.session.query(Image.category, func.count(Image.id)).group_by(Image.category).all():
        cat_rows.append({"category": c or "uncategorized", "count": int(n or 0)})
    cat_rows.sort(key=lambda r: (-r["count"], r["category"]))

    totals = {
        "count": int(total_count),
        "first_created_at": (first_dt_utc.isoformat(timespec="seconds") if first_dt_utc else None),
        "last_created_at": (last_dt_utc.isoformat(timespec="seconds") if last_dt_utc else None),
        "window_days": int(days),
        "window_since": _fmt_date_dmy(start_local),
    }

    return {
        "totals": totals,
        "by_day": by_day,
        "by_week": by_week,
        "size_bins": size_rows,
        "mp_bins": mp_rows,
        "mime": mime_rows,
        "category": cat_rows,
    }


@bp.get("/summary")
def analytics_summary():
    """
    GET /api/analytics/summary
      ?days=60            # 统计窗口天数
      ?tz_min=0           # 本地时间相对 UTC 的分钟偏移（例如东八区传 480）
      ?compact=1          # 去掉 by_week 中 count=0 的周
      ?with_cum=1         # 在 by_day 中增加累计列 cum
    """
    days = _parse_int("days", 60)
    tz_min = _parse_int("tz_min", 0)
    compact = _as_bool("compact")
    with_cum = _as_bool("with_cum")

    data = _compute_summary(days=days, tz_offset_min=tz_min, compact_weeks=compact, with_cum=with_cum)
    return jsonify(data)


# ---- Export ------------------------------------------------------------

def _to_csv_blocks(summary: Dict, with_cum: bool) -> str:
    """
    将 summary 写成“分节”CSV 文本；with_cum 决定 by_day 是否输出 cum 列。
    """
    buf = io.StringIO()
    w = csv.writer(buf)

    # totals
    w.writerow(["[totals]"])
    w.writerow(["key", "value"])
    for k, v in summary["totals"].items():
        w.writerow([k, v])
    w.writerow([])

    # by_day
    w.writerow(["[by_day]"])
    if with_cum:
        w.writerow(["date", "count", "cum"])
        for r in summary["by_day"]:
            w.writerow([r["date"], r["count"], r.get("cum")])
    else:
        w.writerow(["date", "count"])
        for r in summary["by_day"]:
            w.writerow([r["date"], r["count"]])
    w.writerow([])

    # by_week
    w.writerow(["[by_week]"])
    w.writerow(["week", "count"])
    for r in summary["by_week"]:
        w.writerow([r["week"], r["count"]])
    w.writerow([])

    # size_bins
    w.writerow(["[size_bins]"])
    w.writerow(["bin", "count"])
    for r in summary["size_bins"]:
        w.writerow([r["bin"], r["count"]])
    w.writerow([])

    # mp_bins
    w.writerow(["[mp_bins]"])
    w.writerow(["bin", "count"])
    for r in summary["mp_bins"]:
        w.writerow([r["bin"], r["count"]])
    w.writerow([])

    # mime
    w.writerow(["[mime]"])
    w.writerow(["mime", "count"])
    for r in summary["mime"]:
        w.writerow([r["mime"], r["count"]])
    w.writerow([])

    # category
    w.writerow(["[category]"])
    w.writerow(["category", "count"])
    for r in summary["category"]:
        w.writerow([r["category"], r["count"]])

    return buf.getvalue()


@bp.get("/export")
def analytics_export():
    """
    GET /api/analytics/export?target=summary&format=csv&days=60&tz_min=0&compact=1&with_cum=1
      - target=summary（目前仅实现此目标）
      - format=csv|json
      - 其余参数同 /summary
    """
    target = (request.args.get("target") or "summary").lower()
    if target != "summary":
        return jsonify(error="only target=summary is supported for now"), 400

    fmt = (request.args.get("format") or "csv").lower()
    days = _parse_int("days", 60)
    tz_min = _parse_int("tz_min", 0)
    compact = _as_bool("compact")
    with_cum = _as_bool("with_cum")

    data = _compute_summary(days=days, tz_offset_min=tz_min, compact_weeks=compact, with_cum=with_cum)

    if fmt == "json":
        return jsonify(data)

    csv_text = _to_csv_blocks(data, with_cum=with_cum)
    fn = f"analytics_summary_{days}d.csv"
    return Response(
        csv_text,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{fn}"'}
    )
