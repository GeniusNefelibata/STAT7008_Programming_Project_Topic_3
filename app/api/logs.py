# app/api/logs.py
from __future__ import annotations
from flask import Blueprint, request, jsonify, Response
from datetime import datetime
from sqlalchemy import desc
import csv, io, json

from ..extensions import db
from ..models import AuditLog

bp = Blueprint("logs", __name__)

def _parse_dt(s):
    # 接收 ISO 或 “YYYY-MM-DD”
    if not s: return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            return None

@bp.get("/audit")
def audit_list():
    """
    GET /api/audit?limit=200&action=download&level=INFO&from=2025-11-01&to=2025-11-30&user_id=1&q=cat
    """
    q = AuditLog.query

    # 过滤
    action = request.args.get("action")
    level  = request.args.get("level")
    user_id = request.args.get("user_id", type=int)
    status = request.args.get("status")
    s_from = _parse_dt(request.args.get("from"))
    s_to   = _parse_dt(request.args.get("to"))
    kw     = request.args.get("q")

    if action: q = q.filter(AuditLog.action == action)
    if level:  q = q.filter(AuditLog.level  == level)
    if status: q = q.filter(AuditLog.status == status)
    if user_id is not None: q = q.filter(AuditLog.user_id == user_id)
    if s_from: q = q.filter(AuditLog.created_at >= s_from)
    if s_to:   q = q.filter(AuditLog.created_at <= s_to)
    if kw:
        like = f"%{kw}%"
        q = q.filter(
            (AuditLog.message.ilike(like)) |
            (AuditLog.meta_json.ilike(like)) |
            (AuditLog.ua.ilike(like))
        )

    limit = min(request.args.get("limit", 200, type=int), 1000)
    rows = q.order_by(desc(AuditLog.id)).limit(limit).all()

    def _row(x: AuditLog):
        return {
            "id": x.id,
            "ts": x.created_at.isoformat() if x.created_at else None,
            "user_id": x.user_id,
            "action": x.action, "level": x.level, "status": x.status,
            "target_type": x.target_type, "target_id": x.target_id,
            "ip": x.ip, "ua": x.ua,
            "message": x.message,
            "meta": json.loads(x.meta_json or "{}"),
        }

    return jsonify({"items": [_row(r) for r in rows], "count": len(rows)})

@bp.get("/audit/export")
def audit_export():
    """
    导出 CSV/JSON
    /api/audit/export?format=csv&limit=500
    """
    fmt = (request.args.get("format") or "csv").lower()
    req = request.args.copy()
    req["limit"] = request.args.get("limit", 10000)  # 导出默认给大些

    with bp.test_request_context(query_string=req):  # 复用上面的选择逻辑
        data = audit_list().json

    items = data["items"]

    if fmt == "json":
        return jsonify(items)

    # CSV
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "id","ts","user_id","action","level","status",
        "target_type","target_id","ip","ua","message","meta"])
    writer.writeheader()
    for r in items:
        r = r.copy(); r["meta"] = json.dumps(r["meta"], ensure_ascii=False)
        writer.writerow(r)

    csv_bytes = output.getvalue().encode("utf-8-sig")  # 带 BOM 便于 Excel
    return Response(csv_bytes,
        headers={"Content-Disposition": "attachment; filename=audit.csv"},
        mimetype="text/csv; charset=utf-8")
