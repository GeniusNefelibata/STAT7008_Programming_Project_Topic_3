# app/logging_utils.py
from __future__ import annotations
import os, json, logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from flask import current_app, request
from flask_jwt_extended import get_jwt_identity

from .extensions import db
from .models import AuditLog

# —— 配置文件日志（轮转） ————————————————————————————————
def configure_file_logging(app):
    log_dir = app.config.get("LOG_DIR", os.path.join(app.instance_path, "logs"))
    os.makedirs(log_dir, exist_ok=True)

    max_bytes = int(app.config.get("LOG_MAX_BYTES", 10 * 1024 * 1024))  # 10MB
    backup = int(app.config.get("LOG_BACKUP_COUNT", 5))
    level = getattr(logging, str(app.config.get("LOG_LEVEL", "INFO")).upper(), logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S%z",
    )

    # app.log（通用）
    app_handler = RotatingFileHandler(os.path.join(log_dir, "app.log"),
                                      maxBytes=max_bytes, backupCount=backup, encoding="utf-8")
    app_handler.setLevel(level); app_handler.setFormatter(fmt)
    app.logger.addHandler(app_handler)
    app.logger.setLevel(level)

    # 将 werkzeug 访问日志也写入 app.log
    wlog = logging.getLogger("werkzeug")
    wlog.setLevel(level)
    wlog.addHandler(app_handler)

    # error.log（告警及以上）
    err_handler = RotatingFileHandler(os.path.join(log_dir, "error.log"),
                                      maxBytes=max_bytes, backupCount=backup, encoding="utf-8")
    err_handler.setLevel(logging.WARNING); err_handler.setFormatter(fmt)
    app.logger.addHandler(err_handler)

    # 独立审计文件（便于外部查看）
    audit_handler = RotatingFileHandler(os.path.join(log_dir, "audit.log"),
                                        maxBytes=max_bytes, backupCount=backup, encoding="utf-8")
    audit_handler.setLevel(logging.INFO); audit_handler.setFormatter(fmt)
    app.audit_logger = logging.getLogger("audit")
    app.audit_logger.setLevel(logging.INFO)
    app.audit_logger.addHandler(audit_handler)

    app.logger.info("File logging configured: %s", log_dir)


# —— 审计工具 ————————————————————————————————————————————————
def _now_utc():
    return datetime.now(timezone.utc)

def _remote_ip():
    # 简单获取真实 IP（如有代理可改 X-Forwarded-For）
    xf = request.headers.get("X-Forwarded-For")
    return (xf.split(",")[0].strip() if xf else request.remote_addr) or "unknown"

def record_audit(action: str,
                 target_type: str | None = None,
                 target_id: int | None = None,
                 status: str | None = "200",
                 level: str = "INFO",
                 message: str | None = None,
                 meta: dict | None = None,
                 user_id: int | None = None):
    try:
        # 统一准备字段
        payload = {
            "user_id": user_id,
            "action": action,
            "level": level,
            "status": str(status),
            "target_type": target_type,
            "target_id": target_id,
            "ip": (request.remote_addr if request else None),
            "ua": (request.user_agent.string if request else None),
            "message": (message or "")[:512],
        }
        meta_json = json.dumps(meta or {}, ensure_ascii=False)

        # 只给“模型里真的存在”的字段赋值
        colnames = set(c.key for c in AuditLog.__table__.columns)
        if "meta_json" in colnames:
            payload["meta_json"] = meta_json
        elif "meta" in colnames:
            payload["meta"] = meta_json

        log = AuditLog(**{k: v for k, v in payload.items() if k in colnames})

        # 如果模型里有 created_at/ts 字段，再设置（构造之后 set，不作为构造参数）
        now = datetime.now(timezone.utc)
        for ts_field in ("created_at", "ts", "timestamp"):
            if hasattr(log, ts_field):
                setattr(log, ts_field, now)

        db.session.add(log)
        db.session.commit()
    except Exception as e:
        current_app.logger.warning(f"[audit] write failed: {e}")
        db.session.rollback()