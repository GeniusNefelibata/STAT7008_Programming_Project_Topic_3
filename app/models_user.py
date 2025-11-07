# app/models_user.py
from __future__ import annotations

from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

from . import db


class User(db.Model):
    """
    Users table stored in a separate SQLite bind: 'auth'.
    File path is configured in app.config['SQLALCHEMY_BINDS']['auth'].
    """
    __bind_key__  = "auth"
    __tablename__ = "users"

    id            = db.Column(db.Integer, primary_key=True)
    email         = db.Column(db.String(255), unique=True, index=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role          = db.Column(db.String(32), default="user", nullable=False)  # 'user' | 'admin'
    created_at    = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # --- helpers ---
    def set_password(self, raw: str) -> None:
        self.password_hash = generate_password_hash(raw)

    def check_password(self, raw: str) -> bool:
        try:
            return check_password_hash(self.password_hash, raw)
        except Exception:
            return False

    def to_public(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "role": self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r} role={self.role}>"
