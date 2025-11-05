from . import db
from datetime import datetime

# ---------- Users ----------

class User(db.Model):
    __tablename__ = "user"  # keep original name to avoid migration
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r}>"


# ---------- Images & metadata ----------

class Image(db.Model):
    """
    Core image row. We keep __tablename__ = 'image' for backward compatibility.
    """
    __tablename__ = "image"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=True, index=True)

    # content identity & storage
    sha256 = db.Column(db.String(64), unique=True, index=True, nullable=False)
    path = db.Column(db.String(512), nullable=False)
    thumb_path = db.Column(db.String(512), nullable=True)

    # basic properties
    width = db.Column(db.Integer, nullable=True)
    height = db.Column(db.Integer, nullable=True)
    size_bytes = db.Column(db.Integer, nullable=True)
    mime = db.Column(db.String(64), nullable=True)

    # browsing / filtering
    category = db.Column(db.String(64), index=True, nullable=True, default=None)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)

    # one-to-one convenience relationships
    embedding = db.relationship(
        "Embedding",
        uselist=False,
        backref=db.backref("image", lazy=True),
        cascade="all, delete-orphan",
        primaryjoin="Image.id==Embedding.image_id",
        lazy=True,
    )
    ocr_text = db.relationship(
        "OcrText",
        uselist=False,
        backref=db.backref("image", lazy=True),
        cascade="all, delete-orphan",
        primaryjoin="Image.id==OcrText.image_id",
        lazy=True,
    )

    def __repr__(self) -> str:
        cat = self.category or "uncategorized"
        return f"<Image id={self.id} sha={self.sha256[:8]} cat={cat}>"

    def to_dict(self) -> dict:
        """
        Lightweight dict for API responses.
        """
        return {
            "id": self.id,
            "user_id": self.user_id,
            "sha256": self.sha256,
            "path": self.path,
            "thumb_path": self.thumb_path,
            "width": self.width,
            "height": self.height,
            "size_bytes": self.size_bytes,
            "mime": self.mime,
            "category": self.category or "uncategorized",
            "created_at": int(self.created_at.timestamp()) if self.created_at else None,
        }


# Composite index to speed up gallery pagination by category + id order
# (works for both asc/desc with proper query plan)
db.Index("idx_image_category_id", Image.category, Image.id)


class Embedding(db.Model):
    __tablename__ = "embedding"
    image_id = db.Column(db.Integer, db.ForeignKey("image.id"), primary_key=True)
    model_name = db.Column(db.String(128), nullable=False)
    dim = db.Column(db.Integer, nullable=False)
    vector_blob = db.Column(db.LargeBinary, nullable=False)  # typically float32 bytes

    def __repr__(self) -> str:
        return f"<Embedding image_id={self.image_id} dim={self.dim} model={self.model_name}>"


class OcrText(db.Model):
    __tablename__ = "ocr_text"
    image_id = db.Column(db.Integer, db.ForeignKey("image.id"), primary_key=True)
    text = db.Column(db.Text, nullable=True)

    def __repr__(self) -> str:
        return f"<OcrText image_id={self.image_id} len={len(self.text or '')}>"


# ---------- Auditing ----------

class AuditLog(db.Model):
    __tablename__ = "audit_log"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, index=True)
    action = db.Column(db.String(64))
    target_type = db.Column(db.String(64))
    target_id = db.Column(db.Integer)
    ip = db.Column(db.String(128))
    ua = db.Column(db.String(256))
    ts = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    extra_json = db.Column(db.Text)

    def __repr__(self) -> str:
        return f"<AuditLog id={self.id} action={self.action} user={self.user_id}>"
