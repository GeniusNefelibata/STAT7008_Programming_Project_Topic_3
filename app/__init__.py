# app/__init__.py
import os
from importlib import import_module
from flask import Flask, jsonify, current_app, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager

db = SQLAlchemy()
jwt = JWTManager()


def create_app(light: bool = False):
    app = Flask(__name__)

    # ---------- Core config ----------
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret")
    app.config["JWT_SECRET_KEY"] = os.environ.get("JWT_SECRET_KEY", "dev-jwt-secret")
    # 主业务库（图片/向量等），默认走 instance/image_drive.db；也兼容外部传入 DATABASE_URL
    default_image_db = f"sqlite:///{os.path.abspath(os.path.join(app.instance_path, 'image_drive.db')).replace('\\','/')}"
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", default_image_db)
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    # 确保 instance 目录存在（放 auth.db / image_drive.db）
    os.makedirs(app.instance_path, exist_ok=True)

    # 单独的认证库（auth.db），可通过 AUTH_DB 覆盖
    auth_db_file = os.environ.get("AUTH_DB", os.path.join(app.instance_path, "auth.db"))
    app.config["SQLALCHEMY_BINDS"] = {
        "auth": f"sqlite:///{os.path.abspath(auth_db_file).replace('\\','/')}"
    }

    # 资源目录（上传与缩略图）
    app.config["UPLOAD_DIR"] = os.path.abspath(os.environ.get("UPLOAD_DIR", "data/images"))
    app.config["THUMB_DIR"] = os.path.abspath(os.environ.get("THUMB_DIR", "data/thumbs"))
    os.makedirs(app.config["UPLOAD_DIR"], exist_ok=True)
    os.makedirs(app.config["THUMB_DIR"], exist_ok=True)

    # 索引与模型配置：给出稳定默认；支持环境变量覆盖
    default_index_path = os.path.abspath(
        os.environ.get("FAISS_INDEX_PATH")
        or os.path.join(os.path.dirname(app.root_path), "data", "index", "faiss.index")
    )
    os.makedirs(os.path.dirname(default_index_path), exist_ok=True)
    app.config["FAISS_INDEX_PATH"] = default_index_path

    app.config.setdefault("EMBED_MODEL", os.environ.get("EMBED_MODEL", "clip-ViT-B-32"))
    app.config.setdefault("EMBED_DEVICE", os.environ.get("EMBED_DEVICE", "cpu"))

    db.init_app(app)
    jwt.init_app(app)

    # ---------- Blueprint autoload ----------
    def _maybe_register(dotted: str):
        try:
            mod = import_module(dotted)
            bp = getattr(mod, "bp", None)
            if not bp:
                app.logger.warning(f"[blueprint] {dotted} has no 'bp'")
                return
            if bp.name in app.blueprints:
                app.logger.info(f"[blueprint] skip duplicate: {bp.name}")
                return
            app.register_blueprint(bp)
            app.logger.info(f"[blueprint] registered: {bp.name}")
        except Exception as e:
            app.logger.warning(f"[blueprint] skip {dotted}: {e}")

    _maybe_register("app.auth")
    _maybe_register("app.api.uploads")
    _maybe_register("app.api.images")
    _maybe_register("app.api.search")
    _maybe_register("app.api.analytics")
    _maybe_register("app.api.maintenance")

    # ---------- Simple pages ----------
    @app.get("/health")
    def health():
        return jsonify(ok=True)

    @app.get("/analytics")
    def analytics_page():
        root = os.path.dirname(os.path.dirname(__file__))  # project root
        html_path = os.path.join(root, "frontend", "analytics.html")
        if not os.path.exists(html_path):
            return "analytics.html not found. Please create frontend/analytics.html", 404
        return send_file(html_path)

    @app.get("/gallery")
    def gallery_page():
        root = os.path.dirname(os.path.dirname(__file__))  # project root
        path = os.path.join(root, "frontend", "gallery.html")
        if not os.path.exists(path):
            return "gallery.html not found. Please create frontend/gallery.html", 404
        return send_file(path)

    # ---------- Heavy components (Vec + FAISS) ----------
    with app.app_context():
        # 1) 业务库表（图片/embedding/ocr/audit 等）
        from . import models  # noqa: F401
        db.create_all()

        # 2) 用户库表（User 等，位于 models_user.py，并在模型里 __bind_key__="auth"）
        try:
            from . import models_user  # noqa: F401
            # 你的 SQLAlchemy 版本使用 bind_key 关键字
            db.create_all(bind_key="auth")
        except Exception as e:
            app.logger.warning(f"[auth db] create_all(bind_key='auth') failed: {e}")

        # 是否跳过大组件（轻量模式）
        env_light = os.environ.get("LIGHT_MODE") == "1"
        if not (light or env_light):
            # 向量模型
            from .vec import VecModel
            vec_model = VecModel(app.config["EMBED_MODEL"], app.config["EMBED_DEVICE"])

            # 尺寸兜底
            try:
                embed_dim = getattr(vec_model, "dim", None)
                if not embed_dim:
                    from .services import embeddings as EMB
                    embed_dim = getattr(EMB, "DIM", None)
                embed_dim = int(embed_dim or 512)
            except Exception:
                embed_dim = 512

            # FAISS store
            from .faiss_store import FaissStore
            faiss_store = FaissStore(dim=embed_dim, index_path=app.config["FAISS_INDEX_PATH"])

            # 宽容加载：有 load() 就调；否则交由后续脚本/首次写入时处理
            try:
                if hasattr(faiss_store, "load"):
                    faiss_store.load()  # 不传参，避免签名不匹配
                    app.logger.info(
                        "[faiss] index loaded (path=%s, dim=%s, ntotal=%s)",
                        app.config["FAISS_INDEX_PATH"],
                        getattr(faiss_store, "dim", None),
                        getattr(faiss_store, "ntotal", None),
                    )
                else:
                    app.logger.warning("[faiss] FaissStore has no load(); skip explicit load")
            except Exception as e:
                app.logger.warning("[faiss] failed to load index (%s); you can rebuild later", e)

            app.extensions["vec_model"] = vec_model
            app.extensions["faiss_store"] = faiss_store
        else:
            app.extensions["vec_model"] = None
            app.extensions["faiss_store"] = None

    # ---------- Static home ----------
    @app.get("/")
    def home():
        return current_app.send_static_file("index.html")

    return app
