# app/__init__.py
import os
from importlib import import_module
from flask import Flask, jsonify, current_app, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager

db = SQLAlchemy()


def create_app(light: bool = False):
    app = Flask(__name__)

    # ----- Core config -----
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret')
    app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'dev-jwt-secret')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///image_drive.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Folders
    app.config['UPLOAD_DIR'] = os.path.abspath(os.environ.get('UPLOAD_DIR', 'data/images'))
    app.config['THUMB_DIR']  = os.path.abspath(os.environ.get('THUMB_DIR',  'data/thumbs'))
    os.makedirs(app.config['UPLOAD_DIR'], exist_ok=True)
    os.makedirs(app.config['THUMB_DIR'],  exist_ok=True)

    # Index / model configs
    app.config.setdefault('FAISS_INDEX_PATH',
                          os.path.abspath(os.environ.get('FAISS_INDEX_PATH', 'data/index/faiss.index')))
    app.config.setdefault('EMBED_MODEL',  os.environ.get('EMBED_MODEL',  'clip-ViT-B-32'))
    app.config.setdefault('EMBED_DEVICE', os.environ.get('EMBED_DEVICE', 'cpu'))

    db.init_app(app)
    JWTManager(app)

    # ----- Blueprint autoload -----
    def _maybe_register(dotted: str):
        try:
            mod = import_module(dotted)
            bp  = getattr(mod, "bp", None)
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

    # ----- Simple pages -----
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

    # ----- Heavy components (Vec + FAISS) -----
    with app.app_context():
        from . import models  # noqa: F401
        db.create_all()

        env_light = os.environ.get('LIGHT_MODE') == '1'
        if not (light or env_light):
            # Vec model
            from .vec import VecModel
            vec_model = VecModel(app.config['EMBED_MODEL'], app.config['EMBED_DEVICE'])

            # Resolve embedding dim robustly
            try:
                embed_dim = getattr(vec_model, "dim", None)
                if not embed_dim:
                    from .services import embeddings as EMB  # optional helper
                    embed_dim = getattr(EMB, "DIM", None)
                if not embed_dim:
                    embed_dim = 512
                embed_dim = int(embed_dim)
            except Exception:
                embed_dim = 512

            # FAISS store
            from .faiss_store import FaissStore
            faiss_store = FaissStore(dim=embed_dim, index_path=None)

            index_path = app.config.get('FAISS_INDEX_PATH')
            try:
                if index_path and os.path.exists(index_path):
                    faiss_store.open(index_path)  # explicit open with logging
                    app.logger.info(
                        f"[faiss] opened index: {index_path} "
                        f"(dim={faiss_store.dim}, ntotal={getattr(faiss_store, 'ntotal', 0)})"
                    )
                else:
                    app.logger.warning(f"[faiss] index file not found: {index_path}")
            except Exception as e:
                app.logger.error(f"[faiss] failed to open index: {e}")

            app.extensions['vec_model']   = vec_model
            app.extensions['faiss_store'] = faiss_store
        else:
            app.extensions['vec_model']   = None
            app.extensions['faiss_store'] = None

    # ----- Static home -----
    @app.get("/")
    def home():
        return current_app.send_static_file("index.html")

    return app
