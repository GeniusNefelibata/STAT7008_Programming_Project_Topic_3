from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from . import db
from .models import User
from flask_jwt_extended import create_access_token, create_refresh_token

bp = Blueprint("auth", __name__)

@bp.post("/register")
def register():
    data = request.get_json() or {}
    email = data.get("email")
    password = data.get("password")
    if not email or not password:
        return jsonify(error="email and password required"), 400
    if User.query.filter_by(email=email).first():
        return jsonify(error="email already registered"), 400
    user = User(email=email, password_hash=generate_password_hash(password))
    db.session.add(user)
    db.session.commit()
    return jsonify(message="registered")

@bp.post("/login")
def login():
    data = request.get_json() or {}
    email = data.get("email"); password = data.get("password")
    user = User.query.filter_by(email=email).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify(error="invalid credentials"), 401
    access = create_access_token(identity=user.id)
    refresh = create_refresh_token(identity=user.id)
    return jsonify(access_token=access, refresh_token=refresh)
