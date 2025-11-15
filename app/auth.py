# app/auth.py
from __future__ import annotations

from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    jwt_required,
    get_jwt_identity,
    get_jwt,
)

from . import db
from .models_user import User  # ✅ 关键改动：从独立用户模型导入

bp = Blueprint("auth", __name__, url_prefix="/api/auth")


@bp.post("/login")
def login():
    """
    POST /api/auth/login
    JSON: { "email": "...", "password": "..." }
    Return: { "access_token": "...", "user": {id,email,role,...} }
    """
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify(error="email and password required"), 400

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify(error="invalid credentials"), 401

    # 生成 JWT（identity 用 user.id；额外 claims 放 email/role）
    token = create_access_token(
        identity=str(user.id),
        additional_claims={"email": user.email, "role": user.role},
    )
    return jsonify(access_token=token, user=user.to_public())


@bp.get("/me")
@jwt_required()
def me():
    """
    GET /api/auth/me
    Header: Authorization: Bearer <token>
    Return: 当前用户信息（从 DB 读取，以便 role 变化时能反映）
    """
    uid = get_jwt_identity()
    user = None
    if uid is not None:
        try:
            user = db.session.get(User, int(uid))
        except Exception:
            user = None

    if not user:
        return jsonify(error="user not found"), 404

    return jsonify(user=user.to_public(), jwt=get_jwt())


# 可选：简易登出（对无状态 JWT 来说只是前端丢弃 token）
@bp.post("/logout")
@jwt_required()
def logout():
    return jsonify(ok=True)
