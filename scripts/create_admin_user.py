# scripts/create_admin_user.py
from __future__ import annotations

import os
import sys
import argparse

# --- Ensure we import YOUR project package, not a 3rd-party 'app' ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import create_app, db  # your factory & SQLAlchemy instance

# Prefer the dedicated users module (separate auth DB), fallback to legacy
try:
    from app.models_user import User  # User model bound to 'auth' DB
except Exception:
    from app.models import User       # fallback if you kept it in models.py


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an admin (or normal) user.")
    parser.add_argument("--email", required=True, help="User email (must be unique)")
    parser.add_argument("--password", required=True, help="Plain password to be hashed")
    parser.add_argument("--role", default="admin", help="Role to assign (default: admin)")
    args = parser.parse_args()

    # Start app in light mode (no vec/FAISS init)
    app = create_app(light=True)

    with app.app_context():
        binds = app.config.get("SQLALCHEMY_BINDS") or {}
        if "auth" in binds:
            # Create tables for the 'auth' bind (where User usually lives)
            db.create_all(bind_key="auth")
            print(f"[DB] ensured auth bind tables @ {binds['auth']}")
        else:
            # Fallback: create in the primary DB
            db.create_all()
            print(f"[DB] ensured tables @ {app.config.get('SQLALCHEMY_DATABASE_URI')}")

        # Idempotent: skip if the email already exists
        existing = User.query.filter_by(email=args.email).first()
        if existing:
            print(f"[SKIP] user already exists: {existing.email} (id={existing.id}, role={getattr(existing, 'role', None)})")
            return

        # Create the user
        user = User(email=args.email)
        # optional role if the model has it
        if hasattr(user, "role"):
            setattr(user, "role", args.role)

        # Hash password using model API if available, otherwise Werkzeug
        if hasattr(user, "set_password") and callable(getattr(user, "set_password")):
            user.set_password(args.password)
        else:
            # Fallback: write to password_hash directly
            try:
                from werkzeug.security import generate_password_hash
            except Exception as e:
                raise RuntimeError(
                    "werkzeug is required to hash passwords when set_password() "
                    "is not available. Please add 'Werkzeug' to your requirements."
                ) from e

            if hasattr(user, "password_hash"):
                user.password_hash = generate_password_hash(args.password)
            else:
                raise RuntimeError(
                    "User model lacks both set_password() and password_hash field."
                )

        db.session.add(user)
        db.session.commit()
        print(f"[OK] created user {user.email} (id={user.id}, role={getattr(user, 'role', None)})")


if __name__ == "__main__":
    main()
