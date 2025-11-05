# tools/migrate_add_category.py
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import create_app, db

def column_exists(table: str, col: str) -> bool:
    rows = db.session.execute(db.text(f"PRAGMA table_info({table});"))  # SQLite
    # rows: cid, name, type, notnull, dflt_value, pk
    return any((r[1] if isinstance(r, (tuple, list)) else r._mapping['name']).lower() == col.lower()
               for r in rows.fetchall())

def main():
    # 关键：轻量模式，跳过模型与 FAISS
    app = create_app(light=True)
    with app.app_context():
        db.create_all()

        table, col = "image", "category"
        if column_exists(table, col):
            print("Column already exists.")
            return

        db.session.execute(db.text(f"ALTER TABLE {table} ADD COLUMN {col} TEXT;"))
        db.session.commit()
        print("Done. Added column 'category' to table 'image'.")

if __name__ == "__main__":
    main()
