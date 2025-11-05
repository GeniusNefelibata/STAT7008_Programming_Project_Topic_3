# tools/backfill_category.py
import sys, pathlib, re
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import create_app, db
from app.models import Image

def guess_category(p: str) -> str:
    s = p.replace("\\", "/")
    # .../data/images/<cat>/file.jpg  or  .../uploads/<cat>/file.jpg
    m = re.search(r"/(?:images|uploads)/([^/]+)/", s, re.IGNORECASE)
    if m and m.group(1):
        return m.group(1)
    return "uncategorized"

def main():
    app = create_app()
    with app.app_context():
        rows = Image.query.all()
        n_set = 0
        for r in rows:
            if not r.category:
                r.category = guess_category(r.path or "")
                n_set += 1
        db.session.commit()
        print(f"[done] updated categories for {n_set} rows")

if __name__ == "__main__":
    main()
