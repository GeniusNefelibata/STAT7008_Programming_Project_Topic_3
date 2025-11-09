from __future__ import annotations
import os, sys, argparse

# ensure project import
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import create_app, db
from app.models import Image  # your existing image table
from app.services.autotag import predict_labels, canonicalize

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5000, help="max images to process")
    ap.add_argument("--force", action="store_true", help="overwrite existing categories")
    ap.add_argument("--dry_run", action="store_true", help="do not write changes")
    args = ap.parse_args()

    app = create_app(light=True)  # no vec/FAISS needed here
    with app.app_context():
        q = Image.query
        if not args.force:
            # only fill empty / NULL category
            q = q.filter((Image.category.is_(None)) | (Image.category == ""))

        q = q.order_by(Image.id.desc()).limit(args.limit)
        rows = q.all()
        print(f"[SCAN] images to (re)tag: {len(rows)}; force={args.force}, dry_run={args.dry_run}")
        if not rows:
            return

        updated = 0
        batch = 0
        for im in rows:
            try:
                # predict WITHOUT pluralization for storage
                res = predict_labels(im.path, labels=None, top_k=1, threshold=0.30, prefer_plural=False)
                # double-safety: fold to canonical before writing
                canon = canonicalize(res.primary) or res.primary
                if not args.dry_run:
                    im.category = canon
                updated += 1
            except KeyboardInterrupt:
                raise
            except Exception as e:
                # skip this image but continue
                print(f"  ! skip id={im.id}: {e}")

            batch += 1
            if not args.dry_run and batch >= 50:
                db.session.commit()
                print(f"[COMMIT] {updated} rows committed...")
                batch = 0

        if not args.dry_run and batch > 0:
            db.session.commit()
            print(f"[COMMIT] {updated} rows committed...")

        print(f"[DONE] total updated: {updated}")

if __name__ == "__main__":
    main()
