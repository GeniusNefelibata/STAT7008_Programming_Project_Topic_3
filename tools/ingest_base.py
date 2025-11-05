# tools/ingest_base.py
import os
import sys
import argparse
import hashlib
import shutil
from pathlib import Path
from mimetypes import guess_type

from PIL import Image as PILImage

# 启用轻量模式：不加载大模型/FAISS，避免内存和下载
os.environ.setdefault("LIGHT_MODE", "1")

# 保证从项目根目录执行均可找到 app 包
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import create_app, db        # noqa: E402
from app.models import Image as ImageModel  # noqa: E402

# 允许的图片扩展名（大小写都可）
ALLOWED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def make_thumb(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with PILImage.open(src) as im:
            im.thumbnail((512, 512))
            im.convert("RGB").save(dst, "JPEG", quality=85)
    except Exception:
        # 缩略失败不致命
        pass

def ingest_one(app, src_path: Path, category: str) -> dict:
    """复制到仓库并写 DB（不做 embedding / OCR）"""
    upload_dir = Path(app.config["UPLOAD_DIR"])
    thumb_dir  = Path(app.config["THUMB_DIR"])

    sha = sha256_file(src_path)
    sub = Path(sha[:2]) / sha[2:4]
    dst = upload_dir / sub / sha
    dst.parent.mkdir(parents=True, exist_ok=True)

    # 去重：DB 里已有就不重复插入
    existed = ImageModel.query.filter_by(sha256=sha).first()
    if existed and dst.exists():
        return {"image_id": existed.id, "duplicate": True}

    # 复制（保持源数据不动；用 copy2 保留 mtime）
    shutil.copy2(str(src_path), str(dst))

    # 基本元数据
    width = height = None
    try:
        with PILImage.open(dst) as im:
            width, height = im.size
    except Exception:
        pass

    # 缩略图
    thumb_path = thumb_dir / sub / f"{sha}.jpg"
    make_thumb(dst, thumb_path)

    size_bytes = dst.stat().st_size
    mime = guess_type(src_path.name)[0] or "application/octet-stream"

    # 插入 DB
    img = ImageModel(
        user_id=None,
        sha256=sha,
        path=str(dst),
        thumb_path=str(thumb_path) if thumb_path.exists() else None,
        width=width,
        height=height,
        size_bytes=size_bytes,
        mime=mime,
        category=category
    )
    db.session.add(img)
    db.session.commit()
    return {"image_id": img.id, "duplicate": False}

def main(root_dir: str):
    app = create_app(light=True)  # 轻量模式
    root = Path(root_dir).resolve()

    if not root.exists():
        print(f"[ERR] root not found: {root}")
        sys.exit(1)

    count_total = 0
    count_new = 0
    count_dup = 0
    skipped = 0

    with app.app_context():
        print(f"[ingest] root = {root}")
        print(f"[ingest] UPLOAD_DIR = {app.config['UPLOAD_DIR']}")
        print(f"[ingest] THUMB_DIR  = {app.config['THUMB_DIR']}")

        # 目录结构：root/<category>/*.jpg
        for cat_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            category = cat_dir.name.strip()
            num_cat = 0
            new_cat = 0
            dup_cat = 0

            for path in cat_dir.rglob("*"):
                if not path.is_file():
                    continue
                ext = path.suffix.lower()
                if ext not in ALLOWED:
                    skipped += 1
                    continue

                num_cat += 1
                res = ingest_one(app, path, category)
                if res.get("duplicate"):
                    dup_cat += 1
                else:
                    new_cat += 1

                # 每 50 张打点
                if num_cat % 50 == 0:
                    print(f"  [{category}] processed {num_cat} files... (+{new_cat}/dup {dup_cat})")

            count_total += num_cat
            count_new += new_cat
            count_dup += dup_cat
            print(f"[done] {category}: total={num_cat}, new={new_cat}, dup={dup_cat}")

    print("=" * 60)
    print(f"[summary] scanned={count_total}, inserted={count_new}, duplicates={count_dup}, skipped(non-image)={skipped}")
    print("Tip: run `python -m tools.backfill --rebuild` to compute embeddings/OCR and build FAISS.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest base dataset (category folders) into Image Drive.")
    parser.add_argument("--root", required=True, help="Root folder of base images, e.g., data/base")
    args = parser.parse_args()
    main(args.root)
