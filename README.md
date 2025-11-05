# Image Drive Starter (Day-1 runnable)

## Quickstart (Windows / macOS / Linux)

1. Ensure Python 3.11+.
2. Create and activate venv:
   - Windows (PowerShell):
     ```powershell
     cd image_drive_starter
     py -3.11 -m venv .venv
     .venv\Scripts\Activate.ps1
     ```
   - macOS/Linux (bash/zsh):
     ```bash
     cd image_drive_starter
     python3 -m venv .venv
     source .venv/bin/activate
     ```
3. Install dependencies:
   ```bash
   pip install -U pip
   pip install -r requirements.txt
   ```
4. Run:
   ```bash
   python run.py
   ```
5. Health check: open http://127.0.0.1:5000/health

## Test upload (no auth required for Day-1)
```bash
curl -F "file=@/path/to/any.jpg" http://127.0.0.1:5000/api/upload
curl http://127.0.0.1:5000/api/images
```

## Next steps
- Add CLIP embeddings & FAISS/pgvector (only compute at ingest; persist)
- Add OCR (EasyOCR/Tesseract) and FTS search index
- Add text/ocr/image search endpoints and Analytics endpoints
