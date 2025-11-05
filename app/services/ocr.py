# app/services/ocr.py
def extract_text(image_path: str) -> str:
    """
    返回识别到的文本（按行拼接）。失败则返回空串。
    """
    try:
        import easyocr
        reader = easyocr.Reader(['en', 'ch_sim'], gpu=False)  # 英文+简体中文；如只需英文可改为 ['en']
        lines = reader.readtext(image_path, detail=0)
        return "\n".join(lines)
    except Exception:
        return ""
