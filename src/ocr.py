"""
Extract text from an image / screenshot so misinformation shared as pictures
(WhatsApp forwards, social-media screenshots) can be analyzed too.

Prefers Tesseract (via pytesseract); falls back to easyocr if that's what's
installed. Includes light preprocessing (grayscale + upscale small images) to
improve accuracy on phone screenshots. Raises OCRError with an actionable message
when no OCR engine is available.
"""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class OCRError(Exception):
    """Raised when OCR can't run or finds no readable text."""


def _load_image(source):
    """Accept a path, bytes, or a file-like object; return a PIL RGB image."""
    from PIL import Image
    if isinstance(source, (bytes, bytearray)):
        img = Image.open(io.BytesIO(source))
    elif hasattr(source, "read"):
        img = Image.open(source)
    else:
        img = Image.open(source)
    return img.convert("RGB")


def _preprocess(img):
    """Grayscale + upscale small images so text is more legible to the OCR engine."""
    from PIL import Image
    g = img.convert("L")
    w, h = g.size
    if max(w, h) < 1000:  # phone screenshots are often small
        scale = 1000 / max(w, h)
        g = g.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return g


def _tesseract_available():
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _extract_tesseract(img):
    import pytesseract
    return pytesseract.image_to_string(_preprocess(img))


def _extract_easyocr(img):
    import numpy as np
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    lines = reader.readtext(np.array(_preprocess(img)), detail=0, paragraph=True)
    return "\n".join(lines)


def available_engine():
    """Return the name of the OCR engine that will be used, or None."""
    if _tesseract_available():
        return "tesseract"
    try:
        import easyocr  # noqa: F401
        return "easyocr"
    except Exception:
        return None


def extract_text(source):
    """
    Return text extracted from the image `source` (path / bytes / file-like).
    Raises OCRError if no engine is available or no text is found.
    """
    engine = available_engine()
    if engine is None:
        raise OCRError(
            "No OCR engine found. Install one:\n"
            "  brew install tesseract && pip install pytesseract\n"
            "  (or: pip install easyocr)"
        )
    img = _load_image(source)
    text = (_extract_tesseract(img) if engine == "tesseract" else _extract_easyocr(img))
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(text.split()) < 3:
        raise OCRError(
            "Couldn't read enough text from the image. Try a clearer / higher-resolution "
            "screenshot, or paste the text directly."
        )
    return text


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python src/ocr.py <image_path>")
        print("engine available:", available_engine())
        raise SystemExit(0)
    print(f"[engine: {available_engine()}]")
    print(extract_text(sys.argv[1]))
