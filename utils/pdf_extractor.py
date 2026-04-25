"""Extracts text from PDF and image-based medical claim documents."""

import io
import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path


def extract_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using pdfplumber with PyMuPDF fallback."""
    text = _extract_pdfplumber(file_path)
    if len(text.strip()) < 50:
        text = _extract_pymupdf(file_path)
    return text.strip()


def extract_from_bytes(file_bytes: bytes, filename: str = "file.pdf") -> str:
    """Extract text from file bytes (for Streamlit file upload)."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_bytes(file_bytes)
    elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
        return _extract_image_bytes(file_bytes)
    elif suffix == ".txt":
        return file_bytes.decode("utf-8", errors="replace")
    return ""


def _extract_pdfplumber(path: str) -> str:
    lines = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                lines.append(page_text)
    return "\n".join(lines)


def _extract_pymupdf(path: str) -> str:
    doc = fitz.open(path)
    lines = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(lines)


def _extract_pdf_bytes(data: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    if len(text.strip()) < 50:
        doc = fitz.open(stream=data, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
    return text.strip()


def _extract_image_bytes(data: bytes) -> str:
    try:
        import pytesseract
        img = Image.open(io.BytesIO(data))
        return pytesseract.image_to_string(img, lang="eng+hin+tam")
    except ImportError:
        return "[Image OCR unavailable — install pytesseract and Tesseract engine]"
    except Exception as e:
        return f"[OCR error: {e}]"
