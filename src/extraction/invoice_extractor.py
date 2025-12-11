import os
from src.extraction import extract_text_from_pdf, extract_text_from_ocr

def process_invoice(file_path: str) -> str:
    """
    Determine file type and return extracted text as a string.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)

    # treat non-PDF as image → OCR
    return extract_text_from_ocr(file_path)
