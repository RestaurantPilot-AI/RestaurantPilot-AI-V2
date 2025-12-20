from pathlib import Path
from typing import Tuple, Optional
from src.extraction import extract_text_from_pdf, extract_text_from_ocr


def process_invoice(file_path: str) -> Optional[Tuple[str, str, int, int, str]]:
    """
    Determine file type and extract text with metadata.

    Args:
        file_path: Path to the invoice file (PDF or image).

    Returns:
        Tuple containing:
            - extracted_text (str): The raw text extracted from the file
            - filename (str): The base filename
            - text_length (int): Length of extracted text
            - page_count (int): Number of pages processed
            - extraction_timestamp (str): ISO timestamp of extraction
        Returns None if extraction fails.
    """
    ext = Path(file_path).suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(file_path)

    # treat non-PDF as image → OCR
    return extract_text_from_ocr(file_path)
