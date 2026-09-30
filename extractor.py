import pymupdf
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger("docuchat.pdf.extractor")

class PDFExtractionError(Exception):
    pass

class PDFExtractor:
    @staticmethod
    def extract_pages(file_path: Path) -> List[Dict[str, Any]]:
        """
        Extracts text from each page of a PDF using PyMuPDF.
        Preserves page number (1-indexed) and metadata.
        """
        if not file_path.exists():
            raise PDFExtractionError(f"File not found: {file_path}")

        try:
            doc = pymupdf.open(file_path)
        except Exception as e:
            raise PDFExtractionError(f"Failed to open PDF: {str(e)}")

        try:
            if doc.is_encrypted:
                raise PDFExtractionError("PDF is encrypted and password-protected. Please upload an unencrypted file.")

            page_count = len(doc)
            if page_count == 0:
                raise PDFExtractionError("The uploaded PDF has 0 pages.")

            pages_data = []
            total_text_length = 0

            for idx in range(page_count):
                page = doc[idx]
                page_text = page.get_text("text") or ""
                cleaned_preview = page_text.strip()
                total_text_length += len(cleaned_preview)

                pages_data.append({
                    "page_number": idx + 1,
                    "text": page_text,
                    "char_count": len(page_text),
                    "rect": [page.rect.x0, page.rect.y0, page.rect.x1, page.rect.y1]
                })

            if total_text_length < 15:
                raise PDFExtractionError("This PDF does not contain extractable text. It may be scanned or image-only.")

            return pages_data
        finally:
            doc.close()
