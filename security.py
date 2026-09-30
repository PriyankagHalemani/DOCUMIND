import re
import os
from pathlib import Path
from fastapi import HTTPException, UploadFile
from app.config import settings

SAFE_FILENAME_REGEX = re.compile(r'[^a-zA-Z0-9_\-\. ]')

def sanitize_filename(filename: str) -> str:
    """Sanitizes user filename and prevents path traversal attacks."""
    base_name = os.path.basename(filename)
    clean_name = SAFE_FILENAME_REGEX.sub('_', base_name).strip()
    if not clean_name.lower().endswith(".pdf"):
        clean_name += ".pdf"
    return clean_name

def validate_pdf_upload(file: UploadFile, content_length: int = 0):
    """
    Validates MIME type, file extension, and upload size.
    """
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid PDF file. Only .pdf files are supported."
        )

    # Content type check
    valid_content_types = ["application/pdf", "application/x-pdf", "application/acrobat", "applications/vnd.pdf", "text/pdf"]
    if file.content_type and file.content_type.lower() not in valid_content_types:
        # If octet-stream, we will verify PDF magic bytes
        if file.content_type != "application/octet-stream":
            raise HTTPException(
                status_code=400,
                detail="Please upload a valid PDF file."
            )

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if content_length > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB."
        )

def verify_pdf_magic_bytes(header_bytes: bytes):
    """Verifies that the file starts with the %PDF magic header."""
    if not header_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid PDF file. The file header is corrupted or not a PDF."
        )
