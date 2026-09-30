import os
import hashlib
import logging
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from app.config import settings
from app.db.models import DocumentModel, DocumentChunkModel
from app.db.repositories import DocumentRepository, ChunkRepository
from app.pdf.extractor import PDFExtractor, PDFExtractionError
from app.pdf.chunker import TextChunker
from app.rag.embeddings import get_embedding_service
from app.utils.security import sanitize_filename, validate_pdf_upload, verify_pdf_magic_bytes

logger = logging.getLogger("docuchat.api.documents")
router = APIRouter(prefix="/api/documents", tags=["documents"])

async def process_document_pipeline(document_id: str, file_path: Path, filename: str):
    """
    Executes the full PDF processing pipeline:
    1. Extract pages (preserving page numbers)
    2. Clean text
    3. Split into meaningful chunks with metadata
    4. Generate embeddings (if external embedding provider is configured)
    5. Save chunks to DB
    """
    try:
        logger.info("Starting processing pipeline for doc_id=%s, file=%s", document_id, filename)
        await DocumentRepository.update_status(document_id, "extracting")

        # Step 1 & 2: Extract and preserve page-by-page text
        pages_data = PDFExtractor.extract_pages(file_path)
        page_count = len(pages_data)

        await DocumentRepository.update_status(document_id, "processing", page_count=page_count)

        # Step 3 & 4: Clean and split into chunks with metadata
        await DocumentRepository.update_status(document_id, "chunking")
        chunks = TextChunker.chunk_document_pages(
            pages=pages_data,
            document_id=document_id,
            filename=filename
        )

        if not chunks:
            raise PDFExtractionError("No valid text chunks could be created from document.")

        # Step 5: Embeddings & Indexing
        await DocumentRepository.update_status(document_id, "indexing")
        emb_service = get_embedding_service()
        
        # Batch embed if external provider or local dense vectors
        chunk_texts = [c.text for c in chunks]
        embeddings = await emb_service.embed_documents(chunk_texts)
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]

        # Step 6: Store chunks in DB
        # Remove any previous chunks first
        await ChunkRepository.delete_by_document(document_id)
        await ChunkRepository.insert_chunks(chunks)

        # Finished
        await DocumentRepository.update_status(document_id, "ready", page_count=page_count)
        logger.info("Document %s processed successfully! (%d chunks, %d pages)", filename, len(chunks), page_count)

    except PDFExtractionError as pe:
        logger.error("Extraction error for doc_id %s: %s", document_id, str(pe))
        await DocumentRepository.update_status(document_id, "error", error=str(pe))
    except Exception as e:
        logger.exception("Unexpected error processing document %s", document_id)
        await DocumentRepository.update_status(document_id, "error", error=f"Processing failed: {str(e)}")

@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Uploads and queues a PDF for processing.
    Deduplicates identical files based on SHA256.
    """
    validate_pdf_upload(file)

    # Read content
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="The uploaded PDF file is empty.")

    verify_pdf_magic_bytes(contents[:10])

    # Compute sha256
    sha256 = hashlib.sha256(contents).hexdigest()
    clean_name = sanitize_filename(file.filename or "document.pdf")

    # Check deduplication
    existing_doc = await DocumentRepository.get_by_sha256(sha256)
    if existing_doc and existing_doc.status == "ready":
        logger.info("Document %s already processed with identical hash, returning existing record.", clean_name)
        return {
            "message": "Document already processed and ready.",
            "document": existing_doc,
            "already_processed": True
        }

    # Save to uploads directory
    dest_path = settings.UPLOAD_DIR / f"{sha256[:12]}_{clean_name}"
    with open(dest_path, "wb") as f:
        f.write(contents)

    doc_record = DocumentModel(
        filename=clean_name,
        file_size=len(contents),
        sha256=sha256,
        file_path=str(dest_path),
        status="uploading"
    )
    saved_doc = await DocumentRepository.create(doc_record)

    # Trigger processing pipeline
    background_tasks.add_task(
        process_document_pipeline,
        saved_doc.id,
        dest_path,
        clean_name
    )

    return {
        "message": "Document uploaded successfully and queued for processing.",
        "document": saved_doc,
        "already_processed": False
    }

@router.get("")
async def list_documents():
    """List all uploaded documents."""
    docs = await DocumentRepository.list_all()
    return {"documents": docs, "total": len(docs)}

@router.get("/{document_id}")
async def get_document(document_id: str):
    """Retrieve document details and chunk preview."""
    doc = await DocumentRepository.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    chunks = await ChunkRepository.get_by_document(document_id)
    return {
        "document": doc,
        "chunk_count": len(chunks),
        "chunks_preview": chunks[:5]
    }

@router.get("/{document_id}/preview")
async def preview_document_pdf(document_id: str):
    """Stream original PDF for in-app preview."""
    doc = await DocumentRepository.get_by_id(document_id)
    if not doc or not doc.file_path or not Path(doc.file_path).exists():
        raise HTTPException(status_code=404, detail="PDF file not found on server.")

    return FileResponse(
        path=doc.file_path,
        media_type="application/pdf",
        filename=doc.filename
    )

@router.post("/{document_id}/process")
async def reprocess_document(document_id: str, background_tasks: BackgroundTasks):
    """Re-trigger the processing pipeline for a document."""
    doc = await DocumentRepository.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    
    file_path = Path(doc.file_path) if doc.file_path else None
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="Original PDF file is missing.")

    background_tasks.add_task(
        process_document_pipeline,
        doc.id,
        file_path,
        doc.filename
    )
    return {"message": "Document reprocessing started.", "status": "processing"}

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete document, its chunks, and the underlying file."""
    doc = await DocumentRepository.get_by_id(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Delete chunks
    await ChunkRepository.delete_by_document(document_id)

    # Delete physical file
    if doc.file_path and Path(doc.file_path).exists():
        try:
            os.remove(doc.file_path)
        except Exception as e:
            logger.warning("Could not delete file %s: %s", doc.file_path, e)

    # Delete document record
    deleted = await DocumentRepository.delete(document_id)
    return {"message": "Document deleted successfully.", "deleted": deleted}
