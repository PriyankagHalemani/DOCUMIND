from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from app.config import settings
from app.db.client import db_manager
from app.db.repositories import DocumentRepository, ChatRepository

router = APIRouter(prefix="/api", tags=["health"])

class SettingsUpdateRequest(BaseModel):
    ai_provider: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    pinecone_index_host: Optional[str] = None

@router.get("/health")
async def health_check():
    """Returns application health, database mode, and AI provider configurations."""
    total_docs = await DocumentRepository.count()
    total_chats = await ChatRepository.count()

    masked_key = None
    if settings.OPENAI_API_KEY:
        k = settings.OPENAI_API_KEY.strip()
        masked_key = f"{k[:7]}...{k[-4:]}" if len(k) > 12 else "***"

    return {
        "status": "healthy",
        "app_name": "DocuChat AI — Intelligent PDF Question Answering System",
        "database": db_manager.get_status(),
        "stats": {
            "total_documents": total_docs,
            "total_chats": total_chats
        },
        "ai_configuration": {
            "provider": settings.AI_PROVIDER,
            "openai_configured": settings.has_openai(),
            "openai_masked_key": masked_key,
            "openai_model": settings.OPENAI_MODEL,
            "pinecone_configured": settings.has_pinecone(),
            "pinecone_index_host": settings.PINECONE_INDEX_HOST
        }
    }

@router.post("/settings")
async def update_settings(req: SettingsUpdateRequest):
    """Dynamically updates runtime AI and database settings."""
    if req.ai_provider:
        settings.AI_PROVIDER = req.ai_provider
    if req.openai_api_key is not None:
        settings.OPENAI_API_KEY = req.openai_api_key.strip() if req.openai_api_key else None
    if req.openai_model:
        settings.OPENAI_MODEL = req.openai_model
    if req.pinecone_api_key is not None:
        settings.PINECONE_API_KEY = req.pinecone_api_key.strip() if req.pinecone_api_key else None
    if req.pinecone_index_host is not None:
        settings.PINECONE_INDEX_HOST = req.pinecone_index_host.strip() if req.pinecone_index_host else None

    return {
        "message": "Settings updated successfully.",
        "ai_provider": settings.AI_PROVIDER,
        "openai_configured": settings.has_openai(),
        "pinecone_configured": settings.has_pinecone()
    }
