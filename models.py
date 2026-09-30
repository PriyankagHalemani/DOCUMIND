from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import uuid

def generate_id() -> str:
    return str(uuid.uuid4())

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

class SourceReference(BaseModel):
    filename: str
    page_number: int
    chunk_id: str
    snippet: str
    relevance_score: Optional[float] = None

class DocumentModel(BaseModel):
    id: str = Field(default_factory=generate_id)
    filename: str
    file_size: int
    page_count: int = 0
    upload_date: str = Field(default_factory=now_iso)
    status: str = "uploading"  # uploading, extracting, processing, chunking, indexing, ready, error
    processing_error: Optional[str] = None
    sha256: Optional[str] = None
    file_path: Optional[str] = None

class DocumentChunkModel(BaseModel):
    id: str = Field(default_factory=generate_id)
    chunk_id: str = Field(default_factory=generate_id)
    document_id: str
    filename: str
    page_number: int
    chunk_index: int = 0
    text: str
    token_count: int = 0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_iso)

class ChatSessionModel(BaseModel):
    id: str = Field(default_factory=generate_id)
    document_id: Optional[str] = None  # None or "all" represents all documents
    document_name: Optional[str] = "All Documents"
    title: str = "New Conversation"
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)

class ChatMessageModel(BaseModel):
    id: str = Field(default_factory=generate_id)
    chat_id: str
    role: str  # user, assistant, system
    content: str
    sources: List[SourceReference] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_iso)
