import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# Load from backend/.env or root .env
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")

class Settings(BaseModel):
    # Base paths
    BASE_DIR: Path = BASE_DIR
    UPLOAD_DIR: Path = BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")
    DATA_DIR: Path = BASE_DIR / "data"

    # Database
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "docuchat")

    # AI & Embeddings
    AI_PROVIDER: str = os.getenv("AI_PROVIDER", "local")  # 'local' | 'openai'
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", None)
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Vector Storage (Pinecone optional)
    PINECONE_API_KEY: Optional[str] = os.getenv("PINECONE_API_KEY", None)
    PINECONE_INDEX_HOST: Optional[str] = os.getenv("PINECONE_INDEX_HOST", None)
    PINECONE_INDEX_NAME: Optional[str] = os.getenv("PINECONE_INDEX_NAME", None)

    # Processing & Chunks
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "30"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "5"))

    def has_openai(self) -> bool:
        return bool(self.OPENAI_API_KEY and self.OPENAI_API_KEY.strip().startswith("sk-"))

    def has_pinecone(self) -> bool:
        return bool(self.PINECONE_API_KEY and self.PINECONE_INDEX_HOST)

    def ensure_directories(self):
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
settings.ensure_directories()
