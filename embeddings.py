import re
import math
import logging
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from app.config import settings

logger = logging.getLogger("docuchat.rag.embeddings")

class BaseEmbeddingService(ABC):
    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        pass

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

class LocalHashEmbeddingService(BaseEmbeddingService):
    """
    Fast, deterministic, dependency-free local dense vector generator using 
    hashed character n-grams and token frequencies.
    Produces 128-dimensional normalized dense vectors suitable for cosine similarity.
    """
    def __init__(self, dim: int = 128):
        self.dim = dim

    def _hash_vector(self, text: str) -> List[float]:
        cleaned = text.lower()
        tokens = re.findall(r'\b\w+\b', cleaned)
        vec = np.zeros(self.dim, dtype=np.float32)

        for token in tokens:
            h = hash(token) % self.dim
            vec[h] += 1.0

        # Also add bigrams for local context
        for i in range(len(tokens) - 1):
            bigram = f"{tokens[i]}_{tokens[i+1]}"
            h = hash(bigram) % self.dim
            vec[h] += 1.5

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    async def embed_query(self, text: str) -> List[float]:
        return self._hash_vector(text)

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_vector(t) for t in texts]

class OpenAIEmbeddingService(BaseEmbeddingService):
    """
    OpenAI embeddings using text-embedding-3-small or configured model.
    """
    def __init__(self, api_key: str, model: str = None):
        self.api_key = api_key
        self.model = model or settings.OPENAI_EMBEDDING_MODEL

    async def embed_query(self, text: str) -> List[float]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.api_key)
        resp = await client.embeddings.create(
            input=[text],
            model=self.model
        )
        return resp.data[0].embedding

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.api_key)
        # Batch in chunks of 50
        all_embeddings = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = await client.embeddings.create(
                input=batch,
                model=self.model
            )
            all_embeddings.extend([item.embedding for item in resp.data])
        return all_embeddings

def get_embedding_service() -> BaseEmbeddingService:
    if settings.AI_PROVIDER == "openai" and settings.has_openai():
        return OpenAIEmbeddingService(settings.OPENAI_API_KEY)
    return LocalHashEmbeddingService()
