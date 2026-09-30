import re
import math
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import numpy as np
from rank_bm25 import BM25Okapi
from app.db.repositories import ChunkRepository, DocumentRepository
from app.db.models import DocumentChunkModel
from app.rag.embeddings import get_embedding_service
from app.config import settings

logger = logging.getLogger("docuchat.rag.retriever")

class ChunkSearchResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    text: str
    score: float

def tokenize(text: str) -> List[str]:
    # Lowercase, clean alphanumeric tokens
    return [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_-]{2,}\b', text)]

class DocumentRetriever:
    """
    Modular Document Retriever supporting:
    - Local BM25 + keyword density + vector similarity
    - Pinecone semantic search if configured
    - Multi-document retrieval ('all' or specific document_id)
    """

    @staticmethod
    async def retrieve_relevant_chunks(
        query: str,
        document_id: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[ChunkSearchResult]:
        top_k = top_k or settings.TOP_K_CHUNKS

        # Determine target documents
        doc_ids = None
        if document_id and document_id != "all":
            doc_ids = [document_id]

        chunks: List[DocumentChunkModel] = await ChunkRepository.get_all(document_ids=doc_ids)
        if not chunks:
            logger.info("No chunks found in database for document_id=%s", document_id)
            return []

        # Check if Pinecone external retrieval is active
        if settings.has_pinecone():
            try:
                return await DocumentRetriever._pinecone_search(query, chunks, top_k)
            except Exception as e:
                logger.warning("Pinecone search failed (%s), falling back to local retriever", e)

        # Local Hybrid Retrieval (BM25 + Dense vector similarity + phrase match)
        return await DocumentRetriever._local_hybrid_search(query, chunks, top_k)

    @staticmethod
    async def _local_hybrid_search(
        query: str,
        chunks: List[DocumentChunkModel],
        top_k: int
    ) -> List[ChunkSearchResult]:
        query_tokens = tokenize(query)
        if not query_tokens:
            query_tokens = query.lower().split()

        corpus_tokens = [tokenize(c.text) for c in chunks]

        # 1. BM25 scoring
        bm25_scores = np.zeros(len(chunks), dtype=np.float32)
        try:
            bm25 = BM25Okapi(corpus_tokens)
            scores = bm25.get_scores(query_tokens)
            bm25_scores = np.array(scores, dtype=np.float32)
            max_bm = np.max(bm25_scores) if len(bm25_scores) > 0 and np.max(bm25_scores) > 0 else 1.0
            bm25_scores = bm25_scores / max_bm
        except Exception as e:
            logger.debug("BM25 calculation warning: %s", e)

        # 2. Vector Dense Similarity using Embedding Service
        emb_service = get_embedding_service()
        query_vec = await emb_service.embed_query(query)
        q_vec = np.array(query_vec, dtype=np.float32)

        cosine_scores = np.zeros(len(chunks), dtype=np.float32)
        for idx, chunk in enumerate(chunks):
            c_vec = np.array(chunk.embedding or await emb_service.embed_query(chunk.text[:500]), dtype=np.float32)
            denom = (np.linalg.norm(q_vec) * np.linalg.norm(c_vec))
            if denom > 0:
                cosine_scores[idx] = float(np.dot(q_vec, c_vec) / denom)

        # 3. Exact query / phrase matching boost
        lower_q = query.lower().strip()
        phrase_boost = np.zeros(len(chunks), dtype=np.float32)
        for idx, chunk in enumerate(chunks):
            lower_text = chunk.text.lower()
            if lower_q in lower_text:
                phrase_boost[idx] = 1.0
            else:
                # Count matched terms
                match_count = sum(1 for t in query_tokens if t in lower_text)
                phrase_boost[idx] = match_count / max(1, len(query_tokens))

        # Composite score
        final_scores = (0.45 * bm25_scores) + (0.35 * cosine_scores) + (0.20 * phrase_boost)

        # Rank indices
        top_indices = np.argsort(final_scores)[::-1][:top_k]

        results: List[ChunkSearchResult] = []
        for idx in top_indices:
            score = float(final_scores[idx])
            chunk = chunks[idx]
            results.append(ChunkSearchResult(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                text=chunk.text,
                score=round(score, 4)
            ))

        return results

    @staticmethod
    async def _pinecone_search(
        query: str,
        chunks: List[DocumentChunkModel],
        top_k: int
    ) -> List[ChunkSearchResult]:
        """
        External Pinecone vector index search when configured.
        """
        from pinecone import Pinecone
        pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        index = pc.Index(host=settings.PINECONE_INDEX_HOST)

        emb_service = get_embedding_service()
        query_vec = await emb_service.embed_query(query)

        query_res = index.query(
            vector=query_vec,
            top_k=top_k,
            include_metadata=True
        )

        chunk_map = {c.chunk_id: c for c in chunks}
        results: List[ChunkSearchResult] = []

        for match in query_res.get("matches", []):
            cid = match.get("id")
            score = match.get("score", 0.0)
            meta = match.get("metadata", {})
            if cid in chunk_map:
                c = chunk_map[cid]
                results.append(ChunkSearchResult(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    filename=c.filename,
                    page_number=c.page_number,
                    text=c.text,
                    score=round(score, 4)
                ))
            elif meta:
                results.append(ChunkSearchResult(
                    chunk_id=cid,
                    document_id=meta.get("document_id", ""),
                    filename=meta.get("filename", ""),
                    page_number=meta.get("page_number", 1),
                    text=meta.get("text", ""),
                    score=round(score, 4)
                ))

        return results
