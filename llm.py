import re
import math
import logging
from typing import List, Dict, Any, Optional, AsyncGenerator
from pydantic import BaseModel
from app.config import settings
from app.rag.retriever import ChunkSearchResult
from app.db.models import SourceReference, ChatMessageModel

logger = logging.getLogger("docuchat.rag.llm")

SYSTEM_PROMPT = """You are a document question-answering assistant.
Answer the user's question using ONLY the provided document context.
If the answer cannot be found in the provided context, clearly state that the information could not be found in the uploaded document: "I couldn't find information about this in the uploaded document."
Do not invent facts or use unsupported information.
Always mention the source document name and page number when citing facts."""

class AnswerResult(BaseModel):
    answer: str
    sources: List[SourceReference]
    provider: str

class LLMService:
    @staticmethod
    def _build_context_prompt(chunks: List[ChunkSearchResult]) -> str:
        context_parts = []
        for idx, c in enumerate(chunks, 1):
            context_parts.append(
                f"[Source {idx}] File: {c.filename} | Page: {c.page_number}\n{c.text.strip()}"
            )
        return "\n\n---\n\n".join(context_parts)

    @staticmethod
    def _prepare_sources(chunks: List[ChunkSearchResult]) -> List[SourceReference]:
        sources = []
        seen = set()
        for c in chunks:
            key = (c.filename, c.page_number)
            if key not in seen:
                seen.add(key)
                # Take first 160 characters as snippet
                snippet = c.text.strip().replace("\n", " ")
                if len(snippet) > 160:
                    snippet = snippet[:157] + "..."
                sources.append(SourceReference(
                    filename=c.filename,
                    page_number=c.page_number,
                    chunk_id=c.chunk_id,
                    snippet=snippet,
                    relevance_score=c.score
                ))
        return sources

    @staticmethod
    async def generate_answer(
        query: str,
        retrieved_chunks: List[ChunkSearchResult],
        conversation_history: Optional[List[ChatMessageModel]] = None
    ) -> AnswerResult:
        """
        Generates grounded answer using either OpenAI (if configured) or the Local Grounded Engine.
        """
        conversation_history = conversation_history or []

        # Anti-hallucination guard: If no chunks retrieved at all or score is too low
        if not retrieved_chunks:
            return AnswerResult(
                answer="I couldn't find information about this in the uploaded document.",
                sources=[],
                provider=settings.AI_PROVIDER
            )

        # Check max score for relevance
        max_score = max((c.score for c in retrieved_chunks), default=0.0)
        query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
        combined_text = " ".join([c.text.lower() for c in retrieved_chunks])
        has_keyword_match = any(w in combined_text for w in query_words)

        if max_score < 0.10 and not has_keyword_match:
            return AnswerResult(
                answer="I couldn't find information about this in the uploaded document.",
                sources=[],
                provider=settings.AI_PROVIDER
            )

        sources = LLMService._prepare_sources(retrieved_chunks)

        # 1. External LLM via OpenAI
        if settings.AI_PROVIDER == "openai" and settings.has_openai():
            try:
                from openai import AsyncOpenAI
                client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

                context_str = LLMService._build_context_prompt(retrieved_chunks)

                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                ]

                # Include past 6 messages for conversational follow-ups
                for past_msg in conversation_history[-6:]:
                    if past_msg.role in ["user", "assistant"]:
                        messages.append({"role": past_msg.role, "content": past_msg.content})

                user_content = (
                    f"Document Context:\n{context_str}\n\n"
                    f"Question: {query}\n\n"
                    f"Instructions: Answer thoroughly and strictly based on the context above. If the context does not contain the answer, say 'I couldn't find information about this in the uploaded document.' Cite the page number(s) explicitly."
                )
                messages.append({"role": "user", "content": user_content})

                response = await client.chat.completions.create(
                    model=settings.OPENAI_MODEL,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=800
                )
                answer_text = response.choices[0].message.content.strip()

                return AnswerResult(
                    answer=answer_text,
                    sources=sources,
                    provider="openai"
                )
            except Exception as e:
                logger.error("OpenAI call failed (%s), falling back to local grounded engine", e)

        # 2. Local Grounded Synthesis Engine (no external API keys needed)
        answer_text = LLMService._synthesize_local_grounded_answer(query, retrieved_chunks, conversation_history)
        return AnswerResult(
            answer=answer_text,
            sources=sources,
            provider="local"
        )

    @staticmethod
    def _synthesize_local_grounded_answer(
        query: str,
        chunks: List[ChunkSearchResult],
        conversation_history: List[ChatMessageModel]
    ) -> str:
        """
        Extractive, grounded local answer synthesis. Strictly avoids hallucination
        by organizing the extracted sentences from the retrieved chunks.
        """
        query_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()))
        
        # Consider follow-up context if query is very short or contains pronouns
        is_followup = any(pronoun in query.lower().split() for pronoun in ["it", "its", "they", "them", "these", "this", "also"])
        if is_followup and conversation_history:
            last_user_msg = next((m for m in reversed(conversation_history) if m.role == "user"), None)
            if last_user_msg:
                query_words.update(re.findall(r'\b[a-zA-Z]{3,}\b', last_user_msg.content.lower()))

        # Extract sentences from top chunks
        extracted_facts = []
        for c in chunks:
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', c.text)
            for s in sentences:
                s_clean = s.strip()
                if len(s_clean) < 20:
                    continue
                s_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', s_clean.lower()))
                common = query_words.intersection(s_words)
                if common:
                    score = len(common) / (math.log(len(s_words) + 1) + 1)
                    extracted_facts.append((score, c.filename, c.page_number, s_clean))

        if not extracted_facts:
            # Fallback: check if the top chunk has high relevance
            top_chunk = chunks[0]
            if top_chunk.score > 0.15:
                sample_text = top_chunk.text.strip()[:350]
                return (
                    f"According to **{top_chunk.filename}** (Page {top_chunk.page_number}), "
                    f"the document states:\n\n"
                    f"> \"{sample_text}...\"\n\n"
                    f"*Note: Running in local extractive mode. Add an `OPENAI_API_KEY` for generative synthesis.*"
                )
            return "I couldn't find information about this in the uploaded document."

        # Sort facts by matching score
        extracted_facts.sort(key=lambda x: x[0], reverse=True)
        top_facts = extracted_facts[:4]

        # Format into a clean response
        primary_fact = top_facts[0]
        pages_cited = sorted(list(set(f[2] for f in top_facts)))
        pages_str = ", ".join(f"Page {p}" for p in pages_cited)

        bullet_points = []
        for _, fname, pnum, s_text in top_facts:
            bullet_points.append(f"- **[Page {pnum}]**: {s_text}")

        bullet_list_str = "\n".join(bullet_points)

        return (
            f"According to **{primary_fact[1]}** ({pages_str}):\n\n"
            f"{bullet_list_str}\n\n"
            f"> 💡 *Notice: Running in Local Grounded Mode. Information is retrieved and extracted directly from your document without external LLM hallucination.*"
        )
