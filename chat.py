import logging
from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from app.db.models import ChatSessionModel, ChatMessageModel, SourceReference
from app.db.repositories import ChatRepository, MessageRepository, DocumentRepository
from app.rag.retriever import DocumentRetriever
from app.rag.llm import LLMService
from app.config import settings

logger = logging.getLogger("docuchat.api.chat")
router = APIRouter(prefix="/api/chat", tags=["chat"])

class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None
    document_id: Optional[str] = None  # None or "all" for all documents

class CreateChatRequest(BaseModel):
    document_id: Optional[str] = None
    title: Optional[str] = None

@router.post("")
async def send_chat_message(req: ChatRequest):
    """
    Main RAG Question-Answering Endpoint:
    1. Validates query and document availability.
    2. Resolves or creates chat session.
    3. Retrieves top relevant chunks.
    4. Evaluates conversation history for follow-up questions.
    5. Generates grounded answer with anti-hallucination guarantee.
    6. Saves and returns message and sources.
    """
    query = req.message.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Question message cannot be empty.")

    # 1. Document validation
    total_docs = await DocumentRepository.count()
    if total_docs == 0:
        raise HTTPException(
            status_code=400,
            detail="Please upload or select a document first."
        )

    doc_name = "All Documents"
    if req.document_id and req.document_id != "all":
        selected_doc = await DocumentRepository.get_by_id(req.document_id)
        if not selected_doc:
            raise HTTPException(status_code=404, detail="Selected document was not found.")
        if selected_doc.status != "ready":
            raise HTTPException(
                status_code=400,
                detail=f"Document '{selected_doc.filename}' is currently {selected_doc.status}. Please wait until it is ready."
            )
        doc_name = selected_doc.filename

    # 2. Resolve Chat Session
    chat = None
    if req.chat_id:
        chat = await ChatRepository.get_by_id(req.chat_id)

    if not chat:
        # Generate initial title from user query
        words = query.split()
        title = " ".join(words[:5])
        if len(words) > 5:
            title += "..."
        chat = ChatSessionModel(
            document_id=req.document_id,
            document_name=doc_name,
            title=title or "New Conversation"
        )
        await ChatRepository.create(chat)

    # 3. Store User Message
    user_msg = ChatMessageModel(
        chat_id=chat.id,
        role="user",
        content=query,
        sources=[]
    )
    await MessageRepository.create(user_msg)

    # 4. Fetch previous conversation history for follow-ups
    history = await MessageRepository.get_by_chat_id(chat.id, limit=20)
    # Exclude the message we just saved so history contains prior context
    past_history = [m for m in history if m.id != user_msg.id]

    # 5. Retrieve top-k relevant chunks across selected doc(s)
    chunks = await DocumentRetriever.retrieve_relevant_chunks(
        query=query,
        document_id=req.document_id,
        top_k=settings.TOP_K_CHUNKS
    )

    # 6. Generate answer
    ans_result = await LLMService.generate_answer(
        query=query,
        retrieved_chunks=chunks,
        conversation_history=past_history
    )

    # 7. Store Assistant Response
    assistant_msg = ChatMessageModel(
        chat_id=chat.id,
        role="assistant",
        content=ans_result.answer,
        sources=ans_result.sources
    )
    await MessageRepository.create(assistant_msg)

    # 8. Update Chat timestamp
    await ChatRepository.update_timestamp(chat.id)

    return {
        "chat_id": chat.id,
        "user_message": user_msg,
        "assistant_message": assistant_msg,
        "sources": ans_result.sources,
        "provider": ans_result.provider
    }

@router.get("s")
async def list_chats():
    """List all user chat sessions."""
    chats = await ChatRepository.list_all()
    return {"chats": chats, "total": len(chats)}

@router.post("s")
async def create_chat(req: CreateChatRequest):
    """Explicitly create a new chat session."""
    doc_name = "All Documents"
    if req.document_id and req.document_id != "all":
        doc = await DocumentRepository.get_by_id(req.document_id)
        if doc:
            doc_name = doc.filename

    chat = ChatSessionModel(
        document_id=req.document_id,
        document_name=doc_name,
        title=req.title or "New Conversation"
    )
    saved = await ChatRepository.create(chat)
    return saved

@router.get("s/{chat_id}")
async def get_chat_history(chat_id: str):
    """Retrieve full chat session with messages."""
    chat = await ChatRepository.get_by_id(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    messages = await MessageRepository.get_by_chat_id(chat_id)
    return {
        "chat": chat,
        "messages": messages
    }

@router.delete("s/{chat_id}")
async def delete_chat_session(chat_id: str):
    """Delete a chat session and its messages."""
    chat = await ChatRepository.get_by_id(chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat session not found.")

    await MessageRepository.delete_by_chat_id(chat_id)
    deleted = await ChatRepository.delete(chat_id)
    return {"message": "Chat session deleted successfully.", "deleted": deleted}
