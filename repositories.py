from typing import List, Optional, Dict, Any
from app.db.client import db_manager
from app.db.models import DocumentModel, DocumentChunkModel, ChatSessionModel, ChatMessageModel, now_iso

class DocumentRepository:
    @staticmethod
    async def create(doc: DocumentModel) -> DocumentModel:
        data = doc.model_dump()
        await db_manager.documents_col.insert_one(data)
        return doc

    @staticmethod
    async def get_by_id(doc_id: str) -> Optional[DocumentModel]:
        data = await db_manager.documents_col.find_one({"id": doc_id})
        if data:
            data.pop("_id", None)
            return DocumentModel(**data)
        return None

    @staticmethod
    async def get_by_sha256(sha256: str) -> Optional[DocumentModel]:
        data = await db_manager.documents_col.find_one({"sha256": sha256})
        if data:
            data.pop("_id", None)
            return DocumentModel(**data)
        return None

    @staticmethod
    async def list_all() -> List[DocumentModel]:
        if db_manager.is_mongo:
            cursor = db_manager.documents_col.find().sort("upload_date", -1)
            raw = await cursor.to_list(length=1000)
        else:
            raw = await db_manager.documents_col.find(sort_by="upload_date", reverse=True)
        
        docs = []
        for item in raw:
            item.pop("_id", None)
            docs.append(DocumentModel(**item))
        return docs

    @staticmethod
    async def update_status(doc_id: str, status: str, error: Optional[str] = None, page_count: Optional[int] = None) -> bool:
        update_data = {"status": status}
        if error is not None:
            update_data["processing_error"] = error
        if page_count is not None:
            update_data["page_count"] = page_count

        return await db_manager.documents_col.update_one(
            {"id": doc_id},
            {"$set": update_data}
        )

    @staticmethod
    async def delete(doc_id: str) -> bool:
        res = await db_manager.documents_col.delete_one({"id": doc_id})
        # If mongo, delete_one returns a DeleteResult object
        if hasattr(res, "deleted_count"):
            return res.deleted_count > 0
        return bool(res)

    @staticmethod
    async def count() -> int:
        return await db_manager.documents_col.count_documents({})

class ChunkRepository:
    @staticmethod
    async def insert_chunks(chunks: List[DocumentChunkModel]):
        if not chunks:
            return
        data = [chunk.model_dump() for chunk in chunks]
        if db_manager.is_mongo:
            await db_manager.chunks_col.insert_many(data)
        else:
            await db_manager.chunks_col.insert_many(data)

    @staticmethod
    async def get_by_document(doc_id: str) -> List[DocumentChunkModel]:
        query = {"document_id": doc_id}
        if db_manager.is_mongo:
            cursor = db_manager.chunks_col.find(query).sort("chunk_index", 1)
            raw = await cursor.to_list(length=5000)
        else:
            raw = await db_manager.chunks_col.find(query, sort_by="chunk_index", reverse=False)

        res = []
        for item in raw:
            item.pop("_id", None)
            res.append(DocumentChunkModel(**item))
        return res

    @staticmethod
    async def get_all(document_ids: Optional[List[str]] = None) -> List[DocumentChunkModel]:
        query = {}
        if document_ids:
            query = {"document_id": {"$in": document_ids}}
        
        if db_manager.is_mongo:
            cursor = db_manager.chunks_col.find(query)
            raw = await cursor.to_list(length=10000)
        else:
            raw = await db_manager.chunks_col.find(query)

        res = []
        for item in raw:
            item.pop("_id", None)
            res.append(DocumentChunkModel(**item))
        return res

    @staticmethod
    async def delete_by_document(doc_id: str) -> int:
        res = await db_manager.chunks_col.delete_many({"document_id": doc_id})
        if hasattr(res, "deleted_count"):
            return res.deleted_count
        return int(res) if isinstance(res, int) else 0

class ChatRepository:
    @staticmethod
    async def create(chat: ChatSessionModel) -> ChatSessionModel:
        data = chat.model_dump()
        await db_manager.chats_col.insert_one(data)
        return chat

    @staticmethod
    async def get_by_id(chat_id: str) -> Optional[ChatSessionModel]:
        data = await db_manager.chats_col.find_one({"id": chat_id})
        if data:
            data.pop("_id", None)
            return ChatSessionModel(**data)
        return None

    @staticmethod
    async def list_all() -> List[ChatSessionModel]:
        if db_manager.is_mongo:
            cursor = db_manager.chats_col.find().sort("updated_at", -1)
            raw = await cursor.to_list(length=500)
        else:
            raw = await db_manager.chats_col.find(sort_by="updated_at", reverse=True)

        res = []
        for item in raw:
            item.pop("_id", None)
            res.append(ChatSessionModel(**item))
        return res

    @staticmethod
    async def update_timestamp(chat_id: str, title: Optional[str] = None) -> bool:
        update_data = {"updated_at": now_iso()}
        if title:
            update_data["title"] = title
        return await db_manager.chats_col.update_one({"id": chat_id}, {"$set": update_data})

    @staticmethod
    async def delete(chat_id: str) -> bool:
        res = await db_manager.chats_col.delete_one({"id": chat_id})
        if hasattr(res, "deleted_count"):
            return res.deleted_count > 0
        return bool(res)

    @staticmethod
    async def count() -> int:
        return await db_manager.chats_col.count_documents({})

class MessageRepository:
    @staticmethod
    async def create(msg: ChatMessageModel) -> ChatMessageModel:
        data = msg.model_dump()
        await db_manager.messages_col.insert_one(data)
        return msg

    @staticmethod
    async def get_by_chat_id(chat_id: str, limit: int = 100) -> List[ChatMessageModel]:
        query = {"chat_id": chat_id}
        if db_manager.is_mongo:
            cursor = db_manager.messages_col.find(query).sort("created_at", 1)
            raw = await cursor.to_list(length=limit)
        else:
            raw = await db_manager.messages_col.find(query, sort_by="created_at", reverse=False, limit=limit)

        res = []
        for item in raw:
            item.pop("_id", None)
            res.append(ChatMessageModel(**item))
        return res

    @staticmethod
    async def delete_by_chat_id(chat_id: str) -> int:
        res = await db_manager.messages_col.delete_many({"chat_id": chat_id})
        if hasattr(res, "deleted_count"):
            return res.deleted_count
        return int(res) if isinstance(res, int) else 0
