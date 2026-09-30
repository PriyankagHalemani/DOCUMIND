import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

logger = logging.getLogger("docuchat.db")

class LocalJsonCollection:
    """Thread-safe and async-safe local JSON persistent store implementing MongoDB-like operations."""
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._lock = asyncio.Lock()
        self._ensure_file()

    def _ensure_file(self):
        if not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def _read_sync(self) -> List[Dict[str, Any]]:
        self._ensure_file()
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _write_sync(self, data: List[Dict[str, Any]]):
        temp_file = self.file_path.with_suffix(".tmp")
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        temp_file.replace(self.file_path)

    async def insert_one(self, document: Dict[str, Any]):
        async with self._lock:
            docs = self._read_sync()
            docs.append(document)
            self._write_sync(docs)

    async def insert_many(self, documents: List[Dict[str, Any]]):
        async with self._lock:
            docs = self._read_sync()
            docs.extend(documents)
            self._write_sync(docs)

    def _match(self, item: Dict[str, Any], query: Dict[str, Any]) -> bool:
        for k, v in query.items():
            if k == "$or" and isinstance(v, list):
                if not any(self._match(item, cond) for cond in v):
                    return False
            elif k == "$in" and isinstance(v, dict):
                # nested condition
                pass
            elif isinstance(v, dict):
                if "$in" in v and item.get(k) not in v["$in"]:
                    return False
                if "$ne" in v and item.get(k) == v["$ne"]:
                    return False
            else:
                if item.get(k) != v:
                    return False
        return True

    async def find_one(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        async with self._lock:
            docs = self._read_sync()
            for doc in docs:
                if self._match(doc, query):
                    return doc
            return None

    async def find(self, query: Dict[str, Any] = None, sort_by: Optional[str] = None, reverse: bool = False, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        async with self._lock:
            docs = self._read_sync()
            query = query or {}
            results = [doc for doc in docs if self._match(doc, query)]
            if sort_by:
                results.sort(key=lambda x: str(x.get(sort_by, "")), reverse=reverse)
            if limit:
                results = results[:limit]
            return results

    async def update_one(self, query: Dict[str, Any], update: Dict[str, Any]) -> bool:
        async with self._lock:
            docs = self._read_sync()
            found = False
            for idx, doc in enumerate(docs):
                if self._match(doc, query):
                    set_data = update.get("$set", update)
                    doc.update(set_data)
                    docs[idx] = doc
                    found = True
                    break
            if found:
                self._write_sync(docs)
            return found

    async def delete_one(self, query: Dict[str, Any]) -> bool:
        async with self._lock:
            docs = self._read_sync()
            for idx, doc in enumerate(docs):
                if self._match(doc, query):
                    docs.pop(idx)
                    self._write_sync(docs)
                    return True
            return False

    async def delete_many(self, query: Dict[str, Any]) -> int:
        async with self._lock:
            docs = self._read_sync()
            new_docs = [doc for doc in docs if not self._match(doc, query)]
            deleted_count = len(docs) - len(new_docs)
            if deleted_count > 0:
                self._write_sync(new_docs)
            return deleted_count

    async def count_documents(self, query: Dict[str, Any] = None) -> int:
        async with self._lock:
            docs = self._read_sync()
            if not query:
                return len(docs)
            return sum(1 for doc in docs if self._match(doc, query))

class DatabaseManager:
    def __init__(self):
        self.is_mongo: bool = False
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.documents_col = None
        self.chunks_col = None
        self.chats_col = None
        self.messages_col = None

    async def initialize(self):
        # Try connecting to MongoDB first
        try:
            logger.info(f"Connecting to MongoDB at {settings.MONGODB_URI}...")
            client = AsyncIOMotorClient(settings.MONGODB_URI, serverSelectionTimeoutMS=1500)
            # Ping to verify active connection
            await asyncio.wait_for(client.admin.command('ping'), timeout=1.5)
            self.client = client
            self.db = client[settings.MONGODB_DB_NAME]
            self.is_mongo = True
            
            # Setup collections
            self.documents_col = self.db["documents"]
            self.chunks_col = self.db["document_chunks"]
            self.chats_col = self.db["chats"]
            self.messages_col = self.db["messages"]

            # Create MongoDB indexes
            await self.documents_col.create_index("id", unique=True)
            await self.documents_col.create_index("sha256")
            await self.chunks_col.create_index("id", unique=True)
            await self.chunks_col.create_index([("document_id", 1), ("page_number", 1)])
            await self.chats_col.create_index("id", unique=True)
            await self.messages_col.create_index([("chat_id", 1), ("created_at", 1)])

            logger.info("Connected successfully to MongoDB database: %s", settings.MONGODB_DB_NAME)
        except Exception as e:
            logger.warning("MongoDB not active (%s). Activating local persistent JSON store fallback at %s", str(e), settings.DATA_DIR)
            self.is_mongo = False
            self.documents_col = LocalJsonCollection(settings.DATA_DIR / "documents.json")
            self.chunks_col = LocalJsonCollection(settings.DATA_DIR / "document_chunks.json")
            self.chats_col = LocalJsonCollection(settings.DATA_DIR / "chats.json")
            self.messages_col = LocalJsonCollection(settings.DATA_DIR / "messages.json")

    def get_status(self) -> Dict[str, Any]:
        return {
            "mode": "mongodb" if self.is_mongo else "local_json_store",
            "mongodb_uri": settings.MONGODB_URI if self.is_mongo else None,
            "database_name": settings.MONGODB_DB_NAME if self.is_mongo else "local_json",
            "data_directory": str(settings.DATA_DIR)
        }

db_manager = DatabaseManager()
