import re
import uuid
from typing import List, Dict, Any
from app.db.models import DocumentChunkModel
from app.pdf.cleaner import TextCleaner
from app.config import settings

class TextChunker:
    @staticmethod
    def chunk_document_pages(
        pages: List[Dict[str, Any]],
        document_id: str,
        filename: str,
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> List[DocumentChunkModel]:
        """
        Splits page text into meaningful chunks while strictly preserving page numbers.
        """
        chunk_size = chunk_size or settings.CHUNK_SIZE
        chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        chunks: List[DocumentChunkModel] = []
        global_chunk_idx = 0

        for page_data in pages:
            raw_text = page_data.get("text", "")
            cleaned = TextCleaner.clean(raw_text)
            page_num = page_data.get("page_number", 1)

            if not cleaned:
                continue

            # If page text is within chunk size, keep it as single page chunk
            if len(cleaned) <= chunk_size:
                chunks.append(DocumentChunkModel(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    filename=filename,
                    page_number=page_num,
                    chunk_index=global_chunk_idx,
                    text=cleaned,
                    token_count=len(cleaned.split()),
                    metadata={"page": page_num, "char_len": len(cleaned)}
                ))
                global_chunk_idx += 1
                continue

            # Split by paragraphs or sentences
            paragraphs = cleaned.split("\n\n")
            current_buffer = ""

            for p in paragraphs:
                p = p.strip()
                if not p:
                    continue

                if len(current_buffer) + len(p) + 2 <= chunk_size:
                    current_buffer = f"{current_buffer}\n\n{p}".strip()
                else:
                    if current_buffer:
                        chunks.append(DocumentChunkModel(
                            chunk_id=str(uuid.uuid4()),
                            document_id=document_id,
                            filename=filename,
                            page_number=page_num,
                            chunk_index=global_chunk_idx,
                            text=current_buffer,
                            token_count=len(current_buffer.split()),
                            metadata={"page": page_num, "char_len": len(current_buffer)}
                        ))
                        global_chunk_idx += 1

                        # Carry over overlap from the end of the buffer
                        overlap_point = max(0, len(current_buffer) - chunk_overlap)
                        overlap_text = current_buffer[overlap_point:].strip()
                        current_buffer = f"{overlap_text}\n\n{p}".strip()
                    else:
                        # Single paragraph exceeds chunk size -> split by sentence/words
                        sentences = re.split(r'(?<=[.!?])\s+', p)
                        sent_buffer = ""
                        for sent in sentences:
                            if len(sent_buffer) + len(sent) + 1 <= chunk_size:
                                sent_buffer = f"{sent_buffer} {sent}".strip()
                            else:
                                if sent_buffer:
                                    chunks.append(DocumentChunkModel(
                                        chunk_id=str(uuid.uuid4()),
                                        document_id=document_id,
                                        filename=filename,
                                        page_number=page_num,
                                        chunk_index=global_chunk_idx,
                                        text=sent_buffer,
                                        token_count=len(sent_buffer.split()),
                                        metadata={"page": page_num, "char_len": len(sent_buffer)}
                                    ))
                                    global_chunk_idx += 1
                                    overlap_p = max(0, len(sent_buffer) - chunk_overlap)
                                    sent_buffer = f"{sent_buffer[overlap_p:].strip()} {sent}".strip()
                                else:
                                    # Very long sentence without punctuation, slice directly
                                    start = 0
                                    while start < len(sent):
                                        end = start + chunk_size
                                        sub = sent[start:end].strip()
                                        if sub:
                                            chunks.append(DocumentChunkModel(
                                                chunk_id=str(uuid.uuid4()),
                                                document_id=document_id,
                                                filename=filename,
                                                page_number=page_num,
                                                chunk_index=global_chunk_idx,
                                                text=sub,
                                                token_count=len(sub.split()),
                                                metadata={"page": page_num, "char_len": len(sub)}
                                            ))
                                            global_chunk_idx += 1
                                        start += max(1, chunk_size - chunk_overlap)
                                    sent_buffer = ""
                        current_buffer = sent_buffer

            if current_buffer.strip():
                chunks.append(DocumentChunkModel(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document_id,
                    filename=filename,
                    page_number=page_num,
                    chunk_index=global_chunk_idx,
                    text=current_buffer.strip(),
                    token_count=len(current_buffer.split()),
                    metadata={"page": page_num, "char_len": len(current_buffer)}
                ))
                global_chunk_idx += 1

        return chunks
