"""
Chunking pipeline for the knowledge base (staging).

This script:
- Loads full articles and books via `loader.load_all_documents`.
- Splits each document's `clean_text` into overlapping chunks.
- Writes `kb_chunks.jsonl` under `knowldege_base/data/processed/`.

Schema for each chunk (JSONL line):
- chunk_id:        unique id, e.g. "ncmh_article_101_chunk_0"
- parent_doc_id:   original document id (e.g. "ncmh_article_101")
- kb_family:       "article" or "book"
- content_type:    "article" / "book" (as in source)
- source_site:     e.g. "ncmh.org.sa"
- url:             original URL (if available)
- title:           document title
- language:        usually "ar"
- published_date_raw: original date string (if any)
- tags:            list of tags (if any)
- chunk_index:     0-based index of this chunk within the document
- num_chunks_total:total number of chunks for this document
- char_start:      start character offset in original `clean_text`
- char_end:        end character offset (exclusive)
- text:            chunk text
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

from . import loader


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sentence_aware_chunk(
    text: str,
    max_chars: int = 1200,
    overlap_chars: int = 200,
) -> List[Tuple[int, int, str]]:
    """
    Chunking that respects sentence boundaries.
    """
    text = text.strip()
    if not text:
        return []

    # Simple sentence splitting on . ! ? followed by space, or newline
    # This is a heuristic that works well for Arabic and English
    sentence_delimiters = r'(?<=[.!?؟])\s+|\n+'
    parts = re.split(sentence_delimiters, text)
    
    chunks: List[Tuple[int, int, str]] = []
    current_chunk_parts = []
    current_length = 0
    start_offset = 0
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        part_len = len(part)
        
        # If adding this part exceeds max_chars, save current chunk
        if current_length + part_len > max_chars and current_chunk_parts:
            chunk_text = " ".join(current_chunk_parts)
            end_offset = start_offset + len(chunk_text)
            chunks.append((start_offset, end_offset, chunk_text))
            
            # Handle overlap: keep last few parts for overlap
            overlap_target = overlap_chars
            overlap_parts = []
            overlap_len = 0
            for p in reversed(current_chunk_parts):
                if overlap_len + len(p) <= overlap_target:
                    overlap_parts.insert(0, p)
                    overlap_len += len(p) + 1
                else:
                    break
            
            current_chunk_parts = overlap_parts
            current_length = overlap_len
            # Note: actual character offset tracking is tricky with overlap,
            # for now we focus on text quality.
            start_offset = end_offset - overlap_len

        current_chunk_parts.append(part)
        current_length += part_len + 1 # +1 for space
        
    # Add final chunk
    if current_chunk_parts:
        chunk_text = " ".join(current_chunk_parts)
        chunks.append((start_offset, start_offset + len(chunk_text), chunk_text))
        
    return chunks


def simple_overlap_chunk(
    text: str,
    max_chars: int = 1200,
    overlap_chars: int = 200,
) -> List[Tuple[int, int, str]]:
    """
    Very simple character-based chunking with overlap.

    Returns a list of (start_idx, end_idx, chunk_text).
    """
    text = text.strip()
    if not text:
        return []

    n = len(text)
    chunks: List[Tuple[int, int, str]] = []
    start = 0

    while start < n:
        end = min(start + max_chars, n)
        chunk_text = text[start:end].strip()
        chunks.append((start, end, chunk_text))

        if end == n:
            break

        # Move start forward but keep some overlap.
        start = max(0, end - overlap_chars)

    return chunks


def iter_document_chunks(
    docs: Iterable[Dict],
    max_chars: Optional[int] = None,
    overlap_chars: Optional[int] = None,
    article_max_chars: Optional[int] = None,
    article_overlap_chars: Optional[int] = None,
) -> Iterable[Dict]:
    """
    Yield chunk records with metadata for each document.
    """
    try:
        from rag_config import RAGConfig
        use_sentence_chunking = RAGConfig.USE_SENTENCE_CHUNKING
        # Get defaults from config if not provided
        max_chars = max_chars or RAGConfig.CHUNK_SIZE_DEFAULT
        overlap_chars = overlap_chars or RAGConfig.CHUNK_OVERLAP_DEFAULT
        
        # content-type specific sizes
        config_map = {
            "article": (RAGConfig.CHUNK_SIZE_ARTICLE, RAGConfig.CHUNK_OVERLAP_ARTICLE),
            "book": (RAGConfig.CHUNK_SIZE_BOOK, RAGConfig.CHUNK_OVERLAP_BOOK),
            "qa_pair": (RAGConfig.CHUNK_SIZE_QA_PAIR, RAGConfig.CHUNK_OVERLAP_QA_PAIR),
        }
    except ImportError:
        use_sentence_chunking = False
        max_chars = max_chars or 1200
        overlap_chars = overlap_chars or 200
        config_map = {}

    for doc in docs:
        doc_id = doc.get("doc_id")
        if not doc_id:
            continue

        clean_text = (doc.get("clean_text") or "").strip()
        if not clean_text:
            continue

        kb_family = doc.get("kb_family") or doc.get("content_type") or "unknown"
        content_type = doc.get("content_type", kb_family)
        
        # Special handling for QA pairs: keep them as single chunks if possible
        # QA answers should remain intact for better context
        if content_type == "qa_pair":
            # If the answer is reasonable length, keep it as one chunk
            # Only split if it's extremely long (e.g., > 4000 chars)
            if len(clean_text) <= 4000:
                # Keep as single chunk - don't split QA pairs unnecessarily
                chunks = [(0, len(clean_text), clean_text)]
            else:
                # Only split if really long
                chunk_size, chunk_overlap = config_map.get(content_type, (max_chars, overlap_chars))
                if use_sentence_chunking:
                    chunks = sentence_aware_chunk(clean_text, max_chars=chunk_size, overlap_chars=chunk_overlap)
                else:
                    chunks = simple_overlap_chunk(clean_text, max_chars=chunk_size, overlap_chars=chunk_overlap)
        else:
            # Pick sizes based on content type for articles and books
            chunk_size, chunk_overlap = config_map.get(content_type, (max_chars, overlap_chars))
            
            # Override with function args if provided
            if content_type == "article":
                chunk_size = article_max_chars if article_max_chars is not None else chunk_size
                chunk_overlap = article_overlap_chars if article_overlap_chars is not None else chunk_overlap

            # Choose chunking algorithm
            if use_sentence_chunking:
                chunks = sentence_aware_chunk(clean_text, max_chars=chunk_size, overlap_chars=chunk_overlap)
            else:
                chunks = simple_overlap_chunk(clean_text, max_chars=chunk_size, overlap_chars=chunk_overlap)

        num_chunks = len(chunks)
        for idx, (start, end, chunk_text) in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{idx}"
            yield {
                "chunk_id": chunk_id,
                "parent_doc_id": doc_id,
                "kb_family": kb_family,
                "content_type": content_type,
                "source_site": doc.get("source_site"),
                "url": doc.get("url"),
                "title": doc.get("title"),
                "language": doc.get("language"),
                "published_date_raw": doc.get("published_date_raw"),
                "tags": doc.get("tags", []),
                "chunk_index": idx,
                "num_chunks_total": num_chunks,
                "char_start": start,
                "char_end": end,
                "text": chunk_text,
            }


def build_kb_chunks(
    output_filename: str = "kb_chunks.jsonl",
    max_chars: Optional[int] = None,
    overlap_chars: Optional[int] = None,
    article_max_chars: Optional[int] = None,
    article_overlap_chars: Optional[int] = None,
) -> str:
    """
    Build chunks for all documents and write them to JSONL.
    """
    try:
        from rag_config import RAGConfig
        # Get defaults from config if not provided
        max_chars = max_chars or RAGConfig.CHUNK_SIZE_DEFAULT
        overlap_chars = overlap_chars or RAGConfig.CHUNK_OVERLAP_DEFAULT
    except ImportError:
        max_chars = max_chars or 1200
        overlap_chars = overlap_chars or 200

    processed_dir = loader.PROCESSED_DIR
    _ensure_dir(processed_dir)
    out_path = os.path.join(processed_dir, output_filename)

    docs = loader.load_all_documents()

    num_docs = 0
    num_chunks = 0
    article_chunks = 0
    book_chunks = 0
    qa_chunks = 0
    
    # Track seen ids to avoid duplicate processing
    seen_doc_ids = set()
    seen_chunk_ids = set()
    duplicate_count = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for doc in docs:
            doc_id = doc.get("doc_id")
            if not doc_id:
                continue
            
            # Skip if we've already processed this document
            if doc_id in seen_doc_ids:
                print(f"  [WARNING] Skipping duplicate document: {doc_id}")
                continue
            seen_doc_ids.add(doc_id)
            
            num_docs += 1
            kb_family = doc.get("kb_family") or doc.get("content_type") or "unknown"
            
            for chunk in iter_document_chunks(
                [doc],
                max_chars=max_chars,
                overlap_chars=overlap_chars,
                article_max_chars=article_max_chars,
                article_overlap_chars=article_overlap_chars,
            ):
                chunk_id = chunk.get("chunk_id")
                
                # Skip if chunk_id already seen (duplicate)
                if chunk_id in seen_chunk_ids:
                    duplicate_count += 1
                    continue
                seen_chunk_ids.add(chunk_id)
                
                num_chunks += 1
                if kb_family == "article":
                    article_chunks += 1
                elif kb_family == "qa_pair":
                    qa_chunks += 1
                else:
                    book_chunks += 1
                f.write(json.dumps(chunk, ensure_ascii=False))
                f.write("\n")

    print(f"[OK] Built {num_chunks} chunks from {num_docs} documents.")
    print(f"  - Articles: {article_chunks} chunks")
    print(f"  - Books: {book_chunks} chunks")
    if qa_chunks > 0:
        print(f"  - QA Pairs: {qa_chunks} chunks")
    if duplicate_count > 0:
        print(f"  - Skipped {duplicate_count} duplicate chunks")
    print(f"  - Output: {out_path}")
    return out_path


def main() -> None:
    build_kb_chunks()


if __name__ == "__main__":
    main()


