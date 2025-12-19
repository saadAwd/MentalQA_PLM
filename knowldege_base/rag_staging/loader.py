"""
Knowledge base loaders for RAG (staging area).

This module provides small, dependency-light utilities to read the
processed JSONL files under `knowldege_base/data/processed`.

The goal is to:
- Keep I/O logic in one place.
- Expose simple Python-native structures (lists/dicts) that can be
  consumed by later RAG components (chunking, embeddings, retrieval).

Once the design stabilizes, the best pieces can be promoted out of
`rag_staging` into the main codebase.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_ROOT = os.path.dirname(THIS_DIR)

# Processed data directory inside the knowledge base package
PROCESSED_DIR = os.path.join(KNOWLEDGE_BASE_ROOT, "data", "processed")


def _iter_jsonl(path: str) -> Iterable[Dict]:
    """Yield JSON objects line-by-line from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_articles(
    filename: str = "articles_all.jsonl", base_dir: Optional[str] = None
) -> List[Dict]:
    """
    Load all article records from a processed JSONL file.

    Defaults to `articles_all.jsonl`, which appears to contain the full set.
    """
    root = base_dir or PROCESSED_DIR
    path = os.path.join(root, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Articles file not found: {path}")
    return list(_iter_jsonl(path))


def load_books(
    filename: str = "books_all_ragclean.jsonl", base_dir: Optional[str] = None
) -> List[Dict]:
    """
    Load all book records from a processed JSONL file.

    Defaults to `books_all_ragclean.jsonl`, which should contain the RAG-cleaned set.
    Falls back to `books_all_fixed.jsonl` if the cleaned version doesn't exist.
    """
    root = base_dir or PROCESSED_DIR
    path = os.path.join(root, filename)
    
    # Fallback to books_all_fixed.jsonl if ragclean doesn't exist
    if not os.path.exists(path) and filename == "books_all_ragclean.jsonl":
        fallback_path = os.path.join(root, "books_all_fixed.jsonl")
        if os.path.exists(fallback_path):
            print(f"Note: Using fallback file {fallback_path}")
            path = fallback_path
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Books file not found: {path}")
    return list(_iter_jsonl(path))


def load_all_documents(
    articles_filename: str = "articles_all.jsonl",
    books_filename: str = "books_all_ragclean.jsonl",
    qa_filename: str = "shifaa_qa_pairs_all.jsonl",
    base_dir: Optional[str] = None,
) -> List[Dict]:
    """
    Load a unified list of all documents (articles + books + QA pairs).

    Each record is kept as-is from the source files, but we add:
    - `kb_family`: one of {"article", "book", "qa_pair"} as a coarse type.
    """
    root = base_dir or PROCESSED_DIR

    articles_path = os.path.join(root, articles_filename)
    books_path = os.path.join(root, books_filename)
    qa_path = os.path.join(root, qa_filename)

    docs: List[Dict] = []

    # Load Articles
    if os.path.exists(articles_path):
        for obj in _iter_jsonl(articles_path):
            obj = dict(obj)
            obj.setdefault("kb_family", "article")
            docs.append(obj)
    else:
        print(f"Warning: Articles file not found: {articles_path}")

    # Load Books
    if os.path.exists(books_path):
        for obj in _iter_jsonl(books_path):
            obj = dict(obj)
            obj.setdefault("kb_family", "book")
            docs.append(obj)
    else:
        print(f"Warning: Books file not found: {books_path}")

    # Load QA Pairs (Shifaa)
    if os.path.exists(qa_path):
        for idx, obj in enumerate(_iter_jsonl(qa_path)):
            obj = dict(obj)
            obj.setdefault("kb_family", "qa_pair")
            
            # Prepare for chunking: use ANSWER as primary content (for retrieval)
            # Store question separately for metadata, but answer is what we want to retrieve
            if "clean_text" not in obj:
                a = obj.get("answer", "")
                q = obj.get("question", "")
                # Answer is primary - this is what we want to retrieve and use
                obj["clean_text"] = a
                # Store question as title for better retrieval context
                if not obj.get("title"):
                    obj["title"] = q[:200] if q else ""
            
            # Use a consistent, unique doc_id based on index to avoid duplicates
            obj["doc_id"] = f"shifaa_qa_{idx}"
                
            docs.append(obj)
    else:
        print(f"Note: QA pairs file not found at {qa_path}, skipping.")

    return docs


def describe_kb(
    articles_filename: str = "articles_all.jsonl",
    books_filename: str = "books_all_ragclean.jsonl",
    qa_filename: str = "shifaa_qa_pairs_all.jsonl",
    base_dir: Optional[str] = None,
) -> Dict[str, int]:
    """
    Return simple counts for quick sanity checks.
    """
    root = base_dir or PROCESSED_DIR

    # Load everything to get counts
    docs = load_all_documents(articles_filename, books_filename, qa_filename, root)
    
    counts = {
        "num_articles": sum(1 for d in docs if d["kb_family"] == "article"),
        "num_books": sum(1 for d in docs if d["kb_family"] == "book"),
        "num_qa_pairs": sum(1 for d in docs if d["kb_family"] == "qa_pair"),
        "num_documents_total": len(docs),
    }

    return counts


if __name__ == "__main__":
    stats = describe_kb()
    print("Knowledge base summary:")
    for k, v in stats.items():
        print(f"- {k}: {v}")


