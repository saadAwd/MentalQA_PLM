"""
Dense embedding index over KB chunks (staging).

Uses `sentence-transformers` with a multilingual model suitable for Arabic.
For now this keeps everything in memory (vectors + chunks), which is fine
for ~2.6k chunks. Later we can persist to disk or move to FAISS/pgvector.

Usage (from project root, venv activated):

    python -m knowldege_base.rag_staging.dense_index "ما هو الاكتئاب؟"
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer

from . import loader
from .sparse_index import KB_CHUNKS_FILENAME, normalize_arabic


def _create_marbertv2_model():
    """
    Create a SentenceTransformer model from MARBERTv2.
    MARBERTv2 is a BERT model, so we need to add pooling to create sentence embeddings.
    """
    from sentence_transformers import SentenceTransformer
    
    # Load MARBERTv2 base model
    word_embedding_model = Transformer("UBC-NLP/MARBERTv2", max_seq_length=512)
    
    # Add mean pooling to create sentence embeddings
    pooling_model = Pooling(
        word_embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    
    # Create SentenceTransformer model
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    return model


# Embedding model - can be overridden via RAG_EMBEDDING_MODEL env var
import os
_EMBEDDING_MODEL_NAME = os.getenv(
    "RAG_EMBEDDING_MODEL",
    "UBC-NLP/MARBERTv2"  # Arabic-specific BERT model
)

# Use MARBERTv2 if specified, otherwise use the model name directly
if _EMBEDDING_MODEL_NAME == "UBC-NLP/MARBERTv2" or _EMBEDDING_MODEL_NAME.endswith("MARBERTv2"):
    MODEL_NAME = "UBC-NLP/MARBERTv2"  # Will use custom wrapper
    USE_MARBERTV2 = True
else:
    MODEL_NAME = _EMBEDDING_MODEL_NAME
    USE_MARBERTV2 = False


def _load_kb_chunks(filename: str = KB_CHUNKS_FILENAME) -> List[Dict]:
    processed_dir = loader.PROCESSED_DIR
    path = os.path.join(processed_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"KB chunks file not found: {path}")

    chunks: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


class DenseKBIndex:
    """
    Dense index for KB chunks using sentence-transformers.
    
    Uses vector database (ChromaDB) if available, otherwise falls back to
    in-memory computation for backward compatibility.
    """

    def __init__(
        self,
        model: SentenceTransformer,
        chunks: List[Dict] = None,
        embeddings: np.ndarray = None,
        vector_db=None,
    ) -> None:
        self.model = model
        self.chunks = chunks
        self.embeddings = embeddings  # shape: (n_chunks, dim)
        self.vector_db = vector_db  # ChromaDB instance if using vector DB

    @classmethod
    def build(cls, use_vector_db: bool = True, device: str = None) -> "DenseKBIndex":
        """
        Build dense index, preferring vector database if available.
        
        Args:
            use_vector_db: If True, try to use vector database first
            device: Device to use ("cpu" or "cuda"). If None, auto-detect.
        """
        # Load model - use MARBERTv2 wrapper if needed
        if USE_MARBERTV2:
            print(f"Loading MARBERTv2 with sentence-transformers wrapper...")
            model = _create_marbertv2_model()
        else:
            model = SentenceTransformer(MODEL_NAME)
        
        # Check if GPU is available for faster encoding
        try:
            import torch
            if torch and torch.cuda.is_available():
                device = "cuda"
                model = model.to(device)
                print(f"[INFO] Using GPU ({torch.cuda.get_device_name(0)}) for embedding model")
            else:
                device = "cpu"
                model = model.to(device)
                print(f"[INFO] Using CPU for embedding model (GPU not available)")
        except ImportError:
            device = "cpu"
            model = model.to(device)
            print(f"[INFO] Using CPU for embedding model")
        
        # Try to use vector database first
        if use_vector_db:
            try:
                from .vector_db import VectorDB
                print("Loading vector database...")
                # Use the same device as the model for consistency
                vector_db = VectorDB.load(device=device if device else None)
                print(f"[OK] Loaded vector database with {vector_db.get_stats()['total_chunks']} chunks")
                return cls(model=model, vector_db=vector_db)
            except (FileNotFoundError, ImportError) as e:
                if isinstance(e, ImportError):
                    print(f"⚠ ChromaDB not available, falling back to in-memory index")
                else:
                    print(f"⚠ Vector database not found, falling back to in-memory index")
                    print(f"  To build it: python -m knowldege_base.rag_staging.vector_db build")
        
        # Fallback to in-memory computation
        print(f"Loading model: {MODEL_NAME}")
        chunks = _load_kb_chunks()
        texts = [normalize_arabic(c.get("text", "")) for c in chunks]

        print(f"Encoding {len(texts)} chunks...")
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return cls(model=model, chunks=chunks, embeddings=embeddings)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """
        Return top_k chunks with cosine similarity scores for a given query.
        """
        # Use vector database if available
        if self.vector_db is not None:
            return self.vector_db.search(query, top_k=top_k)
        
        # Fallback to in-memory computation
        q_norm = normalize_arabic(query)
        q_vec = self.model.encode(
            [q_norm],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]  # shape: (dim,)

        scores = np.dot(self.embeddings, q_vec)  # cosine sim (since normalized)

        top_indices = np.argsort(-scores)[:top_k]
        return [
            (self.chunks[int(i)], float(scores[int(i)])) for i in top_indices
        ]


def main() -> None:
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m knowldege_base.rag_staging.dense_index \"<arabic query>\"")
        sys.exit(1)

    query = sys.argv[1]
    index = DenseKBIndex.build()
    print(f"Dense index built with {len(index.chunks)} chunks.")

    results = index.search(query, top_k=5)
    print(f"\nTop 5 dense results for query: {query!r}\n")
    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"[{rank}] score={score:.4f} | chunk_id={chunk['chunk_id']} | title={chunk.get('title')}")
        preview = chunk.get("text", "")[:200].replace("\n", " ")
        print(f"    {preview}...")


if __name__ == "__main__":
    main()


