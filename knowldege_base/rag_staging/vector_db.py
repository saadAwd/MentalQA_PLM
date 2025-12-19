"""
Vector database for KB chunks using ChromaDB.

This module provides persistent storage for embeddings, so we don't need to
recompute them every time. ChromaDB is local, simple, and perfect for development.

Usage:
    # Build/rebuild the vector database
    python -m knowldege_base.rag_staging.vector_db build

    # Query the vector database
    from knowldege_base.rag_staging.vector_db import VectorDB
    db = VectorDB.load()
    results = db.search("ما هو الاكتئاب؟", top_k=5)
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None  # torch not available

from . import loader
from .sparse_index import KB_CHUNKS_FILENAME, normalize_arabic


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


def _create_marbertv2_model():
    """
    Create a SentenceTransformer model from MARBERTv2.
    MARBERTv2 is a BERT model, so we need to add pooling to create sentence embeddings.
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Pooling, Transformer
    
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
COLLECTION_NAME = "kb_chunks"
VECTOR_DB_DIR = "vector_db"


def _get_vector_db_path() -> str:
    """Get the path to the vector database directory."""
    processed_dir = loader.PROCESSED_DIR
    return os.path.join(processed_dir, VECTOR_DB_DIR)


class VectorDB:
    """
    Vector database wrapper for ChromaDB.
    
    Stores chunk embeddings persistently and provides fast similarity search.
    """

    def __init__(
        self,
        client: chromadb.ClientAPI,
        collection: chromadb.Collection,
        model: SentenceTransformer,
    ) -> None:
        self.client = client
        self.collection = collection
        self.model = model

    @classmethod
    def build(
        cls,
        model_name: str = MODEL_NAME,
        force_rebuild: bool = False,
    ) -> "VectorDB":
        """
        Build or load the vector database.
        
        Args:
            model_name: Sentence transformer model to use
            force_rebuild: If True, rebuild even if database exists
        
        Returns:
            VectorDB instance
        """
        db_path = _get_vector_db_path()
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        # Use cosine distance for normalized embeddings
        if force_rebuild:
            print(f"[WARNING] Force rebuild requested, deleting existing collection...")
            try:
                client.delete_collection(name=COLLECTION_NAME)
                print(f"[OK] Deleted existing collection")
            except Exception as e:
                # Collection might not exist, that's OK
                print(f"[INFO] Collection did not exist (or already deleted): {type(e).__name__}")
        
        # Now try to get or create the collection
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            if not force_rebuild:
                print(f"[OK] Found existing vector database with {collection.count()} chunks")
                print(f"  To rebuild, use: VectorDB.build(force_rebuild=True)")
            else:
                # If we're here with force_rebuild, collection still exists (delete failed)
                # Try to delete again and create
                print(f"[WARNING] Collection still exists, forcing deletion...")
                try:
                    client.delete_collection(name=COLLECTION_NAME)
                except Exception:
                    pass
                # Now create the collection
                collection = client.create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}  # Use cosine distance for normalized embeddings
                )
                print(f"[INFO] Created collection with cosine distance metric")
                if collection.metadata and collection.metadata.get("hnsw:space") == "cosine":
                    print(f"[OK] Verified: Collection is using cosine distance")
        except Exception:
            # Collection doesn't exist (expected after delete, or never existed), create it
            collection = client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}  # Use cosine distance for normalized embeddings
            )
            print(f"[INFO] Created collection with cosine distance metric")
            # Verify it was set correctly
            if collection.metadata and collection.metadata.get("hnsw:space") == "cosine":
                print(f"[OK] Verified: Collection is using cosine distance")
            else:
                print(f"[WARNING] Could not verify cosine distance setting!")
                print(f"  Collection metadata: {collection.metadata}")
            if not force_rebuild:
                print(f"[OK] Created new vector database collection")

        # Load model - use MARBERTv2 wrapper if needed
        print(f"Loading embedding model: {model_name}")
        if model_name == "UBC-NLP/MARBERTv2" or model_name.endswith("MARBERTv2"):
            print("  Using MARBERTv2 with sentence-transformers wrapper...")
            model = _create_marbertv2_model()
        else:
            model = SentenceTransformer(model_name)
        
        # Check if GPU is available
        if torch and torch.cuda.is_available():
            device = "cuda"
            model = model.to(device)
            print(f"[INFO] Using GPU ({torch.cuda.get_device_name(0)}) for embedding model (faster encoding)")
        else:
            device = "cpu"
            model = model.to(device)
            print(f"[INFO] Using CPU for embedding model (GPU not available)")

        # Check if we need to index chunks
        if collection.count() == 0 or force_rebuild:
            # If force rebuild, regenerate the chunks file first using current config
            if force_rebuild:
                print("Regenerating KB chunks with current configuration...")
                try:
                    from .kb_chunker import build_kb_chunks
                    build_kb_chunks()
                    print("[OK] KB chunks regenerated")
                except ImportError:
                    print("[WARNING] Could not import kb_chunker, using existing chunks file")

            print("Indexing chunks into vector database...")
            cls._index_chunks(collection, model)

        return cls(client=client, collection=collection, model=model)

    @classmethod
    def _index_chunks(
        cls,
        collection: chromadb.Collection,
        model: SentenceTransformer,
    ) -> None:
        """Index all chunks from kb_chunks.jsonl into the vector database."""
        processed_dir = loader.PROCESSED_DIR
        chunks_path = os.path.join(processed_dir, KB_CHUNKS_FILENAME)
        
        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"KB chunks file not found: {chunks_path}")

        # Load chunks
        chunks: List[Dict] = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                chunks.append(json.loads(line))

        print(f"Found {len(chunks)} chunks to index")

        # Prepare data for ChromaDB
        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Dict] = []
        embeddings: List[List[float]] = []

        # Process in batches with progress bar
        batch_size = 64
        num_batches = (len(chunks) + batch_size - 1) // batch_size
        
        print("Encoding chunks with embedding model...")
        sys.stdout.flush()  # Ensure output is visible
        # Use dynamic_ncols for better Windows compatibility
        with tqdm(total=len(chunks), desc="  Encoding", unit="chunk", 
                  dynamic_ncols=True, file=sys.stdout, mininterval=0.1, 
                  disable=False) as pbar:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_texts = [normalize_arabic(c.get("text", "")) for c in batch]
                
                # Generate embeddings
                batch_embeddings = model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )

                # Prepare metadata (ChromaDB stores metadata as dict with string values)
                for chunk, embedding in zip(batch, batch_embeddings):
                    chunk_id = chunk.get("chunk_id", f"chunk_{i}")
                    ids.append(chunk_id)
                    texts.append(chunk.get("text", ""))
                    embeddings.append(embedding.tolist())
                    
                    # Store metadata (all values must be strings, numbers, or bools)
                    metadata = {
                        "parent_doc_id": str(chunk.get("parent_doc_id", "")),
                        "kb_family": str(chunk.get("kb_family", "")),
                        "content_type": str(chunk.get("content_type", "")),
                        "title": str(chunk.get("title", "")),
                        "url": str(chunk.get("url", "")),
                        "language": str(chunk.get("language", "ar")),
                        "chunk_index": str(chunk.get("chunk_index", 0)),
                    }
                    metadatas.append(metadata)
                
                # Update progress bar
                pbar.update(len(batch))

        print(f"\nAdding {len(ids)} chunks to vector database...")
        sys.stdout.flush()  # Ensure output is visible
        
        # Add to ChromaDB (in batches to avoid memory issues) with progress bar
        add_batch_size = 100
        num_add_batches = (len(ids) + add_batch_size - 1) // add_batch_size
        
        with tqdm(total=len(ids), desc="  Adding to DB", unit="chunk", 
                  dynamic_ncols=True, file=sys.stdout, mininterval=0.1,
                  disable=False) as pbar:
            for i in range(0, len(ids), add_batch_size):
                end_idx = min(i + add_batch_size, len(ids))
                collection.add(
                    ids=ids[i:end_idx],
                    embeddings=embeddings[i:end_idx],
                    documents=texts[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                )
                pbar.update(end_idx - i)

        print(f"[OK] Indexed {len(ids)} chunks into vector database")

    @classmethod
    def load(cls, model_name: str = MODEL_NAME, device: str = None) -> "VectorDB":
        """
        Load existing vector database.
        
        Raises:
            FileNotFoundError: If vector database doesn't exist
        """
        db_path = _get_vector_db_path()
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Vector database not found at {db_path}. "
                f"Run VectorDB.build() first."
            )

        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
            # Verify distance metric
            collection_metadata = collection.metadata or {}
            distance_metric = collection_metadata.get("hnsw:space", "l2")
            if distance_metric != "cosine":
                print(f"[WARNING] Collection is using '{distance_metric}' distance, not 'cosine'")
                print(f"  This may cause incorrect similarity scores!")
                print(f"  Rebuild with: python -m knowldege_base.rag_staging.vector_db build --force")
            else:
                print(f"[INFO] Collection using cosine distance metric [OK]")
        except Exception:
            raise FileNotFoundError(
                f"Collection '{COLLECTION_NAME}' not found. "
                f"Run VectorDB.build() first."
            )
        
        # Load model - use MARBERTv2 wrapper if needed
        print(f"[INFO] Loading embedding model: {model_name}")
        if model_name == "UBC-NLP/MARBERTv2" or model_name.endswith("MARBERTv2"):
            print("  Using MARBERTv2 with sentence-transformers wrapper...")
            model = _create_marbertv2_model()
        else:
            model = SentenceTransformer(model_name)
        
        # Check if GPU is available
        if device:
            # Use specified device
            model = model.to(device)
            print(f"[INFO] Embedding model on {device}")
        elif torch and torch.cuda.is_available():
            # Use GPU if available (faster)
            device = "cuda"
            model = model.to(device)
            print(f"[INFO] Embedding model on GPU ({torch.cuda.get_device_name(0)})")
        else:
            # Fall back to CPU
            device = "cpu"
            model = model.to(device)
            print(f"[INFO] Embedding model on CPU (GPU not available)")
        
        return cls(client=client, collection=collection, model=model)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict] = None,
    ) -> List[Tuple[Dict, float]]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query in Arabic
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"content_type": "article"})
        
        Returns:
            List of (chunk_dict, score) tuples, sorted by relevance
        """
        # Encode query
        query_norm = normalize_arabic(query)
        # Ensure model is on correct device for consistent embeddings
        query_embedding = self.model.encode(
            [query_norm],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].tolist()
        
        # Debug: Check embedding stats
        import numpy as np
        emb_array = np.array(query_embedding)
        if len(query_embedding) == 0 or np.all(emb_array == 0):
            print(f"[WARNING] Query embedding is empty or all zeros!")
        elif np.isnan(emb_array).any():
            print(f"[WARNING] Query embedding contains NaN values!")

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata,
        )

        # Format results
        chunks_with_scores: List[Tuple[Dict, float]] = []
        
        # Debug: Check if ChromaDB returned results
        if not results.get("ids") or not results["ids"] or len(results["ids"][0]) == 0:
            if not hasattr(self, "_debug_empty_results"):
                print(f"[WARNING] ChromaDB returned 0 results for query!")
                self._debug_empty_results = True
            return chunks_with_scores  # Return empty list
        
        if results["ids"] and len(results["ids"][0]) > 0:
            # Debug: Check what ChromaDB is returning
            distances = results["distances"][0] if results["distances"] else []
            if distances and len(distances) > 0:
                min_dist = min(distances)
                max_dist = max(distances)
                avg_dist = sum(distances) / len(distances)
                # Convert to scores for debugging
                min_score = 1.0 - max_dist  # Lower distance = higher score
                max_score = 1.0 - min_dist
                avg_score = 1.0 - avg_dist
                # Only print debug for first query to avoid spam
                if not hasattr(self, "_debug_printed"):
                    print(f"[DEBUG] ChromaDB - distances: min={min_dist:.4f}, max={max_dist:.4f}, avg={avg_dist:.4f}")
                    print(f"[DEBUG] ChromaDB - scores: min={min_score:.4f}, max={max_score:.4f}, avg={avg_score:.4f}")
                    self._debug_printed = True
            
            for i, chunk_id in enumerate(results["ids"][0]):
                # Reconstruct chunk dict from ChromaDB data
                chunk = {
                    "chunk_id": chunk_id,
                    "text": results["documents"][0][i] if results["documents"] else "",
                    "parent_doc_id": results["metadatas"][0][i].get("parent_doc_id", ""),
                    "kb_family": results["metadatas"][0][i].get("kb_family", ""),
                    "content_type": results["metadatas"][0][i].get("content_type", ""),
                    "title": results["metadatas"][0][i].get("title", ""),
                    "url": results["metadatas"][0][i].get("url", ""),
                    "language": results["metadatas"][0][i].get("language", "ar"),
                    "chunk_index": int(results["metadatas"][0][i].get("chunk_index", 0)),
                }
                
                # ChromaDB returns distances (lower is better)
                # For normalized embeddings with cosine distance: distance = 1 - cosine_similarity
                # So: cosine_similarity = 1 - distance
                distance = results["distances"][0][i] if results["distances"] else 1.0
                
                # Convert cosine distance to cosine similarity
                # For normalized embeddings with cosine distance:
                # - distance range: [0, 2] (0 = identical, 2 = opposite)
                # - similarity range: [1, -1] (1 = identical, -1 = opposite)
                # But with normalized embeddings pointing in same direction, distance is typically in [0, 1]
                # So: similarity = 1 - distance
                score = max(0.0, 1.0 - distance)  # Ensure non-negative similarity (clamp to [0, 1])
                
                chunks_with_scores.append((chunk, score))

        return chunks_with_scores

    def get_stats(self) -> Dict:
        """Get statistics about the vector database."""
        count = self.collection.count()
        return {
            "total_chunks": count,
            "collection_name": COLLECTION_NAME,
            "model_name": MODEL_NAME,
        }


def main() -> None:
    """CLI for building/rebuilding the vector database."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build or rebuild the vector database for KB chunks."
    )
    parser.add_argument(
        "action",
        type=str,
        choices=["build", "stats"],
        help="Action to perform: 'build' to create/rebuild, 'stats' to show statistics",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if database exists",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_NAME,
        help="Sentence transformer model to use",
    )

    args = parser.parse_args()

    if args.action == "build":
        print("=" * 80)
        print("Building Vector Database")
        print("=" * 80)
        db = VectorDB.build(model_name=args.model_name, force_rebuild=args.force)
        stats = db.get_stats()
        print("\n" + "=" * 80)
        print("Vector Database Ready")
        print("=" * 80)
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Model: {stats['model_name']}")
        print(f"Location: {_get_vector_db_path()}")
        print("=" * 80)
    elif args.action == "stats":
        try:
            db = VectorDB.load(model_name=args.model_name)
            stats = db.get_stats()
            print("=" * 80)
            print("Vector Database Statistics")
            print("=" * 80)
            for key, value in stats.items():
                print(f"{key}: {value}")
            print(f"Location: {_get_vector_db_path()}")
            print("=" * 80)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Run 'python -m knowldege_base.rag_staging.vector_db build' first.")


if __name__ == "__main__":
    main()

