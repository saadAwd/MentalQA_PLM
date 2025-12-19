"""
Hybrid retriever (sparse + dense) over KB chunks (staging).

This module:
- Uses `SparseKBIndex` (BM25) and `DenseKBIndex` (sentence-transformers).
- Combines scores via a simple weighted sum after normalization.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import numpy as np
from sentence_transformers import CrossEncoder

from .dense_index import DenseKBIndex
from .sparse_index import SparseKBIndex


def _normalize_scores(pairs: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
    """
    Min-max normalize scores to [0, 1]. If all equal, return 0.5.
    """
    if not pairs:
        return []
    scores = np.array([s for _, s in pairs], dtype=float)
    s_min = scores.min()
    s_max = scores.max()
    if s_max - s_min < 1e-8:
        norm = np.full_like(scores, 0.5)
    else:
        norm = (scores - s_min) / (s_max - s_min)
    return [(pairs[i][0], float(norm[i])) for i in range(len(pairs))]


class HybridKBRetriever:
    """
    Combines sparse BM25 and dense semantic scores.

    alpha controls the weight of sparse vs dense:
    - alpha = 0.7  => 70% sparse, 30% dense.
    """

    def __init__(
        self,
        sparse_index: SparseKBIndex,
        dense_index: DenseKBIndex,
        alpha: float = None,
        rerank_model: Optional[CrossEncoder] = None,
    ) -> None:
        try:
            from rag_config import RAGConfig
            default_alpha = RAGConfig.HYBRID_ALPHA
        except ImportError:
            default_alpha = 0.7
            
        self.sparse_index = sparse_index
        self.dense_index = dense_index
        self.alpha = alpha if alpha is not None else default_alpha
        self.rerank_model = rerank_model

    @classmethod
    def build(cls, alpha: float = None, use_cpu: bool = True) -> "HybridKBRetriever":
        try:
            from rag_config import RAGConfig
            if alpha is None:
                alpha = RAGConfig.HYBRID_ALPHA
            
            use_reranking = RAGConfig.USE_RERANKING
            rerank_model_name = RAGConfig.RERANKING_MODEL
        except ImportError:
            if alpha is None:
                alpha = 0.7
            use_reranking = False
            rerank_model_name = None
                
        sparse = SparseKBIndex.build()
        # Force CPU for retrieval models to maximize VRAM for the generator
        print(f"[INFO] Initializing Bi-Encoder (MARBERTv2) on CPU...")
        dense = DenseKBIndex.build(device="cpu")
        
        rerank_model = None
        if use_reranking and rerank_model_name:
            # Force CPU for re-ranker
            print(f"Loading re-ranking model: {rerank_model_name} on CPU...")
            rerank_model = CrossEncoder(rerank_model_name, device="cpu")
            
        return cls(sparse_index=sparse, dense_index=dense, alpha=alpha, rerank_model=rerank_model)

    def search(self, query: str, top_k: int = 10, rerank: bool = True) -> List[Dict]:
        """
        Return top_k chunks with combined scores and metadata.
        If rerank is True and a rerank_model is available, re-scores the top results.
        """
        try:
            from rag_config import RAGConfig
            rerank_top_k = RAGConfig.RERANKING_TOP_K
            final_k = top_k
        except ImportError:
            rerank_top_k = top_k * 3
            final_k = top_k

        # 1. Get initial candidates (retrieve more if we're going to rerank)
        retrieve_k = rerank_top_k if (rerank and self.rerank_model) else final_k
        
        sparse_pairs = self.sparse_index.search(query, top_k=retrieve_k * 2)
        dense_pairs = self.dense_index.search(query, top_k=retrieve_k * 2)
        
        # Debug: Track dense retrieval results
        if not hasattr(self, "_debug_query_count"):
            self._debug_query_count = 0
        
        if not dense_pairs:
            if self._debug_query_count < 3:
                print(f"[WARNING] Query {self._debug_query_count + 1}: Dense retrieval returned 0 results!")
                self._debug_query_count += 1
        else:
            if self._debug_query_count < 3:
                sample_scores = [s for _, s in dense_pairs[:3]]
                print(f"[DEBUG] Query {self._debug_query_count + 1}: Dense retrieval - top 3 raw scores: {[f'{s:.4f}' for s in sample_scores]}")
                self._debug_query_count += 1

        sparse_norm = _normalize_scores(sparse_pairs)
        dense_norm = _normalize_scores(dense_pairs)

        # Create maps with both normalized and raw scores
        # Build lookup by chunk_id to match normalized and raw scores
        sparse_raw_map = {c["chunk_id"]: (c, s) for c, s in sparse_pairs}
        sparse_norm_map = {c["chunk_id"]: (c, s) for c, s in sparse_norm}
        
        dense_raw_map = {c["chunk_id"]: (c, s) for c, s in dense_pairs}
        dense_norm_map = {c["chunk_id"]: (c, s) for c, s in dense_norm}
        
        sparse_map = {}
        for cid in sparse_raw_map.keys():
            c_raw, s_raw = sparse_raw_map[cid]
            c_norm, s_norm = sparse_norm_map.get(cid, (c_raw, 0.0))
            sparse_map[cid] = (c_norm, s_norm, s_raw)
        
        dense_map = {}
        for cid in dense_raw_map.keys():
            c_raw, s_raw = dense_raw_map[cid]
            c_norm, s_norm = dense_norm_map.get(cid, (c_raw, 0.0))
            dense_map[cid] = (c_norm, s_norm, s_raw)
        
        # Debug: Check if dense scores are being lost
        if dense_raw_map and not hasattr(self, "_debug_dense_map"):
            sample_cids = list(dense_raw_map.keys())[:3]
            for cid in sample_cids:
                c, s = dense_raw_map[cid]
                print(f"[DEBUG] Dense raw map - CID: {cid[:50]}, score: {s:.4f}")
            self._debug_dense_map = True

        all_ids = set(sparse_map.keys()) | set(dense_map.keys())
        initial_results: List[Dict] = []

        # Debug: Check chunk_id matching
        if not hasattr(self, "_debug_matching"):
            sparse_cids = set(sparse_map.keys())
            dense_cids = set(dense_map.keys())
            overlap = sparse_cids & dense_cids
            sparse_only = sparse_cids - dense_cids
            dense_only = dense_cids - sparse_cids
            print(f"[DEBUG] Chunk ID matching:")
            print(f"  Sparse chunks: {len(sparse_cids)}, Dense chunks: {len(dense_cids)}")
            print(f"  Overlap (in both): {len(overlap)}")
            print(f"  Sparse-only: {len(sparse_only)}, Dense-only: {len(dense_only)}")
            if sparse_only:
                print(f"  Sample sparse-only CIDs: {list(sparse_only)[:3]}")
            if dense_only:
                print(f"  Sample dense-only CIDs: {list(dense_only)[:3]}")
            self._debug_matching = True

        for cid in all_ids:
            sparse_data = sparse_map.get(cid, (None, 0.0, 0.0))
            dense_data = dense_map.get(cid, (None, 0.0, 0.0))
            
            c_sparse, s_sparse_norm, s_sparse_raw = sparse_data
            c_dense, s_dense_norm, s_dense_raw = dense_data
            
            # Prefer whichever chunk dict is available
            chunk = c_sparse or c_dense
            if not chunk:
                continue
            
            # Normalized scores for ranking
            score_hybrid = self.alpha * s_sparse_norm + (1.0 - self.alpha) * s_dense_norm
            
            # Raw dense score is already in [0, 1] range (cosine similarity with normalized embeddings)
            # Use it for threshold checking
            score_raw_dense = s_dense_raw
            
            # Debug: Check if score_raw_dense is 0 when it shouldn't be
            if score_raw_dense == 0.0 and cid in dense_raw_map and not hasattr(self, "_debug_zero_score"):
                print(f"[WARNING] score_raw_dense is 0.0 but chunk exists in dense_raw_map!")
                print(f"  CID: {cid[:50]}")
                print(f"  Dense raw map value: {dense_raw_map[cid][1]:.4f}")
                self._debug_zero_score = True
            
            initial_results.append(
                {
                    "chunk": chunk,
                    "score_hybrid": float(score_hybrid),
                    "score_sparse": float(s_sparse_norm),
                    "score_dense": float(s_dense_norm),
                    "score_raw_dense": float(score_raw_dense),
                }
            )

        # Sort by hybrid score
        initial_results.sort(key=lambda r: r["score_hybrid"], reverse=True)
        candidates = initial_results[:retrieve_k]

        # 2. Re-ranking
        if rerank and self.rerank_model and candidates:
            # Prepare pairs for re-ranking
            sentence_pairs = [[query, c["chunk"]["text"]] for c in candidates]
            
            # Compute cross-encoder scores (logits for BGE)
            rerank_scores = self.rerank_model.predict(sentence_pairs)
            
            # Update scores
            for i, score in enumerate(rerank_scores):
                candidates[i]["score_rerank"] = float(score)
                # BGE scores are logits. We'll map them to a 0-1 range for the threshold
                # Simple sigmoid mapping: 1 / (1 + exp(-x))
                norm_score = 1.0 / (1.0 + np.exp(-float(score)))
                candidates[i]["score_hybrid"] = norm_score

            # Sort by raw rerank score
            candidates.sort(key=lambda x: x.get("score_rerank", -1e9), reverse=True)
            return candidates[:final_k]

        return candidates[:final_k]


def main() -> None:
    import os
    import sys

    if len(sys.argv) < 2:
        # Fallback to a default Arabic test query when no CLI arg is given.
        query = "ما هو الاكتئاب؟"
        print(
            "No query provided on the command line. "
            "Using default test query (Arabic: 'ما هو الاكتئاب؟')"
        )
    else:
        query = sys.argv[1]
    
    retriever = HybridKBRetriever.build(alpha=0.7)

    print("Hybrid retriever ready.")
    results = retriever.search(query, top_k=5)
    
    # Write results to UTF-8 file instead of printing to console
    output_file = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "processed",
        "rag_results_utf8.txt"
    )
    output_file = os.path.abspath(output_file)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Query: {query}\n")
        f.write(f"Top {len(results)} hybrid results:\n\n")
        
        for rank, item in enumerate(results, start=1):
            chunk = item["chunk"]
            f.write(
                f"[{rank}] Hybrid={item['score_hybrid']:.3f} | "
                f"Sparse={item['score_sparse']:.3f} | Dense={item['score_dense']:.3f}\n"
            )
            f.write(f"Chunk ID: {chunk['chunk_id']}\n")
            f.write(f"Title: {chunk.get('title', 'N/A')}\n")
            f.write(f"URL: {chunk.get('url', 'N/A')}\n")
            f.write(f"Content Type: {chunk.get('content_type', 'N/A')}\n")
            f.write(f"\nText Preview:\n{chunk.get('text', '')[:500]}\n")
            f.write("\n" + "="*80 + "\n\n")
    
    print(f"\nResults written to: {output_file}")
    print(f"Found {len(results)} results.")
    print("Note: Query and results are in UTF-8. Open the file above to see full Arabic text.")


if __name__ == "__main__":
    main()


