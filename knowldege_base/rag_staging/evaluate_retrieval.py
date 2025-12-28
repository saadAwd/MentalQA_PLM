"""
Retrieval Evaluation Metrics

This script evaluates retrieval quality using standard information retrieval metrics:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Mean Average Precision (MAP)
- NDCG@K (Normalized Discounted Cumulative Gain)
- Hit Rate@K

It can work with:
1. Ground truth relevance labels (if available)
2. Inferred relevance from answer matching (if ground truth answers exist)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
from tqdm import tqdm

from .hybrid_retriever import HybridKBRetriever


def precision_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    """
    Calculate Precision@K.
    
    Args:
        relevant: Set of relevant chunk IDs
        retrieved: List of retrieved chunk IDs (ordered by relevance)
        k: Number of top results to consider
        
    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if k == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_retrieved = len([cid for cid in retrieved_k if cid in relevant])
    return relevant_retrieved / k


def recall_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    """
    Calculate Recall@K.
    
    Args:
        relevant: Set of relevant chunk IDs
        retrieved: List of retrieved chunk IDs (ordered by relevance)
        k: Number of top results to consider
        
    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if len(relevant) == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_retrieved = len([cid for cid in retrieved_k if cid in relevant])
    return relevant_retrieved / len(relevant)


def mean_reciprocal_rank(relevant: Set[str], retrieved: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).
    
    MRR = 1 / rank of first relevant result
    
    Args:
        relevant: Set of relevant chunk IDs
        retrieved: List of retrieved chunk IDs (ordered by relevance)
        
    Returns:
        MRR score (0.0 to 1.0)
    """
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(relevant: Set[str], retrieved: List[str]) -> float:
    """
    Calculate Average Precision (AP).
    
    AP = (1/|relevant|) * sum(Precision@k for each relevant item at rank k)
    
    Args:
        relevant: Set of relevant chunk IDs
        retrieved: List of retrieved chunk IDs (ordered by relevance)
        
    Returns:
        Average Precision score (0.0 to 1.0)
    """
    if len(relevant) == 0:
        return 0.0
    
    relevant_ranks = []
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            relevant_ranks.append(rank)
    
    if not relevant_ranks:
        return 0.0
    
    # Calculate precision at each relevant rank
    precisions = []
    for rank in relevant_ranks:
        prec = precision_at_k(relevant, retrieved, rank)
        precisions.append(prec)
    
    return np.mean(precisions) if precisions else 0.0


def ndcg_at_k(relevant: Set[str], retrieved: List[str], k: int, scores: List[float] = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain@K (NDCG@K).
    
    NDCG considers both relevance and ranking position.
    
    Args:
        relevant: Set of relevant chunk IDs
        retrieved: List of retrieved chunk IDs (ordered by relevance)
        k: Number of top results to consider
        scores: Optional relevance scores for retrieved items
        
    Returns:
        NDCG@K score (0.0 to 1.0)
    """
    if k == 0:
        return 0.0
    
    retrieved_k = retrieved[:k]
    
    # DCG: Discounted Cumulative Gain
    dcg = 0.0
    for i, cid in enumerate(retrieved_k, start=1):
        if cid in relevant:
            # Relevance = 1 if relevant, 0 otherwise
            rel = 1.0
            # Use score if provided, otherwise binary relevance
            if scores and i <= len(scores):
                rel = scores[i-1]
            dcg += rel / np.log2(i + 1)
    
    # IDCG: Ideal DCG (all relevant items at top)
    idcg = 0.0
    num_relevant = min(len(relevant), k)
    for i in range(1, num_relevant + 1):
        idcg += 1.0 / np.log2(i + 1)
    
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg


def hit_rate_at_k(relevant: Set[str], retrieved: List[str], k: int) -> float:
    """
    Calculate Hit Rate@K.
    
    Hit Rate = 1 if at least one relevant item in top K, else 0.
    
    Args:
        relevant: Set of relevant chunk IDs
        retrieved: List of retrieved chunk IDs (ordered by relevance)
        k: Number of top results to consider
        
    Returns:
        Hit Rate@K (0.0 or 1.0)
    """
    retrieved_k = retrieved[:k]
    return 1.0 if any(cid in relevant for cid in retrieved_k) else 0.0


def infer_relevance_from_answer(
    query: str,
    retrieved_chunks: List[Dict],
    ground_truth_answer: str,
    similarity_threshold: float = 0.7
) -> Set[str]:
    """
    Infer relevant chunks by comparing retrieved chunk text with ground truth answer.
    
    Uses simple text similarity (Jaccard similarity on words) to determine relevance.
    A chunk is considered relevant if its text is similar to the ground truth answer.
    
    Args:
        query: The query string
        retrieved_chunks: List of retrieved chunk dictionaries
        ground_truth_answer: The ground truth answer text
        similarity_threshold: Minimum similarity to consider relevant
        
    Returns:
        Set of relevant chunk IDs
    """
    from sentence_transformers import SentenceTransformer
    
    # Load embedding model for similarity
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    
    # Get embeddings
    gt_embedding = model.encode([ground_truth_answer], normalize_embeddings=True)[0]
    
    relevant_chunks = set()
    
    for chunk in retrieved_chunks:
        chunk_text = chunk.get("text", chunk.get("clean_text", ""))
        if not chunk_text:
            continue
        
        # Compute cosine similarity
        chunk_embedding = model.encode([chunk_text], normalize_embeddings=True)[0]
        similarity = np.dot(gt_embedding, chunk_embedding)
        
        if similarity >= similarity_threshold:
            relevant_chunks.add(chunk.get("chunk_id", ""))
    
    return relevant_chunks


def evaluate_retrieval(
    retriever: HybridKBRetriever,
    test_queries: List[Dict],
    ground_truth: Dict[str, any] = None,
    k_values: List[int] = [1, 3, 5, 10],
    use_answer_matching: bool = True
) -> Dict:
    """
    Evaluate retrieval system on test queries.
    
    Args:
        retriever: HybridKBRetriever instance
        test_queries: List of test queries (dict with 'id', 'text' or 'question')
        ground_truth: Dict mapping query_id to relevant chunk IDs or ground truth answer
        k_values: List of K values to evaluate (e.g., [1, 3, 5, 10])
        use_answer_matching: If True, infer relevance from answer similarity
        
    Returns:
        Dictionary with evaluation metrics
    """
    results = {
        "num_queries": len(test_queries),
        "k_values": k_values,
        "metrics": {}
    }
    
    # Initialize metric accumulators
    for k in k_values:
        results["metrics"][f"precision@{k}"] = []
        results["metrics"][f"recall@{k}"] = []
        results["metrics"][f"ndcg@{k}"] = []
        results["metrics"][f"hit_rate@{k}"] = []
    
    results["metrics"]["mrr"] = []
    results["metrics"]["map"] = []
    
    print(f"\nEvaluating retrieval on {len(test_queries)} queries...")
    
    for query_data in tqdm(test_queries, desc="Evaluating"):
        query_id = query_data.get("id")
        query_text = query_data.get("text") or query_data.get("question", "")
        
        if not query_text:
            continue
        
        # Retrieve chunks
        retrieved_items = retriever.search(query_text, top_k=max(k_values), rerank=False)
        retrieved_chunk_ids = [item.get("chunk", {}).get("chunk_id", "") for item in retrieved_items]
        retrieved_scores = [item.get("score_hybrid", 0.0) for item in retrieved_items]
        
        # Get ground truth relevance
        if ground_truth and query_id in ground_truth:
            gt_data = ground_truth[query_id]
            if isinstance(gt_data, set) or isinstance(gt_data, list):
                # Direct relevance labels (set of chunk IDs)
                relevant_chunk_ids = set(gt_data) if isinstance(gt_data, list) else gt_data
            elif isinstance(gt_data, str) and use_answer_matching:
                # Ground truth answer - infer relevance
                relevant_chunk_ids = infer_relevance_from_answer(
                    query_text,
                    retrieved_items,
                    gt_data
                )
            else:
                relevant_chunk_ids = set()
        else:
            # No ground truth - skip this query
            continue
        
        # Calculate metrics for each K
        for k in k_values:
            prec = precision_at_k(relevant_chunk_ids, retrieved_chunk_ids, k)
            rec = recall_at_k(relevant_chunk_ids, retrieved_chunk_ids, k)
            ndcg = ndcg_at_k(relevant_chunk_ids, retrieved_chunk_ids, k, retrieved_scores)
            hit = hit_rate_at_k(relevant_chunk_ids, retrieved_chunk_ids, k)
            
            results["metrics"][f"precision@{k}"].append(prec)
            results["metrics"][f"recall@{k}"].append(rec)
            results["metrics"][f"ndcg@{k}"].append(ndcg)
            results["metrics"][f"hit_rate@{k}"].append(hit)
        
        # Calculate MRR and MAP
        mrr = mean_reciprocal_rank(relevant_chunk_ids, retrieved_chunk_ids)
        map_score = average_precision(relevant_chunk_ids, retrieved_chunk_ids)
        
        results["metrics"]["mrr"].append(mrr)
        results["metrics"]["map"].append(map_score)
    
    # Calculate averages
    summary = {}
    for metric_name, values in results["metrics"].items():
        if values:
            summary[f"{metric_name}_mean"] = float(np.mean(values))
            summary[f"{metric_name}_std"] = float(np.std(values))
        else:
            summary[f"{metric_name}_mean"] = 0.0
            summary[f"{metric_name}_std"] = 0.0
    
    results["summary"] = summary
    
    return results


def load_test_data(test_path: Path, labels_path: Path = None) -> Tuple[List[Dict], Dict]:
    """
    Load test queries and ground truth.
    
    Args:
        test_path: Path to test queries JSONL file
        labels_path: Optional path to ground truth answers/labels
        
    Returns:
        Tuple of (test_queries, ground_truth_dict)
    """
    test_queries = []
    ground_truth = {}
    
    # Load test queries
    if test_path.exists():
        with test_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    test_queries.append(item)
                except:
                    continue
    
    # Load ground truth (answers)
    if labels_path and labels_path.exists():
        with labels_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    query_id = item.get("id")
                    answer = item.get("text") or item.get("label_answer") or item.get("answer")
                    if query_id and answer:
                        ground_truth[query_id] = answer
                except:
                    continue
    
    return test_queries, ground_truth


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate retrieval system")
    parser.add_argument(
        "--test-path",
        type=str,
        default="data/qtypes/test.jsonl",
        help="Path to test queries JSONL file"
    )
    parser.add_argument(
        "--labels-path",
        type=str,
        default="data/atypes/test.jsonl",
        help="Path to ground truth answers JSONL file"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.4,
        help="Hybrid retrieval alpha parameter"
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="K values for evaluation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="retrieval_evaluation_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable re-ranking for evaluation"
    )
    
    args = parser.parse_args()
    
    # Load test data
    test_path = Path(args.test_path)
    labels_path = Path(args.labels_path) if args.labels_path else None
    
    print(f"[INFO] Loading test data from: {test_path}")
    test_queries, ground_truth = load_test_data(test_path, labels_path)
    print(f"[INFO] Loaded {len(test_queries)} test queries")
    print(f"[INFO] Loaded {len(ground_truth)} ground truth answers")
    
    if not test_queries:
        print("[ERROR] No test queries found!")
        return
    
    # Build retriever
    print(f"\n[INFO] Building hybrid retriever with alpha={args.alpha}...")
    retriever = HybridKBRetriever.build(alpha=args.alpha, use_cpu=True)
    
    # Evaluate
    results = evaluate_retrieval(
        retriever=retriever,
        test_queries=test_queries,
        ground_truth=ground_truth,
        k_values=args.k_values,
        use_answer_matching=True
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 80)
    print(f"Number of queries evaluated: {results['num_queries']}")
    print(f"K values: {args.k_values}")
    print("\nMetrics (Mean ± Std):")
    print("-" * 80)
    
    for k in args.k_values:
        prec_mean = results["summary"][f"precision@{k}_mean"]
        rec_mean = results["summary"][f"recall@{k}_mean"]
        ndcg_mean = results["summary"][f"ndcg@{k}_mean"]
        hit_mean = results["summary"][f"hit_rate@{k}_mean"]
        
        print(f"\nK = {k}:")
        print(f"  Precision@{k}: {prec_mean:.4f} ± {results['summary'][f'precision@{k}_std']:.4f}")
        print(f"  Recall@{k}:    {rec_mean:.4f} ± {results['summary'][f'recall@{k}_std']:.4f}")
        print(f"  NDCG@{k}:      {ndcg_mean:.4f} ± {results['summary'][f'ndcg@{k}_std']:.4f}")
        print(f"  Hit Rate@{k}: {hit_mean:.4f}")
    
    print(f"\nOverall:")
    print(f"  MRR:  {results['summary']['mrr_mean']:.4f} ± {results['summary']['mrr_std']:.4f}")
    print(f"  MAP:  {results['summary']['map_mean']:.4f} ± {results['summary']['map_std']:.4f}")
    print("=" * 80)
    
    # Save results
    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] Results saved to: {output_path}")


if __name__ == "__main__":
    main()



