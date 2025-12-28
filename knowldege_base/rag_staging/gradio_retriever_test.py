"""
Gradio interface for testing and visualizing the hybrid retriever.

This provides an interactive UI to:
- Test queries against the knowledge base
- Visualize retrieval results with scores
- Compare sparse vs dense retrieval
- Adjust retrieval parameters (alpha, top_k)
- View chunk metadata and sources
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr

from .hybrid_retriever import HybridKBRetriever
from .langchain_retriever import HybridKBRetrieverWrapper


class RetrieverTester:
    """Wrapper class to manage retriever state and provide testing interface."""
    
    def __init__(self):
        self.retriever: HybridKBRetriever = None
        self.retriever_wrapper: HybridKBRetrieverWrapper = None
        self._initialized = False
    
    def initialize(self, alpha: float = 0.4, use_cpu: bool = True):
        """Initialize the retriever."""
        if not self._initialized or self.retriever is None:
            print(f"[INFO] Initializing HybridKBRetriever with alpha={alpha}...")
            self.retriever = HybridKBRetriever.build(alpha=alpha, use_cpu=use_cpu)
            self.retriever_wrapper = HybridKBRetrieverWrapper(
                hybrid_retriever=self.retriever,
                return_metadata=True
            )
            self._initialized = True
            print("[OK] Retriever initialized")
        else:
            # Update alpha if changed
            if self.retriever.alpha != alpha:
                print(f"[INFO] Updating alpha from {self.retriever.alpha} to {alpha}")
                self.retriever.alpha = alpha
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.4,
        show_scores: bool = True,
        rerank: bool = True
    ) -> Tuple[str, str]:
        """
        Search the knowledge base and return formatted results.
        
        Returns:
            Tuple of (formatted_results_html, json_results)
        """
        if not query or not query.strip():
            return "Please enter a query.", "{}"
        
        # Initialize if needed
        self.initialize(alpha=alpha)
        
        # Perform search
        try:
            chunks = self.retriever.search(
                query=query.strip(),
                top_k=top_k,
                rerank=rerank
            )
        except Exception as e:
            error_msg = f"Error during retrieval: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return error_msg, "{}"
        
        if not chunks:
            return "No results found.", "{}"
        
        # Format results as HTML
        html_parts = []
        html_parts.append(f"<h3 style='color: #ffffff;'>Query: <em style='color: #66b3ff; direction: rtl; text-align: right; display: inline-block;'>{query}</em></h3>")
        html_parts.append(f"<p style='color: #e0e0e0;'><strong style='color: #ffffff;'>Found {len(chunks)} results</strong></p>")
        html_parts.append("<hr style='border-color: #555;'>")
        
        for i, result_item in enumerate(chunks, 1):
            # The hybrid retriever returns items with nested "chunk" key
            chunk = result_item.get("chunk", result_item)  # Fallback to item itself if no nested chunk
            
            # Extract information from the actual chunk dict
            text = chunk.get("text", chunk.get("clean_text", ""))
            title = chunk.get("title", "No title")
            kb_family = chunk.get("kb_family", "unknown")
            chunk_id = chunk.get("chunk_id", "unknown")
            url = chunk.get("url", "")
            
            # Scores are at the top level of result_item
            # Use original hybrid score if available (before reranking), otherwise use current hybrid score
            hybrid_score_original = result_item.get("score_hybrid_original", result_item.get("score_hybrid", result_item.get("hybrid_score", 0.0)))
            hybrid_score = result_item.get("score_hybrid", result_item.get("hybrid_score", 0.0))
            sparse_score = result_item.get("score_sparse", result_item.get("sparse_score", 0.0))
            # Use score_raw_dense (actual cosine similarity) instead of score_dense (normalized)
            # score_raw_dense is the actual meaningful similarity score (0-1 range)
            dense_score = result_item.get("score_raw_dense", result_item.get("score_dense", result_item.get("dense_score", 0.0)))
            rerank_score = result_item.get("score_rerank", result_item.get("rerank_score"))
            rerank_score_norm = result_item.get("score_rerank_norm")
            
            # Check if scores are 0 because chunk wasn't retrieved by that method
            # (This happens when sparse and dense return different chunks)
            # If score_raw_dense is exactly 0.0, chunk wasn't retrieved by dense
            # If sparse_score is 0.0 and dense_score > 0, likely not retrieved by sparse
            # Note: Normalized scores can be 0.0 even if retrieved (if it's the min score)
            # But score_raw_dense being 0.0 definitively means not retrieved by dense
            dense_retrieved = dense_score > 1e-6  # Use small threshold to avoid floating point issues
            # For sparse, if it's 0.0 and dense is > 0, likely not retrieved by sparse
            # But we can't be 100% sure without raw sparse score
            sparse_retrieved = sparse_score > 1e-6 or (dense_score <= 1e-6)  # If dense is also 0, sparse might be legit
            
            # Build HTML for this result
            html_parts.append(f"<div style='margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background: #2b2b2b; color: #e0e0e0;'>")
            html_parts.append(f"<h4 style='color: #ffffff; margin-top: 0;'>Result #{i}</h4>")
            
            # Scores section
            html_parts.append("<div style='background: #3a3a3a; padding: 10px; margin-bottom: 10px; border-radius: 3px; color: #e0e0e0;'>")
            html_parts.append(f"<strong style='color: #ffffff;'>Scores:</strong><br>")
            # Show original hybrid score (before reranking)
            if rerank_score is not None:
                html_parts.append(f"  • Hybrid (original): <span style='color: #66b3ff; font-weight: bold;'>{hybrid_score_original:.4f}</span> <span style='color: #888; font-size: 0.9em;'>(before rerank)</span><br>")
            else:
                html_parts.append(f"  • Hybrid: <span style='color: #66b3ff; font-weight: bold;'>{hybrid_score:.4f}</span><br>")
            
            # Sparse score with indicator if not retrieved
            sparse_display = f"{sparse_score:.4f}" if sparse_retrieved else f"{sparse_score:.4f} <span style='color: #888; font-size: 0.9em;'>(not retrieved)</span>"
            html_parts.append(f"  • Sparse (BM25): <span style='color: #ff9966; font-weight: bold;'>{sparse_display}</span><br>")
            
            # Dense score with indicator if not retrieved
            dense_display = f"{dense_score:.4f}" if dense_retrieved else f"{dense_score:.4f} <span style='color: #888; font-size: 0.9em;'>(not retrieved)</span>"
            html_parts.append(f"  • Dense (MARBERTv2): <span style='color: #66ff66; font-weight: bold;'>{dense_display}</span><br>")
            
            if rerank_score is not None:
                html_parts.append(f"  • Rerank: <span style='color: #ff66ff; font-weight: bold;'>{rerank_score:.4f}</span><br>")
            html_parts.append("</div>")
            
            # Metadata
            html_parts.append("<div style='margin-bottom: 10px; color: #e0e0e0;'>")
            html_parts.append(f"<strong style='color: #ffffff;'>Source:</strong> <span style='color: #b0b0b0;'>{kb_family}</span><br>")
            html_parts.append(f"<strong style='color: #ffffff;'>Title:</strong> <span style='color: #b0b0b0; direction: rtl; text-align: right; display: inline-block;'>{title}</span><br>")
            html_parts.append(f"<strong style='color: #ffffff;'>Chunk ID:</strong> <span style='color: #b0b0b0; font-family: monospace;'>{chunk_id}</span><br>")
            if url:
                html_parts.append(f"<strong style='color: #ffffff;'>URL:</strong> <a href='{url}' target='_blank' style='color: #66b3ff;'>{url}</a><br>")
            html_parts.append("</div>")
            
            # Text content - show full chunk (no truncation)
            # Add expandable section for very long chunks
            chunk_length = len(text)
            if chunk_length > 2000:
                # For very long chunks, show first 2000 chars with expand option
                preview_text = text[:2000]
                remaining_text = text[2000:]
                chunk_id_safe = chunk_id.replace(" ", "_").replace(".", "_")
                html_parts.append(f"<div style='background: #1e1e1e; padding: 10px; border-left: 3px solid #0066cc; color: #e0e0e0;'>")
                html_parts.append(f"<strong style='color: #ffffff;'>Content:</strong> <span style='color: #888; font-size: 0.9em;'>({chunk_length} chars)</span><br>")
                html_parts.append(f"<p id='preview_{chunk_id_safe}' style='white-space: pre-wrap; color: #d0d0d0; line-height: 1.6; direction: rtl; text-align: right; font-size: 1.05em;'>{preview_text}...</p>")
                html_parts.append(f"<p id='full_{chunk_id_safe}' style='white-space: pre-wrap; color: #d0d0d0; line-height: 1.6; direction: rtl; text-align: right; font-size: 1.05em; display: none;'>{text}</p>")
                html_parts.append(f"<button onclick=\"document.getElementById('preview_{chunk_id_safe}').style.display='none'; document.getElementById('full_{chunk_id_safe}').style.display='block'; this.style.display='none';\" style='background: #0066cc; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer;'>Show Full Text</button>")
                html_parts.append("</div>")
            else:
                # Show full text for shorter chunks
                html_parts.append(f"<div style='background: #1e1e1e; padding: 10px; border-left: 3px solid #0066cc; color: #e0e0e0;'>")
                html_parts.append(f"<strong style='color: #ffffff;'>Content:</strong> <span style='color: #888; font-size: 0.9em;'>({chunk_length} chars)</span><br>")
                html_parts.append(f"<p style='white-space: pre-wrap; color: #d0d0d0; line-height: 1.6; direction: rtl; text-align: right; font-size: 1.05em;'>{text}</p>")
                html_parts.append("</div>")
            
            html_parts.append("</div>")
        
        html_result = "\n".join(html_parts)
        
        # Also return JSON for programmatic access
        json_result = json.dumps({
            "query": query,
            "num_results": len(chunks),
            "alpha": alpha,
            "top_k": top_k,
            "results": [
                {
                    "rank": i,
                    "chunk_id": result_item.get("chunk", result_item).get("chunk_id"),
                    "title": result_item.get("chunk", result_item).get("title"),
                    "kb_family": result_item.get("chunk", result_item).get("kb_family"),
                    "text": result_item.get("chunk", result_item).get("text", result_item.get("chunk", result_item).get("clean_text", ""))[:200] + "...",
                    "scores": {
                        "hybrid": result_item.get("score_hybrid", result_item.get("hybrid_score", 0.0)),
                        "sparse": result_item.get("score_sparse", result_item.get("sparse_score", 0.0)),
                        "dense": result_item.get("score_raw_dense", result_item.get("score_dense", result_item.get("dense_score", 0.0))),
                        "rerank": result_item.get("score_rerank", result_item.get("rerank_score"))
                    }
                }
                for i, result_item in enumerate(chunks, 1)
            ]
        }, indent=2, ensure_ascii=False)
        
        return html_result, json_result
    
    def compare_retrieval_methods(
        self,
        query: str,
        top_k: int = 5
    ) -> str:
        """Compare sparse-only, dense-only, and hybrid retrieval."""
        if not query or not query.strip():
            return "Please enter a query."
        
        self.initialize()
        
        # Get sparse and dense results separately
        sparse_pairs = self.retriever.sparse_index.search(query, top_k=top_k * 2)
        dense_pairs = self.retriever.dense_index.search(query, top_k=top_k * 2)
        
        html_parts = []
        html_parts.append(f"<h3 style='color: #ffffff;'>Comparison for Query: <em style='color: #66b3ff; direction: rtl; text-align: right; display: inline-block;'>{query}</em></h3>")
        
        # Sparse results
        html_parts.append("<h4 style='color: #ff9966;'>🔍 Sparse Retrieval (BM25) - Top Results:</h4>")
        for i, (chunk, score) in enumerate(sparse_pairs[:top_k], 1):
            text = chunk.get("text", chunk.get("clean_text", ""))[:200]
            html_parts.append(f"<p style='color: #e0e0e0;'><strong style='color: #ffffff;'>{i}.</strong> [Score: <span style='color: #ff9966;'>{score:.4f}</span>] <span style='color: #d0d0d0; direction: rtl; text-align: right; display: inline-block;'>{text}...</span></p>")
        
        # Dense results
        html_parts.append("<h4 style='color: #66ff66;'>🧠 Dense Retrieval (MARBERTv2) - Top Results:</h4>")
        for i, (chunk, score) in enumerate(dense_pairs[:top_k], 1):
            text = chunk.get("text", chunk.get("clean_text", ""))[:200]
            html_parts.append(f"<p style='color: #e0e0e0;'><strong style='color: #ffffff;'>{i}.</strong> [Score: <span style='color: #66ff66;'>{score:.4f}</span>] <span style='color: #d0d0d0; direction: rtl; text-align: right; display: inline-block;'>{text}...</span></p>")
        
        # Hybrid results
        html_parts.append("<h4 style='color: #66b3ff;'>⚡ Hybrid Retrieval (Combined) - Top Results:</h4>")
        hybrid_results = self.retriever.search(query, top_k=top_k, rerank=False)
        for i, result_item in enumerate(hybrid_results[:top_k], 1):
            chunk = result_item.get("chunk", result_item)
            text = chunk.get("text", chunk.get("clean_text", ""))[:200]
            score = result_item.get("score_hybrid", result_item.get("hybrid_score", 0.0))
            html_parts.append(f"<p style='color: #e0e0e0;'><strong style='color: #ffffff;'>{i}.</strong> [Score: <span style='color: #66b3ff;'>{score:.4f}</span>] <span style='color: #d0d0d0; direction: rtl; text-align: right; display: inline-block;'>{text}...</span></p>")
        
        return "\n".join(html_parts)


def create_gradio_interface():
    """Create and launch the Gradio interface."""
    tester = RetrieverTester()
    
    # Define the interface
    with gr.Blocks(title="KB Retriever Tester", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🔍اختبار قاعدة البيانات المعرفية للصحة النفسية
        
        الغرض هو اختبار وعرض استرجاع البيانات من قاعدة البيانات المعرفية للصحة النفسية.
        - **Sparse Retrieval**: BM25 keyword matching
        - **Dense Retrieval**: MARBERTv2 semantic embeddings
        - **Hybrid Retrieval**: Combined sparse + dense with configurable alpha
        - **Reranking**: Optional Cross-Encoder re-scoring
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="Query (Arabic)",
                    placeholder="مثال: ما هو الاكتئاب؟",
                    lines=2
                )
                
                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Top K Results"
                    )
                    alpha_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.4,
                        step=0.1,
                        label="Alpha (0=all dense, 1=all sparse)"
                    )
                
                with gr.Row():
                    rerank_checkbox = gr.Checkbox(
                        value=True,
                        label="Use Reranking"
                    )
                    search_btn = gr.Button("🔍 Search", variant="primary")
            
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Retrieval Configuration")
                gr.Markdown(f"""
                - **Alpha = {alpha_slider.value}**: 
                  - {int((1-alpha_slider.value)*100)}% Dense (semantic)
                  - {int(alpha_slider.value*100)}% Sparse (keyword)
                """)
        
        with gr.Tabs():
            with gr.Tab("📋 Results"):
                results_html = gr.HTML(label="Retrieval Results")
                results_json = gr.JSON(label="Results (JSON)", visible=False)
            
            with gr.Tab("🔬 Comparison"):
                comparison_html = gr.HTML(label="Method Comparison")
                compare_btn = gr.Button("Compare Methods", variant="secondary")
            
            with gr.Tab("📊 Statistics"):
                stats_md = gr.Markdown("### KB Statistics\n\nClick 'Get Stats' to load statistics.")
                stats_btn = gr.Button("Get Stats")
        
        # Event handlers
        def search_handler(query, top_k, alpha, rerank):
            html, json_data = tester.search(
                query=query,
                top_k=int(top_k),
                alpha=float(alpha),
                rerank=rerank
            )
            return html, json_data
        
        def compare_handler(query, top_k):
            return tester.compare_retrieval_methods(query, top_k=int(top_k))
        
        def stats_handler():
            if not tester._initialized:
                tester.initialize()
            
            try:
                sparse_stats = tester.retriever.sparse_index.get_stats()
                
                # Try to get vector DB stats if available
                vector_db_stats = {}
                try:
                    if hasattr(tester.retriever.dense_index, 'vector_db') and tester.retriever.dense_index.vector_db:
                        vector_db_stats = tester.retriever.dense_index.vector_db.get_stats()
                except:
                    pass
                
                stats_text = f"""
                ### Knowledge Base Statistics
                
                - **Total Chunks**: {sparse_stats.get('total_chunks', 'N/A')}
                - **KB Families**: {sparse_stats.get('kb_families', 'N/A')}
                - **Documents**: {sparse_stats.get('total_docs', 'N/A')}
                
                ### Index Information
                - **BM25 Index**: Ready ({sparse_stats.get('index_type', 'BM25')})
                - **Vector DB**: {'Ready' if vector_db_stats else 'Not loaded'} (ChromaDB)
                - **Embedding Model**: MARBERTv2
                - **Retriever Alpha**: {tester.retriever.alpha} ({(1-tester.retriever.alpha)*100:.0f}% dense, {tester.retriever.alpha*100:.0f}% sparse)
                """
                return stats_text
            except Exception as e:
                return f"Error loading stats: {str(e)}"
        
        search_btn.click(
            fn=search_handler,
            inputs=[query_input, top_k_slider, alpha_slider, rerank_checkbox],
            outputs=[results_html, results_json]
        )
        
        compare_btn.click(
            fn=compare_handler,
            inputs=[query_input, top_k_slider],
            outputs=[comparison_html]
        )
        
        stats_btn.click(
            fn=stats_handler,
            outputs=[stats_md]
        )
        
        # Example queries
        gr.Markdown("### 💡 Example Queries")
        examples = gr.Examples(
            examples=[
                ["ما هو الاكتئاب؟"],
                ["كيف أتعامل مع القلق؟"],
                ["أعراض اضطراب الهلع"],
                ["علاج الرهاب الاجتماعي"],
                ["مشاكل النوم والأرق"]
            ],
            inputs=query_input
        )
    
    return demo


def main():
    """Main entry point to launch the Gradio interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Gradio interface for KB retriever testing")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Server name")
    parser.add_argument("--server-port", type=int, default=7860, help="Server port")
    args = parser.parse_args()
    
    demo = create_gradio_interface()
    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port
    )


if __name__ == "__main__":
    main()

