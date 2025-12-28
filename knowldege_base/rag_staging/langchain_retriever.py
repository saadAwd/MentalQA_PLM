"""
LangChain retriever wrapper for the existing hybrid retriever.

This module provides a LangChain-compatible retriever interface that wraps
our existing HybridKBRetriever, allowing integration with LangChain chains
and tools while maintaining our custom retrieval logic.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import PrivateAttr, Field

from .hybrid_retriever import HybridKBRetriever


class HybridKBRetrieverWrapper(BaseRetriever):
    """
    LangChain retriever wrapper for HybridKBRetriever.
    
    This allows our custom hybrid retrieval system to be used with LangChain
    chains and tools while maintaining all our custom logic (BM25 + dense,
    reranking, etc.).
    """
    
    # Use PrivateAttr for non-serializable fields
    _hybrid_retriever: HybridKBRetriever = PrivateAttr()
    
    # Declare return_metadata as a proper Pydantic field
    return_metadata: bool = Field(default=True, description="Whether to include metadata in returned documents")
    
    def __init__(
        self,
        hybrid_retriever: HybridKBRetriever,
        return_metadata: bool = True,
        **kwargs
    ):
        """
        Initialize the wrapper.
        
        Args:
            hybrid_retriever: The HybridKBRetriever instance to wrap
            return_metadata: Whether to include metadata in returned documents
            **kwargs: Additional arguments passed to BaseRetriever
        """
        super().__init__(return_metadata=return_metadata, **kwargs)
        self._hybrid_retriever = hybrid_retriever
    
    @classmethod
    def build(
        cls,
        alpha: Optional[float] = None,
        use_cpu: bool = True,
        return_metadata: bool = True
    ) -> "HybridKBRetrieverWrapper":
        """
        Build a new retriever wrapper with a fresh HybridKBRetriever.
        
        Args:
            alpha: Weight for sparse vs dense retrieval (0-1, higher = more sparse)
            use_cpu: Whether to force CPU usage
            return_metadata: Whether to include metadata in returned documents
            
        Returns:
            HybridKBRetrieverWrapper instance
        """
        hybrid_retriever = HybridKBRetriever.build(alpha=alpha, use_cpu=use_cpu)
        return cls(hybrid_retriever=hybrid_retriever, return_metadata=return_metadata)
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        top_k: int = 5,
        rerank: bool = True
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: The query string
            run_manager: LangChain callback manager
            top_k: Number of documents to retrieve
            rerank: Whether to use reranking if available
            
        Returns:
            List of LangChain Document objects
        """
        # Use our hybrid retriever to get chunks
        chunks = self._hybrid_retriever.search(
            query=query,
            top_k=top_k,
            rerank=rerank
        )
        
        # Convert chunks to LangChain Documents
        documents = []
        for chunk in chunks:
            # Extract text content
            text = chunk.get("text", chunk.get("clean_text", ""))
            
            # Build metadata
            metadata = {
                "chunk_id": chunk.get("chunk_id", ""),
                "kb_family": chunk.get("kb_family", ""),
                "doc_id": chunk.get("doc_id", ""),
                "title": chunk.get("title", ""),
                "url": chunk.get("url", ""),
                "score": chunk.get("score", 0.0),
                "sparse_score": chunk.get("sparse_score", 0.0),
                "dense_score": chunk.get("dense_score", 0.0),
                "hybrid_score": chunk.get("hybrid_score", 0.0),
            }
            
            # Add rerank score if available
            if "rerank_score" in chunk:
                metadata["rerank_score"] = chunk.get("rerank_score")
            
            # Create LangChain Document
            doc = Document(
                page_content=text,
                metadata=metadata if self.return_metadata else {}
            )
            documents.append(doc)
        
        return documents
    
    def get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        top_k: int = 5,
        rerank: bool = True
    ) -> List[Document]:
        """
        Public method to retrieve documents (LangChain interface).
        
        Args:
            query: The query string
            run_manager: LangChain callback manager
            top_k: Number of documents to retrieve
            rerank: Whether to use reranking if available
            
        Returns:
            List of LangChain Document objects
        """
        return self._get_relevant_documents(
            query=query,
            run_manager=run_manager,
            top_k=top_k,
            rerank=rerank
        )
    
    def search_with_scores(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = True
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents with their scores.
        
        Args:
            query: The query string
            top_k: Number of documents to retrieve
            rerank: Whether to use reranking if available
            
        Returns:
            List of (Document, score) tuples
        """
        chunks = self._hybrid_retriever.search(
            query=query,
            top_k=top_k,
            rerank=rerank
        )
        
        results = []
        for chunk in chunks:
            text = chunk.get("text", chunk.get("clean_text", ""))
            score = chunk.get("score", chunk.get("hybrid_score", 0.0))
            
            metadata = {
                "chunk_id": chunk.get("chunk_id", ""),
                "kb_family": chunk.get("kb_family", ""),
                "doc_id": chunk.get("doc_id", ""),
                "title": chunk.get("title", ""),
                "sparse_score": chunk.get("sparse_score", 0.0),
                "dense_score": chunk.get("dense_score", 0.0),
            }
            
            doc = Document(page_content=text, metadata=metadata)
            results.append((doc, score))
        
        return results

