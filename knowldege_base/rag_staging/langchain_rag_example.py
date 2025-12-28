"""
Example: Using LangChain with our custom hybrid retriever.

This demonstrates how to:
1. Use our HybridKBRetrieverWrapper with LangChain
2. Create a simple RAG chain
3. Build more complex chains with memory, tools, etc.
"""

from __future__ import annotations

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .langchain_retriever import HybridKBRetrieverWrapper


def simple_rag_chain_example():
    """Simple RAG chain using our retriever."""
    print("[INFO] Building retriever...")
    retriever = HybridKBRetrieverWrapper.build(alpha=0.4, use_cpu=True)
    
    print("[INFO] Creating simple RAG chain...")
    
    # Define a prompt template
    template = """استخدم المعلومات التالية من قاعدة المعرفة للإجابة على السؤال.
    إذا لم تجد الإجابة في المعلومات المقدمة، قل أنك لا تعرف.

    المعلومات:
    {context}

    السؤال: {question}

    الإجابة:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Create a simple chain
    # Note: You'll need to add your LLM here
    # For now, this is just the structure
    
    def format_docs(docs):
        """Format retrieved documents into context."""
        return "\n\n".join([doc.page_content for doc in docs])
    
    # This is a template - you'll need to add your LLM
    # chain = (
    #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm  # Your LLM here (e.g., from langchain.llms or langchain.chat_models)
    #     | StrOutputParser()
    # )
    
    print("[OK] Chain structure created (add LLM to complete)")
    return retriever, prompt


def retrieval_only_example():
    """Example of using the retriever standalone."""
    print("[INFO] Building retriever...")
    retriever = HybridKBRetrieverWrapper.build(alpha=0.4, use_cpu=True)
    
    query = "ما هو الاكتئاب؟"
    print(f"\n[QUERY] {query}")
    
    # Retrieve documents
    docs = retriever.get_relevant_documents(query, top_k=5, rerank=True)
    
    print(f"\n[RESULTS] Found {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Document {i} ---")
        print(f"Score: {doc.metadata.get('score', 'N/A')}")
        print(f"Source: {doc.metadata.get('kb_family', 'N/A')}")
        print(f"Title: {doc.metadata.get('title', 'N/A')}")
        print(f"Content: {doc.page_content[:200]}...")
    
    return docs


def retrieval_with_scores_example():
    """Example of retrieving with scores."""
    print("[INFO] Building retriever...")
    retriever = HybridKBRetrieverWrapper.build(alpha=0.4, use_cpu=True)
    
    query = "كيف أتعامل مع القلق؟"
    print(f"\n[QUERY] {query}")
    
    # Retrieve with scores
    results = retriever.search_with_scores(query, top_k=5, rerank=True)
    
    print(f"\n[RESULTS] Found {len(results)} documents with scores:")
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n--- Document {i} (Score: {score:.4f}) ---")
        print(f"Source: {doc.metadata.get('kb_family', 'N/A')}")
        print(f"Title: {doc.metadata.get('title', 'N/A')}")
        print(f"Sparse: {doc.metadata.get('sparse_score', 0):.4f}")
        print(f"Dense: {doc.metadata.get('dense_score', 0):.4f}")
        print(f"Content: {doc.page_content[:200]}...")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        mode = "retrieval"
    
    if mode == "retrieval":
        retrieval_only_example()
    elif mode == "scores":
        retrieval_with_scores_example()
    elif mode == "chain":
        simple_rag_chain_example()
    else:
        print("Usage: python langchain_rag_example.py [retrieval|scores|chain]")
        print("\nExamples:")
        print("  python langchain_rag_example.py retrieval  # Simple retrieval")
        print("  python langchain_rag_example.py scores     # Retrieval with scores")
        print("  python langchain_rag_example.py chain      # RAG chain structure")



