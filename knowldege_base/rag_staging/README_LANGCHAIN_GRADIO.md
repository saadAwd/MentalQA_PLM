# LangChain & Gradio Integration for KB Retriever

This directory now includes LangChain integration and a Gradio interface for testing and visualizing the knowledge base retriever.

## 🎯 What's New

### 1. LangChain Retriever Wrapper (`langchain_retriever.py`)
- Wraps our custom `HybridKBRetriever` to work with LangChain
- Maintains all custom logic (BM25 + dense, reranking, etc.)
- Compatible with LangChain chains, agents, and tools

### 2. Gradio Testing Interface (`gradio_retriever_test.py`)
- Interactive web UI for testing retrieval
- Visualize retrieval results with scores
- Compare sparse vs dense vs hybrid retrieval
- Adjust parameters (alpha, top_k) in real-time
- View chunk metadata and sources

## 📦 Installation

Add to your `requirements.txt`:
```bash
langchain>=0.1.0
langchain-core>=0.1.0
gradio>=4.0.0
```

Install:
```bash
pip install langchain langchain-core gradio
```

## 🚀 Quick Start

### Launch Gradio Interface

```bash
# From project root
python -m knowldege_base.rag_staging.gradio_retriever_test

# Or with custom port
python -m knowldege_base.rag_staging.gradio_retriever_test --server-port 8080

# Or with public link (for sharing)
python -m knowldege_base.rag_staging.gradio_retriever_test --share
```

The interface will open at `http://127.0.0.1:7860` (or your specified port).

### Use LangChain Retriever in Code

```python
from knowldege_base.rag_staging.langchain_retriever import HybridKBRetrieverWrapper

# Build retriever
retriever = HybridKBRetrieverWrapper.build(alpha=0.4, use_cpu=True)

# Simple retrieval
docs = retriever.get_relevant_documents("ما هو الاكتئاب؟", top_k=5)

# Retrieval with scores
results = retriever.search_with_scores("كيف أتعامل مع القلق؟", top_k=5)

# Use with LangChain chains
from langchain.chains import RetrievalQA
# ... (see langchain_rag_example.py for full example)
```

## 🎨 Gradio Interface Features

### Main Search Tab
- **Query Input**: Enter Arabic questions
- **Top K Slider**: Control number of results (1-20)
- **Alpha Slider**: Adjust sparse/dense balance (0=all dense, 1=all sparse)
- **Reranking Toggle**: Enable/disable Cross-Encoder reranking
- **Results Display**: 
  - Formatted HTML with scores
  - JSON export for programmatic access
  - Chunk metadata (source, title, URL)
  - Score breakdown (hybrid, sparse, dense, rerank)

### Comparison Tab
- Compare sparse-only, dense-only, and hybrid retrieval
- See how different methods rank the same query
- Understand which method works best for your queries

### Statistics Tab
- View KB statistics (total chunks, documents, families)
- Check index status

## 📊 Example Queries

The interface includes example queries:
- "ما هو الاكتئاب؟" (What is depression?)
- "كيف أتعامل مع القلق؟" (How do I deal with anxiety?)
- "أعراض اضطراب الهلع" (Panic disorder symptoms)
- "علاج الرهاب الاجتماعي" (Social phobia treatment)
- "مشاكل النوم والأرق" (Sleep problems and insomnia)

## 🔧 Advanced Usage

### Custom Alpha Values

Test different alpha values to find optimal balance:
- `alpha=0.0`: Pure dense retrieval (semantic only)
- `alpha=0.4`: 60% dense, 40% sparse (recommended)
- `alpha=0.7`: 30% dense, 70% sparse (more keyword-focused)
- `alpha=1.0`: Pure sparse retrieval (BM25 only)

### Integration with LangChain Chains

See `langchain_rag_example.py` for examples of:
- Simple RAG chains
- Retrieval with custom prompts
- Integration with LLMs
- Building complex multi-step chains

## 🐛 Troubleshooting

### "Vector database not found"
Run:
```bash
python -m knowldege_base.rag_staging.vector_db build
```

### "ChromaDB collection not found"
Rebuild the vector database:
```bash
python -m knowldege_base.rag_staging.vector_db build --force
```

### Import errors
Make sure you've installed:
```bash
pip install langchain langchain-core gradio
```

## 📝 Notes

- The retriever uses your existing `HybridKBRetriever` - no changes to core logic
- All custom features (MARBERTv2, reranking, etc.) are preserved
- Gradio interface is for testing/development - not production-ready
- LangChain wrapper is production-ready and can be used in any LangChain application

## 🎯 Next Steps

1. **Test the Gradio interface** to understand retrieval behavior
2. **Experiment with alpha values** to optimize for your use case
3. **Use LangChain wrapper** in your own applications
4. **Build custom chains** using the retriever with your LLM



