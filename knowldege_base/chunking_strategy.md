# Best Strategy for Chunking Books

## Overview
Effective chunking is crucial for RAG (Retrieval-Augmented Generation) systems. The goal is to create semantically coherent chunks that preserve context while fitting within embedding model limits.

## Key Principles

### 1. **Semantic Coherence**
- ✅ Preserve complete sentences and paragraphs
- ✅ Don't break mid-thought or mid-concept
- ✅ Respect natural text boundaries

### 2. **Optimal Size**
- **Target:** 400-800 characters (or 100-300 tokens)
- **Why:** 
  - Too small: Loses context, poor semantic meaning
  - Too large: Exceeds embedding limits, less precise retrieval
- **For Arabic:** Slightly larger chunks (600-1000 chars) due to character density

### 3. **Overlap Strategy**
- **Overlap:** 50-150 characters between chunks
- **Why:** Preserves context across boundaries, improves retrieval
- **Trade-off:** More chunks = more storage, but better retrieval

### 4. **Structure Awareness**
- Respect document structure (chapters, sections, headings)
- Keep related content together
- Don't mix different topics in one chunk

## Chunking Strategies

### Strategy 1: **Sentence-Based Chunking** (Recommended)
```
Pros:
- Preserves semantic meaning
- Natural boundaries
- Good for Arabic text

How:
1. Split by sentence boundaries (. ! ? ؟)
2. Group sentences until ~600-800 chars
3. Add overlap (last 2-3 sentences of previous chunk)
```

### Strategy 2: **Paragraph-Based Chunking**
```
Pros:
- Maintains topic coherence
- Natural text structure
- Good for structured documents

How:
1. Split by paragraph breaks (\n\n)
2. Combine paragraphs until size limit
3. Overlap with last paragraph
```

### Strategy 3: **Sliding Window**
```
Pros:
- Maximum context preservation
- Good for dense technical content

How:
1. Fixed-size windows (e.g., 600 chars)
2. Slide with overlap (e.g., 100 chars)
3. Break at sentence boundaries when possible
```

### Strategy 4: **Hierarchical Chunking**
```
Pros:
- Respects document structure
- Multiple granularities
- Best for long documents

How:
1. Split by major sections first
2. Then by paragraphs
3. Then by sentences if needed
```

## Recommended Approach for Books

### **Hybrid Strategy: Sentence + Paragraph Aware**

1. **Primary:** Split by paragraphs (respects structure)
2. **Secondary:** If paragraph too large, split by sentences
3. **Overlap:** Last 2-3 sentences of previous chunk
4. **Size:** Target 600-1000 characters per chunk
5. **Boundaries:** Always break at sentence ends

### Implementation Steps:

```python
1. Split text into paragraphs
2. For each paragraph:
   - If size < 600 chars: Keep as-is or combine with next
   - If size 600-1000 chars: Keep as chunk
   - If size > 1000 chars: Split by sentences
3. Add overlap (last sentences from previous chunk)
4. Create metadata (chunk_id, position, parent_section)
```

## Arabic-Specific Considerations

### 1. **RTL Text Flow**
- Arabic reads right-to-left
- Chunk boundaries should respect this
- Sentence endings: `.` `!` `؟` `،` (comma for pauses)

### 2. **Character Density**
- Arabic text is more compact than English
- 600 Arabic chars ≈ 300-400 English words
- Adjust size targets accordingly

### 3. **Diacritics & Formatting**
- Preserve diacritics (they affect meaning)
- Handle formatting characters (ـ) properly
- Clean but preserve structure

### 4. **Mixed Content**
- Books may have Arabic + English
- Preserve both languages in chunks
- Don't split mid-translation

## Chunk Metadata

Each chunk should include:

```json
{
  "chunk_id": "doc_id_chunk_0001",
  "doc_id": "ncmh_book_1",
  "chunk_number": 1,
  "text": "...",
  "text_length": 750,
  "char_count": 750,
  "word_count": ~150,
  "start_position": 0,
  "end_position": 750,
  "overlap_with_previous": false,
  "overlap_with_next": true,
  "metadata": {
    "title": "...",
    "url": "...",
    "section": "Chapter 1",
    "language": "ar"
  }
}
```

## Size Guidelines

| Content Type | Chunk Size | Overlap | Use Case |
|-------------|------------|---------|----------|
| Dense technical | 400-600 chars | 100-150 | Precise retrieval |
| General content | 600-800 chars | 100-150 | Balanced |
| Narrative/books | 800-1000 chars | 150-200 | Context preservation |
| Long documents | 1000-1500 chars | 200-300 | Maximum context |

## Best Practices

### ✅ DO:
- Break at sentence boundaries
- Preserve paragraph structure
- Add meaningful overlap
- Include metadata for context
- Test chunk quality with sample queries

### ❌ DON'T:
- Break mid-sentence
- Create chunks smaller than 200 chars
- Ignore document structure
- Create chunks larger than 1500 chars
- Split related concepts

## Implementation Priority

1. **Start Simple:** Sentence-based with overlap
2. **Add Structure:** Respect paragraphs
3. **Optimize:** Adjust size based on retrieval quality
4. **Refine:** Add hierarchical chunking if needed

## Testing Chunk Quality

Test with:
- Sample queries from your domain
- Check if relevant chunks are retrieved
- Verify context preservation
- Measure retrieval precision/recall

