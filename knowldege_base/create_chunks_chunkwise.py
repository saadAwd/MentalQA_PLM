# create_chunks_chunkwise.py
# Using ChunkWise library for Arabic-optimized chunking
# https://github.com/h9-tec/ChunkWise

import json
import os
from typing import List, Dict, Optional
from config import PROCESSED_DIR

try:
    from chunkwise import Chunker, RecursiveChunker, SentenceChunker, ParagraphChunker
    from chunkwise.language.arabic.preprocessor import normalize_arabic, remove_diacritics
    from chunkwise.language.detector import detect_language
    CHUNKWISE_AVAILABLE = True
except ImportError:
    CHUNKWISE_AVAILABLE = False
    print("⚠ ChunkWise not installed. Install with: pip install chunkwise")
    print("   For Arabic support: pip install chunkwise[arabic]")


def chunk_book_with_chunkwise(
    input_file: str,
    output_file: Optional[str] = None,
    strategy: str = "recursive",
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    language: str = "ar"
):
    """
    Chunk a book using ChunkWise library (optimized for Arabic).
    
    Args:
        input_file: Path to book JSONL file
        output_file: Output path (default: adds _chunks_chunkwise.jsonl)
        strategy: Chunking strategy (recursive, sentence, paragraph, etc.)
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        language: Language code ('ar' for Arabic, 'auto' for auto-detect)
    """
    if not CHUNKWISE_AVAILABLE:
        print("Error: ChunkWise is not installed")
        return None
    
    if output_file is None:
        base_name = input_file.replace('.jsonl', '')
        output_file = f"{base_name}_chunks_chunkwise.jsonl"
    
    print(f"\n{'='*70}")
    print(f"CREATING CHUNKS WITH CHUNKWISE")
    print(f"{'='*70}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Strategy: {strategy}")
    print(f"Chunk size: {chunk_size} chars, Overlap: {chunk_overlap} chars")
    print(f"Language: {language}")
    
    # Initialize chunker based on strategy
    if strategy == "recursive":
        # Recursive chunker - best for general text, respects structure
        chunker = RecursiveChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "؟", "!", " ", ""]  # Arabic-aware separators
        )
    elif strategy == "sentence":
        # Sentence chunker - groups sentences
        chunker = SentenceChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    elif strategy == "paragraph":
        # Paragraph chunker - respects paragraph boundaries
        chunker = ParagraphChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    else:
        # Use main Chunker class for other strategies
        chunker = Chunker(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            language=language
        )
    
    total_chunks = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            doc = json.loads(line)
            text = doc.get('clean_text', '')
            
            if not text:
                print(f"  ⚠ Document {line_num}: No text content")
                continue
            
            # Auto-detect language if needed
            if language == "auto":
                detected = detect_language(text)
                print(f"  Detected language: {detected}")
            
            # Create chunks using ChunkWise
            try:
                chunks = chunker.chunk(text)
                
                # Convert ChunkWise chunks to our format
                for i, chunk in enumerate(chunks, 1):
                    chunk_doc = {
                        "chunk_id": f"{doc.get('doc_id', 'doc')}_chunk_{i:04d}",
                        "doc_id": doc.get('doc_id'),
                        "chunk_number": i,
                        "text": chunk.content if hasattr(chunk, 'content') else str(chunk),
                        "text_length": len(chunk.content) if hasattr(chunk, 'content') else len(str(chunk)),
                        "char_count": len(chunk.content) if hasattr(chunk, 'content') else len(str(chunk)),
                        "word_count": len(chunk.content.split()) if hasattr(chunk, 'content') else len(str(chunk).split()),
                        "start_position": chunk.start_char if hasattr(chunk, 'start_char') else None,
                        "end_position": chunk.end_char if hasattr(chunk, 'end_char') else None,
                        "overlap_with_previous": i > 1,
                        "overlap_with_next": i < len(chunks),
                        "title": doc.get('title'),
                        "url": doc.get('url'),
                        "language": language if language != "auto" else detect_language(text),
                        "content_type": doc.get('content_type', 'book'),
                        "chunking_strategy": strategy,
                        "chunking_library": "chunkwise"
                    }
                    
                    # Add metadata if available
                    if hasattr(chunk, 'metadata'):
                        chunk_doc['chunk_metadata'] = chunk.metadata
                    
                    outfile.write(json.dumps(chunk_doc, ensure_ascii=False) + '\n')
                    total_chunks += 1
                
                print(f"  ✓ Document {line_num}: Created {len(chunks)} chunks")
                
            except Exception as e:
                print(f"  ✗ Error chunking document {line_num}: {e}")
                continue
    
    print(f"\n{'='*70}")
    print("CHUNKING COMPLETE")
    print(f"{'='*70}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Output saved to: {output_file}")
    
    return output_file


def compare_chunking_strategies(input_file: str):
    """Compare different ChunkWise strategies on the same text."""
    if not CHUNKWISE_AVAILABLE:
        print("Error: ChunkWise is not installed")
        return
    
    # Read sample text
    with open(input_file, 'r', encoding='utf-8') as f:
        doc = json.loads(f.readline().strip())
        text = doc.get('clean_text', '')[:5000]  # First 5000 chars for comparison
    
    print(f"\n{'='*70}")
    print("COMPARING CHUNKWISE STRATEGIES")
    print(f"{'='*70}")
    print(f"Sample text length: {len(text)} characters")
    
    strategies = [
        ("recursive", RecursiveChunker(chunk_size=800, chunk_overlap=150)),
        ("sentence", SentenceChunker(chunk_size=800, chunk_overlap=150)),
        ("paragraph", ParagraphChunker(chunk_size=800, chunk_overlap=150)),
    ]
    
    for name, chunker in strategies:
        try:
            chunks = chunker.chunk(text)
            avg_size = sum(len(c.content if hasattr(c, 'content') else str(c)) for c in chunks) / len(chunks) if chunks else 0
            print(f"\n{name.upper()}:")
            print(f"  Chunks: {len(chunks)}")
            print(f"  Avg size: {avg_size:.0f} chars")
            if chunks:
                print(f"  First chunk preview: {chunks[0].content[:100] if hasattr(chunks[0], 'content') else str(chunks[0])[:100]}...")
        except Exception as e:
            print(f"\n{name.upper()}: Error - {e}")


if __name__ == "__main__":
    import sys
    
    # Default: chunk the fixed book file
    books_file = os.path.join(PROCESSED_DIR, "books_poc_fixed.jsonl")
    
    if len(sys.argv) > 1:
        books_file = sys.argv[1]
    
    if not os.path.exists(books_file):
        print(f"Error: File not found: {books_file}")
        sys.exit(1)
    
    if not CHUNKWISE_AVAILABLE:
        print("\nInstalling ChunkWise...")
        print("Run: pip install chunkwise")
        print("For Arabic support: pip install chunkwise[arabic]")
        sys.exit(1)
    
    # Compare strategies first
    print("\n" + "="*70)
    print("STEP 1: Comparing Strategies")
    print("="*70)
    compare_chunking_strategies(books_file)
    
    # Create chunks with recommended strategy
    print("\n" + "="*70)
    print("STEP 2: Creating Chunks with Recursive Strategy")
    print("="*70)
    chunk_book_with_chunkwise(
        books_file,
        strategy="recursive",  # Best for Arabic books
        chunk_size=800,
        chunk_overlap=150,
        language="ar"
    )
    
    print("\n✅ Chunking complete!")
    print("\nNext steps:")
    print("1. Review chunk quality: python show_chunk.py [number]")
    print("2. Compare with previous chunks")
    print("3. Generate embeddings from chunks")

