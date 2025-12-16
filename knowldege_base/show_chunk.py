# show_chunk.py
# Display a specific chunk in a readable format

import json
import sys
import os

def show_chunk(chunk_number, chunks_file=None):
    """Display a chunk in a nicely formatted way."""
    if chunks_file is None:
        # Try fixed file first, fallback to original
        fixed_file = "data/processed/book_chunks_example_fixed.jsonl"
        original_file = "data/processed/book_chunks_example.jsonl"
        chunks_file = fixed_file if os.path.exists(fixed_file) else original_file
    
    # Read all chunks
    chunks = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line.strip()))
    
    # Get the requested chunk (chunk_number is 1-indexed, list is 0-indexed)
    if chunk_number < 1 or chunk_number > len(chunks):
        print(f"Error: Chunk {chunk_number} not found. Available chunks: 1-{len(chunks)}")
        return
    
    chunk = chunks[chunk_number - 1]
    
    # Format and display
    print("\n" + "="*80)
    print(f"CHUNK #{chunk_number}")
    print("="*80)
    print()
    
    # Metadata
    print("📋 METADATA:")
    print("-" * 80)
    print(f"  Chunk ID:      {chunk['chunk_id']}")
    print(f"  Document ID:   {chunk['doc_id']}")
    print(f"  Title:         {chunk['title']}")
    print(f"  URL:           {chunk['url']}")
    print(f"  Chunk Number:  {chunk['chunk_number']}")
    print(f"  Text Length:   {chunk['text_length']:,} characters")
    print()
    
    # Text content
    print("📄 TEXT CONTENT:")
    print("-" * 80)
    print()
    
    # Display text with proper formatting
    text = chunk['text']
    
    # Split into lines and display
    lines = text.split('\n')
    for i, line in enumerate(lines, 1):
        if line.strip():  # Only show non-empty lines
            print(f"  {line}")
        else:
            print()  # Empty line
    
    print()
    print("="*80)
    print()
    
    # Statistics
    print("📊 STATISTICS:")
    print("-" * 80)
    words = text.split()
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    print(f"  Total words:        ~{len(words)}")
    print(f"  Arabic characters: {arabic_chars:,}")
    print(f"  Lines:             {len([l for l in lines if l.strip()])}")
    print()


if __name__ == "__main__":
    chunk_num = 264
    if len(sys.argv) > 1:
        try:
            chunk_num = int(sys.argv[1])
        except ValueError:
            print(f"Error: '{sys.argv[1]}' is not a valid chunk number")
            sys.exit(1)
    
    show_chunk(chunk_num)

