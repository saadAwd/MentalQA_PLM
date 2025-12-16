# examples_read_book.py
# Simple examples of how to read and work with book JSON data

import json
import os

# Path to the book JSONL file
BOOKS_FILE = "data/processed/books_poc.jsonl"


# ============================================================================
# EXAMPLE 1: Basic - Read one book from JSONL
# ============================================================================
print("="*70)
print("EXAMPLE 1: Basic Reading")
print("="*70)

with open(BOOKS_FILE, "r", encoding="utf-8") as f:
    # Read first line (first book)
    line = f.readline().strip()
    book = json.loads(line)
    
    print(f"Title: {book['title']}")
    print(f"URL: {book['url']}")
    print(f"Text Length: {book['text_length']:,} characters")
    print(f"\nFirst 200 chars of text:")
    print(book['clean_text'][:200])


# ============================================================================
# EXAMPLE 2: Read all books
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 2: Read All Books")
print("="*70)

books = []
with open(BOOKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            books.append(json.loads(line))

print(f"Total books: {len(books)}")
for book in books:
    print(f"  - {book['doc_id']}: {book['title']}")


# ============================================================================
# EXAMPLE 3: Get specific book by ID
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 3: Get Book by ID")
print("="*70)

def get_book_by_id(book_id):
    """Get a book by its ID."""
    with open(BOOKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                book = json.loads(line)
                if book.get("doc_id") == f"ncmh_book_{book_id}":
                    return book
    return None

book_1 = get_book_by_id(1)
if book_1:
    print(f"Found book: {book_1['title']}")
    print(f"Text length: {book_1['text_length']:,} chars")


# ============================================================================
# EXAMPLE 4: Extract just the text content
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 4: Extract Text Content")
print("="*70)

with open(BOOKS_FILE, "r", encoding="utf-8") as f:
    book = json.loads(f.readline().strip())
    text = book['clean_text']
    
    print(f"Full text length: {len(text):,} characters")
    print(f"\nSample from middle of book:")
    middle_start = len(text) // 2
    print(text[middle_start:middle_start + 300])


# ============================================================================
# EXAMPLE 5: Search in book text
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 5: Search in Book Text")
print("="*70)

with open(BOOKS_FILE, "r", encoding="utf-8") as f:
    book = json.loads(f.readline().strip())
    text = book['clean_text']
    
    # Search for a term
    search_term = "المساعدة"
    if search_term in text:
        index = text.find(search_term)
        context = text[max(0, index-50):index+len(search_term)+50]
        print(f"Found '{search_term}' at position {index}")
        print(f"Context: ...{context}...")
    else:
        print(f"'{search_term}' not found")


# ============================================================================
# EXAMPLE 6: Split text into chunks
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 6: Split into Chunks")
print("="*70)

def chunk_text(text, chunk_size=800):
    """Split text into chunks of approximately chunk_size characters."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        # Try to end at sentence boundary
        if end < len(text):
            # Look for period or newline in last 100 chars
            for i in range(end, max(start, end-100), -1):
                if text[i] in '.\n؟':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({
                "chunk_number": len(chunks) + 1,
                "start": start,
                "end": end,
                "text": chunk,
                "length": len(chunk)
            })
        start = end
    
    return chunks

with open(BOOKS_FILE, "r", encoding="utf-8") as f:
    book = json.loads(f.readline().strip())
    chunks = chunk_text(book['clean_text'], chunk_size=800)
    
    print(f"Total chunks: {len(chunks)}")
    print(f"\nFirst chunk:")
    print(f"  Length: {chunks[0]['length']} chars")
    print(f"  Preview: {chunks[0]['text'][:150]}...")


# ============================================================================
# EXAMPLE 7: Save chunks to new file
# ============================================================================
print("\n" + "="*70)
print("EXAMPLE 7: Save Chunks to File")
print("="*70)

with open(BOOKS_FILE, "r", encoding="utf-8") as f:
    book = json.loads(f.readline().strip())
    chunks = chunk_text(book['clean_text'], chunk_size=800)
    
    # Save chunks to JSONL
    chunks_file = "data/processed/book_chunks_example.jsonl"
    os.makedirs(os.path.dirname(chunks_file), exist_ok=True)
    
    with open(chunks_file, "w", encoding="utf-8") as out_f:
        for chunk in chunks:
            chunk_doc = {
                "chunk_id": f"{book['doc_id']}_chunk_{chunk['chunk_number']:04d}",
                "doc_id": book['doc_id'],
                "title": book['title'],
                "url": book['url'],
                "chunk_number": chunk['chunk_number'],
                "text": chunk['text'],
                "text_length": chunk['length']
            }
            out_f.write(json.dumps(chunk_doc, ensure_ascii=False) + "\n")
    
    print(f"Saved {len(chunks)} chunks to {chunks_file}")

