# read_book_data.py
# Helper script to read and work with book JSON data

import json
import os
from config import PROCESSED_DIR


def read_books_jsonl(file_path=None):
    """
    Read all books from a JSONL file.
    
    Args:
        file_path: Path to JSONL file. If None, uses default books_poc.jsonl
    
    Returns:
        List of book dictionaries
    """
    if file_path is None:
        file_path = os.path.join(PROCESSED_DIR, "books_poc.jsonl")
    
    books = []
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    books.append(json.loads(line))
    return books


def get_book_by_id(book_id, file_path=None):
    """
    Get a specific book by its ID.
    
    Args:
        book_id: Book ID (integer)
        file_path: Path to JSONL file
    
    Returns:
        Book dictionary or None if not found
    """
    books = read_books_jsonl(file_path)
    for book in books:
        if book.get("doc_id") == f"ncmh_book_{book_id}":
            return book
    return None


def get_book_text(book_id=None, book_data=None):
    """
    Extract clean text from a book.
    
    Args:
        book_id: Book ID (if book_data is None)
        book_data: Book dictionary (if provided, book_id is ignored)
    
    Returns:
        Clean text string
    """
    if book_data is None:
        if book_id is None:
            raise ValueError("Either book_id or book_data must be provided")
        book_data = get_book_by_id(book_id)
    
    if book_data is None:
        return None
    
    return book_data.get("clean_text", "")


def get_book_info(book_id=None, book_data=None):
    """
    Get book metadata.
    
    Returns:
        Dictionary with book info (title, url, text_length, etc.)
    """
    if book_data is None:
        if book_id is None:
            raise ValueError("Either book_id or book_data must be provided")
        book_data = get_book_by_id(book_id)
    
    if book_data is None:
        return None
    
    return {
        "doc_id": book_data.get("doc_id"),
        "title": book_data.get("title"),
        "url": book_data.get("url"),
        "language": book_data.get("language"),
        "text_length": book_data.get("text_length", 0),
        "text_source": book_data.get("text_source"),
        "pdf_url": book_data.get("pdf_url"),
    }


def search_in_book(text_query, book_id=None, book_data=None, case_sensitive=False):
    """
    Search for text within a book.
    
    Args:
        text_query: Text to search for
        book_id: Book ID (if book_data is None)
        book_data: Book dictionary
        case_sensitive: Whether search should be case sensitive
    
    Returns:
        List of tuples (start_index, end_index, context)
    """
    book_text = get_book_text(book_id=book_id, book_data=book_data)
    if not book_text:
        return []
    
    if not case_sensitive:
        book_text_lower = book_text.lower()
        text_query = text_query.lower()
    else:
        book_text_lower = book_text
    
    results = []
    start = 0
    context_length = 100  # Characters before/after match
    
    while True:
        index = book_text_lower.find(text_query, start)
        if index == -1:
            break
        
        # Get context around the match
        context_start = max(0, index - context_length)
        context_end = min(len(book_text), index + len(text_query) + context_length)
        context = book_text[context_start:context_end]
        
        results.append((index, index + len(text_query), context))
        start = index + 1
    
    return results


def chunk_book_text(book_id=None, book_data=None, chunk_size=800, overlap=100):
    """
    Split book text into chunks.
    
    Args:
        book_id: Book ID
        book_data: Book dictionary
        chunk_size: Maximum characters per chunk
        overlap: Characters to overlap between chunks
    
    Returns:
        List of dictionaries with chunk info
    """
    book_text = get_book_text(book_id=book_id, book_data=book_data)
    if not book_text:
        return []
    
    if book_data is None:
        if book_id is None:
            raise ValueError("Either book_id or book_data must be provided")
        book_data = get_book_by_id(book_id)
    
    chunks = []
    start = 0
    chunk_num = 1
    
    while start < len(book_text):
        # Try to end at sentence boundary
        end = start + chunk_size
        
        if end < len(book_text):
            # Look for sentence end in the last 200 chars
            sentence_end_chars = book_text[max(start, end - 200):end]
            sentence_end = max(
                sentence_end_chars.rfind('.\n'),
                sentence_end_chars.rfind('.\n'),
                sentence_end_chars.rfind('؟'),
                sentence_end_chars.rfind('!'),
            )
            
            if sentence_end != -1:
                end = max(start, end - 200) + sentence_end + 1
        
        chunk_text = book_text[start:end].strip()
        
        if chunk_text:
            chunks.append({
                "chunk_id": f"{book_data.get('doc_id', 'book')}_chunk_{chunk_num:04d}",
                "doc_id": book_data.get("doc_id"),
                "chunk_number": chunk_num,
                "start_position": start,
                "end_position": end,
                "text": chunk_text,
                "text_length": len(chunk_text),
                "title": book_data.get("title"),
                "url": book_data.get("url"),
            })
            chunk_num += 1
        
        # Move start position with overlap
        start = end - overlap if end < len(book_text) else end
    
    return chunks


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("BOOK DATA READER - EXAMPLE USAGE")
    print("="*70)
    
    # Read all books
    print("\n1. Reading all books:")
    books = read_books_jsonl()
    print(f"   Found {len(books)} book(s)")
    
    if books:
        book = books[0]
        
        # Get book info
        print("\n2. Book Information:")
        info = get_book_info(book_data=book)
        for key, value in info.items():
            print(f"   {key}: {value}")
        
        # Get text sample
        print("\n3. Text Sample (first 500 chars):")
        text = get_book_text(book_data=book)
        print(f"   {text[:500]}...")
        
        # Search example
        print("\n4. Search Example (searching for 'الصحة النفسية'):")
        results = search_in_book("الصحة النفسية", book_data=book)
        print(f"   Found {len(results)} match(es)")
        if results:
            print(f"   First match context: ...{results[0][2][:150]}...")
        
        # Chunking example
        print("\n5. Chunking Example (first 3 chunks):")
        chunks = chunk_book_text(book_data=book, chunk_size=800)
        print(f"   Total chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n   Chunk {i}:")
            print(f"   - ID: {chunk['chunk_id']}")
            print(f"   - Length: {chunk['text_length']} chars")
            print(f"   - Text preview: {chunk['text'][:150]}...")

