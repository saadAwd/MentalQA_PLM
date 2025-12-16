# create_chunks.py
# Advanced chunking strategy for books - optimized for Arabic text and RAG

import json
import re
import os
from typing import List, Dict, Optional
from config import PROCESSED_DIR


class BookChunker:
    """
    Advanced chunker for book text with multiple strategies.
    Optimized for Arabic text and RAG systems.
    """
    
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        min_chunk_size: int = 200,
        max_chunk_size: int = 1500,
        strategy: str = "hybrid"
    ):
        """
        Initialize chunker with parameters.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            min_chunk_size: Minimum chunk size (merge if smaller)
            max_chunk_size: Maximum chunk size (split if larger)
            strategy: "sentence", "paragraph", "hybrid", or "sliding"
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.strategy = strategy
        
        # Arabic sentence endings
        self.sentence_endings = r'[.!?؟]\s+'
        # Paragraph break
        self.paragraph_break = r'\n\s*\n'
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, preserving punctuation."""
        # Split by sentence endings
        sentences = re.split(self.sentence_endings, text)
        # Add punctuation back
        result = []
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Find the punctuation that was removed
            if i < len(sentences) - 1:
                # Try to find what punctuation was used
                match = re.search(self.sentence_endings, text[text.find(sentence):text.find(sentence)+len(sentence)+5])
                if match:
                    sentence += match.group(0).strip()
            
            if sentence:
                result.append(sentence)
        
        return result
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(self.paragraph_break, text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def chunk_by_sentences(self, text: str) -> List[Dict]:
        """Chunk text by grouping sentences."""
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_num = 1
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If adding this sentence exceeds max size and we have content
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'start': len(' '.join(chunks)) if chunks else 0,
                    'size': current_size,
                    'chunk_num': chunk_num
                })
                chunk_num += 1
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk[-1:]
                current_chunk = overlap_sentences + [sentence]
                current_size = sum(len(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
            
            # If we've reached target size, create chunk
            if current_size >= self.chunk_size:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'start': sum(c['size'] for c in chunks),
                    'size': current_size,
                    'chunk_num': chunk_num
                })
                chunk_num += 1
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk[-1:]
                current_chunk = overlap_sentences
                current_size = sum(len(s) for s in current_chunk)
        
        # Add remaining content
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    'text': chunk_text,
                    'start': sum(c['size'] for c in chunks),
                    'size': current_size,
                    'chunk_num': chunk_num
                })
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str) -> List[Dict]:
        """Chunk text by grouping paragraphs."""
        paragraphs = self.split_into_paragraphs(text)
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_num = 1
        
        for para in paragraphs:
            para_size = len(para)
            
            # If paragraph is too large, split it by sentences
            if para_size > self.max_chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'start': sum(c['size'] for c in chunks),
                        'size': current_size,
                        'chunk_num': chunk_num
                    })
                    chunk_num += 1
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph by sentences
                para_sentences = self.split_into_sentences(para)
                for sentence in para_sentences:
                    if current_size + len(sentence) > self.max_chunk_size and current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append({
                            'text': chunk_text,
                            'start': sum(c['size'] for c in chunks),
                            'size': current_size,
                            'chunk_num': chunk_num
                        })
                        chunk_num += 1
                        current_chunk = current_chunk[-1:] + [sentence]  # Overlap
                        current_size = sum(len(s) for s in current_chunk)
                    else:
                        current_chunk.append(sentence)
                        current_size += len(sentence)
                
                continue
            
            # If adding this paragraph exceeds max size
            if current_size + para_size > self.max_chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'start': sum(c['size'] for c in chunks),
                    'size': current_size,
                    'chunk_num': chunk_num
                })
                chunk_num += 1
                
                # Start new chunk with overlap (last paragraph)
                current_chunk = [current_chunk[-1], para] if current_chunk else [para]
                current_size = sum(len(p) for p in current_chunk)
            else:
                current_chunk.append(para)
                current_size += para_size
            
            # If we've reached target size, create chunk
            if current_size >= self.chunk_size:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'start': sum(c['size'] for c in chunks),
                    'size': current_size,
                    'chunk_num': chunk_num
                })
                chunk_num += 1
                
                # Start new chunk with overlap
                current_chunk = current_chunk[-1:] if current_chunk else []
                current_size = sum(len(p) for p in current_chunk)
        
        # Add remaining content
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append({
                    'text': chunk_text,
                    'start': sum(c['size'] for c in chunks),
                    'size': current_size,
                    'chunk_num': chunk_num
                })
        
        return chunks
    
    def chunk_hybrid(self, text: str) -> List[Dict]:
        """
        Hybrid strategy: Use paragraphs primarily, sentences for large paragraphs.
        Best balance for books.
        """
        return self.chunk_by_paragraphs(text)
    
    def create_chunks(self, text: str, doc_metadata: Dict) -> List[Dict]:
        """
        Create chunks from text using selected strategy.
        
        Args:
            text: Text to chunk
            doc_metadata: Document metadata (doc_id, title, url, etc.)
        
        Returns:
            List of chunk dictionaries
        """
        if not text or len(text.strip()) < self.min_chunk_size:
            return []
        
        # Choose strategy
        if self.strategy == "sentence":
            raw_chunks = self.chunk_by_sentences(text)
        elif self.strategy == "paragraph":
            raw_chunks = self.chunk_by_paragraphs(text)
        elif self.strategy == "hybrid":
            raw_chunks = self.chunk_hybrid(text)
        else:
            raw_chunks = self.chunk_by_sentences(text)
        
        # Convert to final format with metadata
        chunks = []
        for i, raw_chunk in enumerate(raw_chunks):
            chunk = {
                "chunk_id": f"{doc_metadata.get('doc_id', 'doc')}_chunk_{raw_chunk['chunk_num']:04d}",
                "doc_id": doc_metadata.get('doc_id'),
                "chunk_number": raw_chunk['chunk_num'],
                "text": raw_chunk['text'],
                "text_length": len(raw_chunk['text']),
                "char_count": len(raw_chunk['text']),
                "word_count": len(raw_chunk['text'].split()),
                "start_position": raw_chunk['start'],
                "end_position": raw_chunk['start'] + raw_chunk['size'],
                "overlap_with_previous": i > 0,
                "overlap_with_next": i < len(raw_chunks) - 1,
                "title": doc_metadata.get('title'),
                "url": doc_metadata.get('url'),
                "language": doc_metadata.get('language', 'ar'),
                "content_type": doc_metadata.get('content_type', 'book'),
            }
            chunks.append(chunk)
        
        return chunks


def chunk_book_file(
    input_file: str,
    output_file: Optional[str] = None,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    strategy: str = "hybrid"
):
    """
    Chunk a book JSONL file.
    
    Args:
        input_file: Path to book JSONL file
        output_file: Output path (default: adds _chunks.jsonl)
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        strategy: Chunking strategy
    """
    if output_file is None:
        base_name = input_file.replace('.jsonl', '')
        output_file = f"{base_name}_chunks.jsonl"
    
    print(f"\n{'='*70}")
    print(f"CREATING CHUNKS")
    print(f"{'='*70}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Strategy: {strategy}")
    print(f"Chunk size: {chunk_size} chars, Overlap: {chunk_overlap} chars")
    
    chunker = BookChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=strategy
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
            
            # Create chunks
            doc_metadata = {
                'doc_id': doc.get('doc_id'),
                'title': doc.get('title'),
                'url': doc.get('url'),
                'language': doc.get('language', 'ar'),
                'content_type': doc.get('content_type', 'book'),
            }
            
            chunks = chunker.create_chunks(text, doc_metadata)
            
            # Write chunks
            for chunk in chunks:
                outfile.write(json.dumps(chunk, ensure_ascii=False) + '\n')
                total_chunks += 1
            
            print(f"  ✓ Document {line_num}: Created {len(chunks)} chunks")
    
    print(f"\n{'='*70}")
    print("CHUNKING COMPLETE")
    print(f"{'='*70}")
    print(f"Total chunks created: {total_chunks}")
    print(f"Output saved to: {output_file}")
    
    return output_file


if __name__ == "__main__":
    import sys
    
    # Default: chunk the fixed book file
    books_file = os.path.join(PROCESSED_DIR, "books_poc_fixed.jsonl")
    
    if len(sys.argv) > 1:
        books_file = sys.argv[1]
    
    if not os.path.exists(books_file):
        print(f"Error: File not found: {books_file}")
        sys.exit(1)
    
    # Create chunks with recommended settings
    chunk_book_file(
        books_file,
        chunk_size=800,      # Good balance for Arabic
        chunk_overlap=150,   # Preserves context
        strategy="hybrid"    # Best for books
    )
    
    print("\n✅ Chunking complete!")
    print("\nNext steps:")
    print("1. Review chunk quality with: python show_chunk.py [number]")
    print("2. Generate embeddings from chunks")
    print("3. Test retrieval quality")

