"""
Clean books text for RAG processing.

This script:
- Reads books_all_fixed.jsonl
- Applies text cleanup on clean_text field
- Writes books_all_ragclean.jsonl

The cleanup includes:
- Normalizing whitespace and line breaks
- Removing excessive formatting
- Making text more suitable for RAG chunking
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict

# Import the corruption fix function
# We need to go up one level from rag_staging to knowldege_base
KNOWLEDGE_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KNOWLEDGE_BASE_DIR)
from fix_arabic_corruption import fix_arabic_corruption

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_BASE_ROOT = os.path.dirname(THIS_DIR)
PROCESSED_DIR = os.path.join(KNOWLEDGE_BASE_ROOT, "data", "processed")


def clean_text_for_rag(text: str) -> str:
    """
    Clean text for RAG processing.
    
    This combines OCR-style cleanup with normalization suitable for chunking.
    """
    if not text:
        return ""
    
    # First apply Arabic corruption fixes
    cleaned = fix_arabic_corruption(text)
    
    # Normalize line breaks (keep paragraph breaks, remove excessive ones)
    lines = cleaned.split("\n")
    cleaned_lines = []
    prev_empty = False
    
    for line in lines:
        line = line.strip()
        if not line:
            if not prev_empty:
                cleaned_lines.append("")
            prev_empty = True
        else:
            cleaned_lines.append(line)
            prev_empty = False
    
    # Join with single newline between paragraphs
    cleaned = "\n".join(cleaned_lines)
    
    # Normalize multiple spaces to single space (but keep newlines)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    
    # Remove trailing spaces from lines
    lines = cleaned.split("\n")
    cleaned = "\n".join(line.rstrip() for line in lines)
    
    # Remove excessive blank lines (more than 2 consecutive)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    
    # Clean up any remaining weird characters that might interfere
    # Keep Arabic characters, basic punctuation, and whitespace
    # Remove control characters except newline and tab
    cleaned = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", cleaned)
    
    return cleaned.strip()


def clean_books_file(
    input_filename: str = "books_all_fixed.jsonl",
    output_filename: str = "books_all_ragclean.jsonl",
) -> str:
    """
    Clean all books in the input file and write to output file.
    
    Returns the output file path.
    """
    input_path = os.path.join(PROCESSED_DIR, input_filename)
    output_path = os.path.join(PROCESSED_DIR, output_filename)
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"\n{'='*70}")
    print(f"CLEANING BOOKS FOR RAG")
    print(f"{'='*70}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    
    num_books = 0
    num_cleaned = 0
    total_chars_before = 0
    total_chars_after = 0
    
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:
        
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                book = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  ⚠ Skipping line {line_num}: JSON decode error: {e}")
                continue
            
            num_books += 1
            doc_id = book.get("doc_id", f"book_{num_books}")
            
            original_text = book.get("clean_text", "")
            if not original_text:
                # Keep book even if no text
                outfile.write(json.dumps(book, ensure_ascii=False) + "\n")
                continue
            
            total_chars_before += len(original_text)
            
            # Clean the text
            cleaned_text = clean_text_for_rag(original_text)
            total_chars_after += len(cleaned_text)
            
            # Update book record
            updated_book = book.copy()
            updated_book["clean_text"] = cleaned_text
            updated_book["text_length"] = len(cleaned_text)
            updated_book["rag_cleaned"] = True
            
            # Check if text actually changed
            if cleaned_text != original_text:
                num_cleaned += 1
                if num_cleaned <= 5:  # Show first 5 changes
                    print(f"  ✓ Cleaned {doc_id}: {len(original_text)} → {len(cleaned_text)} chars")
            
            outfile.write(json.dumps(updated_book, ensure_ascii=False) + "\n")
    
    print(f"\n{'='*70}")
    print("CLEANING COMPLETE")
    print(f"{'='*70}")
    print(f"Books processed: {num_books}")
    print(f"Books with text changes: {num_cleaned}")
    print(f"Total characters before: {total_chars_before:,}")
    print(f"Total characters after: {total_chars_after:,}")
    print(f"Output file: {output_path}")
    
    return output_path


def main() -> None:
    """Main entry point."""
    clean_books_file()


if __name__ == "__main__":
    main()

