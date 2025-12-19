"""
OCR-based re-extraction of books for clean Arabic text.

This script:
1. Reads books from books_all_fixed.jsonl
2. Downloads PDFs if needed (from pdf_url)
3. Converts PDF pages to images
4. Runs Tesseract OCR with Arabic language support
5. Outputs clean text to books_all_ocr.jsonl

Requirements:
- Tesseract OCR installed on system (https://github.com/UB-Mannheim/tesseract/wiki for Windows)
- Arabic language pack for Tesseract (ara.traineddata)
"""

from __future__ import annotations

import os
import json
import sys
from typing import Dict, List, Optional
from pathlib import Path

try:
    import pytesseract
    from PIL import Image
    import pypdfium2 as pdfium
    import requests
    from io import BytesIO
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pytesseract pillow pypdfium2 requests")
    sys.exit(1)

# Import loader to get processed directory path
from .loader import PROCESSED_DIR

THIS_DIR = Path(__file__).parent
KNOWLEDGE_BASE_ROOT = THIS_DIR.parent
RAW_BOOKS_DIR = KNOWLEDGE_BASE_ROOT / "data" / "raw" / "books"
OUTPUT_FILE = PROCESSED_DIR / "books_all_ocr.jsonl"


def check_tesseract() -> bool:
    """Check if Tesseract is installed and Arabic language pack is available."""
    # Try common Windows installation paths
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    
    for path in common_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"✓ Found Tesseract at: {path}")
            break
    
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✓ Tesseract version: {version}")
        
        # Check for Arabic language pack
        langs = pytesseract.get_languages()
        if "ara" in langs:
            print(f"✓ Arabic language pack found. Available languages: {', '.join(langs)}")
            return True
        else:
            print("⚠ Arabic language pack (ara) not found!")
            print("  Download from: https://github.com/tesseract-ocr/tessdata")
            print("  Place ara.traineddata in Tesseract tessdata directory")
            return False
    except Exception as e:
        print(f"✗ Tesseract not found: {e}")
        print("\n" + "=" * 80)
        print("SETUP REQUIRED:")
        print("=" * 80)
        print("1. Download Tesseract OCR for Windows:")
        print("   https://github.com/UB-Mannheim/tesseract/wiki")
        print("\n2. Install it (make sure to check 'Arabic' language pack)")
        print("\n3. If installed in non-standard location, set path:")
        print("   pytesseract.pytesseract.tesseract_cmd = r'C:\\Path\\To\\tesseract.exe'")
        print("\n4. Verify installation:")
        print("   tesseract --version")
        print("   tesseract --list-langs  # Should include 'ara'")
        print("=" * 80)
        return False


def download_pdf(pdf_url: str, pdf_path: Path) -> bool:
    """Download PDF if it doesn't exist locally."""
    if pdf_path.exists():
        print(f"  ✓ PDF already exists: {pdf_path}")
        return True
    
    try:
        print(f"  [Downloading PDF] {pdf_url}")
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        
        print(f"  ✓ Downloaded: {pdf_path} ({len(response.content)} bytes)")
        return True
    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False


def pdf_to_images(pdf_path: Path) -> List[Image.Image]:
    """Convert PDF pages to PIL Images."""
    try:
        pdf = pdfium.PdfDocument(str(pdf_path))
        images = []
        
        for page_num in range(len(pdf)):
            page = pdf.get_page(page_num)
            # Render at 300 DPI for good OCR quality
            bitmap = page.render(scale=300/72)  # 72 is default DPI
            pil_image = bitmap.to_pil()
            images.append(pil_image)
        
        pdf.close()
        print(f"  ✓ Converted {len(images)} pages to images")
        return images
    except Exception as e:
        print(f"  ✗ PDF to image conversion failed: {e}")
        return []


def ocr_image(image: Image.Image, lang: str = "ara") -> str:
    """Extract text from image using OCR."""
    try:
        # Use Arabic language, with page segmentation mode 6 (uniform block of text)
        text = pytesseract.image_to_string(
            image,
            lang=lang,
            config="--psm 6"  # Assume uniform block of text
        )
        return text
    except Exception as e:
        print(f"  ⚠ OCR error: {e}")
        return ""


def clean_ocr_text(text: str) -> str:
    """Clean OCR output: normalize spaces, remove excessive line breaks."""
    if not text:
        return ""
    
    # Normalize line breaks (keep paragraph breaks, remove excessive ones)
    lines = text.split("\n")
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
    text = "\n".join(cleaned_lines)
    
    # Normalize multiple spaces to single space (but keep newlines)
    import re
    text = re.sub(r"[ \t]+", " ", text)
    
    # Remove trailing spaces from lines
    lines = text.split("\n")
    text = "\n".join(line.rstrip() for line in lines)
    
    return text.strip()


def extract_book_with_ocr(book: Dict) -> Optional[Dict]:
    """Re-extract book text using OCR."""
    doc_id = book.get("doc_id", "")
    pdf_url = book.get("pdf_url")
    pdf_path_str = book.get("pdf_path", "")
    
    if not pdf_url:
        print(f"  ⚠ No PDF URL for {doc_id}, skipping")
        return None
    
    # Determine local PDF path
    if pdf_path_str:
        pdf_path = KNOWLEDGE_BASE_ROOT / pdf_path_str
    else:
        # Extract book ID from doc_id
        book_id = doc_id.replace("ncmh_book_", "")
        pdf_path = RAW_BOOKS_DIR / f"book_{book_id}.pdf"
    
    print(f"\n[Processing] {doc_id}")
    print(f"  Title: {book.get('title', 'N/A')}")
    
    # Download PDF if needed
    if not download_pdf(pdf_url, pdf_path):
        return None
    
    # Convert PDF to images
    images = pdf_to_images(pdf_path)
    if not images:
        return None
    
    # OCR each page
    print(f"  [Running OCR on {len(images)} pages]...")
    ocr_texts = []
    for i, img in enumerate(images, 1):
        print(f"    Page {i}/{len(images)}...", end="\r")
        page_text = ocr_image(img, lang="ara")
        if page_text:
            ocr_texts.append(page_text)
    
    print(f"\n  ✓ OCR completed for {len(ocr_texts)} pages")
    
    # Combine and clean
    full_text = "\n\n".join(ocr_texts)
    clean_text = clean_ocr_text(full_text)
    
    if not clean_text:
        print(f"  ⚠ No text extracted via OCR")
        return None
    
    # Create updated book dict
    updated_book = book.copy()
    updated_book["clean_text"] = clean_text
    updated_book["text_length"] = len(clean_text)
    updated_book["text_source"] = "ocr"
    updated_book["ocr_extracted"] = True
    
    print(f"  ✓ Extracted {len(clean_text)} characters (was {book.get('text_length', 0)})")
    
    return updated_book


def main():
    """Main function to re-extract all books with OCR."""
    print("=" * 80)
    print("OCR-based Book Re-extraction")
    print("=" * 80)
    
    # Check Tesseract
    if not check_tesseract():
        print("\n✗ Cannot proceed without Tesseract OCR and Arabic language pack")
        sys.exit(1)
    
    # Load books
    input_file = PROCESSED_DIR / "books_all_fixed.jsonl"
    if not input_file.exists():
        print(f"\n✗ Input file not found: {input_file}")
        sys.exit(1)
    
    print(f"\n[Loading books] {input_file}")
    books = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                books.append(json.loads(line))
    
    print(f"✓ Loaded {len(books)} books")
    
    # Process each book
    print(f"\n[Processing books with OCR]...")
    updated_books = []
    failed = []
    
    for i, book in enumerate(books, 1):
        print(f"\n[{i}/{len(books)}]")
        updated = extract_book_with_ocr(book)
        if updated:
            updated_books.append(updated)
        else:
            failed.append(book.get("doc_id", "unknown"))
            # Keep original if OCR fails
            updated_books.append(book)
    
    # Write output
    print(f"\n[Writing output] {OUTPUT_FILE}")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for book in updated_books:
            f.write(json.dumps(book, ensure_ascii=False) + "\n")
    
    print(f"✓ Wrote {len(updated_books)} books to {OUTPUT_FILE}")
    
    if failed:
        print(f"\n⚠ Failed to OCR {len(failed)} books: {', '.join(failed)}")
        print("  (Original text kept for failed books)")
    
    print("\n" + "=" * 80)
    print("Done! Next steps:")
    print(f"1. Review {OUTPUT_FILE}")
    print("2. Update kb_chunker.py to use 'books_all_ocr.jsonl' instead of 'books_all_fixed.jsonl'")
    print("3. Re-run chunker and re-build RAG indices")
    print("=" * 80)


if __name__ == "__main__":
    main()

