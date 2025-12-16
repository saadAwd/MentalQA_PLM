# scraper_books_poc.py
# Proof of concept: Scrape one book (ID=1) as proof of concept

import os
import json
import time
import re
from urllib.parse import urljoin
from functools import wraps

import requests
from bs4 import BeautifulSoup

try:
    import pdfplumber
    PDFPLUMBER_SUPPORT = True
except ImportError:
    PDFPLUMBER_SUPPORT = False
    print("⚠ pdfplumber not installed. PDF extraction will be skipped.")
    print("   Install with: pip install pdfplumber")

try:
    import fitz  # PyMuPDF
    PYMUPDF_SUPPORT = True
except ImportError:
    PYMUPDF_SUPPORT = False
    print("⚠ PyMuPDF (fitz) not installed. Will use pdfplumber only.")
    print("   Install with: pip install pymupdf (better Arabic support)")

PDF_SUPPORT = PDFPLUMBER_SUPPORT or PYMUPDF_SUPPORT

from config import (
    BASE_URL,
    BOOK_DETAIL_URL,
    BOOK_POC_ID,
    BOOK_TITLE_SELECTOR,
    BOOK_BODY_SELECTOR,
    BOOK_DESCRIPTION_SELECTOR,
    BOOK_PDF_LINK_SELECTOR,
    RAW_BOOKS_DIR,
    PROCESSED_DIR,
    HEADERS,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
    RETRY_BACKOFF,
    REQUEST_DELAY,
)


# Create session for connection pooling
SESSION = requests.Session()
SESSION.headers.update(HEADERS)


def retry_on_failure(max_attempts=MAX_RETRIES, delay=RETRY_DELAY, backoff=RETRY_BACKOFF):
    """Decorator for retrying failed requests."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 404:
                        print(f"  ✗ 404 Not Found")
                        return None
                    elif e.response.status_code == 429:
                        wait_time = delay * (backoff ** attempts)
                        print(f"  ⚠ Rate limited, waiting {wait_time}s before retry {attempts + 1}/{max_attempts}...")
                        time.sleep(wait_time)
                        attempts += 1
                        continue
                    elif attempts < max_attempts - 1:
                        wait_time = delay * (backoff ** attempts)
                        print(f"  ⚠ HTTP {e.response.status_code}, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        attempts += 1
                        continue
                    raise
                except requests.exceptions.RequestException as e:
                    if attempts < max_attempts - 1:
                        wait_time = delay * (backoff ** attempts)
                        print(f"  ⚠ Request failed: {e}, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        attempts += 1
                        continue
                    raise
            return None
        return wrapper
    return decorator


@retry_on_failure()
def fetch_url(url):
    """Fetch a URL and return the HTML content."""
    response = SESSION.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


@retry_on_failure()
def download_file(url, save_path):
    """Download a file (e.g., PDF) from URL."""
    response = SESSION.get(url, timeout=REQUEST_TIMEOUT, stream=True)
    response.raise_for_status()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return save_path


def clean_cid_codes(text):
    """Remove CID codes only - minimal cleaning to preserve Arabic text."""
    if not text:
        return text
    
    # Only remove CID codes like (cid:123) or (cid:224)
    # Don't do aggressive cleaning that might corrupt Arabic text
    text = re.sub(r'\(cid:\d+\)', '', text)
    
    # Clean up multiple spaces that might result from CID removal
    text = re.sub(r'  +', ' ', text)  # Multiple spaces to single space
    
    # Clean up excessive newlines (more than 3 consecutive)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    
    return text


def extract_pdf_text(pdf_path):
    """Extract text from a PDF file using best available method."""
    if not PDF_SUPPORT:
        return None
    
    # Try PyMuPDF first (better Arabic support)
    if PYMUPDF_SUPPORT:
        try:
            print(f"  [Trying PyMuPDF extraction]...")
            doc = fitz.open(pdf_path)
            text_parts = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                if page_text:
                    text_parts.append(page_text)
            doc.close()
            
            if text_parts:
                text = "\n\n".join(text_parts)
                # Only remove CID codes, keep everything else as-is
                text = clean_cid_codes(text)
                print(f"  ✓ PyMuPDF extracted {len(text)} characters")
                return text
        except Exception as e:
            print(f"  ⚠ PyMuPDF extraction failed: {e}, trying pdfplumber...")
    
    # Fallback to pdfplumber
    if PDFPLUMBER_SUPPORT:
        try:
            print(f"  [Trying pdfplumber extraction]...")
            text_parts = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
            
            if text_parts:
                text = "\n\n".join(text_parts)
                # Clean CID codes
                text = clean_cid_codes(text)
                print(f"  ✓ pdfplumber extracted {len(text)} characters")
                return text
        except Exception as e:
            print(f"  ⚠ Error extracting PDF text with pdfplumber: {e}")
            return None
    
    return None


def parse_book(book_id):
    """Parse a book page and extract content."""
    url = BOOK_DETAIL_URL.format(book_id=book_id)
    print(f"\n[Parsing book] ID: {book_id}")
    print(f"  URL: {url}")
    
    try:
        html = fetch_url(url)
        if html is None:
            return None
        
        # Save raw HTML
        raw_path = os.path.join(RAW_BOOKS_DIR, f"book_{book_id}.html")
        os.makedirs(RAW_BOOKS_DIR, exist_ok=True)
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        soup = BeautifulSoup(html, "lxml")
        
        # Extract title
        title = None
        if BOOK_TITLE_SELECTOR:
            title_el = soup.select_one(BOOK_TITLE_SELECTOR)
            if title_el:
                title = title_el.get_text(strip=True)
        
        if not title:
            print(f"  ⚠ No title found with selector '{BOOK_TITLE_SELECTOR}'")
        
        # Extract body/description
        body_text = None
        
        # Try body selector first
        if BOOK_BODY_SELECTOR:
            body_el = soup.select_one(BOOK_BODY_SELECTOR)
            if body_el:
                for tag in body_el(["script", "style", "nav", "header", "footer"]):
                    tag.decompose()
                body_text = body_el.get_text("\n", strip=True)
        
        # If no body, try description selector
        if not body_text and BOOK_DESCRIPTION_SELECTOR:
            desc_el = soup.select_one(BOOK_DESCRIPTION_SELECTOR)
            if desc_el:
                for tag in desc_el(["script", "style", "nav", "header", "footer"]):
                    tag.decompose()
                body_text = desc_el.get_text("\n", strip=True)
        
        if not body_text:
            print(f"  ⚠ No body/description found with configured selectors")
        
        # Check for PDF download link
        pdf_url = None
        pdf_path = None
        pdf_text = None
        
        # First, try to extract PDF URL from JavaScript (books use PDF flipbook)
        script_tags = soup.find_all("script")
        for script in script_tags:
            script_text = script.string
            if script_text:
                # Look for: const pdfUrl = "..." or var pdfUrl = "..."
                match = re.search(r'(?:const|var)\s+pdfUrl\s*=\s*["\']([^"\']+\.pdf)["\']', script_text)
                if match:
                    pdf_url = match.group(1)
                    # Make sure it's absolute
                    if not pdf_url.startswith('http'):
                        pdf_url = urljoin(BASE_URL, pdf_url)
                    print(f"  ✓ Found PDF URL in JavaScript: {pdf_url}")
                    break
        
        # If not found in JS, try selector
        if not pdf_url and BOOK_PDF_LINK_SELECTOR:
            pdf_link_el = soup.select_one(BOOK_PDF_LINK_SELECTOR)
            if pdf_link_el:
                pdf_href = pdf_link_el.get("href")
                if pdf_href:
                    pdf_url = urljoin(BASE_URL, pdf_href)
                    print(f"  ✓ Found PDF link: {pdf_url}")
        
        # If still not found, search for any PDF links in HTML
        if not pdf_url:
            all_links = soup.find_all("a", href=True)
            for link in all_links:
                href = link.get("href", "")
                if ".pdf" in href.lower():
                    pdf_url = urljoin(BASE_URL, href)
                    print(f"  ✓ Found PDF link (general search): {pdf_url}")
                    break
        
        # Download and extract PDF if found
        if pdf_url and PDF_SUPPORT:
            try:
                pdf_path = os.path.join(RAW_BOOKS_DIR, f"book_{book_id}.pdf")
                print(f"  [Downloading PDF] {pdf_url}")
                download_file(pdf_url, pdf_path)
                print(f"  ✓ PDF downloaded to: {pdf_path}")
                
                print(f"  [Extracting PDF text]...")
                pdf_text = extract_pdf_text(pdf_path)
                if pdf_text:
                    print(f"  ✓ Extracted {len(pdf_text)} characters from PDF")
                else:
                    print(f"  ⚠ Could not extract text from PDF")
            except Exception as e:
                print(f"  ✗ Error downloading/processing PDF: {e}")
        
        # Use PDF text if available, otherwise use HTML body
        final_text = pdf_text if pdf_text else body_text
        
        # Create document
        doc = {
            "doc_id": f"ncmh_book_{book_id}",
            "content_type": "book",
            "source_site": "ncmh.org.sa",
            "url": url,
            "title": title,
            "language": "ar",
            "raw_html_path": raw_path,
            "pdf_url": pdf_url,
            "pdf_path": pdf_path,
            "clean_text": final_text,
            "text_length": len(final_text) if final_text else 0,
            "text_source": "pdf" if pdf_text else ("html" if body_text else None),
        }
        
        if final_text:
            print(f"  ✓ Extracted: title={bool(title)}, text_length={len(final_text)} chars")
        else:
            print(f"  ⚠ No text content extracted")
        
        return doc
    
    except Exception as e:
        print(f"  ✗ Error parsing book {book_id}: {e}")
        return None


def main():
    """Main scraping function for POC."""
    print("\n" + "="*60)
    print("BOOK SCRAPER - PROOF OF CONCEPT")
    print("="*60)
    print(f"\nScraping book ID: {BOOK_POC_ID}")
    print(f"Output: {os.path.join(PROCESSED_DIR, 'books_poc.jsonl')}")
    
    # Ensure directories exist
    os.makedirs(RAW_BOOKS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Check if already scraped
    output_path = os.path.join(PROCESSED_DIR, "books_poc.jsonl")
    completed_ids = set()
    
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        doc = json.loads(line)
                        book_id = doc.get("doc_id", "").replace("ncmh_book_", "")
                        if book_id.isdigit():
                            completed_ids.add(int(book_id))
        except Exception:
            pass
    
    if BOOK_POC_ID in completed_ids:
        print(f"\n✓ Book {BOOK_POC_ID} already scraped!")
        return
    
    # Scrape book
    doc = parse_book(BOOK_POC_ID)
    
    if doc and doc.get("clean_text"):
        with open(output_path, "a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        print(f"\n{'='*60}")
        print("SCRAPING COMPLETE")
        print(f"{'='*60}")
        print(f"✓ Successfully scraped book {BOOK_POC_ID}")
        print(f"📄 Output saved to: {output_path}")
    else:
        print(f"\n{'='*60}")
        print("SCRAPING COMPLETE")
        print(f"{'='*60}")
        print(f"✗ Could not extract content from book {BOOK_POC_ID}")
        print(f"  This might mean:")
        print(f"  - The book doesn't exist (404)")
        print(f"  - The selectors in config.py need to be updated")
        print(f"  - Run explore_structure.py to identify correct selectors")


if __name__ == "__main__":
    main()

