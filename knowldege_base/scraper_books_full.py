# scraper_books_full.py
# Scrape all books from ncmh.org.sa using ID range (BOOK_FIRST_ID..BOOK_LAST_ID)
#
# This reuses the robust parsing and PDF extraction logic from scraper_books_poc.py
# and extends it to iterate over all book IDs, with a simple checkpoint.

import os
import json
import time

from config import (
    BOOK_FIRST_ID,
    BOOK_LAST_ID,
    PROCESSED_DIR,
)
from scraper_books_poc import parse_book


def load_checkpoint(output_path: str):
    """Load already scraped book IDs from an existing JSONL file."""
    completed_ids = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    doc = json.loads(line)
                    doc_id = doc.get("doc_id", "").replace("ncmh_book_", "")
                    if doc_id.isdigit():
                        completed_ids.add(int(doc_id))
        except Exception:
            # If anything goes wrong, just start fresh
            completed_ids = set()
    return completed_ids


def main():
    print("\n" + "=" * 60)
    print("BOOK SCRAPER - FULL RANGE")
    print("=" * 60)
    print(f"\nScraping book IDs: {BOOK_FIRST_ID}..{BOOK_LAST_ID}")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    output_path = os.path.join(PROCESSED_DIR, "books_all.jsonl")

    # Load checkpoint from existing file (if any)
    completed_ids = load_checkpoint(output_path)
    if completed_ids:
        print(f"\n✓ Found checkpoint: {len(completed_ids)} already scraped book(s)")

    success_count = 0
    fail_count = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for book_id in range(BOOK_FIRST_ID, BOOK_LAST_ID + 1):
            if book_id in completed_ids:
                print(f"\n[Skipping book {book_id}] Already scraped")
                continue

            doc = parse_book(book_id)

            # parse_book handles 404 and errors, returns None when not found/failed
            if doc and doc.get("clean_text"):
                out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                out_f.flush()
                completed_ids.add(book_id)
                success_count += 1
            else:
                fail_count += 1
                print(f"  ✗ Skipping book {book_id} (no content extracted)")

            # Be polite, avoid hammering the server
            time.sleep(1)

    print(f"\n{'=' * 60}")
    print("SCRAPING COMPLETE")
    print(f"{'=' * 60}")
    print(f"✓ Successfully scraped: {success_count} book(s)")
    print(f"✗ Failed / empty / 404: {fail_count} book(s)")
    print(f"📄 Output saved to: {output_path}")


if __name__ == "__main__":
    main()


