# scraper_articles_full.py
# Scrape all article pages (1..N) from ncmh.org.sa
#
# This builds on the proof-of-concept scraper and extends it to all pages.
# It reuses the same checkpoint file as the POC, so already-scraped
# articles (from page 11) will be skipped automatically.

import os
import json
import time
from urllib.parse import urljoin
from functools import wraps

import requests
from bs4 import BeautifulSoup

from config import (
    BASE_URL,
    ARTICLE_LIST_URL,
    ARTICLE_LIST_FIRST_PAGE,
    ARTICLE_LIST_LAST_PAGE,
    ARTICLE_LINK_SELECTOR,
    ARTICLE_LINK_HREF_ATTR,
    ARTICLE_TITLE_SELECTOR,
    ARTICLE_BODY_SELECTOR,
    ARTICLE_DATE_SELECTOR,
    ARTICLE_TAG_SELECTOR,
    RAW_ARTICLES_DIR,
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
                    if e.response.status_code == 429:
                        wait_time = delay * (backoff ** attempts)
                        print(
                            f"  ⚠ Rate limited, waiting {wait_time}s before retry {attempts + 1}/{max_attempts}..."
                        )
                        time.sleep(wait_time)
                        attempts += 1
                        continue
                    elif e.response.status_code == 404:
                        print("  [ERROR] 404 Not Found")
                        return None
                    elif attempts < max_attempts - 1:
                        wait_time = delay * (backoff ** attempts)
                        print(
                            f"  ⚠ HTTP {e.response.status_code}, retrying in {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        attempts += 1
                        continue
                    raise
                except requests.exceptions.RequestException as e:
                    if attempts < max_attempts - 1:
                        wait_time = delay * (backoff ** attempts)
                        print(
                            f"  ⚠ Request failed: {e}, retrying in {wait_time}s..."
                        )
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


def get_article_links(page_num):
    """Extract article URLs from a list page."""
    url = ARTICLE_LIST_URL.format(page=page_num)
    print(f"\n[Fetching article list] Page {page_num}: {url}")

    try:
        html = fetch_url(url)
        if html is None:
            return []

        soup = BeautifulSoup(html, "lxml")
        links_set = set()  # Use set to avoid duplicates

        # Find article links using configured selector
        link_elements = soup.select(ARTICLE_LINK_SELECTOR)

        if not link_elements:
            print(
                f"  ⚠ No links found with selector '{ARTICLE_LINK_SELECTOR}' on page {page_num}"
            )
            return []

        for link_el in link_elements:
            href = link_el.get(ARTICLE_LINK_HREF_ATTR)
            if not href:
                continue

            # Clean and normalize URL (remove trailing spaces)
            href = href.strip()

            # Skip if this is a nested link (inside article-title paragraph)
            # The outer link wraps the card, inner links are for "اقرأ المزيد"
            parent = link_el.parent
            if parent and parent.name == "p" and "article-title" in parent.get(
                "class", []
            ):
                continue

            # Convert relative URLs to absolute
            full_url = urljoin(BASE_URL, href)
            # Remove trailing spaces from final URL
            full_url = full_url.rstrip()
            links_set.add(full_url)

        links = sorted(list(links_set))  # Convert to sorted list
        print(f"  [OK] Found {len(links)} unique article link(s) on page {page_num}")
        return links

    except Exception as e:
        print(f"  [ERROR] Error fetching article list page {page_num}: {e}")
        return []


def parse_article(url):
    """Parse a single article page and extract content."""
    print(f"\n[Parsing article] {url}")

    try:
        html = fetch_url(url)
        if html is None:
            return None

        # Save raw HTML
        article_id = url.rstrip("/").split("/")[-1]
        # Clean article_id to be filesystem-safe
        article_id = "".join(c for c in article_id if c.isalnum() or c in ("-", "_"))[
            :50
        ]
        raw_path = os.path.join(RAW_ARTICLES_DIR, f"{article_id}.html")

        os.makedirs(RAW_ARTICLES_DIR, exist_ok=True)
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(html)

        soup = BeautifulSoup(html, "lxml")

        # Extract title
        title = None
        if ARTICLE_TITLE_SELECTOR:
            title_el = soup.select_one(ARTICLE_TITLE_SELECTOR)
            if title_el:
                title = title_el.get_text(strip=True)

        if not title:
            print(f"  ⚠ No title found with selector '{ARTICLE_TITLE_SELECTOR}'")

        # Extract body/content
        body_text = None
        if ARTICLE_BODY_SELECTOR:
            body_el = soup.select_one(ARTICLE_BODY_SELECTOR)
            if body_el:
                # Remove script and style tags
                for tag in body_el(["script", "style", "nav", "header", "footer"]):
                    tag.decompose()
                body_text = body_el.get_text("\n", strip=True)

        if not body_text:
            print(
                f"  ⚠ No body content found with selector '{ARTICLE_BODY_SELECTOR}'"
            )

        # Extract date (optional)
        date_text = None
        if ARTICLE_DATE_SELECTOR:
            # Get all matching elements and find the one with date-like content
            date_elements = soup.select(ARTICLE_DATE_SELECTOR)
            for date_el in date_elements:
                text = date_el.get_text(strip=True)
                # Check if it looks like a date (contains numbers and dashes or colons)
                if any(char.isdigit() for char in text) and (
                    "-" in text or ":" in text
                ):
                    date_text = text
                    break
            # Also check for datetime attribute on any date element
            if not date_text:
                for date_el in date_elements:
                    datetime_attr = date_el.get("datetime")
                    if datetime_attr:
                        date_text = datetime_attr
                        break

        # Extract tags (optional)
        tags = []
        if ARTICLE_TAG_SELECTOR:
            tag_elements = soup.select(ARTICLE_TAG_SELECTOR)
            tags = [
                tag.get_text(strip=True)
                for tag in tag_elements
                if tag.get_text(strip=True)
            ]

        # Create document
        doc = {
            "doc_id": f"ncmh_article_{article_id}",
            "content_type": "article",
            "source_site": "ncmh.org.sa",
            "url": url,
            "title": title,
            "language": "ar",
            "published_date_raw": date_text,
            "tags": tags,
            "raw_html_path": raw_path,
            "clean_text": body_text,
            "text_length": len(body_text) if body_text else 0,
        }

        if body_text:
            print(
                f"  [OK] Extracted: title={bool(title)}, text_length={len(body_text)} chars"
            )
        else:
            print("  ⚠ No text content extracted")

        return doc

    except Exception as e:
        print(f"  [ERROR] Error parsing article {url}: {e}")
        return None


def load_checkpoint():
    """Load already scraped URLs from checkpoint file."""
    checkpoint_file = os.path.join(PROCESSED_DIR, ".articles_checkpoint.json")
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return set(data.get("completed_urls", []))
        except Exception:
            return set()
    return set()


def save_checkpoint(completed_urls):
    """Save checkpoint of completed URLs."""
    checkpoint_file = os.path.join(PROCESSED_DIR, ".articles_checkpoint.json")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump({"completed_urls": list(completed_urls)}, f, ensure_ascii=False)


def main():
    """Main scraping function for all article pages."""
    print("\n" + "=" * 60)
    print("ARTICLE SCRAPER - FULL RANGE")
    print("=" * 60)
    print(
        f"\nScraping articles from pages {ARTICLE_LIST_FIRST_PAGE} to {ARTICLE_LIST_LAST_PAGE}"
    )
    print(f"Output: {os.path.join(PROCESSED_DIR, 'articles_all.jsonl')}")

    # Ensure directories exist
    os.makedirs(RAW_ARTICLES_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Load checkpoint (shared with POC scraper)
    completed_urls = load_checkpoint()
    if completed_urls:
        print(f"\n[OK] Found checkpoint: {len(completed_urls)} already scraped article(s)")

    # Collect article URLs from all pages
    all_article_urls = set()
    for page in range(ARTICLE_LIST_FIRST_PAGE, ARTICLE_LIST_LAST_PAGE + 1):
        links = get_article_links(page)
        all_article_urls.update(links)
        time.sleep(REQUEST_DELAY)

    if not all_article_urls:
        print("\n[ERROR] No article URLs found on any page.")
        return

    print(f"\nTotal unique article URLs across all pages: {len(all_article_urls)}")

    # Filter out already completed URLs
    new_urls = [url for url in sorted(all_article_urls) if url not in completed_urls]
    if new_urls:
        print(f"\nScraping {len(new_urls)} new article(s)...")
    else:
        print("\n[OK] All articles already scraped according to checkpoint!")
        return

    # Scrape articles
    output_path = os.path.join(PROCESSED_DIR, "articles_all.jsonl")
    success_count = 0
    fail_count = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for url in new_urls:
            doc = parse_article(url)

            if doc and doc.get("clean_text"):
                out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                out_f.flush()
                completed_urls.add(url)
                success_count += 1
            else:
                fail_count += 1
                print("  [ERROR] Skipping article (no content extracted)")

            # Rate limiting
            time.sleep(REQUEST_DELAY)

    # Save checkpoint
    save_checkpoint(completed_urls)

    # Summary
    print(f"\n{'=' * 60}")
    print("SCRAPING COMPLETE")
    print(f"{'=' * 60}")
    print(f"[OK] Successfully scraped: {success_count} article(s)")
    print(f"[ERROR] Failed: {fail_count} article(s)")
    print(f"📄 Output saved to: {output_path}")
    print(f"💾 Checkpoint saved: {len(completed_urls)} total article(s)")


if __name__ == "__main__":
    main()


