# explore_structure.py
# Tool to analyze HTML structure and help identify CSS selectors

import os
import sys
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from config import (
    BASE_URL,
    ARTICLE_LIST_URL,
    ARTICLE_LIST_POC_PAGE,
    BOOK_DETAIL_URL,
    BOOK_POC_ID,
    RAW_ARTICLES_DIR,
    RAW_BOOKS_DIR,
    HEADERS,
    REQUEST_TIMEOUT,
)


def fetch_url(url, description):
    """Fetch a URL and return the HTML content."""
    print(f"\n{'='*60}")
    print(f"Fetching: {description}")
    print(f"URL: {url}")
    print(f"{'='*60}")
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        print(f"✓ Successfully fetched (Status: {response.status_code})")
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching URL: {e}")
        return None


def analyze_article_list_page(html, page_num):
    """Analyze the article list page structure."""
    print(f"\n{'='*60}")
    print(f"ANALYZING ARTICLE LIST PAGE {page_num}")
    print(f"{'='*60}")
    
    soup = BeautifulSoup(html, "lxml")
    
    # Save raw HTML for manual inspection
    os.makedirs(RAW_ARTICLES_DIR, exist_ok=True)
    raw_file = os.path.join(RAW_ARTICLES_DIR, f"article_list_page_{page_num}.html")
    with open(raw_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✓ Saved raw HTML to: {raw_file}")
    
    # Find all <a> tags (potential article links)
    print(f"\n--- All <a> tags found ---")
    links = soup.find_all("a", href=True)
    print(f"Total <a> tags with href: {len(links)}")
    
    # Group by common patterns
    article_candidates = []
    for link in links[:50]:  # Show first 50
        href = link.get("href", "")
        text = link.get_text(strip=True)
        classes = link.get("class", [])
        link_id = link.get("id", "")
        
        # Check if it might be an article link
        if any(keyword in href.lower() for keyword in ["article", "artical", "post", "news"]):
            article_candidates.append({
                "href": href,
                "text": text[:50],
                "classes": classes,
                "id": link_id,
                "full_url": urljoin(BASE_URL, href)
            })
    
    print(f"\n--- Potential article links (first 10) ---")
    for i, candidate in enumerate(article_candidates[:10], 1):
        print(f"\n{i}. Href: {candidate['href']}")
        print(f"   Text: {candidate['text']}")
        print(f"   Classes: {candidate['classes']}")
        print(f"   ID: {candidate['id']}")
        print(f"   Full URL: {candidate['full_url']}")
    
    # Find common container patterns
    print(f"\n--- Common container divs ---")
    containers = soup.find_all("div", class_=True)
    class_counts = {}
    for div in containers:
        classes = div.get("class", [])
        for cls in classes:
            class_counts[cls] = class_counts.get(cls, 0) + 1
    
    print("Most common div classes:")
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  .{cls}: {count} occurrences")
    
    # Find heading tags
    print(f"\n--- Heading tags found ---")
    for level in range(1, 7):
        headings = soup.find_all(f"h{level}")
        if headings:
            print(f"\n<h{level}> tags ({len(headings)} found):")
            for h in headings[:5]:
                text = h.get_text(strip=True)[:60]
                classes = h.get("class", [])
                print(f"  - {text} (classes: {classes})")
    
    return article_candidates


def analyze_article_detail_page(html, url):
    """Analyze a single article detail page structure."""
    print(f"\n{'='*60}")
    print(f"ANALYZING ARTICLE DETAIL PAGE")
    print(f"URL: {url}")
    print(f"{'='*60}")
    
    soup = BeautifulSoup(html, "lxml")
    
    # Extract article ID from URL for filename
    article_id = url.rstrip("/").split("/")[-1]
    raw_file = os.path.join(RAW_ARTICLES_DIR, f"article_detail_{article_id}.html")
    with open(raw_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✓ Saved raw HTML to: {raw_file}")
    
    # Find title candidates
    print(f"\n--- Title candidates ---")
    for selector in ["h1", "h2", ".title", ".article-title", "[class*='title']"]:
        elements = soup.select(selector)
        if elements:
            print(f"\nSelector '{selector}' found {len(elements)} element(s):")
            for el in elements[:3]:
                text = el.get_text(strip=True)[:80]
                print(f"  - {text}")
    
    # Find body/content candidates
    print(f"\n--- Content/body candidates ---")
    content_selectors = [
        ".article-body", ".article-content", ".content", ".post-content",
        "article", "main", "[class*='article']", "[class*='content']"
    ]
    
    for selector in content_selectors:
        elements = soup.select(selector)
        if elements:
            for el in elements[:2]:
                text = el.get_text(strip=True)
                text_preview = text[:100].replace("\n", " ")
                print(f"\nSelector '{selector}':")
                print(f"  Text length: {len(text)} chars")
                print(f"  Preview: {text_preview}...")
    
    # Find date candidates
    print(f"\n--- Date candidates ---")
    date_selectors = [
        ".date", ".published", ".article-date", "[class*='date']",
        "time", "[datetime]"
    ]
    for selector in date_selectors:
        elements = soup.select(selector)
        if elements:
            print(f"\nSelector '{selector}' found:")
            for el in elements[:3]:
                text = el.get_text(strip=True)
                datetime_attr = el.get("datetime", "")
                print(f"  - Text: {text}")
                if datetime_attr:
                    print(f"    datetime attr: {datetime_attr}")


def analyze_book_page(html, book_id):
    """Analyze a book detail page structure."""
    print(f"\n{'='*60}")
    print(f"ANALYZING BOOK PAGE (ID: {book_id})")
    print(f"{'='*60}")
    
    soup = BeautifulSoup(html, "lxml")
    
    # Save raw HTML
    raw_file = os.path.join(RAW_BOOKS_DIR, f"book_{book_id}.html")
    os.makedirs(RAW_BOOKS_DIR, exist_ok=True)
    with open(raw_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✓ Saved raw HTML to: {raw_file}")
    
    # Find title
    print(f"\n--- Book title candidates ---")
    for selector in ["h1", "h2", ".book-title", ".title", "[class*='book']"]:
        elements = soup.select(selector)
        if elements:
            print(f"\nSelector '{selector}':")
            for el in elements[:2]:
                print(f"  - {el.get_text(strip=True)[:80]}")
    
    # Find description/content
    print(f"\n--- Book description/content candidates ---")
    content_selectors = [
        ".book-description", ".book-body", ".description", ".content",
        "[class*='book']", "[class*='description']"
    ]
    
    for selector in content_selectors:
        elements = soup.select(selector)
        if elements:
            for el in elements[:2]:
                text = el.get_text(strip=True)
                print(f"\nSelector '{selector}':")
                print(f"  Text length: {len(text)} chars")
                print(f"  Preview: {text[:100].replace(chr(10), ' ')}...")
    
    # Find PDF links
    print(f"\n--- PDF download links ---")
    pdf_links = soup.find_all("a", href=True)
    pdf_candidates = [a for a in pdf_links if ".pdf" in a.get("href", "").lower()]
    
    if pdf_candidates:
        print(f"Found {len(pdf_candidates)} potential PDF link(s):")
        for link in pdf_candidates[:5]:
            href = link.get("href")
            text = link.get_text(strip=True)
            print(f"  - {href}")
            print(f"    Link text: {text}")
    else:
        print("No PDF links found")


def main():
    """Main exploration function."""
    print("\n" + "="*60)
    print("KNOWLEDGE BASE STRUCTURE EXPLORATION TOOL")
    print("="*60)
    print("\nThis tool will help you identify the correct CSS selectors")
    print("by analyzing the HTML structure of the website.")
    print("\nAfter running this, inspect the saved HTML files and")
    print("update the selectors in config.py accordingly.")
    
    # Explore article list page
    article_list_url = ARTICLE_LIST_URL.format(page=ARTICLE_LIST_POC_PAGE)
    html = fetch_url(article_list_url, f"Article List Page {ARTICLE_LIST_POC_PAGE}")
    
    if html:
        article_candidates = analyze_article_list_page(html, ARTICLE_LIST_POC_PAGE)
        
        # If we found potential article links, try to analyze one
        if article_candidates:
            print(f"\n{'='*60}")
            print("Attempting to analyze first article detail page...")
            print(f"{'='*60}")
            
            first_article_url = article_candidates[0]["full_url"]
            article_html = fetch_url(first_article_url, "First Article Detail Page")
            
            if article_html:
                analyze_article_detail_page(article_html, first_article_url)
    
    # Explore book page
    book_url = BOOK_DETAIL_URL.format(book_id=BOOK_POC_ID)
    book_html = fetch_url(book_url, f"Book Page (ID: {BOOK_POC_ID})")
    
    if book_html:
        analyze_book_page(book_html, BOOK_POC_ID)
    else:
        print(f"\n⚠ Could not fetch book page. It might not exist (404) or")
        print(f"  you might need to try a different book ID.")
    
    print(f"\n{'='*60}")
    print("EXPLORATION COMPLETE")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. Inspect the saved HTML files in data/raw/")
    print("2. Identify the correct CSS selectors based on the analysis above")
    print("3. Update the selectors in config.py")
    print("4. Run the POC scrapers to test")


if __name__ == "__main__":
    main()

