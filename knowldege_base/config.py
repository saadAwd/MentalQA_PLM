# config.py
# Central configuration for knowledge base scraper

import os

# Base URL
BASE_URL = "https://ncmh.org.sa"

# ARTICLE PAGES
ARTICLE_LIST_URL = BASE_URL + "/articles?page={page}"
ARTICLE_LIST_POC_PAGE = 11  # Proof of concept: only page 11

# Full range of article list pages (verified manually)
# The site currently has pages 1 through 19 under /articles?page={page}
ARTICLE_LIST_FIRST_PAGE = 1
ARTICLE_LIST_LAST_PAGE = 19

# CSS SELECTORS (updated based on HTML structure analysis)
# Article list page: links are inside div.articles-list, pointing to /articals/{id}
ARTICLE_LINK_SELECTOR = "div.articles-list a[href*='/articals/']"  # Links to article detail pages
ARTICLE_LINK_HREF_ATTR = "href"

# Article detail page selectors (verified with actual article page)
ARTICLE_TITLE_SELECTOR = "h1.content-title"  # Main article title
ARTICLE_BODY_SELECTOR = "div.article"  # Article content container
ARTICLE_DATE_SELECTOR = "div.content-info span"  # Date in content-info div (second span contains date)
ARTICLE_TAG_SELECTOR = None  # Tags not found in article pages

# BOOKS
BOOK_DETAIL_URL = BASE_URL + "/Book/{book_id}"
BOOK_POC_ID = 1  # Proof of concept: only book ID 1

# Full range of book IDs (based on site structure /Book/{id})
# Some IDs may return 404 and will be skipped gracefully
BOOK_FIRST_ID = 1
BOOK_LAST_ID = 20

# Book page selectors (verified with actual book page)
BOOK_TITLE_SELECTOR = "h1.text-center"  # Book title
BOOK_BODY_SELECTOR = None  # Books are PDF-only, no HTML body
BOOK_DESCRIPTION_SELECTOR = None  # No description in HTML
BOOK_PDF_LINK_SELECTOR = None  # PDF URL is in JavaScript, extracted separately

# Directory paths
DATA_DIR = "data"
RAW_ARTICLES_DIR = os.path.join(DATA_DIR, "raw", "articles")
RAW_BOOKS_DIR = os.path.join(DATA_DIR, "raw", "books")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Request settings
HEADERS = {
    "User-Agent": "MentalHealthKBBot/0.1 (Educational/Research Purpose)"
}

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
RETRY_BACKOFF = 2  # multiplier

# Request timeout
REQUEST_TIMEOUT = 20  # seconds

# Rate limiting
REQUEST_DELAY = 1  # seconds between requests

