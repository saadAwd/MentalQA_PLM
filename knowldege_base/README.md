# Knowledge Base Scraper - Proof of Concept

This is a proof-of-concept scraper system for extracting articles and books from ncmh.org.sa to build a knowledge base. The POC focuses on scraping articles from page 11 and one book (ID=1) to validate the approach before scaling to the full dataset.

## Project Structure

```
Knowldege_Base/
├── config.py                 # Central configuration with CSS selectors
├── requirements.txt          # Python dependencies
├── explore_structure.py      # Tool to analyze HTML structure
├── scraper_articles_poc.py   # POC: Scrape page 11 articles
├── scraper_books_poc.py      # POC: Scrape one book
├── validate_output.py        # Validate scraped data quality
├── data/
│   ├── raw/
│   │   ├── articles/        # Raw HTML files
│   │   └── books/           # Raw HTML/PDF files
│   └── processed/           # JSONL output files
└── README.md                # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `requests` - HTTP requests
- `beautifulsoup4` - HTML parsing
- `lxml` - Fast XML/HTML parser
- `pdfplumber` - PDF text extraction (optional, for books with PDFs)

### 2. Directory Structure

The directories are created automatically when you run the scripts, but you can also create them manually:

```bash
mkdir -p data/raw/articles data/raw/books data/processed
```

## Usage Workflow

### Step 1: Explore Website Structure

Before scraping, you need to identify the correct CSS selectors for the website. Run the exploration tool:

```bash
python explore_structure.py
```

This will:
- Fetch page 11 article list and analyze its structure
- Fetch one article detail page and analyze its structure
- Fetch book page (ID=1) and analyze its structure
- Save raw HTML files to `data/raw/` for manual inspection
- Print analysis of potential selectors

**Output:**
- Raw HTML files saved for inspection
- Console output showing potential CSS selectors
- Statistics about links, headings, and content containers

### Step 2: Update Configuration

After running the exploration tool, inspect the saved HTML files and the console output. Then update the selectors in `config.py`:

```python
# Example updates based on findings:
ARTICLE_LINK_SELECTOR = "a.article-link"  # Update this
ARTICLE_TITLE_SELECTOR = "h1.article-title"  # Update this
ARTICLE_BODY_SELECTOR = "div.article-content"  # Update this
# ... etc
```

### Step 3: Scrape Articles (Page 11)

Run the article scraper:

```bash
python scraper_articles_poc.py
```

This will:
- Fetch article list from page 11
- Extract all article URLs
- Scrape each article (title, body, date, tags)
- Save to `data/processed/articles_poc.jsonl`
- Save raw HTML to `data/raw/articles/`
- Use checkpoint system to resume if interrupted

**Output:**
- `data/processed/articles_poc.jsonl` - Scraped articles in JSONL format
- `data/raw/articles/*.html` - Raw HTML files for each article

### Step 4: Scrape One Book

Run the book scraper:

```bash
python scraper_books_poc.py
```

This will:
- Fetch book page (ID=1)
- Extract title and description
- Detect and download PDF if available
- Extract text from PDF if present
- Save to `data/processed/books_poc.jsonl`

**Output:**
- `data/processed/books_poc.jsonl` - Scraped book in JSONL format
- `data/raw/books/book_1.html` - Raw HTML
- `data/raw/books/book_1.pdf` - PDF file (if available)

### Step 5: Validate Output

Check the quality of scraped data:

```bash
python validate_output.py
```

This will:
- Validate all articles and books
- Check for missing fields, empty content, duplicate URLs
- Generate statistics (text lengths, success rates)
- Print validation report

**Output:**
- Console report with validation results
- Statistics about data quality
- List of issues and warnings

## Configuration

Edit `config.py` to customize:

- **Base URL**: Website base URL
- **CSS Selectors**: Update after running `explore_structure.py`
- **Retry Settings**: Number of retries, delays
- **Rate Limiting**: Delay between requests
- **File Paths**: Where to save data

## Output Format

### Articles JSONL

Each line is a JSON object:

```json
{
  "doc_id": "ncmh_article_123",
  "content_type": "article",
  "source_site": "ncmh.org.sa",
  "url": "https://ncmh.org.sa/articles/123",
  "title": "Article Title",
  "language": "ar",
  "published_date_raw": "2024-01-01",
  "tags": ["tag1", "tag2"],
  "raw_html_path": "data/raw/articles/123.html",
  "clean_text": "Article content...",
  "text_length": 1500
}
```

### Books JSONL

Each line is a JSON object:

```json
{
  "doc_id": "ncmh_book_1",
  "content_type": "book",
  "source_site": "ncmh.org.sa",
  "url": "https://ncmh.org.sa/Book/1",
  "title": "Book Title",
  "language": "ar",
  "raw_html_path": "data/raw/books/book_1.html",
  "pdf_url": "https://ncmh.org.sa/books/book_1.pdf",
  "pdf_path": "data/raw/books/book_1.pdf",
  "clean_text": "Book content...",
  "text_length": 5000,
  "text_source": "pdf"
}
```

## Features

### Error Handling
- Automatic retry on network failures (3 attempts with exponential backoff)
- Graceful handling of 404 errors
- Checkpoint system to resume interrupted scrapes

### Rate Limiting
- Configurable delay between requests (default: 1 second)
- Respects rate limits (429 responses)

### Data Quality
- Validation script checks for:
  - Missing required fields
  - Empty or very short content
  - Duplicate URLs
  - Encoding issues
  - Content quality metrics

## Troubleshooting

### No articles/links found
- Run `explore_structure.py` to identify correct selectors
- Check the saved HTML files in `data/raw/`
- Update `ARTICLE_LINK_SELECTOR` in `config.py`

### Empty content extracted
- Check if selectors are correct in `config.py`
- Inspect raw HTML files to verify structure
- Website structure might have changed

### 404 errors
- Some articles/books might not exist
- Check the URL pattern in `config.py`
- Verify the base URL is correct

### PDF extraction fails
- Install `pdfplumber`: `pip install pdfplumber`
- Some PDFs might be image-based (OCR needed)
- Check if PDF link selector is correct

## Next Steps

After validating the POC:

1. **Update selectors** based on findings
2. **Extend to full scraping**:
   - All article pages (1-19)
   - All books (1-20)
3. **Add text processing**:
   - Clean and normalize Arabic text
   - Chunk text for embeddings
4. **Generate embeddings** for RAG system
5. **Store in database** (Postgres + pgvector, FAISS, etc.)

## Notes

- The scraper includes a User-Agent header identifying it as a research/educational bot
- Raw HTML is saved for debugging and reprocessing
- Checkpoint files (`.articles_checkpoint.json`) allow resuming interrupted scrapes
- All scripts include progress logging and error messages

## License

This is a proof-of-concept tool for educational/research purposes. Ensure you have permission to scrape the target website and comply with their terms of service and robots.txt.

