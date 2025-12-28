# OCR Setup Guide for Book Re-extraction

This guide explains how to set up OCR (Optical Character Recognition) to re-extract clean Arabic text from book PDFs.

## Why OCR?

The original PDF extraction (`pdfplumber`/`PyMuPDF`) produces corrupted text with misordered letters and broken words because the PDFs use complex font encodings. OCR reads the PDF as images and extracts text cleanly.

## Prerequisites

### 1. Install Tesseract OCR

**Windows:**
1. Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer (e.g., `tesseract-ocr-w64-setup-5.x.x.exe`)
3. **Important:** During installation, check "Arabic" language pack
4. Note the installation path (usually `C:\Program Files\Tesseract-OCR`)

**Linux/Mac:**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-ara

# macOS
brew install tesseract tesseract-lang
```

### 2. Verify Installation

```bash
tesseract --version
tesseract --list-langs
```

You should see `ara` in the language list.

### 3. Set Tesseract Path (Windows only)

If Python can't find Tesseract, set the path in your script or environment:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

Or set environment variable:
```powershell
$env:TESSDATA_PREFIX = "C:\Program Files\Tesseract-OCR\tessdata"
```

## Usage

Once Tesseract is installed:

```bash
cd C:\Users\PCD\Documents\Researches\Code\Shifaa\mentalqa-mlc
.\.venv\Scripts\Activate.ps1
python -m knowldege_base.rag_staging.ocr_re_extract_books
```

The script will:
1. Check Tesseract installation
2. Load books from `books_all_fixed.jsonl`
3. Download PDFs if needed (from `pdf_url`)
4. Convert PDF pages to images
5. Run OCR with Arabic language support
6. Output clean text to `books_all_ocr.jsonl`

## Output

- **Input:** `knowldege_base/data/processed/books_all_fixed.jsonl`
- **Output:** `knowldege_base/data/processed/books_all_ocr.jsonl`

Each book will have:
- `clean_text`: OCR-extracted Arabic text (clean, properly ordered)
- `text_source`: `"ocr"`
- `ocr_extracted`: `true`

## After OCR Extraction

1. **Review the output:**
   ```bash
   # Check a sample
   python -m knowldege_base.rag_staging.inspect_kb
   ```

2. **Update chunker to use OCR books:**
   Edit `knowldege_base/rag_staging/kb_chunker.py`:
   ```python
   books = loader.load_books(filename="books_all_ocr.jsonl")
   ```

3. **Re-chunk:**
   ```bash
   python -m knowldege_base.rag_staging.kb_chunker
   ```

4. **Re-build RAG indices:**
   ```bash
   python -m knowldege_base.rag_staging.hybrid_retriever
   ```

## Troubleshooting

### "Tesseract not found"
- Install Tesseract (see above)
- On Windows, set `pytesseract.pytesseract.tesseract_cmd` to full path

### "Arabic language pack not found"
- Download `ara.traineddata` from: https://github.com/tesseract-ocr/tessdata
- Place in Tesseract `tessdata` directory (usually `C:\Program Files\Tesseract-OCR\tessdata`)

### OCR quality is poor
- Ensure PDFs are high quality (not scanned images)
- Try adjusting DPI in `ocr_re_extract_books.py` (currently 300 DPI)
- Some PDFs may need manual review/correction

### Slow processing
- OCR is CPU-intensive
- Processing 18 books may take 10-30 minutes depending on PDF size
- Consider processing in batches if needed








