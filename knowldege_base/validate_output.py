# validate_output.py
# Validate scraped data quality and generate validation report

import os
import json
import sys
from collections import defaultdict
from urllib.parse import urlparse

from config import PROCESSED_DIR


def iter_jsonl(path):
    """Iterate over JSONL file, yielding each JSON object."""
    if not os.path.exists(path):
        return
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                yield json.loads(line), line_num
            except json.JSONDecodeError as e:
                print(f"  ⚠ Line {line_num}: Invalid JSON - {e}")


def validate_article(doc, line_num):
    """Validate a single article document."""
    issues = []
    warnings = []
    
    # Required fields
    required_fields = ["doc_id", "content_type", "url", "title", "clean_text"]
    for field in required_fields:
        if field not in doc or not doc[field]:
            issues.append(f"Missing or empty required field: {field}")
    
    # Check title
    if doc.get("title"):
        title_len = len(doc["title"])
        if title_len < 3:
            warnings.append(f"Title very short ({title_len} chars)")
        elif title_len > 200:
            warnings.append(f"Title very long ({title_len} chars)")
    
    # Check text content
    text = doc.get("clean_text", "")
    text_len = len(text) if text else 0
    
    if text_len == 0:
        issues.append("No text content (clean_text is empty)")
    elif text_len < 50:
        warnings.append(f"Text content very short ({text_len} chars)")
    elif text_len < 100:
        warnings.append(f"Text content short ({text_len} chars)")
    
    # Check URL format
    url = doc.get("url", "")
    if url:
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                issues.append(f"Invalid URL format: {url}")
        except Exception:
            issues.append(f"URL parsing error: {url}")
    
    # Check encoding (look for common encoding issues)
    if text:
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            issues.append("Text contains invalid UTF-8 characters")
    
    # Check for suspiciously short or repetitive content
    if text_len > 0:
        words = text.split()
        if len(words) < 10:
            warnings.append(f"Very few words ({len(words)} words)")
        
        # Check for excessive repetition
        if len(set(words)) < len(words) * 0.3 and len(words) > 20:
            warnings.append("Possible repetitive content detected")
    
    return issues, warnings


def validate_book(doc, line_num):
    """Validate a single book document."""
    issues = []
    warnings = []
    
    # Required fields
    required_fields = ["doc_id", "content_type", "url", "title"]
    for field in required_fields:
        if field not in doc or not doc[field]:
            issues.append(f"Missing or empty required field: {field}")
    
    # clean_text is optional for books (might be PDF only)
    text = doc.get("clean_text", "")
    text_len = len(text) if text else 0
    
    if text_len == 0:
        if not doc.get("pdf_path"):
            warnings.append("No text content and no PDF path")
        else:
            warnings.append("No extracted text (PDF might need manual processing)")
    elif text_len < 50:
        warnings.append(f"Text content very short ({text_len} chars)")
    
    # Check title
    if doc.get("title"):
        title_len = len(doc["title"])
        if title_len < 3:
            warnings.append(f"Title very short ({title_len} chars)")
    
    # Check URL format
    url = doc.get("url", "")
    if url:
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                issues.append(f"Invalid URL format: {url}")
        except Exception:
            issues.append(f"URL parsing error: {url}")
    
    return issues, warnings


def validate_file(file_path, content_type):
    """Validate all documents in a JSONL file."""
    print(f"\n{'='*60}")
    print(f"Validating: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    
    if not os.path.exists(file_path):
        print(f"  ✗ File not found: {file_path}")
        return None
    
    stats = {
        "total": 0,
        "valid": 0,
        "with_issues": 0,
        "with_warnings": 0,
        "issues": [],
        "warnings": [],
        "urls": set(),
        "duplicate_urls": [],
        "text_lengths": [],
        "title_lengths": [],
    }
    
    validate_func = validate_article if content_type == "article" else validate_book
    
    for doc, line_num in iter_jsonl(file_path):
        stats["total"] += 1
        
        # Check for duplicate URLs
        url = doc.get("url", "")
        if url:
            if url in stats["urls"]:
                stats["duplicate_urls"].append((url, line_num))
            else:
                stats["urls"].add(url)
        
        # Validate document
        issues, warnings = validate_func(doc, line_num)
        
        if issues:
            stats["with_issues"] += 1
            for issue in issues:
                stats["issues"].append({
                    "line": line_num,
                    "doc_id": doc.get("doc_id", "unknown"),
                    "issue": issue
                })
        else:
            stats["valid"] += 1
        
        if warnings:
            stats["with_warnings"] += 1
            for warning in warnings:
                stats["warnings"].append({
                    "line": line_num,
                    "doc_id": doc.get("doc_id", "unknown"),
                    "warning": warning
                })
        
        # Collect statistics
        text_len = doc.get("text_length") or len(doc.get("clean_text", ""))
        if text_len > 0:
            stats["text_lengths"].append(text_len)
        
        title = doc.get("title", "")
        if title:
            stats["title_lengths"].append(len(title))
    
    return stats


def print_statistics(stats, content_type):
    """Print validation statistics."""
    if stats is None:
        return
    
    print(f"\n--- Statistics ---")
    print(f"Total documents: {stats['total']}")
    print(f"Valid documents: {stats['valid']} ({stats['valid']/max(stats['total'],1)*100:.1f}%)")
    print(f"Documents with issues: {stats['with_issues']}")
    print(f"Documents with warnings: {stats['with_warnings']}")
    
    if stats['duplicate_urls']:
        print(f"\n⚠ Duplicate URLs found: {len(stats['duplicate_urls'])}")
        for url, line_num in stats['duplicate_urls'][:5]:
            print(f"  - Line {line_num}: {url}")
        if len(stats['duplicate_urls']) > 5:
            print(f"  ... and {len(stats['duplicate_urls']) - 5} more")
    
    if stats['text_lengths']:
        avg_len = sum(stats['text_lengths']) / len(stats['text_lengths'])
        min_len = min(stats['text_lengths'])
        max_len = max(stats['text_lengths'])
        print(f"\n--- Text Length Statistics ---")
        print(f"Average: {avg_len:.0f} characters")
        print(f"Min: {min_len} characters")
        print(f"Max: {max_len} characters")
    
    if stats['title_lengths']:
        avg_title = sum(stats['title_lengths']) / len(stats['title_lengths'])
        print(f"\n--- Title Length Statistics ---")
        print(f"Average: {avg_title:.1f} characters")
    
    if stats['issues']:
        print(f"\n--- Issues ({len(stats['issues'])} total) ---")
        for issue_info in stats['issues'][:10]:
            print(f"  Line {issue_info['line']} ({issue_info['doc_id']}): {issue_info['issue']}")
        if len(stats['issues']) > 10:
            print(f"  ... and {len(stats['issues']) - 10} more issues")
    
    if stats['warnings']:
        print(f"\n--- Warnings ({len(stats['warnings'])} total) ---")
        # Group warnings by type
        warning_types = defaultdict(int)
        for warning_info in stats['warnings']:
            warning_types[warning_info['warning']] += 1
        
        for warning, count in sorted(warning_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {warning}: {count} occurrence(s)")
        if len(warning_types) > 10:
            print(f"  ... and {len(warning_types) - 10} more warning types")


def generate_report(articles_stats, books_stats):
    """Generate a summary validation report."""
    print(f"\n{'='*60}")
    print("VALIDATION REPORT SUMMARY")
    print(f"{'='*60}")
    
    total_docs = 0
    total_valid = 0
    total_issues = 0
    
    if articles_stats:
        total_docs += articles_stats['total']
        total_valid += articles_stats['valid']
        total_issues += len(articles_stats['issues'])
    
    if books_stats:
        total_docs += books_stats['total']
        total_valid += books_stats['valid']
        total_issues += len(books_stats['issues'])
    
    print(f"\nOverall Statistics:")
    print(f"  Total documents: {total_docs}")
    print(f"  Valid documents: {total_valid} ({total_valid/max(total_docs,1)*100:.1f}%)")
    print(f"  Total issues: {total_issues}")
    
    if total_issues == 0 and total_docs > 0:
        print(f"\n✓ All documents passed validation!")
    elif total_issues > 0:
        print(f"\n⚠ Some documents have issues that need attention.")
    
    return total_issues == 0


def main():
    """Main validation function."""
    print("\n" + "="*60)
    print("DATA VALIDATION TOOL")
    print("="*60)
    
    articles_path = os.path.join(PROCESSED_DIR, "articles_poc.jsonl")
    books_path = os.path.join(PROCESSED_DIR, "books_poc.jsonl")
    
    articles_stats = None
    books_stats = None
    
    # Validate articles
    if os.path.exists(articles_path):
        articles_stats = validate_file(articles_path, "article")
        print_statistics(articles_stats, "article")
    else:
        print(f"\n⚠ Articles file not found: {articles_path}")
    
    # Validate books
    if os.path.exists(books_path):
        books_stats = validate_file(books_path, "book")
        print_statistics(books_stats, "book")
    else:
        print(f"\n⚠ Books file not found: {books_path}")
    
    # Generate summary report
    all_valid = generate_report(articles_stats, books_stats)
    
    # Exit code
    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()

