"""
Quick inspection utilities for the scraped knowledge base (staging).

Usage (from project root):

    python -m knowldege_base.rag_staging.inspect_kb

This will:
- Print basic counts (articles, books).
- Show a few sample records (titles + truncated text).
"""

from __future__ import annotations

from typing import Iterable, List

from . import loader


def _truncate(text: str, max_len: int = 300) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _print_samples(
    records: Iterable[dict],
    label: str,
    n: int = 3,
    include_text: bool = False,
) -> None:
    print(f"\n=== {label} (showing up to {n}) ===")
    for i, rec in enumerate(records):
        if i >= n:
            break
        title = rec.get("title", "<no title>")
        doc_id = rec.get("doc_id", "<no id>")
        content_type = rec.get("content_type", "<no type>")
        clean_text = rec.get("clean_text", "")
        print(f"\n[{i+1}] {content_type} | {doc_id}")
        print(f"Title : {title}")
        if include_text:
            # Avoid crashing on narrow-console encodings by replacing
            # unprintable characters.
            safe_text = _truncate(clean_text).encode(
                "utf-8", errors="replace"
            ).decode("utf-8", errors="replace")
            print(f"Text  : {safe_text}")


def _collect_samples(records: List[dict], n: int = 3) -> List[dict]:
    return list(records[:n])


def _write_samples_report(
    articles: List[dict],
    books: List[dict],
    path: str = "kb_samples_utf8.txt",
) -> None:
    """
    Write a small UTF-8 report with sample articles and books.
    This is easier to inspect in an editor that supports Arabic.
    """
    article_samples = _collect_samples(articles, n=3)
    book_samples = _collect_samples(books, n=3)

    lines: List[str] = []
    lines.append("=== Sample Articles ===")
    for i, rec in enumerate(article_samples, start=1):
        lines.append(f"\n[{i}] {rec.get('doc_id', '<no id>')}")
        lines.append(f"Title : {rec.get('title', '<no title>')}")
        lines.append(_truncate(rec.get("clean_text", "")))

    lines.append("\n\n=== Sample Books ===")
    for i, rec in enumerate(book_samples, start=1):
        lines.append(f"\n[{i}] {rec.get('doc_id', '<no id>')}")
        lines.append(f"Title : {rec.get('title', '<no title>')}")
        lines.append(_truncate(rec.get("clean_text", "")))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nSample report written to: {path} (UTF-8)")


def main() -> None:
    stats = loader.describe_kb()
    print("Knowledge base summary:")
    for k, v in stats.items():
        print(f"- {k}: {v}")

    articles = loader.load_articles()
    books = loader.load_books()

    # Print only IDs and titles to avoid console encoding issues.
    _print_samples(articles, "Articles", include_text=False)
    _print_samples(books, "Books", include_text=False)

    # Write a UTF-8 report with truncated text for deeper inspection.
    _write_samples_report(articles, books)


if __name__ == "__main__":
    main()


