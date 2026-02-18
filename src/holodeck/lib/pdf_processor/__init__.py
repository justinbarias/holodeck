"""PDF processing utilities for HoloDeck.

This package provides PDF-specific file processing operations:

- **Heading Extraction**: Text extraction using pdfminer that produces
  markdown with proper heading markers (##). Supports two strategies:

  - **Bookmark-based detection** (preferred): Uses PDF outline/bookmark
    entries to identify headings and their hierarchy. More reliable when
    bookmarks are present.
  - **Font-size-based detection** (fallback): Analyzes font sizes against
    configurable thresholds. Used when bookmarks are absent or disabled.

- **Page Extraction**: Extract specific pages from PDF files using pypdf,
  producing temporary PDF files with the selected pages.

These operations are used by FileProcessor to provide enhanced PDF handling
beyond what markitdown offers natively.

Example:
    from holodeck.lib.pdf_processor import extract_pdf_with_headings, extract_pdf_pages

    # Extract text with heading detection (bookmarks preferred, font-size fallback)
    markdown = extract_pdf_with_headings(Path("document.pdf"))

    # Force font-size-only detection
    markdown = extract_pdf_with_headings(Path("document.pdf"), use_bookmarks=False)

    # Extract specific pages
    temp_path = extract_pdf_pages(Path("document.pdf"), pages=[0, 1, 2])

Functions:
    extract_pdf_with_headings: Extract PDF text with heading detection
    extract_pdf_pages: Extract specific pages from a PDF file
"""

from holodeck.lib.pdf_processor.heading_extractor import extract_pdf_with_headings
from holodeck.lib.pdf_processor.page_extractor import extract_pdf_pages

__all__ = [
    "extract_pdf_pages",
    "extract_pdf_with_headings",
]
