"""PDF processing utilities for HoloDeck.

This package provides PDF-specific file processing operations:

- **Heading Extraction**: Font-size-aware text extraction using pdfminer
  that produces markdown with proper heading markers (##) based on
  configurable font-size thresholds.

- **Page Extraction**: Extract specific pages from PDF files using pypdf,
  producing temporary PDF files with the selected pages.

These operations are used by FileProcessor to provide enhanced PDF handling
beyond what markitdown offers natively.

Example:
    from holodeck.lib.pdf_processor import extract_pdf_with_headings, extract_pdf_pages

    # Extract text with heading detection
    markdown = extract_pdf_with_headings(Path("document.pdf"))

    # Extract specific pages
    temp_path = extract_pdf_pages(Path("document.pdf"), pages=[0, 1, 2])

Functions:
    extract_pdf_with_headings: Extract PDF text with font-size heading detection
    extract_pdf_pages: Extract specific pages from a PDF file
"""

from holodeck.lib.pdf_processor.heading_extractor import extract_pdf_with_headings
from holodeck.lib.pdf_processor.page_extractor import extract_pdf_pages

__all__ = [
    "extract_pdf_pages",
    "extract_pdf_with_headings",
]
