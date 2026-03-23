# PDF Processor

The `holodeck.lib.pdf_processor` package provides PDF-specific processing
utilities for text extraction with heading detection and page extraction.
These operations are used by `FileProcessor` to deliver enhanced PDF handling
beyond what markitdown offers natively.

## Overview

The package exposes two public functions:

| Function | Purpose |
|---|---|
| `extract_pdf_with_headings` | Extract text from a PDF while detecting headings via bookmarks or font sizes |
| `extract_pdf_pages` | Extract a subset of pages from a PDF into a temporary file |

### Heading Detection Strategies

`extract_pdf_with_headings` supports two strategies, applied in priority order:

1. **Bookmark-based detection** (preferred) -- Uses PDF outline/bookmark entries
   to identify headings and their hierarchy levels. More reliable when the PDF
   contains bookmarks.
2. **Font-size-based detection** (fallback) -- Analyzes font sizes against
   configurable thresholds. Used automatically when bookmarks are absent or
   when bookmark matching falls below a 30% match rate.

The output is Markdown text with heading markers (`#`, `##`, etc.) suitable for
downstream processing by tools such as `StructuredChunker`.

## Package Exports

::: holodeck.lib.pdf_processor
    options:
      docstring_style: google
      show_source: true
      show_root_heading: false
      members:
        - extract_pdf_with_headings
        - extract_pdf_pages

## Heading Extractor

::: holodeck.lib.pdf_processor.heading_extractor.extract_pdf_with_headings
    options:
      docstring_style: google
      show_source: true

## Page Extractor

::: holodeck.lib.pdf_processor.page_extractor.extract_pdf_pages
    options:
      docstring_style: google
      show_source: true

## Usage Examples

### Extract text with heading detection

```python
from pathlib import Path
from holodeck.lib.pdf_processor import extract_pdf_with_headings

# Bookmark-preferred extraction (default)
markdown = extract_pdf_with_headings(Path("document.pdf"))

# Force font-size-only detection
markdown = extract_pdf_with_headings(Path("document.pdf"), use_bookmarks=False)

# Custom font-size thresholds (18pt+ = h1, 14pt+ = h2, 12pt+ = h3)
markdown = extract_pdf_with_headings(
    Path("document.pdf"),
    heading_thresholds={18.0: 1, 14.0: 2, 12.0: 3},
)
```

### Extract specific pages

```python
from pathlib import Path
from holodeck.lib.pdf_processor import extract_pdf_pages

# Extract pages 0, 1, and 4 (0-indexed) into a temporary PDF
temp_path = extract_pdf_pages(Path("large-report.pdf"), pages=[0, 1, 4])

# Use the temporary file for further processing
print(f"Extracted pages saved to: {temp_path}")
```

## Dependencies

| Dependency | Used By | Purpose |
|---|---|---|
| `pdfminer.six` | `heading_extractor` | PDF text and font-size extraction, bookmark/outline parsing |
| `pypdf` | `page_extractor` | PDF page-level read/write operations |
