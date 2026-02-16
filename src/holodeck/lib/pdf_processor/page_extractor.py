"""PDF page extraction using pypdf.

This module provides functionality to extract specific pages from PDF files,
creating temporary PDF files containing only the requested pages.

This is used as a preprocessing step before PDF-to-markdown conversion,
allowing users to process only a subset of pages from large documents.
"""

import tempfile
from pathlib import Path

from holodeck.lib.logging_config import get_logger

logger = get_logger(__name__)


def extract_pdf_pages(file_path: Path, pages: list[int]) -> Path:
    """Extract specific pages from PDF into temporary file.

    Args:
        file_path: Path to original PDF file
        pages: List of page numbers to extract (0-indexed)

    Returns:
        Path to temporary PDF file with extracted pages

    Raises:
        ImportError: If pypdf is not installed
        ValueError: If page numbers are invalid or out of range
    """
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as e:
        raise ImportError(
            "pypdf is required for PDF page extraction. "
            "Install with: pip install 'markitdown[all]'"
        ) from e

    logger.debug(f"Extracting pages {pages} from PDF: {file_path}")

    try:
        reader = PdfReader(str(file_path))
        writer = PdfWriter()
        total_pages = len(reader.pages)

        # Validate page numbers
        for page_num in pages:
            if page_num < 0 or page_num >= total_pages:
                raise ValueError(
                    f"Page {page_num} out of range (PDF has {total_pages} pages)"
                )

        # Extract specified pages
        for page_num in pages:
            writer.add_page(reader.pages[page_num])

        # Create temporary file
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)  # noqa: SIM115
        tmp_path = Path(tmp.name)
        writer.write(tmp)
        tmp.close()

        logger.debug(f"Extracted {len(pages)} pages from PDF to temp file: {tmp_path}")
        return tmp_path

    except Exception as e:
        logger.error(f"PDF page extraction failed: {e}")
        raise
