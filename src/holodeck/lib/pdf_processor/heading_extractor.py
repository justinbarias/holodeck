"""PDF heading extraction using pdfminer with bookmark and font-size detection.

This module extracts text from PDF files while detecting headings. It supports
two strategies:

1. **Bookmark-based detection** (preferred): Uses PDF outline/bookmark entries
   to identify headings and their hierarchy levels. More reliable when the PDF
   contains bookmarks.

2. **Font-size-based detection** (fallback): Analyzes font sizes to identify
   headings based on configurable thresholds. Used when bookmarks are absent.

The output is markdown with proper heading markers (#, ##, etc.) based on the
detected heading hierarchy, enabling downstream tools like StructuredChunker
to create hierarchical chunks.
"""

import re
from pathlib import Path

from pdfminer.pdfdocument import PDFDocument, PDFNoOutlines
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdftypes import PDFObjRef

from holodeck.lib.logging_config import get_logger

logger = get_logger(__name__)


def extract_pdf_with_headings(
    file_path: Path,
    heading_thresholds: dict[float, int] | None = None,
    use_bookmarks: bool = True,
) -> str:
    """Extract PDF text with heading detection.

    Prefers bookmark-based heading detection when bookmarks are available,
    falling back to font-size-based detection otherwise.

    Args:
        file_path: Path to PDF file
        heading_thresholds: Font size -> heading level mapping.
            Default: {14.0: 1, 12.0: 2} (14pt+ = h1, 12pt+ = h2)
        use_bookmarks: Whether to attempt bookmark-based detection first.
            Default: True

    Returns:
        Markdown text with heading markers based on detected headings.

    Raises:
        Exception: If PDF parsing fails (caller should handle fallback).
    """
    if use_bookmarks:
        try:
            has_bookmarks, outlines = _has_bookmarks(file_path)
            if has_bookmarks:
                result = _extract_with_bookmarks(file_path, outlines)
                if result is not None:
                    return result
                logger.debug(
                    "Bookmark extraction returned None, "
                    "falling back to font-size detection"
                )
        except Exception:
            logger.debug(
                "Bookmark detection failed, falling back to font-size detection",
                exc_info=True,
            )

    return _extract_with_font_sizes(file_path, heading_thresholds)


def _has_bookmarks(file_path: Path) -> tuple[bool, list]:
    """Check if a PDF has bookmarks (outline entries).

    Args:
        file_path: Path to PDF file

    Returns:
        Tuple of (has_bookmarks, outline_entries). If no bookmarks exist,
        returns (False, []).
    """
    with open(file_path, "rb") as fp:
        parser = PDFParser(fp)
        document = PDFDocument(parser)

        try:
            outlines = list(document.get_outlines())
        except PDFNoOutlines:
            return False, []

    if not outlines:
        return False, []

    return True, outlines


def _resolve_page_number(
    dest: object,
    action: object,
    objid_to_pagenum: dict[object, int],
) -> int | None:
    """Resolve a bookmark destination to a 0-indexed page number.

    Args:
        dest: Outline destination (may be a list with a PDFObjRef, or None)
        action: Outline action (alternative to dest, may contain a 'D' key)
        objid_to_pagenum: Mapping from PDF page object IDs to page numbers

    Returns:
        0-indexed page number, or None if resolution fails.
    """
    # Try dest first (most common case)
    ref = None
    if isinstance(dest, list) and len(dest) > 0:
        ref = dest[0]
    elif isinstance(dest, PDFObjRef):
        ref = dest

    # Fall back to action dict
    if ref is None and isinstance(action, dict):
        d = action.get("D")
        if isinstance(d, list) and len(d) > 0:
            ref = d[0]
        elif isinstance(d, PDFObjRef):
            ref = d

    if ref is None:
        return None

    if isinstance(ref, PDFObjRef):
        return objid_to_pagenum.get(ref.objid)

    return None


def _extract_text_by_page(file_path: Path) -> dict[int, list[str]]:
    """Extract text lines grouped by 0-indexed page number.

    Args:
        file_path: Path to PDF file

    Returns:
        Dict mapping page number to list of text lines on that page.
    """
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer, LTTextLine

    pages: dict[int, list[str]] = {}

    for page_num, page_layout in enumerate(extract_pages(str(file_path))):
        lines: list[str] = []
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for text_line in element:
                    if isinstance(text_line, LTTextLine):
                        line_text = text_line.get_text().strip()
                        if line_text:
                            lines.append(line_text)
        pages[page_num] = lines

    return pages


def _normalize_text(text: str) -> str:
    """Normalize text for fuzzy comparison.

    Lowercases, collapses whitespace, and strips.

    Args:
        text: Text to normalize

    Returns:
        Normalized text string.
    """
    return re.sub(r"\s+", " ", text.lower().strip())


def _extract_with_bookmarks(
    file_path: Path,
    outlines: list,
) -> str | None:
    """Extract PDF text using bookmark-based heading detection.

    Parses outline entries into (title, level, page_num) tuples, extracts
    text page-by-page, and matches bookmark titles to text lines to insert
    heading markers.

    Args:
        file_path: Path to PDF file
        outlines: List of outline entries from PDFDocument.get_outlines()

    Returns:
        Markdown text with heading markers, or None if match rate is too low.
    """
    # Build objid -> page number mapping
    with open(file_path, "rb") as fp:
        parser = PDFParser(fp)
        document = PDFDocument(parser)
        objid_to_pagenum = {
            page.pageid: page_num
            for page_num, page in enumerate(PDFPage.create_pages(document))
        }

    # Parse outlines into structured bookmarks
    bookmarks: list[tuple[str, int, int | None]] = []
    for level, title, dest, action, _se in outlines:
        page_num = _resolve_page_number(dest, action, objid_to_pagenum)
        bookmarks.append((title, level, page_num))

    if not bookmarks:
        return None

    # Build per-page bookmark lookup: {page_num: [(normalized_title, level)]}
    page_bookmarks: dict[int, list[tuple[str, int]]] = {}
    for title, level, page_num in bookmarks:
        if page_num is not None:
            page_bookmarks.setdefault(page_num, []).append(
                (_normalize_text(title), level)
            )

    # Also build a global lookup for bookmarks with unresolved pages
    global_bookmarks: list[tuple[str, int]] = []
    for title, level, page_num in bookmarks:
        if page_num is None:
            global_bookmarks.append((_normalize_text(title), level))

    # Extract text by page
    text_by_page = _extract_text_by_page(file_path)

    # Match text lines to bookmarks and build output
    output_lines: list[str] = []
    total_bookmarks = len(bookmarks)
    matched_count = 0

    for page_num in sorted(text_by_page.keys()):
        page_lines = text_by_page[page_num]
        page_bm = page_bookmarks.get(page_num, [])

        for line in page_lines:
            normalized_line = _normalize_text(line)
            heading_level = _match_bookmark(normalized_line, page_bm)

            # Try global bookmarks if no page-specific match
            if heading_level is None and global_bookmarks:
                heading_level = _match_bookmark(normalized_line, global_bookmarks)

            if heading_level is not None:
                matched_count += 1
                prefix = "#" * heading_level
                output_lines.append(f"{prefix} {line}")
            else:
                output_lines.append(line)

    # Check match rate - fall back if too low
    if total_bookmarks > 0 and matched_count / total_bookmarks < 0.3:
        logger.warning(
            "Bookmark match rate too low (%d/%d = %.0f%%), "
            "falling back to font-size detection",
            matched_count,
            total_bookmarks,
            matched_count / total_bookmarks * 100,
        )
        return None

    logger.debug(
        "Bookmark-based extraction matched %d/%d bookmarks",
        matched_count,
        total_bookmarks,
    )
    return "\n".join(output_lines)


def _match_bookmark(
    normalized_line: str,
    bookmark_entries: list[tuple[str, int]],
) -> int | None:
    """Check if a normalized text line matches any bookmark title.

    Matching strategy (in priority order):
    1. Exact match
    2. Substring match (longest matching title wins to avoid short-title
       false positives, e.g. "section" matching "subsection")

    Args:
        normalized_line: Normalized text line to check
        bookmark_entries: List of (normalized_title, level) tuples

    Returns:
        Heading level if matched, None otherwise.
    """
    if not normalized_line:
        return None

    # 1. Exact match (highest priority)
    for norm_title, level in bookmark_entries:
        if norm_title and norm_title == normalized_line:
            return level

    # 2. Substring match — prefer longest matching title
    best_match: tuple[int, int] | None = None  # (title_len, level)
    for norm_title, level in bookmark_entries:
        if not norm_title:
            continue
        if norm_title in normalized_line or normalized_line in norm_title:
            title_len = len(norm_title)
            if best_match is None or title_len > best_match[0]:
                best_match = (title_len, level)

    if best_match is not None:
        return best_match[1]

    return None


def _extract_with_font_sizes(
    file_path: Path,
    heading_thresholds: dict[float, int] | None = None,
) -> str:
    """Extract PDF text with font-size-based heading detection.

    Uses pdfminer to analyze font sizes and produces markdown with proper
    heading markers based on font size thresholds.

    Args:
        file_path: Path to PDF file
        heading_thresholds: Font size -> heading level mapping.
            Default: {14.0: 1, 12.0: 2} (14pt+ = h1, 12pt+ = h2)

    Returns:
        Markdown text with heading markers based on font sizes.
    """
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTChar, LTTextContainer, LTTextLine

    if heading_thresholds is None:
        heading_thresholds = {14.0: 1, 12.0: 2}

    # Sort thresholds descending for proper matching
    sorted_thresholds = sorted(heading_thresholds.items(), reverse=True)
    lines: list[str] = []

    for page_layout in extract_pages(str(file_path)):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for text_line in element:
                    if isinstance(text_line, LTTextLine):
                        line_text = text_line.get_text().strip()
                        if not line_text:
                            continue

                        # Get font size from first character
                        font_size: float | None = None
                        for char in text_line:
                            if isinstance(char, LTChar):
                                font_size = char.size
                                break

                        # Determine heading level based on font size
                        heading_level = 0
                        if font_size:
                            for threshold, level in sorted_thresholds:
                                if font_size >= threshold:
                                    heading_level = level
                                    break

                        # Format line with heading markers if applicable
                        if heading_level > 0:
                            lines.append(f"{'#' * heading_level} {line_text}")
                        else:
                            lines.append(line_text)

    return "\n".join(lines)
