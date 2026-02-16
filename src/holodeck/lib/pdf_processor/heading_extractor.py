"""Font-size-aware PDF heading extraction using pdfminer.

This module extracts text from PDF files while detecting headings based on
font size thresholds. It produces markdown output with proper heading markers
(#, ##, etc.) based on the font sizes found in the document.

This provides better structure preservation than generic PDF-to-text conversion,
enabling downstream tools like StructuredChunker to create hierarchical chunks.
"""

from pathlib import Path

from holodeck.lib.logging_config import get_logger

logger = get_logger(__name__)


def extract_pdf_with_headings(
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

    Raises:
        Exception: If PDF parsing fails (caller should handle fallback).
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
