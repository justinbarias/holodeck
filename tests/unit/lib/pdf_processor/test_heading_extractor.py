"""Tests for PDF heading extraction standalone function.

This module tests the extract_pdf_with_headings function that uses
pdfminer to extract PDF text with font-size-based heading detection.
"""

from pathlib import Path
from unittest import mock

from pdfminer.layout import LTChar, LTTextContainer, LTTextLine

from holodeck.lib.pdf_processor import extract_pdf_with_headings


class TestExtractPdfWithHeadings:
    """Tests for the extract_pdf_with_headings function."""

    def test_extract_with_default_thresholds(self) -> None:
        """Test extraction uses default thresholds (14pt=h1, 12pt=h2)."""
        # Create mock characters with font sizes
        mock_char_14pt = mock.MagicMock(spec=LTChar)
        mock_char_14pt.size = 14.0

        mock_char_12pt = mock.MagicMock(spec=LTChar)
        mock_char_12pt.size = 12.0

        mock_char_10pt = mock.MagicMock(spec=LTChar)
        mock_char_10pt.size = 10.0

        # Create mock text lines
        mock_line_h1 = mock.MagicMock(spec=LTTextLine)
        mock_line_h1.get_text.return_value = "Title Heading\n"
        mock_line_h1.__iter__ = lambda self: iter([mock_char_14pt])

        mock_line_h2 = mock.MagicMock(spec=LTTextLine)
        mock_line_h2.get_text.return_value = "Section Heading\n"
        mock_line_h2.__iter__ = lambda self: iter([mock_char_12pt])

        mock_line_body = mock.MagicMock(spec=LTTextLine)
        mock_line_body.get_text.return_value = "Body text content\n"
        mock_line_body.__iter__ = lambda self: iter([mock_char_10pt])

        # Create mock text container
        mock_container = mock.MagicMock(spec=LTTextContainer)
        mock_container.__iter__ = lambda self: iter(
            [mock_line_h1, mock_line_h2, mock_line_body]
        )

        # Create mock page
        mock_page = mock.MagicMock()
        mock_page.__iter__ = lambda self: iter([mock_container])

        with mock.patch("pdfminer.high_level.extract_pages") as mock_extract:
            mock_extract.return_value = iter([mock_page])

            result = extract_pdf_with_headings(Path("/fake/path.pdf"))

        assert "# Title Heading" in result
        assert "## Section Heading" in result
        assert "Body text content" in result

    def test_extract_with_custom_thresholds(self) -> None:
        """Test extraction with custom font size thresholds."""
        mock_char_16pt = mock.MagicMock(spec=LTChar)
        mock_char_16pt.size = 16.0

        mock_line = mock.MagicMock(spec=LTTextLine)
        mock_line.get_text.return_value = "Large Heading\n"
        mock_line.__iter__ = lambda self: iter([mock_char_16pt])

        mock_container = mock.MagicMock(spec=LTTextContainer)
        mock_container.__iter__ = lambda self: iter([mock_line])

        mock_page = mock.MagicMock()
        mock_page.__iter__ = lambda self: iter([mock_container])

        with mock.patch("pdfminer.high_level.extract_pages") as mock_extract:
            mock_extract.return_value = iter([mock_page])

            # Custom thresholds: 16pt = h1, 14pt = h2, 12pt = h3
            result = extract_pdf_with_headings(
                Path("/fake/path.pdf"),
                heading_thresholds={16.0: 1, 14.0: 2, 12.0: 3},
            )

        assert "# Large Heading" in result

    def test_extract_empty_lines_skipped(self) -> None:
        """Test that empty lines are skipped."""
        mock_char = mock.MagicMock(spec=LTChar)
        mock_char.size = 10.0

        mock_empty_line = mock.MagicMock(spec=LTTextLine)
        mock_empty_line.get_text.return_value = "   \n"
        mock_empty_line.__iter__ = lambda self: iter([mock_char])

        mock_content_line = mock.MagicMock(spec=LTTextLine)
        mock_content_line.get_text.return_value = "Content\n"
        mock_content_line.__iter__ = lambda self: iter([mock_char])

        mock_container = mock.MagicMock(spec=LTTextContainer)
        mock_container.__iter__ = lambda self: iter(
            [mock_empty_line, mock_content_line]
        )

        mock_page = mock.MagicMock()
        mock_page.__iter__ = lambda self: iter([mock_container])

        with mock.patch("pdfminer.high_level.extract_pages") as mock_extract:
            mock_extract.return_value = iter([mock_page])

            result = extract_pdf_with_headings(Path("/fake/path.pdf"))

        # Only content line should be present
        lines = [line for line in result.split("\n") if line.strip()]
        assert len(lines) == 1
        assert "Content" in result

    def test_extract_no_font_info_treated_as_body(self) -> None:
        """Test that lines without font info are treated as body text."""
        # Line with no LTChar children (no font info)
        mock_line = mock.MagicMock(spec=LTTextLine)
        mock_line.get_text.return_value = "No font info\n"
        mock_line.__iter__ = lambda self: iter([])  # No chars

        mock_container = mock.MagicMock(spec=LTTextContainer)
        mock_container.__iter__ = lambda self: iter([mock_line])

        mock_page = mock.MagicMock()
        mock_page.__iter__ = lambda self: iter([mock_container])

        with mock.patch("pdfminer.high_level.extract_pages") as mock_extract:
            mock_extract.return_value = iter([mock_page])

            result = extract_pdf_with_headings(Path("/fake/path.pdf"))

        # Should be plain text without heading marker
        assert result == "No font info"
        assert not result.startswith("#")
