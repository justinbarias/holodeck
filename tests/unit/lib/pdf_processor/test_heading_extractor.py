"""Tests for PDF heading extraction standalone function.

This module tests the extract_pdf_with_headings function that uses
pdfminer to extract PDF text with heading detection, including both
bookmark-based and font-size-based strategies.
"""

from pathlib import Path
from unittest import mock

from pdfminer.layout import LTChar, LTTextContainer, LTTextLine
from pdfminer.pdfdocument import PDFNoOutlines

from holodeck.lib.pdf_processor import extract_pdf_with_headings
from holodeck.lib.pdf_processor.heading_extractor import (
    _extract_with_bookmarks,
    _has_bookmarks,
    _match_bookmark,
    _normalize_text,
)

# === Helpers for building pdfminer mocks ===


def _make_mock_page(lines: list[tuple[str, float]]) -> mock.MagicMock:
    """Create a mock pdfminer page with text lines and font sizes.

    Args:
        lines: List of (text, font_size) tuples.

    Returns:
        Mock page layout object.
    """
    mock_lines = []
    for text, font_size in lines:
        char = mock.MagicMock(spec=LTChar)
        char.size = font_size

        line = mock.MagicMock(spec=LTTextLine)
        line.get_text.return_value = f"{text}\n"
        line.__iter__ = (lambda c: lambda self: iter([c]))(char)
        mock_lines.append(line)

    container = mock.MagicMock(spec=LTTextContainer)
    container.__iter__ = (lambda ls: lambda self: iter(ls))(mock_lines)

    page = mock.MagicMock()
    page.__iter__ = (lambda c: lambda self: iter([c]))(container)
    return page


# === Font-size-based detection tests (existing behavior) ===


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


# === Bookmark-based detection tests ===


class TestBookmarkDetection:
    """Tests for bookmark-based heading detection."""

    def test_extract_with_bookmarks_available(self) -> None:
        """Test that bookmark-based detection is used when bookmarks exist."""
        from pdfminer.pdftypes import PDFObjRef

        # Mock outline entries: (level, title, dest, action, se)
        mock_ref = mock.MagicMock(spec=PDFObjRef)
        mock_ref.objid = 100
        outlines = [
            (1, "Chapter One", [mock_ref], None, None),
            (2, "Section 1.1", [mock_ref], None, None),
        ]

        with (
            mock.patch(
                "holodeck.lib.pdf_processor.heading_extractor._has_bookmarks"
            ) as mock_has_bm,
            mock.patch(
                "holodeck.lib.pdf_processor.heading_extractor._extract_with_bookmarks"
            ) as mock_extract_bm,
            mock.patch(
                "holodeck.lib.pdf_processor.heading_extractor._extract_with_font_sizes"
            ) as mock_extract_fs,
        ):
            mock_has_bm.return_value = (True, outlines)
            mock_extract_bm.return_value = "# Chapter One\n## Section 1.1\nBody text"

            result = extract_pdf_with_headings(Path("/fake/path.pdf"))

        assert "# Chapter One" in result
        assert "## Section 1.1" in result
        mock_extract_fs.assert_not_called()

    def test_extract_no_bookmarks_falls_back_to_font_sizes(self) -> None:
        """Test that font-size detection is used when no bookmarks exist."""
        mock_page = _make_mock_page([("Title", 14.0), ("Body", 10.0)])

        with (
            mock.patch(
                "holodeck.lib.pdf_processor.heading_extractor._has_bookmarks"
            ) as mock_has_bm,
            mock.patch("pdfminer.high_level.extract_pages") as mock_extract,
        ):
            mock_has_bm.return_value = (False, [])
            mock_extract.return_value = iter([mock_page])

            result = extract_pdf_with_headings(Path("/fake/path.pdf"))

        assert "# Title" in result
        assert "Body" in result

    def test_extract_bookmarks_disabled(self) -> None:
        """Test that use_bookmarks=False skips bookmark detection entirely."""
        mock_page = _make_mock_page([("Title", 14.0), ("Body", 10.0)])

        with (
            mock.patch(
                "holodeck.lib.pdf_processor.heading_extractor._has_bookmarks"
            ) as mock_has_bm,
            mock.patch("pdfminer.high_level.extract_pages") as mock_extract,
        ):
            mock_extract.return_value = iter([mock_page])

            result = extract_pdf_with_headings(
                Path("/fake/path.pdf"), use_bookmarks=False
            )

        # _has_bookmarks should never be called
        mock_has_bm.assert_not_called()
        assert "# Title" in result

    def test_bookmark_extraction_error_falls_back(self) -> None:
        """Test that bookmark errors gracefully fall back to font-size method."""
        mock_page = _make_mock_page([("Title", 14.0)])

        with (
            mock.patch(
                "holodeck.lib.pdf_processor.heading_extractor._has_bookmarks"
            ) as mock_has_bm,
            mock.patch("pdfminer.high_level.extract_pages") as mock_extract,
        ):
            mock_has_bm.side_effect = Exception("Corrupt PDF outline")
            mock_extract.return_value = iter([mock_page])

            result = extract_pdf_with_headings(Path("/fake/path.pdf"))

        assert "# Title" in result


class TestHasBookmarks:
    """Tests for the _has_bookmarks helper."""

    def test_returns_true_with_outlines(self) -> None:
        """Test returns (True, entries) when outlines exist."""
        mock_outlines = [(1, "Chapter", None, None, None)]

        with (
            mock.patch("builtins.open", mock.mock_open(read_data=b"")),
            mock.patch("holodeck.lib.pdf_processor.heading_extractor.PDFParser"),
            mock.patch(
                "holodeck.lib.pdf_processor.heading_extractor.PDFDocument"
            ) as mock_doc_cls,
        ):
            mock_doc = mock_doc_cls.return_value
            mock_doc.get_outlines.return_value = iter(mock_outlines)

            has_bm, entries = _has_bookmarks(Path("/fake.pdf"))

        assert has_bm is True
        assert len(entries) == 1

    def test_returns_false_with_no_outlines_exception(self) -> None:
        """Test returns (False, []) when PDFNoOutlines is raised."""
        with (
            mock.patch("builtins.open", mock.mock_open(read_data=b"")),
            mock.patch("holodeck.lib.pdf_processor.heading_extractor.PDFParser"),
            mock.patch(
                "holodeck.lib.pdf_processor.heading_extractor.PDFDocument"
            ) as mock_doc_cls,
        ):
            mock_doc = mock_doc_cls.return_value
            mock_doc.get_outlines.side_effect = PDFNoOutlines()

            has_bm, entries = _has_bookmarks(Path("/fake.pdf"))

        assert has_bm is False
        assert entries == []

    def test_returns_false_with_empty_outlines(self) -> None:
        """Test returns (False, []) when outlines are empty."""
        with (
            mock.patch("builtins.open", mock.mock_open(read_data=b"")),
            mock.patch("holodeck.lib.pdf_processor.heading_extractor.PDFParser"),
            mock.patch(
                "holodeck.lib.pdf_processor.heading_extractor.PDFDocument"
            ) as mock_doc_cls,
        ):
            mock_doc = mock_doc_cls.return_value
            mock_doc.get_outlines.return_value = iter([])

            has_bm, entries = _has_bookmarks(Path("/fake.pdf"))

        assert has_bm is False
        assert entries == []


class TestBookmarkLevelMapping:
    """Tests for bookmark level -> heading marker mapping."""

    def test_level_1_becomes_h1(self) -> None:
        """Test level 1 bookmark becomes # heading."""
        result = _extract_with_bookmarks_helper(
            bookmarks=[("Chapter One", 1, 0)],
            page_lines={0: ["Chapter One", "Body text"]},
        )
        assert result is not None
        assert "# Chapter One" in result

    def test_level_2_becomes_h2(self) -> None:
        """Test level 2 bookmark becomes ## heading."""
        result = _extract_with_bookmarks_helper(
            bookmarks=[("Section A", 2, 0)],
            page_lines={0: ["Section A", "Body text"]},
        )
        assert result is not None
        assert "## Section A" in result

    def test_level_3_becomes_h3(self) -> None:
        """Test level 3 bookmark becomes ### heading."""
        result = _extract_with_bookmarks_helper(
            bookmarks=[("Subsection", 3, 0)],
            page_lines={0: ["Subsection", "Body text"]},
        )
        assert result is not None
        assert "### Subsection" in result

    def test_multiple_levels(self) -> None:
        """Test mixed heading levels in output."""
        result = _extract_with_bookmarks_helper(
            bookmarks=[
                ("Chapter", 1, 0),
                ("Section", 2, 0),
                ("Subsection", 3, 0),
            ],
            page_lines={0: ["Chapter", "Section", "Subsection", "Body"]},
        )
        assert result is not None
        assert "# Chapter" in result
        assert "## Section" in result
        assert "### Subsection" in result
        # Body should not have heading marker
        lines = result.split("\n")
        body_line = [ln for ln in lines if "Body" in ln][0]
        assert not body_line.startswith("#")


class TestBookmarkMatchingNormalized:
    """Tests for normalized text matching between bookmarks and text lines."""

    def test_exact_match(self) -> None:
        """Test exact title match."""
        level = _match_bookmark("chapter one", [("chapter one", 1)])
        assert level == 1

    def test_case_insensitive_match(self) -> None:
        """Test case-insensitive matching."""
        level = _match_bookmark("chapter one", [("Chapter One", 1)])
        # _normalize_text lowercases, so both should be lowercase when called properly
        # Test via the actual normalize path
        norm_line = _normalize_text("Chapter One")
        norm_title = _normalize_text("CHAPTER ONE")
        level = _match_bookmark(norm_line, [(norm_title, 1)])
        assert level == 1

    def test_whitespace_normalized(self) -> None:
        """Test whitespace normalization in matching."""
        norm_line = _normalize_text("Chapter   One")
        norm_title = _normalize_text("Chapter One")
        level = _match_bookmark(norm_line, [(norm_title, 1)])
        assert level == 1

    def test_substring_match_title_in_line(self) -> None:
        """Test that bookmark title substring of line matches."""
        level = _match_bookmark("1. chapter one - introduction", [("chapter one", 1)])
        assert level == 1

    def test_substring_match_line_in_title(self) -> None:
        """Test that line substring of bookmark title matches."""
        level = _match_bookmark("chapter", [("chapter one", 1)])
        assert level == 1

    def test_no_match_returns_none(self) -> None:
        """Test that unrelated text returns None."""
        level = _match_bookmark("random body text", [("chapter one", 1)])
        assert level is None

    def test_empty_line_returns_none(self) -> None:
        """Test that empty line returns None."""
        level = _match_bookmark("", [("chapter one", 1)])
        assert level is None

    def test_empty_bookmarks_returns_none(self) -> None:
        """Test that empty bookmark list returns None."""
        level = _match_bookmark("chapter one", [])
        assert level is None


class TestNormalizeText:
    """Tests for the _normalize_text helper."""

    def test_lowercases(self) -> None:
        """Test that text is lowercased."""
        assert _normalize_text("HELLO") == "hello"

    def test_collapses_whitespace(self) -> None:
        """Test that multiple spaces are collapsed."""
        assert _normalize_text("hello   world") == "hello world"

    def test_strips(self) -> None:
        """Test that leading/trailing whitespace is stripped."""
        assert _normalize_text("  hello  ") == "hello"

    def test_tabs_and_newlines(self) -> None:
        """Test that tabs and newlines are collapsed."""
        assert _normalize_text("hello\t\nworld") == "hello world"


class TestExtractWithBookmarksLowMatchRate:
    """Tests for bookmark match rate fallback."""

    def test_low_match_rate_returns_none(self) -> None:
        """Test that <30% match rate returns None (triggers fallback)."""
        # 10 bookmarks but only 2 matching text lines
        bookmarks = [(f"Bookmark {i}", 1, 0) for i in range(10)]
        page_lines = {
            0: ["Bookmark 0", "Bookmark 1", "Unrelated text A", "Unrelated text B"]
        }

        result = _extract_with_bookmarks_helper(
            bookmarks=bookmarks,
            page_lines=page_lines,
        )
        assert result is None

    def test_high_match_rate_returns_content(self) -> None:
        """Test that good match rate returns content."""
        bookmarks = [
            ("Chapter One", 1, 0),
            ("Chapter Two", 1, 0),
        ]
        page_lines = {0: ["Chapter One", "Body text", "Chapter Two", "More body"]}

        result = _extract_with_bookmarks_helper(
            bookmarks=bookmarks,
            page_lines=page_lines,
        )
        assert result is not None
        assert "# Chapter One" in result
        assert "# Chapter Two" in result


# === Helper for bookmark tests ===


def _extract_with_bookmarks_helper(
    bookmarks: list[tuple[str, int, int | None]],
    page_lines: dict[int, list[str]],
) -> str | None:
    """Helper to test _extract_with_bookmarks with pre-built data.

    Mocks all pdfminer I/O and calls _extract_with_bookmarks directly.

    Args:
        bookmarks: List of (title, level, page_num) tuples.
        page_lines: Dict of page_num -> list of text lines.

    Returns:
        Result from _extract_with_bookmarks.
    """
    from pdfminer.pdftypes import PDFObjRef

    # Assign one objid per unique page number (multiple bookmarks can share a page)
    objid_base = 100
    pagenum_to_objid: dict[int, int] = {}
    for _title, _level, page_num in bookmarks:
        if page_num is not None and page_num not in pagenum_to_objid:
            pagenum_to_objid[page_num] = objid_base + page_num

    # Build outlines in pdfminer format: (level, title, dest, action, se)
    outlines = []
    for title, level, page_num in bookmarks:
        ref = mock.MagicMock(spec=PDFObjRef)
        if page_num is not None:
            ref.objid = pagenum_to_objid[page_num]
        else:
            ref.objid = -1  # unresolvable
        outlines.append((level, title, [ref], None, None))

    # Create mock PDF pages for PDFPage.create_pages (one per unique page)
    sorted_page_nums = sorted(pagenum_to_objid.keys())
    mock_pdf_pages = []
    for pn in sorted_page_nums:
        mock_page = mock.MagicMock()
        mock_page.pageid = pagenum_to_objid[pn]
        mock_pdf_pages.append(mock_page)

    # Build pdfminer page layouts for _extract_text_by_page
    mock_page_layouts = []
    for page_num in sorted(page_lines.keys()):
        lines = page_lines[page_num]
        mock_text_lines = []
        for text in lines:
            lt_line = mock.MagicMock(spec=LTTextLine)
            lt_line.get_text.return_value = f"{text}\n"
            mock_text_lines.append(lt_line)

        container = mock.MagicMock(spec=LTTextContainer)
        container.__iter__ = (lambda ls: lambda self: iter(ls))(mock_text_lines)

        page_layout = mock.MagicMock()
        page_layout.__iter__ = (lambda c: lambda self: iter([c]))(container)
        mock_page_layouts.append(page_layout)

    with (
        mock.patch("builtins.open", mock.mock_open(read_data=b"")),
        mock.patch("holodeck.lib.pdf_processor.heading_extractor.PDFParser"),
        mock.patch(
            "holodeck.lib.pdf_processor.heading_extractor.PDFDocument"
        ) as mock_doc_cls,
        mock.patch(
            "holodeck.lib.pdf_processor.heading_extractor.PDFPage"
        ) as mock_page_cls,
        mock.patch("pdfminer.high_level.extract_pages") as mock_extract,
    ):
        mock_doc = mock_doc_cls.return_value  # noqa: F841

        # Set up page ID mapping
        mock_page_cls.create_pages.return_value = iter(mock_pdf_pages)

        mock_extract.return_value = iter(mock_page_layouts)

        return _extract_with_bookmarks(Path("/fake/path.pdf"), outlines)
