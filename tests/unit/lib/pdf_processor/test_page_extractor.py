"""Tests for PDF page extraction standalone function.

This module tests the extract_pdf_pages function that uses
pypdf to extract specific pages from PDF files.
"""

import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from holodeck.lib.pdf_processor import extract_pdf_pages


class TestExtractPdfPages:
    """Tests for the extract_pdf_pages function."""

    def test_single_page_extraction(self) -> None:
        """Test extracting a single page from PDF."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            from pypdf import PdfReader, PdfWriter

            # Create a minimal valid PDF with 3 pages
            writer = PdfWriter()
            for _ in range(3):
                writer.add_blank_page(width=200, height=200)

            with open(tmp_path, "wb") as f:
                writer.write(f)

            # Test extracting page 1
            result_path = extract_pdf_pages(tmp_path, [1])

            # Verify the result is a valid PDF
            assert result_path.exists()
            assert result_path.suffix == ".pdf"

            # Verify it has exactly 1 page
            reader = PdfReader(str(result_path))
            assert len(reader.pages) == 1

            # Clean up temp file
            result_path.unlink()

        finally:
            tmp_path.unlink(missing_ok=True)

    def test_out_of_range_page(self) -> None:
        """Test PDF page extraction with page number out of range."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            from pypdf import PdfWriter

            # Create a PDF with only 2 pages
            writer = PdfWriter()
            for _ in range(2):
                writer.add_blank_page(width=200, height=200)

            with open(tmp_path, "wb") as f:
                writer.write(f)

            # Try to extract page 5 (out of range)
            with pytest.raises(ValueError, match="Page .* out of range"):
                extract_pdf_pages(tmp_path, [5])

        finally:
            tmp_path.unlink(missing_ok=True)

    def test_import_error_when_pypdf_missing(self) -> None:
        """Test page extraction when pypdf is not available."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # Mock pypdf import to fail
            with (
                mock.patch.dict(sys.modules, {"pypdf": None}),
                mock.patch("builtins.__import__", side_effect=ImportError("No pypdf")),
                pytest.raises(ImportError, match="pypdf is required"),
            ):
                extract_pdf_pages(tmp_path, [0])

        finally:
            tmp_path.unlink(missing_ok=True)
