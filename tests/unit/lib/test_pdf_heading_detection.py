"""Tests for PDF heading detection integration with FileProcessor.

This module tests the integration between FileProcessor and the pdf_processor
package for font-size-aware PDF heading detection. Unit tests for the
standalone extract_pdf_with_headings function are in
tests/unit/lib/pdf_processor/test_heading_extractor.py.
"""

import tempfile
from pathlib import Path
from unittest import mock

import pytest

from holodeck.lib.file_processor import FileProcessor
from holodeck.models.test_case import FileInput


class TestFileProcessorPDFIntegration:
    """Integration tests for PDF processing through FileProcessor."""

    def test_process_pdf_uses_heading_detection(self) -> None:
        """Test that process_file uses heading detection for PDFs."""
        processor = FileProcessor()

        with mock.patch.object(processor, "_extract_pdf_with_headings") as mock_extract:
            mock_extract.return_value = "# Heading\nBody text"

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(b"%PDF-1.4 fake pdf content")
                tmp_path = tmp.name

            try:
                result = processor.process_file(FileInput(path=tmp_path, type="pdf"))

                mock_extract.assert_called_once()
                assert result.metadata.get("pdf_heading_detection") is True
                assert "# Heading" in result.markdown_content
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    def test_process_pdf_fallback_on_error(self) -> None:
        """Test that PDF processing falls back to markitdown on error."""
        processor = FileProcessor()

        with mock.patch.object(processor, "_extract_pdf_with_headings") as mock_extract:
            mock_extract.side_effect = Exception("PDF parsing failed")

            with mock.patch.object(processor, "_get_markitdown") as mock_md:
                mock_instance = mock.MagicMock()
                mock_instance.convert.return_value = mock.MagicMock(
                    text_content="Fallback content"
                )
                mock_md.return_value = mock_instance

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(b"%PDF-1.4 fake pdf content")
                    tmp_path = tmp.name

                try:
                    result = processor.process_file(
                        FileInput(path=tmp_path, type="pdf")
                    )

                    assert result.metadata.get("pdf_heading_detection") is False
                    assert result.markdown_content == "Fallback content"
                finally:
                    Path(tmp_path).unlink(missing_ok=True)

    def test_process_non_pdf_uses_markitdown(self) -> None:
        """Test that non-PDF files still use markitdown."""
        processor = FileProcessor()

        with (
            mock.patch.object(processor, "_extract_pdf_with_headings") as mock_extract,
            mock.patch.object(processor, "_get_markitdown") as mock_md,
        ):
            mock_instance = mock.MagicMock()
            mock_instance.convert.return_value = mock.MagicMock(
                text_content="Text file content"
            )
            mock_md.return_value = mock_instance

            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
                tmp.write(b"Plain text content")
                tmp_path = tmp.name

            try:
                result = processor.process_file(FileInput(path=tmp_path, type="text"))

                mock_extract.assert_not_called()
                assert "pdf_heading_detection" not in result.metadata
            finally:
                Path(tmp_path).unlink(missing_ok=True)


@pytest.mark.integration
class TestRealPDFFixture:
    """Integration tests using the real HR8847 PDF fixture."""

    @pytest.fixture
    def hr8847_pdf_path(self) -> Path:
        """Get the path to the HR8847 PDF test fixture."""
        path = Path("tests/fixtures/hierarchical/HR8847_CNIMDT_AIGCE_STD_Act.pdf")
        if not path.exists():
            pytest.skip("HR8847 PDF fixture not found")
        return path

    def test_hr8847_has_heading_markers(self, hr8847_pdf_path: Path) -> None:
        """Test that processing HR8847 PDF produces heading markers."""
        processor = FileProcessor()
        result = processor.process_file(
            FileInput(path=str(hr8847_pdf_path), type="pdf")
        )

        assert result.error is None
        assert result.metadata.get("pdf_heading_detection") is True

        # Should have heading markers
        content = result.markdown_content
        assert "# " in content or "## " in content

        # Count headings
        lines = content.split("\n")
        headings = [line for line in lines if line.startswith("#")]
        assert len(headings) > 50  # HR8847 has many sections

    def test_hr8847_title_headings_detected(self, hr8847_pdf_path: Path) -> None:
        """Test that TITLE headings are detected in HR8847 PDF."""
        processor = FileProcessor()
        result = processor.process_file(
            FileInput(path=str(hr8847_pdf_path), type="pdf")
        )

        content = result.markdown_content

        # TITLE headings should be present (16pt font = h1)
        assert "TITLE I" in content
        assert "TITLE II" in content

    def test_hr8847_section_headings_detected(self, hr8847_pdf_path: Path) -> None:
        """Test that SEC. headings are detected in HR8847 PDF."""
        processor = FileProcessor()
        result = processor.process_file(
            FileInput(path=str(hr8847_pdf_path), type="pdf")
        )

        content = result.markdown_content

        # SEC. headings should be present (12pt font = h2)
        assert "SEC. 101" in content
        assert "SEC. 102" in content

    def test_hr8847_body_text_no_heading_markers(self, hr8847_pdf_path: Path) -> None:
        """Test that body text doesn't have heading markers."""
        processor = FileProcessor()
        result = processor.process_file(
            FileInput(path=str(hr8847_pdf_path), type="pdf")
        )

        content = result.markdown_content
        lines = content.split("\n")

        # Count lines with and without heading markers
        heading_lines = [line for line in lines if line.startswith("#")]
        body_lines = [
            line for line in lines if line.strip() and not line.startswith("#")
        ]

        # Body text should outnumber headings significantly
        assert len(body_lines) > len(heading_lines) * 2

    def test_hr8847_structured_chunker_compatibility(
        self, hr8847_pdf_path: Path
    ) -> None:
        """Test that output is compatible with StructuredChunker."""
        from holodeck.lib.structured_chunker import StructuredChunker

        processor = FileProcessor()
        result = processor.process_file(
            FileInput(path=str(hr8847_pdf_path), type="pdf")
        )

        # StructuredChunker should be able to parse the output
        chunker = StructuredChunker(max_tokens=800)
        chunks = chunker.parse(result.markdown_content, str(hr8847_pdf_path))

        # Should produce chunks with parent chains
        assert len(chunks) > 0

        # At least some chunks should have parent chains (from heading hierarchy)
        chunks_with_parents = [c for c in chunks if c.parent_chain]
        assert len(chunks_with_parents) > 0
