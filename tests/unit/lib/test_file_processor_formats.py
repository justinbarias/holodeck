"""Every file type FileProcessor accepts converts without a missing parser."""

from pathlib import Path

import pytest

from holodeck.lib.file_processor import FileProcessor

SAMPLES = Path(__file__).resolve().parents[2] / "fixtures" / "golden" / "parsers"
# FileInput.type values that go through MarkItDown, by sample extension.
EXTENSIONS = ["txt", "md", "json", "html", "csv", "docx", "pdf", "pptx", "xlsx"]


@pytest.mark.unit
@pytest.mark.parametrize("ext", EXTENSIONS)
def test_markitdown_converts_sample(ext: str) -> None:
    processor = FileProcessor(cache_dir=".holodeck/cache")
    result = processor._get_markitdown().convert(str(SAMPLES / f"sample.{ext}"))
    assert result.text_content.strip()
