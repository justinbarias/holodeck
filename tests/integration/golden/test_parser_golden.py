"""Golden extracted text for ``holodeck.lib.file_processor.FileProcessor``.

Each case runs ``FileProcessor.process_file`` on a tracked sample file and
freezes the markdown plus the deterministic metadata. Cases cover every
``FileInput.type`` that can be exercised offline: text (txt, md, json, html),
csv, pdf (font-size headings, page selection), word, powerpoint (slide
selection) and excel (sheet and range selection), plus error paths.

Samples live in ``tests/fixtures/golden/parsers``. They were generated once
(python-pptx, openpyxl, a hand-built PDF and DOCX) and are committed as bytes,
so the goldens do not depend on those libraries. The large PDF
``tests/fixtures/hierarchical/HR8847_CNIMDT_AIGCE_STD_Act.pdf`` is stored as a
length, SHA-256 and head. Images are not covered: they need OCR or a vision
model.

Page numbers: ``FileInput.pages`` must be positive integers, but the PDF and
PowerPoint extractors index from zero. The cases freeze that behavior as it is.

Regenerate with ``HOLODECK_UPDATE_GOLDEN=1`` (see ``_golden.py``).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pytest

from holodeck.lib.file_processor import FileProcessor
from holodeck.models.test_case import FileInput

from ._golden import GOLDEN_ROOT, assert_golden

pytestmark = pytest.mark.integration

PARSER_DIR = GOLDEN_ROOT / "parsers"
EXPECTED = PARSER_DIR / "expected.json"
BIG_PDF = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "hierarchical"
    / "HR8847_CNIMDT_AIGCE_STD_Act.pdf"
)
LARGE_THRESHOLD = 4000  # chars; longer output is stored as length/sha/head/tail

# (case id, file, FileInput.type, extra FileInput fields)
CASES: list[tuple[str, Path, str, dict[str, Any]]] = [
    ("txt", PARSER_DIR / "sample.txt", "text", {}),
    ("md", PARSER_DIR / "sample.md", "text", {}),
    ("json", PARSER_DIR / "sample.json", "text", {}),
    ("html", PARSER_DIR / "sample.html", "text", {}),
    ("csv", PARSER_DIR / "sample.csv", "csv", {}),
    ("docx", PARSER_DIR / "sample.docx", "word", {}),
    ("pdf_all_pages", PARSER_DIR / "sample.pdf", "pdf", {}),
    ("pdf_pages_1", PARSER_DIR / "sample.pdf", "pdf", {"pages": [1]}),
    ("pdf_pages_1_2", PARSER_DIR / "sample.pdf", "pdf", {"pages": [1, 2]}),
    ("pdf_page_out_of_range", PARSER_DIR / "sample.pdf", "pdf", {"pages": [9]}),
    ("pdf_large_all_pages", BIG_PDF, "pdf", {}),
    ("pdf_large_pages_1_2", BIG_PDF, "pdf", {"pages": [1, 2]}),
    ("pptx_all_slides", PARSER_DIR / "sample.pptx", "powerpoint", {}),
    ("pptx_slides_1_2", PARSER_DIR / "sample.pptx", "powerpoint", {"pages": [1, 2]}),
    (
        "pptx_slide_out_of_range",
        PARSER_DIR / "sample.pptx",
        "powerpoint",
        {"pages": [3]},
    ),
    ("xlsx_all_sheets", PARSER_DIR / "sample.xlsx", "excel", {}),
    ("xlsx_sheet_notes", PARSER_DIR / "sample.xlsx", "excel", {"sheet": "Notes"}),
    (
        "xlsx_default_sheet_range",
        PARSER_DIR / "sample.xlsx",
        "excel",
        {"range": "A1:C3"},
    ),
    (
        "xlsx_sheet_and_range",
        PARSER_DIR / "sample.xlsx",
        "excel",
        {"sheet": "Inventory", "range": "B2:D4"},
    ),
    ("xlsx_unknown_sheet", PARSER_DIR / "sample.xlsx", "excel", {"sheet": "Missing"}),
    ("missing_file", PARSER_DIR / "does_not_exist.txt", "text", {}),
]

# Metadata keys that are stable across machines (path and size are not frozen).
_STABLE_METADATA = (
    "type",
    "preprocessed",
    "pages_extracted",
    "sheet_extracted",
    "range_extracted",
    "pdf_heading_detection",
)


def _markdown_record(markdown: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "length": len(markdown),
        "sha256": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
    }
    if len(markdown) > LARGE_THRESHOLD:
        record["head"] = markdown[:600]
        record["tail"] = markdown[-300:]
    else:
        record["text"] = markdown
    return record


def _scrub(message: str | None) -> str | None:
    """Replace machine-specific absolute paths and temp names in error text."""
    if message is None:
        return None
    message = message.replace(str(PARSER_DIR), "<parsers>")
    message = message.replace(str(BIG_PDF.parent), "<hierarchical>")
    return re.sub(r"/(?:private/)?(?:var|tmp)/[^\s'\"]+", "<tmp>", message)


def _run(
    processor: FileProcessor, path: Path, file_type: str, extra: dict[str, Any]
) -> dict[str, Any]:
    result = processor.process_file(FileInput(path=str(path), type=file_type, **extra))
    metadata = result.metadata or {}
    return {
        "input_sha256": (
            hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        ),
        "file_type": file_type,
        "options": extra,
        "error": _scrub(result.error),
        "metadata": {k: metadata[k] for k in _STABLE_METADATA if k in metadata},
        "markdown": _markdown_record(result.markdown_content),
    }


def test_file_processor_golden(tmp_path: Path) -> None:
    """Extracted markdown and metadata for every sample match the goldens."""
    processor = FileProcessor(cache_dir=str(tmp_path / "cache"))
    actual = {
        case_id: _run(processor, path, file_type, extra)
        for case_id, path, file_type, extra in CASES
    }
    assert_golden(EXPECTED, actual, label="file_processor")
