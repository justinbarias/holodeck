"""Tests for the effective date of a source document (spec 043, ING-03)."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from holodeck.lib.memory.chunk_source import SourceDocument
from holodeck.lib.memory.effective_date import effective_date, strip_front_matter
from holodeck.lib.memory.errors import MemoryValueError

pytestmark = pytest.mark.unit

NOW = datetime(2026, 11, 11, 12, 0, tzinfo=UTC)


def _document(path: Path, *, mtime_reliable: bool = True) -> SourceDocument:
    return SourceDocument(
        source_tool="docs",
        source_uri=str(path),
        path=path,
        mtime_reliable=mtime_reliable,
    )


def _write(tmp_path: Path, text: str, name: str = "doc.md") -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_ing03_front_matter_date_wins(tmp_path: Path) -> None:
    path = _write(tmp_path, "---\neffective_date: 2022-07-01\n---\n\n# Act\n")

    assert effective_date(_document(path), "effective_date", NOW) == (
        datetime(2022, 7, 1, tzinfo=UTC),
        "front_matter",
    )


def test_ing03_front_matter_custom_key(tmp_path: Path) -> None:
    path = _write(tmp_path, "---\ncommenced: 2022-07-01\n---\n\n# Act\n")

    assert effective_date(_document(path), "commenced", NOW) == (
        datetime(2022, 7, 1, tzinfo=UTC),
        "front_matter",
    )


def test_ing03_front_matter_date_wins_over_unreliable_mtime(tmp_path: Path) -> None:
    path = _write(tmp_path, "---\neffective_date: 2022-07-01\n---\nBody\n")

    document = _document(path, mtime_reliable=False)

    assert effective_date(document, "effective_date", NOW) == (
        datetime(2022, 7, 1, tzinfo=UTC),
        "front_matter",
    )


def test_ing03_front_matter_value_not_a_date_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "---\neffective_date: last July\n---\n\n# Act\n")

    with pytest.raises(MemoryValueError, match="effective_date is not a date"):
        effective_date(_document(path), "effective_date", NOW)


def test_ing03_front_matter_non_text_value_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "---\neffective_date: 2022\n---\n\n# Act\n")

    with pytest.raises(MemoryValueError, match="effective_date is not a date"):
        effective_date(_document(path), "effective_date", NOW)


def test_ing03_invalid_front_matter_yaml_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "---\neffective_date: [2022\n---\n\n# Act\n")

    with pytest.raises(MemoryValueError, match="not valid YAML"):
        effective_date(_document(path), "effective_date", NOW)


def test_ing03_mtime_when_no_front_matter(tmp_path: Path) -> None:
    path = _write(tmp_path, "# Act\n\nBody\n")
    os.utime(path, (1_600_000_000, 1_600_000_000))

    assert effective_date(_document(path), "effective_date", NOW) == (
        datetime.fromtimestamp(1_600_000_000, UTC),
        "mtime",
    )


def test_ing03_mtime_when_front_matter_lacks_key(tmp_path: Path) -> None:
    path = _write(tmp_path, "---\ntitle: Act\n---\n\nBody\n")
    os.utime(path, (1_600_000_000, 1_600_000_000))

    assert effective_date(_document(path), "effective_date", NOW) == (
        datetime.fromtimestamp(1_600_000_000, UTC),
        "mtime",
    )


def test_ing03_ingestion_time_when_mtime_unreliable(tmp_path: Path) -> None:
    path = _write(tmp_path, "# Act\n\nBody\n")

    assert effective_date(
        _document(path, mtime_reliable=False), "effective_date", NOW
    ) == (NOW, "ingestion")


def test_ing03_binary_file_has_no_front_matter(tmp_path: Path) -> None:
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.7\n\xff\xfe binary")

    assert effective_date(
        _document(path, mtime_reliable=False), "effective_date", NOW
    ) == (NOW, "ingestion")


def test_ing03_naive_now_raises(tmp_path: Path) -> None:
    path = _write(tmp_path, "Body\n")

    with pytest.raises(MemoryValueError, match="now must be"):
        effective_date(_document(path), "effective_date", datetime(2026, 1, 1))


def test_a6_strip_front_matter_removes_block() -> None:
    assert strip_front_matter("---\neffective_date: 2022-07-01\n---\n\nBody") == "Body"


def test_a6_strip_front_matter_leaves_other_text_unchanged() -> None:
    for text in ("  Body\n", "---\nnot a mapping\n---\nBody", "---\n[bad\n---\nx"):
        assert strip_front_matter(text) == text


def test_ing03_a_missing_file_falls_back_to_ingestion_time(tmp_path: Path) -> None:
    gone = _document(tmp_path / "removed.md", mtime_reliable=False)
    assert effective_date(gone, "effective_date", NOW) == (NOW, "ingestion")

    local = _document(tmp_path / "removed.md", mtime_reliable=True)
    assert effective_date(local, "effective_date", NOW) == (NOW, "ingestion")


def test_ing03_an_impossible_yaml_date_raises_a_memory_error(tmp_path: Path) -> None:
    path = _write(tmp_path, "---\neffective_date: 2022-13-01\n---\nBody\n")
    with pytest.raises(MemoryValueError, match="not valid YAML"):
        effective_date(_document(path), "effective_date", NOW)


def test_a6_strip_front_matter_handles_a_byte_order_mark() -> None:
    assert strip_front_matter("﻿---\neffective_date: 2022-07-01\n---\nBody") == "Body"


@pytest.mark.parametrize(
    "text",
    [
        "----\nRate: $100\n----\nBody",
        "---\neffective_date: 2022-13-01\n---\nBody",
        "--- not a delimiter\nRate: 1\n---\nBody",
    ],
)
def test_a6_text_that_is_not_front_matter_is_kept(text: str) -> None:
    assert strip_front_matter(text) == text
