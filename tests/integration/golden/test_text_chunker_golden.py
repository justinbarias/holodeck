"""Golden chunk lists for ``holodeck.lib.text_chunker.TextChunker``.

Four inputs (short, multi-paragraph, one long paragraph, unicode) are split
with several ``(chunk_size, chunk_overlap)`` settings. The exact chunk lists
are frozen in ``tests/fixtures/golden/text_chunker/expected.json``.

Note: the current splitter ignores ``chunk_overlap`` (it only uses
``chunk_size``), so settings that differ only in overlap give equal chunks.
The overlap value is still part of the case so a replacement splitter that
starts honoring it shows up as a reviewed golden change.

Regenerate with ``HOLODECK_UPDATE_GOLDEN=1`` (see ``_golden.py``).
"""

from __future__ import annotations

from typing import Any

import pytest

from holodeck.lib.text_chunker import TextChunker

from ._golden import GOLDEN_ROOT, assert_golden, sha12

pytestmark = pytest.mark.integration

CHUNKER_DIR = GOLDEN_ROOT / "text_chunker"
INPUTS = ["short", "multi_paragraph", "long_single_paragraph", "unicode"]
SETTINGS = [(32, 0), (64, 8), (128, 16), (512, 50)]


def _chunks(name: str, chunk_size: int, overlap: int) -> list[str]:
    text = (CHUNKER_DIR / f"{name}.txt").read_text(encoding="utf-8")
    return TextChunker(chunk_size=chunk_size, chunk_overlap=overlap).split_text(text)


def test_text_chunker_golden() -> None:
    """Chunk lists for every input and setting match the frozen goldens."""
    actual: dict[str, Any] = {}
    for name in INPUTS:
        text = (CHUNKER_DIR / f"{name}.txt").read_text(encoding="utf-8")
        per_setting: dict[str, Any] = {}
        for chunk_size, overlap in SETTINGS:
            chunks = _chunks(name, chunk_size, overlap)
            per_setting[f"size={chunk_size},overlap={overlap}"] = {
                "chunk_count": len(chunks),
                "chunks": chunks,
            }
        actual[name] = {"input_sha": sha12(text), "settings": per_setting}
    assert_golden(CHUNKER_DIR / "expected.json", actual, label="text_chunker")


@pytest.mark.parametrize("text", ["", "   \n\t  "])
def test_blank_text_is_rejected(text: str) -> None:
    """Blank input raises ValueError with the current message."""
    with pytest.raises(ValueError, match="Text cannot be empty"):
        TextChunker().split_text(text)
