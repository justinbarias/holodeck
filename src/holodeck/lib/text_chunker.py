"""Text chunking utilities for document processing.

HoloDeck's own plain-text splitter. Token counts use tiktoken's
``cl100k_base`` encoding, ``chunk_size`` is a hard upper bound on the tokens of
every chunk, and ``chunk_overlap`` tokens from the end of one chunk are
repeated at the start of the next.

Text is split at the coarsest boundary that keeps a piece within the budget:
paragraph, then line, then sentence, then word, then a hard split by tokens
for a single unit that is larger than the budget.

The bound holds on the final, stripped text of every chunk: stripping can
change tokenization (``" differ"`` is one token, ``"differ"`` two), so a chunk
that exceeds the budget once stripped is split again. A single character that
needs more tokens than ``chunk_size`` raises ``ValueError``.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache

import tiktoken

logger = logging.getLogger(__name__)

_ENCODING_NAME = "cl100k_base"

# Boundary patterns, coarsest first. Each piece keeps its trailing separator so
# that joining pieces reproduces the input exactly.
_BOUNDARIES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\n\s*\n"),  # paragraph
    re.compile(r"\n"),  # line
    re.compile(r"(?<=[.!?…])\s+|(?<=[。！？])\s*"),  # sentence
    re.compile(r"\s+"),  # word
)

_DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]

SPLITTER_POLICY = "plain-tiktoken-cl100k/1"
"""Pipeline policy ID of this splitter. Bump it when chunk boundaries change."""

LEGACY_SPLITTER_POLICY = "plain-sk-char4/0"
"""Policy of the Semantic Kernel splitter used before 042: tokens counted as
characters divided by four, overlap ignored. Collections it wrote have no
pipeline fingerprint; see ``holodeck.lib.retrieval.guard``."""


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    """Load the tiktoken encoding once, on first use."""
    return tiktoken.get_encoding(_ENCODING_NAME)


def _encode(text: str) -> list[int]:
    tokens: list[int] = _encoding().encode(text, disallowed_special=())
    return tokens


def _count(text: str) -> int:
    return len(_encode(text))


def _split_keep(text: str, pattern: re.Pattern[str]) -> list[str]:
    """Split text after each match of pattern, keeping the separator."""
    pieces: list[str] = []
    last = 0
    for match in pattern.finditer(text):
        end = match.end()
        if end <= last or end >= len(text):
            continue
        pieces.append(text[last:end])
        last = end
    pieces.append(text[last:])
    return pieces


def _token_bytes(tokens: list[int]) -> bytes:
    enc = _encoding()
    return b"".join(enc.decode_single_token_bytes(t) for t in tokens)


def _hard_split(text: str, budget: int) -> list[str]:
    """Split text into pieces of at most budget tokens, on character boundaries."""
    tokens = _encode(text)
    pieces: list[str] = []
    i = 0
    while i < len(tokens):
        j = min(i + budget, len(tokens))
        piece = ""
        # Shrink until the slice is valid UTF-8 and re-encodes within budget.
        end = j
        while end > i:
            try:
                candidate = _token_bytes(tokens[i:end]).decode("utf-8")
            except UnicodeDecodeError:
                end -= 1
                continue
            if end == i + 1 or _count(candidate) <= budget:
                piece = candidate
                break
            end -= 1
        if not piece:
            # A single character spans more tokens than the budget: keep it whole.
            end = j
            while end < len(tokens):
                try:
                    piece = _token_bytes(tokens[i:end]).decode("utf-8")
                    break
                except UnicodeDecodeError:
                    end += 1
            if not piece:
                piece = _token_bytes(tokens[i:end]).decode("utf-8", errors="replace")
        pieces.append(piece)
        i = end
    return pieces


def _units(text: str, level: int, budget: int) -> list[tuple[str, int]]:
    """Break text into (piece, tokens) units of at most budget tokens."""
    tokens = _count(text)
    if tokens <= budget:
        return [(text, tokens)]
    if level >= len(_BOUNDARIES):
        return [(piece, _count(piece)) for piece in _hard_split(text, budget)]
    pieces = _split_keep(text, _BOUNDARIES[level])
    if len(pieces) == 1:
        return _units(text, level + 1, budget)
    units: list[tuple[str, int]] = []
    for piece in pieces:
        units.extend(_units(piece, level + 1, budget))
    return units


def _fit(text: str, limit: int) -> list[str]:
    """Split ``text`` into pieces whose stripped text is at most ``limit`` tokens.

    Stripping can change tokenization (``" differ"`` is one token, ``"differ"``
    two), so a unit within budget can exceed it once stripped. Such a unit is
    split again at a smaller budget until every stripped piece fits.

    Raises:
        ValueError: A single character needs more than ``limit`` tokens.
    """
    stripped = text.strip()
    if not stripped or _count(stripped) <= limit:
        return [text]
    budget = min(limit, _count(text))
    while True:
        parts = [piece for piece, _ in _units(text, 0, budget)]
        if len(parts) > 1:
            break
        budget -= 1
        if budget <= 0:
            raise ValueError(
                f"Text {stripped[:20]!r} cannot be split below "
                f"{_count(stripped)} tokens; it needs a chunk_size of at least "
                f"{_count(stripped)} (chunk_size is {limit})."
            )
    fitted: list[str] = []
    for part in parts:
        fitted.extend(_fit(part, limit))
    return fitted


def _tail(text: str, max_tokens: int) -> str:
    """Return the last max_tokens tokens of text as clean, stripped text."""
    if max_tokens <= 0:
        return ""
    tokens = _encode(text)
    raw = _token_bytes(tokens[-max_tokens:])
    start = 0
    while start < len(raw) and 0x80 <= raw[start] <= 0xBF:
        start += 1
    tail = raw[start:].decode("utf-8", errors="ignore").lstrip()
    while tail and _count(tail) > max_tokens:
        tail = tail[1:].lstrip()
    return tail


class TextChunker:
    """Split plain text into token-bounded, optionally overlapping chunks.

    Attributes:
        chunk_size: Maximum number of tokens in a chunk (cl100k_base)
        chunk_overlap: Tokens repeated from the end of the previous chunk
        separator_list: Boundary order used (fixed: paragraph, line, word)
    """

    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separator_list: list[str] | None = None,
    ) -> None:
        """Initialize text chunker.

        Args:
            chunk_size: Maximum tokens per chunk (default: 512)
            chunk_overlap: Tokens repeated at the start of the next chunk
                (default: 50)
            separator_list: Must be None or the default list. The splitter
                uses a fixed boundary order, so other values are rejected.

        Raises:
            ValueError: If chunk_size <= 0, chunk_overlap < 0,
                chunk_overlap >= chunk_size, or separator_list is not the
                default.
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be smaller than "
                f"chunk_size ({chunk_size})"
            )
        if separator_list is not None and separator_list != _DEFAULT_SEPARATORS:
            raise ValueError(
                "Custom separator_list is not supported; the splitter always "
                "splits at paragraph, line, sentence, word, then token boundaries"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator_list = list(_DEFAULT_SEPARATORS)

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks of at most chunk_size tokens.

        Args:
            text: Text to split into chunks

        Returns:
            Non-empty chunks without leading or trailing whitespace. Each chunk
            after the first starts with up to chunk_overlap tokens copied from
            the end of the previous chunk.

        Raises:
            ValueError: If text is empty, or a single character needs more
                tokens than chunk_size (the message names the minimum size).
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        units = _units(text, 0, self.chunk_size - self.chunk_overlap)
        chunks: list[str] = []
        prev = ""
        prev_trailing = ""
        pos = 0
        while pos < len(units):
            prefix = _tail(prev, self.chunk_overlap) if prev else ""
            lead = prefix + prev_trailing if prefix else ""
            budget = self.chunk_size - _count(lead) if lead else self.chunk_size
            end = pos
            used = 0
            while end < len(units) and (end == pos or used + units[end][1] <= budget):
                used += units[end][1]
                end += 1
            # Real token counts can differ from the per-unit sum at joins.
            while True:
                body = "".join(u for u, _ in units[pos:end])
                chunk = (lead + body).strip()
                if _count(chunk) <= self.chunk_size:
                    break
                if end - pos > 1:
                    end -= 1
                elif lead:
                    lead = ""
                else:
                    # One unit that exceeds the budget once stripped: split it
                    # further and assemble again from its first piece.
                    pieces = _fit(units[pos][0], self.chunk_size)
                    units[pos : pos + 1] = [(p, _count(p)) for p in pieces]
                    end = pos + 1
            raw = lead + "".join(u for u, _ in units[pos:end])
            pos = end
            if not chunk:
                continue
            chunks.append(chunk)
            prev = chunk
            prev_trailing = raw[len(raw.rstrip()) :]
        return chunks
