"""The read-only seam between a source tool and the temporal memory (spec 043).

The memory reads documents and chunks for ingestion, and ranked chunks for
retrieval, through ``ChunkSource``. This module holds the protocol and its
three types only. It does not import ``holodeck.tools``: adapters live with
the tools they wrap.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

UNKNOWN_ORDINAL = -1

# The id the structured chunker gives a chunk with no heading of its own:
# ``chunk_<index>`` (``StructuredChunker._generate_section_id``). The
# subsection-pattern path gives one to every body it splits from its heading.
# It names a position in one document, not a section.
_POSITIONAL_SECTION_ID = re.compile(r"chunk_\d+")


def is_positional_section_id(section_id: str) -> bool:
    """True for a section id that is only a chunk position (``chunk_<n>``).

    Two equal positional ids in two documents, or in two versions of one
    document, say nothing about the section: the memory treats such an id
    as no section id and compares the section number and path instead.
    """
    return _POSITIONAL_SECTION_ID.fullmatch(section_id) is not None


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """One document a source tool has stored."""

    source_tool: str
    source_uri: str
    """Stable key for the document (the tool's source key)."""
    path: Path
    """Local file, for front matter and mtime."""
    mtime_reliable: bool
    """False for a downloaded remote source, whose mtime is the download time."""


@dataclass(frozen=True, slots=True)
class ChunkRef:
    """One stored chunk of a document."""

    source_tool: str
    chunk_id: str
    """The id the tool's store uses."""
    source_uri: str
    section_id: str
    ordinal: int
    """Position of the chunk in its document, or ``UNKNOWN_ORDINAL`` for a
    search hit, where the tool does not report it."""
    text: str
    """Raw chunk content with front matter removed."""
    section_number: str = ""
    """Leading number of the section's own heading ("57GJ", "3A"); empty when
    the heading has none or the tool does not report it."""
    section_path: tuple[str, ...] = ()
    """Headings above the section, outermost first; empty when unknown."""


@dataclass(frozen=True, slots=True)
class ChunkMatch:
    """A chunk returned by a source tool's search."""

    chunk: ChunkRef
    score: float
    """Higher is better. Use for order only."""


@runtime_checkable
class ChunkSource(Protocol):
    """Read-only access to a source tool's documents, chunks and search."""

    @property
    def name(self) -> str:
        """Name of the source tool."""
        ...

    async def documents(self) -> Sequence[SourceDocument]:
        """Return every document the tool has stored."""
        ...

    async def document(self, source_uri: str) -> SourceDocument | None:
        """Return the document stored under ``source_uri``, or None.

        It does not read the store: a search calls it for each hit's document
        to learn the document's date, and a store listing there would scan
        the whole corpus. None means the source file no longer exists.
        """
        ...

    async def chunks(self, document: SourceDocument) -> Sequence[ChunkRef]:
        """Return the chunks of ``document`` in ascending ordinal."""
        ...

    async def find(self, query: str, top_k: int) -> list[ChunkMatch]:
        """Return the tool's ranked chunks for ``query``."""
        ...
