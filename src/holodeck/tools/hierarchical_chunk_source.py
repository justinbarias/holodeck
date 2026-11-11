"""Adapter from ``HierarchicalDocumentTool`` to the memory's ``ChunkSource``.

Read-only. It reads stored chunks through ``document_chunks()`` and search
hits through ``search()``; it never starts ingestion. Text on both paths goes
through ``strip_front_matter``, so a chunk has the same text whether it was
listed or found.

With ``preview`` it reads the source files through
``preview_document_chunks()`` and needs no initialized tool: no embedding and
no store call. A dry run uses it (ING-08).
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from pathlib import Path

from holodeck.lib.memory.chunk_source import (
    UNKNOWN_ORDINAL,
    ChunkMatch,
    ChunkRef,
    SourceDocument,
)
from holodeck.lib.memory.effective_date import strip_front_matter
from holodeck.lib.structured_chunker import ChunkType, DocumentChunk
from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool

_HEADING_LINE = re.compile(r"^#{1,6}\s+(.*\S)")
_SECTION_NUMBER = re.compile(r"(\d+(?:\.\d+)*[A-Za-z]*)(?=$|[\s.:)\u2013\u2014-])")


def _is_content(chunk: DocumentChunk) -> bool:
    """Return True for a chunk with content (the tool's header-only rule)."""
    return chunk.chunk_type != ChunkType.HEADER


def section_number(heading: str) -> str:
    """Return the leading number of a heading, or "" when it has none.

    "57GJ Benefit restriction notice" gives "57GJ", "3A Applicable
    percentage" gives "3A" and "1.2 Scope" gives "1.2". "Part 2" gives "":
    the number must start the heading.
    """
    match = _SECTION_NUMBER.match(heading.strip())
    return match.group(1) if match else ""


def _own_heading(first: DocumentChunk) -> tuple[str, tuple[str, ...]]:
    """Return a section's own heading and the headings above it.

    ``first`` is the section's first stored chunk. The store keeps no
    heading level, so the heading is read from the content: the chunker
    starts a section's first chunk with its heading line. A chunk with no
    heading line and a parent chain is a body the chunker split from its
    heading (subsection patterns); the last entry of the chain is then its
    own heading.
    """
    first_line = strip_front_matter(first.content).lstrip().split("\n", 1)[0]
    match = _HEADING_LINE.match(first_line)
    if match:
        return match.group(1), tuple(first.parent_chain)
    if first.parent_chain:
        return first.parent_chain[-1], tuple(first.parent_chain[:-1])
    return "", ()


def _sections(
    chunks: Sequence[DocumentChunk],
) -> dict[str, tuple[str, tuple[str, ...]]]:
    """Map each section id of one document to its number and path."""
    firsts: dict[str, DocumentChunk] = {}
    for chunk in sorted(chunks, key=lambda c: c.chunk_index):
        if chunk.section_id:
            firsts.setdefault(chunk.section_id, chunk)
    sections: dict[str, tuple[str, tuple[str, ...]]] = {}
    for section_id, first in firsts.items():
        heading, path = _own_heading(first)
        sections[section_id] = (section_number(heading), path)
    return sections


class HierarchicalChunkSource:
    """``ChunkSource`` over an initialized ``HierarchicalDocumentTool``.

    Stored chunks are read once and kept until ``documents()`` is called
    again, so listing every document does not scan the store once per
    document.

    Args:
        tool: The wrapped tool. Initialized, unless ``preview`` is set.
        preview: List the chunks an ingestion would store, read from the
            source files. ``find`` is not available.
    """

    def __init__(
        self, tool: HierarchicalDocumentTool, *, preview: bool = False
    ) -> None:
        self._tool = tool
        self._preview = preview
        self._stored: dict[str, list[DocumentChunk]] | None = None

    @property
    def name(self) -> str:
        """Name of the wrapped tool."""
        return self._tool.config.name

    def _require_initialized(self) -> None:
        if self._preview:
            return
        if not self._tool._initialized:
            from holodeck.lib.backends.base import BackendSessionError

            raise BackendSessionError(f"Tool '{self.name}' is not initialized")

    async def _load(self) -> dict[str, list[DocumentChunk]]:
        if self._preview:
            self._stored = await self._tool.preview_document_chunks()
        else:
            self._stored = await self._tool.document_chunks()
        return self._stored

    async def _stored_chunks(self) -> dict[str, list[DocumentChunk]]:
        if self._stored is None:
            return await self._load()
        return self._stored

    def _path_of(self, source_uri: str) -> Path:
        """Return the local file of a source key."""
        root = self._tool._source_root
        if self._tool._is_remote and root is not None:
            return root / source_uri
        return Path(source_uri)

    def _ref(
        self,
        chunk: DocumentChunk,
        sections: dict[str, tuple[str, tuple[str, ...]]],
    ) -> ChunkRef:
        number, path = sections.get(chunk.section_id, ("", ()))
        return ChunkRef(
            source_tool=self.name,
            chunk_id=chunk.id,
            source_uri=chunk.source_path,
            section_id=chunk.section_id,
            ordinal=chunk.chunk_index,
            text=strip_front_matter(chunk.content),
            section_number=number,
            section_path=path,
        )

    async def documents(self) -> Sequence[SourceDocument]:
        """Return every stored document, ordered by source key."""
        self._require_initialized()
        stored = await self._load()
        return [
            SourceDocument(
                source_tool=self.name,
                source_uri=source_uri,
                path=self._path_of(source_uri),
                mtime_reliable=not self._tool._is_remote,
            )
            for source_uri in sorted(stored)
        ]

    async def document(self, source_uri: str) -> SourceDocument | None:
        """Return the document for a source key, or None when its file is gone.

        The path comes from the tool's own mapping, so no store is read. The
        file check runs in a thread.
        """
        path = self._path_of(source_uri)
        if not await asyncio.to_thread(path.is_file):
            return None
        return SourceDocument(
            source_tool=self.name,
            source_uri=source_uri,
            path=path,
            mtime_reliable=not self._tool._is_remote,
        )

    async def chunks(self, document: SourceDocument) -> Sequence[ChunkRef]:
        """Return the content chunks of ``document`` in ascending ordinal.

        Header-only chunks, and chunks that are only front matter, are left
        out: they have no fact to extract. Each chunk carries its section's
        number and path, read from the section's first stored chunk.
        """
        self._require_initialized()
        stored = await self._stored_chunks()
        chunks = stored.get(document.source_uri, [])
        sections = _sections(chunks)
        refs = [self._ref(chunk, sections) for chunk in chunks if _is_content(chunk)]
        refs.sort(key=lambda ref: ref.ordinal)
        return [ref for ref in refs if ref.text.strip()]

    async def find(self, query: str, top_k: int) -> list[ChunkMatch]:
        """Return the tool's search hits in the tool's order and scores.

        It reads nothing but the search result, so a query never loads the
        corpus. A hit does not carry its position: its ordinal is
        ``UNKNOWN_ORDINAL``. A hit whose text is empty or whitespace once
        front matter is removed is left out, so fewer than ``top_k`` hits may
        come back.

        Raises:
            BackendSessionError: The tool is not initialized, or the source
                is a preview.
        """
        if self._preview:
            from holodeck.lib.backends.base import BackendSessionError

            raise BackendSessionError(
                f"Tool '{self.name}' is a preview source and cannot search"
            )
        self._require_initialized()
        results = await self._tool.search(query, top_k)
        matches = [
            ChunkMatch(
                chunk=ChunkRef(
                    source_tool=self.name,
                    chunk_id=result.chunk_id,
                    source_uri=result.source_path,
                    section_id=result.section_id,
                    ordinal=UNKNOWN_ORDINAL,
                    text=strip_front_matter(result.content),
                ),
                score=result.fused_score,
            )
            for result in results
        ]
        return [match for match in matches if match.chunk.text.strip()]
