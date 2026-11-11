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


def _is_content(chunk: DocumentChunk) -> bool:
    """Return True for a chunk with content (the tool's header-only rule)."""
    return chunk.chunk_type != ChunkType.HEADER


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

    def _ref(self, chunk: DocumentChunk) -> ChunkRef:
        return ChunkRef(
            source_tool=self.name,
            chunk_id=chunk.id,
            source_uri=chunk.source_path,
            section_id=chunk.section_id,
            ordinal=chunk.chunk_index,
            text=strip_front_matter(chunk.content),
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

    async def chunks(self, document: SourceDocument) -> Sequence[ChunkRef]:
        """Return the content chunks of ``document`` in ascending ordinal.

        Header-only chunks, and chunks that are only front matter, are left
        out: they have no fact to extract.
        """
        self._require_initialized()
        stored = await self._stored_chunks()
        refs = [
            self._ref(chunk)
            for chunk in stored.get(document.source_uri, [])
            if _is_content(chunk)
        ]
        refs.sort(key=lambda ref: ref.ordinal)
        return [ref for ref in refs if ref.text]

    async def find(self, query: str, top_k: int) -> list[ChunkMatch]:
        """Return the tool's search hits in the tool's order and scores.

        It reads nothing but the search result, so a query never loads the
        corpus. A hit does not carry its position: its ordinal is
        ``UNKNOWN_ORDINAL``.

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
        return [
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
