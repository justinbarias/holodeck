"""HierarchicalDocumentTool for intelligent document retrieval.

This module provides the main tool that orchestrates all components for
hierarchical document processing and retrieval. It integrates structured
chunking, hybrid search, definition extraction, and context generation
into a unified Semantic Kernel tool.

Key Features:
- Complete document processing pipeline
- YAML-configurable tool parameters
- Integration with Semantic Kernel agent framework
- Support for multiple document formats
- Automatic definition and cross-reference handling
- Hybrid search with configurable weights

Usage:
    In agent.yaml:
        tools:
          - name: document_search
            type: hierarchical_document
            source: ./docs/legislation.md
            search_mode: hybrid
            semantic_weight: 0.5
            keyword_weight: 0.3

    Programmatic:
        from holodeck.tools.hierarchical_document_tool import HierarchicalDocumentTool
        from holodeck.models.tool import HierarchicalDocumentToolConfig

        config = HierarchicalDocumentToolConfig(
            name="doc_search",
            description="Search legal docs",
            source="./docs/policy.md"
        )
        tool = HierarchicalDocumentTool(config)
        await tool.initialize()
        results = await tool.search(query)

Classes:
    HierarchicalDocumentTool: Main tool class for Semantic Kernel integration.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from holodeck.lib.keyword_search import HybridSearchExecutor
from holodeck.lib.retrieval import (
    Operation,
    Record,
    RecordFilter,
    UnsupportedOperationError,
    VectorQuery,
    VectorStoreAdapter,
    clamp_similarity,
)
from holodeck.lib.retrieval.schemas import (
    chunk_from_record,
    chunk_to_record,
    float_field,
    hierarchical_schema,
    text_field,
)

if TYPE_CHECKING:
    from holodeck.lib.backends.base import ContextGenerator
    from holodeck.lib.definition_extractor import DefinitionEntry
    from holodeck.lib.file_processor import FileProcessor
    from holodeck.lib.structured_chunker import (
        DocumentChunk,
        StructuredChunker,
        SubsectionPattern,
    )
    from holodeck.models.config import ExecutionConfig

from holodeck.lib.hybrid_search import SearchResult
from holodeck.lib.structured_chunker import DOMAIN_PATTERNS, ChunkType
from holodeck.models.tool import (
    DocumentDomain,
    HierarchicalDocumentToolConfig,
    SearchMode,
)
from holodeck.tools.base_tool import DatabaseConfigMixin, EmbeddingServiceMixin
from holodeck.tools.common import (
    SUPPORTED_EXTENSIONS,
    discover_files,
    generate_placeholder_embeddings,
    get_file_type,
    resolve_source_path,
)

logger = logging.getLogger(__name__)

# Operations ingestion and search use. Incremental ingestion looks records up
# and deletes them by source_path, so FIND is required: without it a changed
# file would be re-ingested on every start and its stale chunks never removed.
# Checked before ingestion starts (FR-014).
_REQUIRED_OPERATIONS = (
    Operation.UPSERT,
    Operation.DELETE,
    Operation.FIND,
    Operation.VECTOR_SEARCH,
)


class HierarchicalDocumentTool(EmbeddingServiceMixin, DatabaseConfigMixin):
    """Semantic Kernel tool for hierarchical document retrieval.

    This tool provides intelligent document search that understands
    document structure, extracts definitions, and generates optimized
    context for LLM consumption.

    Inherits from:
        EmbeddingServiceMixin: Provides set_embedding_service() method
        DatabaseConfigMixin: Provides database config resolution and collection creation

    Attributes:
        config: Tool configuration from HierarchicalDocumentToolConfig.
        chunks: Indexed document chunks.

    Example:
        >>> from holodeck.models.tool import HierarchicalDocumentToolConfig
        >>> config = HierarchicalDocumentToolConfig(
        ...     name="doc_search",
        ...     description="Search policy documents",
        ...     source="./docs/policy.md"
        ... )
        >>> tool = HierarchicalDocumentTool(config)
        >>> await tool.initialize()
        >>> results = await tool.search("What are the reporting requirements?")
    """

    def __init__(
        self,
        config: HierarchicalDocumentToolConfig,
        base_dir: str | None = None,
        execution_config: ExecutionConfig | None = None,
    ) -> None:
        """Initialize the hierarchical document tool.

        Args:
            config: Tool configuration from HierarchicalDocumentToolConfig.
            base_dir: Optional base directory for resolving relative source paths.
                If None, source paths are resolved relative to current working
                directory or agent_base_dir context variable.
            execution_config: Execution configuration for file processing
                timeouts and caching. When provided, the lazy FileProcessor
                used by ``_convert_to_markdown`` is built via
                ``FileProcessor.from_execution_config`` so that
                ``execution.file_timeout`` (and download_timeout / cache_dir)
                from agent.yaml are honored. When None, FileProcessor's
                hardcoded defaults (30s timeouts) apply.
        """
        self.config = config
        self._base_dir = base_dir
        self._execution_config = execution_config
        self._chunker: StructuredChunker | None = None
        self._searcher: Any = None  # HybridSearcher (future)
        self._context_generator: ContextGenerator | None = None
        self._glossary: dict[str, DefinitionEntry] | None = None
        self._chunks: list[DocumentChunk] = []
        self._initialized = False

        # Service injection attributes
        self._embedding_service: Any = None
        self._adapter: VectorStoreAdapter | None = None
        self._provider: str = "in-memory"
        self._embedding_dimensions: int | None = None

        # Lazily constructed FileProcessor (see _get_file_processor)
        self._file_processor: FileProcessor | None = None

        # Hybrid search executor (initialized during ingestion)
        self._hybrid_executor: HybridSearchExecutor | None = None

        # Source context for stable record keys (remote sources)
        self._source_root: Path | None = None
        self._is_remote: bool = False

    def _get_file_processor(self) -> FileProcessor:
        """Get or create the FileProcessor instance (lazy init).

        Mirrors VectorStoreTool._get_file_processor: when an
        ``execution_config`` was supplied at construction, build via the
        ``from_execution_config`` factory so per-agent timeout/cache
        settings reach the processor; otherwise fall back to defaults.
        """
        from holodeck.lib.file_processor import FileProcessor

        if self._file_processor is None:
            if self._execution_config is not None:
                self._file_processor = FileProcessor.from_execution_config(
                    self._execution_config
                )
            else:
                self._file_processor = FileProcessor()
        return self._file_processor

    # set_embedding_service is inherited from EmbeddingServiceMixin

    def set_context_generator(self, generator: Any) -> None:
        """Set the context generator for contextual embeddings.

        Accepts any ContextGenerator protocol implementation
        (LLMContextGenerator, ClaudeSDKContextGenerator, etc.).

        Args:
            generator: A ContextGenerator protocol implementation.
        """
        self._context_generator = generator

    def set_source_context(self, source_root: Path, is_remote: bool) -> None:
        """Set source context for stable record keys.

        Called by tool_initializer when a remote source is resolved.
        Enables stable keys (relative paths) and content-hash-based
        change detection for remote sources.

        Args:
            source_root: Root directory of the resolved source.
            is_remote: Whether the source was resolved from a remote URI.
        """
        self._source_root = source_root
        self._is_remote = is_remote

    def _compute_source_key(self, file_path: Path) -> str:
        """Compute a stable source key for a file.

        For remote sources, returns the path relative to source_root
        (stable across temp dirs). For local sources, returns the
        absolute path string (backward compatible with existing records).

        Args:
            file_path: Absolute path to the file.

        Returns:
            Stable string key for record storage and lookup.
        """
        if self._is_remote and self._source_root is not None:
            return str(file_path.relative_to(self._source_root))
        return str(file_path)

    def _compute_content_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file content for change detection.

        Only used for remote sources where mtime is unreliable (files are
        freshly downloaded each time). Returns empty string for local
        sources which use mtime instead.

        Args:
            file_path: Path to the file to hash.

        Returns:
            SHA-256 hex digest for remote sources, empty string for local.
        """
        if not self._is_remote:
            return ""
        import hashlib

        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                h.update(block)
        return h.hexdigest()

    @staticmethod
    def _build_searchable_text(chunk: DocumentChunk) -> str:
        """Build the concatenated text indexed for native hybrid keyword search.

        Native hybrid search consults one full-text field, so we fold every
        text-bearing field into a single ``searchable_text`` string and let
        the store's full-text index handle the lexical pass.

        The boost recipe mirrors :func:`_build_bm25_document` so the
        BM25-fallback path and the qdrant-native path index the same shape:

        * contextualized_content (or content fallback) — 1× (the body)
        * parent_chain — 2× (heading-path queries)
        * section_id — 2× (section lookups)
        * defined_term — 3× (definition queries)
        * cross_references — 1×
        * source filename — 1×

        ``content`` is intentionally omitted: ``contextualized_content``
        already contains it verbatim when context-gen is on, and equals it
        when off (see ``_ingest_documents`` line 710).
        """
        primary = chunk.contextualized_content or chunk.content
        parent_chain = " > ".join(chunk.parent_chain)
        cross_refs = " ".join(chunk.cross_references)
        source_file = Path(chunk.source_path).name if chunk.source_path else ""

        parts: list[str] = []
        if primary:
            parts.append(primary)
        if parent_chain:
            parts.extend([parent_chain, parent_chain])  # 2×
        if chunk.section_id:
            parts.extend([chunk.section_id, chunk.section_id])  # 2×
        if chunk.defined_term:
            parts.extend([chunk.defined_term] * 3)  # 3×
        if cross_refs:
            parts.append(cross_refs)
        if source_file:
            parts.append(source_file)

        return "\n\n".join(parts)

    def _coerce_chunk_ids_to_uuid(self, chunks: list[DocumentChunk]) -> None:
        """Rewrite chunk IDs to deterministic UUIDv5 for qdrant compatibility.

        Qdrant point IDs must be UUIDs or unsigned 64-bit ints; the chunker
        emits human-readable IDs like ``{path}_chunk_{N}``. UUIDv5 over
        (tool-name, original-id) is stable across runs, so re-ingestion of
        unchanged content lands on the same point.

        Mutates ``chunks`` in place.
        """
        import uuid

        namespace = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"holodeck/hierarchical_document/{self.config.name}",
        )
        for chunk in chunks:
            chunk.id = str(uuid.uuid5(namespace, chunk.id))

    def _remap_chunk_keys(self, chunks: list[DocumentChunk], file_path: Path) -> None:
        """Remap chunk source_path and id to use stable source keys.

        For remote sources, rewrites the source_path from absolute temp
        dir path to a relative path, and updates chunk IDs accordingly.
        For local sources, this is a no-op.

        Args:
            chunks: List of DocumentChunks to remap (modified in-place).
            file_path: Absolute path to the source file.
        """
        source_key = self._compute_source_key(file_path)
        original_source = str(file_path)

        if source_key == original_source:
            return  # No remapping needed (local source)

        # Compute the ID prefix replacements
        old_prefix = original_source.replace("/", "_").replace("\\", "_")
        new_prefix = source_key.replace("/", "_").replace("\\", "_")

        for chunk in chunks:
            chunk.source_path = source_key
            chunk.id = chunk.id.replace(old_prefix, new_prefix, 1)

    def _setup_collection(self, provider_type: str = "openai") -> None:
        """Set up the vector store adapter for chunk storage.

        Uses DatabaseConfigMixin methods for config resolution and adapter
        creation with automatic fallback to in-memory storage, then checks
        that the provider supports every operation ingestion needs.

        Args:
            provider_type: LLM provider type for dimension resolution
                (e.g., "openai", "azure_openai", "ollama").

        Raises:
            UnsupportedOperationError: The provider lacks an operation that
                ingestion or search needs. Raised before any ingestion.
        """
        # Resolve embedding dimensions if not set
        if self._embedding_dimensions is None:
            from holodeck.config.defaults import get_embedding_dimensions

            self._embedding_dimensions = get_embedding_dimensions(
                model_name=None, provider=provider_type
            )

        # Use mixin to resolve database config
        provider, connection_kwargs = self._resolve_database_config(
            self.config.database
        )

        schema = hierarchical_schema(
            self._embedding_dimensions, self.config.name, connection_kwargs
        )

        # Use mixin to create the adapter with fallback
        self._adapter = self._create_adapter_with_fallback(
            provider=provider,
            schema=schema,
            connection_kwargs=connection_kwargs,
        )
        self._adapter.capabilities.require(*_REQUIRED_OPERATIONS)

        logger.debug(
            f"Collection setup: provider={self._provider}, "
            f"dimensions={self._embedding_dimensions}"
        )

    def _require_adapter(self) -> VectorStoreAdapter:
        """Return the adapter or raise when the collection is not set up."""
        if self._adapter is None:
            raise RuntimeError("Collection not initialized")
        return self._adapter

    def _uses_native_hybrid(self) -> bool:
        """Return True when the store runs the keyword leg itself."""
        return self._adapter is not None and self._adapter.capabilities.supports(
            Operation.HYBRID_SEARCH
        )

    def _resolve_source_path(self) -> Path:
        """Resolve the source path relative to base directory.

        Delegates to common.resolve_source_path utility.

        Returns:
            Resolved absolute Path to the source.
        """
        return resolve_source_path(self.config.source, self._base_dir)

    def _discover_files(self) -> list[Path]:
        """Discover files to ingest from configured source.

        Delegates to common.discover_files utility.

        Returns:
            List of Path objects for files to process.
        """
        source_path = self._resolve_source_path()
        return discover_files(source_path)

    async def _needs_reingest(self, file_path: Path) -> bool:
        """Check if file needs re-ingestion based on modification time.

        Compares file's current mtime against stored record mtime.

        Args:
            file_path: Path to file to check.

        Returns:
            True if file needs re-ingestion (modified or not in store).
            False if file is up-to-date.
        """
        if self._adapter is None:
            return True  # No collection, must ingest

        source_key = self._compute_source_key(file_path)

        try:
            records = await self._adapter.find(
                RecordFilter.where(source_path=source_key), limit=1
            )

            if not records:
                return True  # Not in store, must ingest

            if self._is_remote:
                # For remote sources, compare content hash
                current_hash = self._compute_content_hash(file_path)
                stored_hash = text_field(records[0], "content_hash")
                if not stored_hash:
                    return True  # No hash stored (old record)
                return current_hash != stored_hash
            # For local sources, compare mtime (backward compat)
            current_mtime = file_path.stat().st_mtime
            stored_mtime = float_field(records[0], "mtime")
            return current_mtime - stored_mtime > 0.001

        except UnsupportedOperationError:
            raise
        except Exception as e:
            logger.debug(f"Could not retrieve record for {file_path}: {e}")
            return True  # Error = must ingest

    async def _delete_file_records(self, file_path: Path) -> int:
        """Delete all records for a source file from vector store.

        Args:
            file_path: Path to source file whose records should be deleted.

        Returns:
            Number of records deleted.
        """
        if self._adapter is None:
            return 0

        source_key = self._compute_source_key(file_path)
        deleted_count = 0

        try:
            deleted_count = await self._adapter.delete_where(
                RecordFilter.where(source_path=source_key)
            )
        except UnsupportedOperationError:
            raise
        except Exception as e:
            logger.debug(f"Error deleting records for {file_path}: {e}")

        if deleted_count > 0:
            logger.debug(f"Deleted {deleted_count} records for {file_path}")

        return deleted_count

    @staticmethod
    def _chunk_from_record(record: Record) -> DocumentChunk:
        """Reconstruct a DocumentChunk from a stored record.

        Shared by the full-corpus reload path and the native-hybrid search
        path. See :func:`holodeck.lib.retrieval.schemas.chunk_from_record`.
        """
        return chunk_from_record(record)

    async def _load_chunks_from_store(self) -> list[DocumentChunk]:
        """Load all document chunks from the vector store.

        Reconstructs DocumentChunk objects from stored records, enabling
        BM25 index rebuild on startup when files are unchanged (skipped
        via mtime check). This is necessary because the BM25 index is
        in-memory only and does not survive restarts. Native-hybrid
        providers skip this path entirely — see ``initialize``.

        Returns:
            List of reconstructed DocumentChunk objects, or empty list
            if collection doesn't exist or an error occurs.
        """
        if self._adapter is None:
            return []

        try:
            if not await self._adapter.collection_exists():
                return []

            records = await self._adapter.find(RecordFilter())

            if not records:
                return []

            chunks = [chunk_from_record(r) for r in records]
            logger.info(f"Loaded {len(chunks)} chunks from vector store")
            return chunks

        except UnsupportedOperationError:
            raise
        except Exception as e:
            logger.warning(f"Failed to load chunks from store: {e}")
            return []

    async def _convert_to_markdown(self, file_path: str) -> str:
        """Convert a file to markdown using FileProcessor.

        Args:
            file_path: Path to the file to convert.

        Returns:
            Markdown content string.
        """
        from holodeck.models.test_case import FileInput

        path = Path(file_path)

        file_input = FileInput(
            path=str(path),
            url=None,
            type=get_file_type(path),
            description=None,
            pages=None,
            sheet=None,
            range=None,
            cache=None,
        )

        processor = self._get_file_processor()
        processed = processor.process_file(file_input)

        if processed.error:
            logger.warning(f"Error processing file {file_path}: {processed.error}")
            # Fall back to reading file directly for text files
            if path.suffix.lower() in (".txt", ".md"):
                return path.read_text()
            return ""

        return processed.markdown_content

    def _get_subsection_patterns(self) -> list[SubsectionPattern] | None:
        """Get subsection patterns based on document_domain configuration.

        Returns:
            List of SubsectionPattern objects for the configured domain,
            or None if document_domain is 'none' (default).
        """
        if self.config.document_domain == DocumentDomain.NONE:
            return None

        domain_key = self.config.document_domain.value
        return DOMAIN_PATTERNS.get(domain_key)

    async def _ingest_documents(
        self,
        force_ingest: bool = False,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> int:
        """Ingest all configured documents through the ingestion pipeline.

        Pipeline stages:
        1. Document Discovery - resolve source path and discover files
        2. Incremental Check - skip unchanged files (unless force_ingest)
        3. Conversion - markitdown via FileProcessor
        4. Structure Parsing - StructuredChunker.parse()
        5. LLM Context Generation - LLMContextGenerator.contextualize_batch()
        6. Embedding Generation - self._embed_chunks()
        7. Index Construction - semantic only for MVP
        8. Persistence - self._store_chunks()

        Args:
            force_ingest: If True, re-ingest all files regardless of modification
                time. Existing records will be deleted before re-ingestion.
            progress_callback: Optional callback invoked after each file is
                processed (or skipped). Called as ``callback(current, total)``.

        Returns:
            Number of files skipped (unchanged since last ingestion).

        Raises:
            FileNotFoundError: If source path does not exist.
            RuntimeError: If no supported files found in source.
        """
        from holodeck.lib.structured_chunker import StructuredChunker

        # Initialize chunker if needed
        if self._chunker is None:
            # Get subsection patterns based on document domain
            subsection_patterns = self._get_subsection_patterns()

            self._chunker = StructuredChunker(
                max_tokens=self.config.max_chunk_tokens,
                subsection_patterns=subsection_patterns,
                max_subsection_depth=self.config.max_subsection_depth,
            )

        # 1. Resolve source path (handles absolute/relative with base_dir)
        source_path = self._resolve_source_path()

        if not source_path.exists():
            # Include resolution info in error message
            raise FileNotFoundError(
                f"Source path does not exist: {source_path} "
                f"(configured as: {self.config.source})"
            )

        # Discover files (handles single file or directory with extension filter)
        files = self._discover_files()

        if not files:
            logger.warning(
                f"No supported files found in source: {self.config.source}. "
                f"Supported extensions: {SUPPORTED_EXTENSIONS}"
            )
            # Still mark as initialized even with no files
            return 0

        logger.info(f"Discovered {len(files)} files for ingestion")

        skipped_files = 0
        ingested_files = 0
        processed_count = 0
        total_files = len(files)

        for file_path in files:
            # Check if file needs re-ingestion (unless force_ingest)
            if not force_ingest:
                needs_reingest = await self._needs_reingest(file_path)
                if not needs_reingest:
                    logger.debug(f"Skipping unchanged file: {file_path}")
                    skipped_files += 1
                    processed_count += 1
                    if progress_callback is not None:
                        progress_callback(processed_count, total_files)
                    continue
            else:
                # Force ingest: delete existing records first
                await self._delete_file_records(file_path)

            # 1-2. Convert to markdown
            markdown_content = await self._convert_to_markdown(str(file_path))
            if not markdown_content.strip():
                logger.warning(f"Empty content from {file_path}, skipping")
                processed_count += 1
                if progress_callback is not None:
                    progress_callback(processed_count, total_files)
                continue

            mtime = file_path.stat().st_mtime

            # 3. Parse structure
            chunks = self._chunker.parse(markdown_content, str(file_path), mtime)

            # Remap chunk keys for stable record IDs (remote sources)
            self._remap_chunk_keys(chunks, file_path)

            # Qdrant rejects arbitrary-string point IDs (only UUID/uint64).
            # Coerce human-readable chunk IDs to deterministic UUIDv5 so
            # re-ingestion stays idempotent and the chunk_map (built later
            # from these same chunks) matches what qdrant returns at search.
            # Collections written before the adapter contract hold exactly
            # these point IDs, so the mapping stays in the tool.
            if self._provider == "qdrant":
                self._coerce_chunk_ids_to_uuid(chunks)

            # 3.5 Filter out header-only chunks (no substantive content)
            chunks = [c for c in chunks if c.chunk_type != ChunkType.HEADER]

            if not chunks:
                logger.debug(
                    f"No content chunks after filtering headers from {file_path}"
                )
                processed_count += 1
                if progress_callback is not None:
                    progress_callback(processed_count, total_files)
                continue

            # 4. Context generation (if enabled and generator available)
            if (
                self.config.contextual_embeddings
                and self._context_generator is not None
            ):
                logger.info(
                    f"Using contextual embeddings for {file_path} "
                    f"({len(chunks)} chunks)"
                )
                contextualized = await self._context_generator.contextualize_batch(
                    chunks, document_text=markdown_content
                )
                # Update chunks with contextualized content
                for chunk, ctx_content in zip(chunks, contextualized, strict=False):
                    chunk.contextualized_content = ctx_content
            else:
                # Fall back to original content
                if not self.config.contextual_embeddings:
                    reason = "disabled in config"
                else:
                    reason = "no context generator configured"
                logger.info(
                    f"Skipping contextual embeddings for {file_path} ({reason})"
                )
                for chunk in chunks:
                    chunk.contextualized_content = chunk.content

            # 5-6. Embed chunks
            await self._embed_chunks(chunks)

            # 7. Store
            content_hash = self._compute_content_hash(file_path)
            await self._store_chunks(chunks, content_hash=content_hash)
            self._chunks.extend(chunks)
            ingested_files += 1

            processed_count += 1
            if progress_callback is not None:
                progress_callback(processed_count, total_files)

            logger.debug(f"Ingested {len(chunks)} chunks from {file_path}")

        logger.info(
            f"Ingestion complete: {ingested_files} files processed, "
            f"{skipped_files} skipped (up-to-date), {len(self._chunks)} total chunks"
        )

        # Build hybrid search indices if we have chunks
        if self._chunks:
            await self._build_hybrid_indices()

        return skipped_files

    async def _embed_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Generate embeddings for document chunks.

        Uses contextualized_content for embedding when available,
        falls back to original content.

        Args:
            chunks: List of DocumentChunks to embed.
        """
        if not chunks:
            return

        # Use contextualized content for embedding
        texts = [
            c.contextualized_content if c.contextualized_content else c.content
            for c in chunks
        ]

        if self._embedding_service is not None:
            # Provider failures propagate (EmbeddingServiceError from the
            # LiteLLM shim); never store placeholder vectors for real data.
            embeddings = await self._embedding_service.generate_embeddings(texts)
            for chunk, emb in zip(chunks, embeddings, strict=False):
                chunk.embedding = list(emb)
            logger.debug(f"Generated {len(chunks)} embeddings")
            return

        # No service injected: placeholder embeddings
        dims = self._embedding_dimensions or 1536
        placeholders = generate_placeholder_embeddings(len(chunks), dims)
        for chunk, emb in zip(chunks, placeholders, strict=False):
            chunk.embedding = emb

    async def _store_chunks(
        self, chunks: list[DocumentChunk], content_hash: str = ""
    ) -> int:
        """Store document chunks in the vector store.

        The chunks of each source file replace that file's stored chunks, so
        chunks left over from a longer earlier version are removed.

        Args:
            chunks: List of DocumentChunks to store.
            content_hash: SHA-256 content hash for remote source change
                detection. Empty string for local sources.

        Returns:
            Number of chunks stored.

        Raises:
            RuntimeError: If collection is not initialized.
        """
        adapter = self._require_adapter()

        by_source: dict[str, list[Record]] = {}
        for chunk in chunks:
            by_source.setdefault(chunk.source_path, []).append(
                chunk_to_record(
                    chunk,
                    searchable_text=self._build_searchable_text(chunk),
                    content_hash=content_hash,
                )
            )

        await adapter.ensure_collection()
        for source_path, records in by_source.items():
            await adapter.replace_source(
                RecordFilter.where(source_path=source_path), records
            )

        logger.debug(f"Stored {len(chunks)} chunks")
        return len(chunks)

    async def _embed_query(self, query: str) -> list[float]:
        """Generate embedding for a query string.

        Args:
            query: Query text to embed.

        Returns:
            Query embedding vector.

        Raises:
            Exception: Any error raised by the embedding service; a failed
                provider call surfaces as the tool error rather than
                searching with a zero vector.
        """
        if self._embedding_service is not None:
            embeddings = await self._embedding_service.generate_embeddings([query])
            return list(embeddings[0])

        # No service injected: placeholder embedding
        dims = self._embedding_dimensions or 1536
        return [0.0] * dims

    async def _semantic_search(
        self, query_embedding: list[float], top_k: int
    ) -> list[SearchResult]:
        """Perform semantic-only vector search.

        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects sorted by score.
        """
        adapter = self._require_adapter()
        hits = await adapter.vector_search(
            VectorQuery(vector=query_embedding, top_k=top_k)
        )

        results: list[SearchResult] = []
        for hit in hits:
            chunk = chunk_from_record(hit.record)
            results.append(
                SearchResult(
                    chunk_id=chunk.id,
                    content=chunk.content,
                    fused_score=clamp_similarity(hit.score),
                    source_path=chunk.source_path,
                    parent_chain=chunk.parent_chain,
                    section_id=chunk.section_id,
                    subsection_ids=chunk.subsection_ids,
                    semantic_score=hit.score,
                    keyword_score=None,  # Semantic-only mode
                    exact_match=False,
                    definitions_context=[],
                )
            )

        # Sort by score descending
        results.sort(key=lambda r: r.fused_score, reverse=True)
        return results

    def _setup_hybrid_executor(self) -> None:
        """Create the HybridSearchExecutor (idempotent).

        Splits executor construction from keyword-index building so
        native-hybrid providers (e.g. qdrant) can route search through
        the executor without paying for an in-process BM25/OpenSearch
        index build. Safe to call multiple times — second call is a
        no-op.
        """
        if self._hybrid_executor is not None or self._adapter is None:
            return

        self._hybrid_executor = HybridSearchExecutor(
            self._adapter,
            semantic_weight=self.config.semantic_weight,
            keyword_weight=self.config.keyword_weight,
            rrf_k=self.config.rrf_k,
            keyword_index_config=self.config.keyword_index,
            # Native-hybrid providers return records inline; the decoder
            # lets the executor populate _chunk_map as it ranks results —
            # no separate retrieve call required.
            chunk_decoder=chunk_from_record,
        )

    async def _build_hybrid_indices(self) -> None:
        """Build hybrid search indices (BM25 and exact match).

        Ensures the HybridSearchExecutor exists, then populates the
        in-process keyword index (BM25 fallback or OpenSearch) from
        the stored chunks. For native-hybrid providers, the executor
        skips the index build internally — the retrieval store owns
        the keyword leg.
        """
        if not self._chunks:
            logger.debug("No chunks to build hybrid indices from")
            return

        self._setup_hybrid_executor()
        if self._hybrid_executor is None:  # defensive — _setup_ already sets it
            return

        # Build keyword index (no-op for native-hybrid providers — they
        # bypass the in-process index at query time).
        await self._hybrid_executor.build_keyword_index(self._chunks)

        logger.info(
            f"Built hybrid search indices: {len(self._chunks)} documents indexed"
        )

    async def _hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[SearchResult]:
        """Perform hybrid search combining semantic, keyword, and exact match.

        Uses the HybridSearchExecutor to route to native hybrid search or
        BM25 fallback based on provider capabilities.

        Args:
            query: Search query string.
            query_embedding: Query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects sorted by fused score.
        """
        if self._hybrid_executor is None:
            # Fall back to semantic-only if hybrid executor not available
            logger.warning("Hybrid executor not initialized, using semantic search")
            return await self._semantic_search(query_embedding, top_k)

        # Get fused results from hybrid executor. Native-hybrid paths
        # populate _chunk_map inline from the returned records, so the
        # lookup below is a pure dict hit — no follow-up retrieve needed.
        fused_results = await self._hybrid_executor.search(
            query, query_embedding, top_k
        )

        # Convert to SearchResult objects via executor's chunk map
        results: list[SearchResult] = []

        for chunk_id, score in fused_results:
            chunk = self._hybrid_executor.get_chunk(chunk_id)
            if chunk is None:
                continue

            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    content=chunk.content,
                    fused_score=max(0.0, min(1.0, score)),
                    source_path=chunk.source_path,
                    parent_chain=chunk.parent_chain,
                    section_id=chunk.section_id,
                    subsection_ids=chunk.subsection_ids,
                    semantic_score=None,  # Fused score combines all modalities
                    keyword_score=None,
                    exact_match=False,
                    definitions_context=[],
                )
            )

        return results

    async def _keyword_search(
        self,
        query: str,
        top_k: int,
    ) -> list[SearchResult]:
        """Perform keyword-only BM25 search.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of SearchResult objects sorted by BM25 score.
        """
        if self._hybrid_executor is None:
            logger.warning("BM25 index not available, returning empty results")
            return []

        # Get BM25 results via public API
        bm25_results = await self._hybrid_executor.keyword_search(query, top_k)

        # Normalize BM25 scores using max-score normalization
        max_score = max((s for _, s in bm25_results), default=0.0)

        # Convert to SearchResult objects
        results: list[SearchResult] = []

        for chunk_id, score in bm25_results:
            chunk = self._hybrid_executor.get_chunk(chunk_id)
            if chunk is None:
                continue

            normalized_score = score / max_score if max_score > 0 else 0.0

            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    content=chunk.content,
                    fused_score=normalized_score,
                    source_path=chunk.source_path,
                    parent_chain=chunk.parent_chain,
                    section_id=chunk.section_id,
                    subsection_ids=chunk.subsection_ids,
                    semantic_score=None,
                    keyword_score=score,
                    exact_match=False,
                    definitions_context=[],
                )
            )

        return results

    async def initialize(
        self,
        force_ingest: bool = True,
        provider_type: str | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> None:
        """Initialize the tool by processing all configured documents.

        This method should be called before any search operations.
        It loads documents, chunks them, extracts definitions, and
        indexes content for search.

        Uses mtime-based incremental ingestion to skip unchanged files.
        Files are only re-ingested if their modification time is newer
        than the stored record's mtime.

        Args:
            force_ingest: If True, re-ingest all files regardless of
                modification time. Existing records will be deleted
                before re-ingestion.
            provider_type: LLM provider for dimension auto-detection
                (defaults to "openai" if not specified).
            progress_callback: Optional callback invoked after each file is
                processed (or skipped). Called as ``callback(current, total)``
                where *current* is the 1-based file index and *total* is the
                total number of discovered files.

        Raises:
            FileNotFoundError: If a document file is not found.
        """
        # Default to openai if not specified
        if provider_type is None:
            provider_type = "openai"
            logger.debug(
                f"Defaulting to '{provider_type}' for dimension auto-detection"
            )

        # Set up collection with provider type for dimension resolution
        self._setup_collection(provider_type)

        # Ingest all documents (with incremental check)
        skipped_files = await self._ingest_documents(
            force_ingest=force_ingest, progress_callback=progress_callback
        )

        # Always set up the hybrid executor — search routes through it.
        self._setup_hybrid_executor()

        # Decide whether to reload the corpus and rebuild the in-process
        # keyword index. Only the in-process BM25 fallback actually needs
        # this: it's ephemeral and dies with the process. For native-hybrid
        # providers (qdrant), keyword retrieval happens inside the store —
        # the in-process index is never read, so loading every chunk into
        # Python on every restart is pure waste. For in-memory providers,
        # the collection itself is reseeded by _ingest_documents (nothing
        # is "skipped"), so there's no reload to do.
        is_native_hybrid = self._uses_native_hybrid()
        needs_corpus_reload = (
            skipped_files > 0 and self._provider != "in-memory" and not is_native_hybrid
        )

        if needs_corpus_reload:
            stored_chunks = await self._load_chunks_from_store()
            if stored_chunks:
                self._chunks = stored_chunks
                await self._build_hybrid_indices()
        elif is_native_hybrid:
            logger.debug(
                "Skipped startup corpus reload for native-hybrid provider "
                f"'{self._provider}'; lazy chunk fetch on miss."
            )

        self._initialized = True
        logger.info(
            f"HierarchicalDocumentTool initialized: "
            f"{len(self._chunks)} chunks from {self.config.source}"
        )

    async def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Search documents for content relevant to query.

        Args:
            query: Search query string.
            top_k: Override configured top_k.

        Returns:
            List of SearchResult objects.

        Raises:
            RuntimeError: If tool is not initialized.
            ValueError: If query is empty.
        """
        if not self._initialized:
            raise RuntimeError(
                "HierarchicalDocumentTool must be initialized before search. "
                "Call initialize() first."
            )

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        effective_top_k = top_k or self.config.top_k

        # Generate query embedding (needed for semantic and hybrid modes)
        query_embedding = await self._embed_query(query)

        # Route to appropriate search mode
        if self.config.search_mode == SearchMode.KEYWORD:
            results = await self._keyword_search(query, effective_top_k)
        elif self.config.search_mode == SearchMode.HYBRID:
            results = await self._hybrid_search(query, query_embedding, effective_top_k)
        else:
            # SEMANTIC mode (default)
            results = await self._semantic_search(query_embedding, effective_top_k)

        # Filter by min_score if configured
        if self.config.min_score is not None:
            results = [r for r in results if r.fused_score >= self.config.min_score]

        return results

    async def get_context(self, query: str, max_tokens: int | None = None) -> str:
        """Get LLM-ready context for a query.

        This is a convenience method that searches and formats results
        into a single context string suitable for LLM prompts.

        Args:
            query: Query to get context for.
            max_tokens: Maximum tokens for context (currently unused).

        Returns:
            Formatted context string.

        Raises:
            RuntimeError: If tool is not initialized.
        """
        results = await self.search(query)

        if not results:
            return f"No relevant context found for: {query}"

        lines = [f"Context for query: {query}", ""]
        for i, result in enumerate(results, 1):
            lines.append(f"[{i}] {result.format()}")
            lines.append("")

        return "\n".join(lines).rstrip()

    def get_definition(self, term: str) -> dict[str, str] | None:
        """Look up a term's definition.

        Args:
            term: Term to look up.

        Returns:
            Dictionary with 'term' and 'definition' keys, or None.
        """
        if self._glossary is None:
            return None

        normalized = term.lower().replace(" ", "_")
        entry = self._glossary.get(normalized)
        if entry:
            return {"term": entry.term, "definition": entry.definition_text}
        return None
