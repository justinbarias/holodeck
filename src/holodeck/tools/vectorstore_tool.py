"""VectorStoreTool for semantic search over documents and structured data.

This module provides the VectorStoreTool class that enables agents to perform
semantic search over files and directories containing text data or structured
data (CSV, JSON, JSONL files with field mapping).

Features:
- Automatic file discovery (single files or directories)
- Support for multiple file formats (.txt, .md, .pdf, .csv, .json)
- Text chunking with configurable size and overlap
- Embedding generation via the injected embedding service
- Vector storage through the retrieval adapter contract (any provider)
- Modification time tracking for incremental ingestion
- Structured data ingestion with field mapping (vector_field, meta_fields, id_field)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from holodeck.lib.file_processor import FileProcessor, SourceFile
from holodeck.lib.retrieval import (
    CollectionSchema,
    Operation,
    ScoredRecord,
    UnsupportedOperationError,
    VectorQuery,
    VectorStoreAdapter,
    clamp_similarity,
)
from holodeck.lib.retrieval.schemas import (
    DocumentChunkRecord,
    document_schema,
    float_field,
    int_field,
    structured_record,
    structured_schema,
    text_field,
)
from holodeck.lib.text_chunker import TextChunker
from holodeck.lib.vector_store import QueryResult, StructuredQueryResult
from holodeck.tools.base_tool import DatabaseConfigMixin, EmbeddingServiceMixin
from holodeck.tools.common import (
    SUPPORTED_EXTENSIONS,
    discover_files,
    generate_placeholder_embeddings,
    get_file_type,
    resolve_source_path,
)

if TYPE_CHECKING:
    from holodeck.lib.structured_loader import StructuredDataLoader
    from holodeck.models.config import ExecutionConfig
    from holodeck.models.tool import VectorstoreTool as VectorstoreToolConfig

logger = logging.getLogger(__name__)

# Operations each mode uses. Checked before ingestion starts (FR-014).
_DOCUMENT_OPERATIONS = (
    Operation.UPSERT,
    Operation.GET,
    Operation.DELETE,
    Operation.VECTOR_SEARCH,
)
_STRUCTURED_OPERATIONS = (Operation.UPSERT, Operation.VECTOR_SEARCH)


def _query_result(hit: ScoredRecord) -> QueryResult:
    """Convert a document search hit to a ``QueryResult``."""
    record = hit.record
    return QueryResult(
        content=text_field(record, "content"),
        score=clamp_similarity(hit.score),
        source_path=text_field(record, "source_path"),
        chunk_index=int_field(record, "chunk_index"),
        metadata={
            "file_type": text_field(record, "file_type"),
            "file_size_bytes": int_field(record, "file_size_bytes"),
            "mtime": float_field(record, "mtime"),
        },
    )


def _is_qdrant_native_id(record_id: str) -> bool:
    """Return True when Qdrant accepts *record_id* as a point ID as-is.

    Qdrant point IDs are UUIDs or unsigned 64-bit integers. Record IDs are
    strings on the HoloDeck side, and Qdrant rejects a numeric *string*
    (``"42"``) where it would accept the integer ``42``, so only UUID strings
    count as native here; numeric keys are hashed like any other business key.
    """
    import uuid

    try:
        uuid.UUID(record_id)
    except ValueError:
        return False
    return True


class VectorStoreTool(EmbeddingServiceMixin, DatabaseConfigMixin):
    """Vectorstore tool for semantic search over unstructured data.

    This tool enables agents to perform semantic search over documents by:
    1. Discovering files from configured source (file or directory)
    2. Converting files to markdown using FileProcessor
    3. Chunking text for optimal embedding generation
    4. Generating embeddings via Semantic Kernel services
    5. Storing document chunks in a vector database
    6. Performing similarity search on queries

    Inherits from:
        EmbeddingServiceMixin: Provides set_embedding_service() method
        DatabaseConfigMixin: Provides database config resolution and collection creation

    Attributes:
        config: Tool configuration from agent.yaml
        is_initialized: Whether the tool has been initialized
        document_count: Number of document chunks stored
        last_ingest_time: Timestamp of last ingestion

    Example:
        >>> config = VectorstoreTool(
        ...     name="knowledge_base",
        ...     description="Search product docs",
        ...     source="data/docs/"
        ... )
        >>> tool = VectorStoreTool(config)
        >>> await tool.initialize()
        >>> results = await tool.search("How do I authenticate?")
    """

    def __init__(
        self,
        config: VectorstoreToolConfig,
        base_dir: str | None = None,
        execution_config: ExecutionConfig | None = None,
    ) -> None:
        """Initialize VectorStoreTool with configuration.

        Args:
            config: VectorstoreTool configuration from agent.yaml containing:
                - name: Tool identifier
                - description: Tool description
                - source: File or directory path to ingest
                - embedding_model: Optional custom embedding model
                - database: Optional database configuration
                - top_k: Number of results to return (default: 5)
                - min_similarity_score: Minimum score threshold (optional)
                - chunk_size: Text chunk size in tokens (optional)
                - chunk_overlap: Chunk overlap in tokens (optional)
            base_dir: Base directory for resolving relative source paths.
                If None, source paths are resolved relative to current
                working directory.
            execution_config: Execution configuration for file processing
                timeouts and caching. If None, default FileProcessor
                settings are used.
        """
        self.config = config
        self._base_dir = base_dir
        self._execution_config = execution_config

        # State tracking
        self.is_initialized: bool = False
        self.document_count: int = 0
        self.last_ingest_time: datetime | None = None

        # Embedding dimensions (resolved during initialization)
        self._embedding_dimensions: int | None = None

        # Initialize components (lazy initialization for some)
        chunk_size = config.chunk_size or TextChunker.DEFAULT_CHUNK_SIZE
        chunk_overlap = config.chunk_overlap or TextChunker.DEFAULT_CHUNK_OVERLAP
        self._text_chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._file_processor: FileProcessor | None = None

        # Embedding service (initialized lazily by AgentFactory)
        self._embedding_service: Any = None

        # Vector store adapter, owned by the tool until close()
        self._adapter: VectorStoreAdapter | None = None
        self._provider: str = "in-memory"
        self._warned_id_coercion: bool = False

        # Source context for stable record keys (remote sources)
        self._source_root: Path | None = None
        self._is_remote: bool = False

        logger.debug(
            f"VectorStoreTool initialized: name={config.name}, "
            f"source={config.source}, base_dir={base_dir}, top_k={config.top_k}"
        )

    # set_embedding_service is inherited from EmbeddingServiceMixin

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

    def _is_structured_mode(self) -> bool:
        """Check if tool is configured for structured data mode.

        Structured mode is enabled when vector_field is specified, indicating
        the source is a structured data file (CSV, JSON, JSONL) with field mapping.

        Returns:
            True if structured data mode is enabled.
        """
        return self.config.vector_field is not None

    def _resolve_dimensions(self, provider: str) -> int:
        """Resolve embedding dimensions from config or auto-detect.

        Resolution order:
        1. Explicit config.embedding_dimensions
        2. Auto-detect from config.embedding_model + provider
        3. Provider default

        Args:
            provider: LLM provider ("openai", "azure_openai", "ollama")

        Returns:
            Resolved embedding dimensions
        """
        # Explicit configuration takes precedence
        if self.config.embedding_dimensions is not None:
            logger.debug(
                f"Using explicit "
                f"embedding_dimensions={self.config.embedding_dimensions}"
            )
            return self.config.embedding_dimensions

        # Auto-detect from model name
        from holodeck.config.defaults import get_embedding_dimensions

        dimensions = get_embedding_dimensions(
            self.config.embedding_model, provider=provider
        )

        logger.debug(
            f"Auto-detected embedding_dimensions={dimensions} "
            f"for model={self.config.embedding_model}, provider={provider}"
        )
        return dimensions

    def _setup_collection(self, provider_type: str) -> None:
        """Set up the vector store adapter for the configured mode.

        Uses DatabaseConfigMixin methods for config resolution and adapter
        creation with automatic fallback to in-memory storage, then checks
        that the provider supports every operation the mode needs.

        Args:
            provider_type: LLM provider type for dimension resolution

        Raises:
            UnsupportedOperationError: The provider lacks an operation the
                configured mode needs. Raised before any ingestion.
            SchemaError: A structured metadata field name is invalid.
        """
        # Resolve dimensions before creating the adapter
        if self._embedding_dimensions is None:
            self._embedding_dimensions = self._resolve_dimensions(provider_type)

        # Use mixin to resolve database config
        provider, connection_kwargs = self._resolve_database_config(
            self.config.database
        )

        schema: CollectionSchema
        if self._is_structured_mode():
            schema = structured_schema(
                self._embedding_dimensions,
                self.config.meta_fields,
                self.config.name,
                connection_kwargs,
            )
        else:
            schema = document_schema(self._embedding_dimensions, connection_kwargs)

        # Use mixin to create the adapter with fallback
        self._adapter = self._create_adapter_with_fallback(
            provider=provider,
            schema=schema,
            connection_kwargs=connection_kwargs,
        )
        self._adapter.capabilities.require(
            *(
                _STRUCTURED_OPERATIONS
                if self._is_structured_mode()
                else _DOCUMENT_OPERATIONS
            )
        )

        logger.debug(
            f"Collection created for provider: {self._provider}, "
            f"dimensions={self._embedding_dimensions}"
        )

    def _require_adapter(self) -> VectorStoreAdapter:
        """Return the adapter or raise when the collection is not set up."""
        if self._adapter is None:
            raise RuntimeError("Collection not initialized")
        return self._adapter

    def _get_file_processor(self) -> FileProcessor:
        """Get or create FileProcessor instance (lazy initialization).

        Uses ExecutionConfig for timeout and cache settings if available,
        otherwise falls back to FileProcessor defaults.
        """
        if self._file_processor is None:
            if self._execution_config:
                self._file_processor = FileProcessor.from_execution_config(
                    self._execution_config
                )
            else:
                self._file_processor = FileProcessor()
        return self._file_processor

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

    async def _process_file(self, file_path: Path) -> SourceFile | None:
        """Process a single file into a SourceFile with chunks.

        Args:
            file_path: Path to the file to process.

        Returns:
            SourceFile with content and chunks populated, or None if processing fails.
        """
        try:
            # Get file metadata
            stat = file_path.stat()
            source_file = SourceFile(
                path=file_path,
                # Round mtime to 6 decimal places (microsecond precision) to avoid
                # float precision loss when stored/retrieved from vector databases
                mtime=round(stat.st_mtime, 6),
                size_bytes=stat.st_size,
                file_type=file_path.suffix.lower(),
            )

            # Warn for large files
            size_mb = source_file.size_bytes / (1024 * 1024)
            if size_mb > 100:
                logger.warning(
                    f"Large file detected: {file_path} ({size_mb:.2f}MB). "
                    "Processing may take longer."
                )

            # Convert to markdown using FileProcessor
            from holodeck.models.test_case import FileInput

            file_input = FileInput(
                path=str(file_path),
                url=None,
                type=get_file_type(file_path),
                description=None,
                pages=None,
                sheet=None,
                range=None,
                cache=None,
            )
            processor = self._get_file_processor()
            processed = processor.process_file(file_input)

            if processed.error:
                logger.warning(
                    f"Error processing file {file_path}: {processed.error}. Skipping."
                )
                return None

            source_file.content = processed.markdown_content

            # Skip empty files
            if not source_file.content or not source_file.content.strip():
                logger.warning(
                    f"File {file_path} is empty or whitespace-only. Skipping."
                )
                return None

            # Chunk the content
            try:
                source_file.chunks = self._text_chunker.split_text(source_file.content)
            except ValueError as e:
                logger.warning(f"Error chunking file {file_path}: {e}. Skipping.")
                return None

            logger.debug(
                f"Processed file: {file_path} -> {len(source_file.chunks)} chunks"
            )
            return source_file

        except PermissionError:
            logger.warning(f"Permission denied reading file {file_path}. Skipping.")
            return None
        except Exception as e:
            logger.warning(
                f"Unexpected error processing file {file_path}: {e}. Skipping."
            )
            return None

    async def _embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of text chunks.

        Uses the injected embedding service when one is set. Provider
        failures propagate unchanged (``EmbeddingServiceError`` from the
        LiteLLM shim) so a broken provider never silently stores or queries
        placeholder vectors. Without a service, placeholder embeddings are
        returned (for testing without an LLM).

        Validates that embedding dimensions match configuration.

        Args:
            chunks: List of text chunks to embed.

        Returns:
            List of embedding vectors (one per chunk).

        Raises:
            ValueError: If embedding dimensions don't match configuration.
            Exception: Any error raised by the embedding service.
        """
        if self._embedding_service is not None:
            embeddings = await self._embedding_service.generate_embeddings(chunks)
            # Convert to list of lists (service may return different types)
            result = [list(emb) for emb in embeddings]

            # Validate dimensions match configuration
            if result and self._embedding_dimensions is not None:
                actual_dim = len(result[0])
                if actual_dim != self._embedding_dimensions:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected "
                        f"{self._embedding_dimensions}, got {actual_dim}. "
                        f"This usually means:\n"
                        f"1. Your embedding_model produces different dimensions\n"
                        f"2. The embedding_dimensions setting is incorrect\n"
                        f"Fix: Set 'embedding_dimensions: {actual_dim}' "
                        f"in tool config"
                    )

            if result:
                logger.debug(
                    f"Generated {len(result)} embeddings, dim={len(result[0])}"
                )
            return result

        # No service injected: placeholder embeddings using configured dimensions
        if self._embedding_dimensions is None:
            self._embedding_dimensions = 1536

        placeholder = generate_placeholder_embeddings(
            len(chunks), self._embedding_dimensions
        )
        return placeholder

    async def _store_chunks(
        self,
        source_file: SourceFile,
        embeddings: list[list[float]],
    ) -> int:
        """Store document chunks with embeddings in vector store.

        Replaces the file's previous chunks: chunks beyond the new count,
        left over from a longer earlier version of the file, are deleted.

        Args:
            source_file: SourceFile with chunks to store.
            embeddings: Embedding vectors corresponding to chunks.

        Returns:
            Number of chunks stored.

        Raises:
            RuntimeError: If collection is not initialized.
        """
        adapter = self._require_adapter()

        if self._embedding_dimensions is None:
            self._embedding_dimensions = 1536

        source_key = self._compute_source_key(source_file.path)
        content_hash = self._compute_content_hash(source_file.path)

        records = [
            DocumentChunkRecord(
                id=self._record_id(source_key, idx),
                source_path=source_key,
                chunk_index=idx,
                content=chunk,
                embedding=embedding,
                mtime=source_file.mtime,
                file_type=source_file.file_type,
                file_size_bytes=source_file.size_bytes,
                content_hash=content_hash,
            ).to_record()
            for idx, (chunk, embedding) in enumerate(
                zip(source_file.chunks, embeddings, strict=False)
            )
        ]

        await adapter.ensure_collection()
        await adapter.upsert(records)
        stale = await self._delete_chunks_from(adapter, source_key, len(records))
        if stale:
            logger.debug(f"Deleted {stale} stale chunks of {source_file.path}")

        logger.debug(f"Stored {len(records)} chunks from {source_file.path}")
        return len(records)

    def _record_id(self, source_key: str, chunk_index: int) -> str:
        """Return the vector-store record ID for one chunk of a source file.

        Records are keyed ``{source_key}_chunk_{index}`` so the store can be
        walked per file. Qdrant only accepts UUID or unsigned-int point IDs, so
        on that provider the readable key is mapped to a deterministic UUIDv5
        (namespace derived from the tool name). The mapping is stable across
        runs, so re-ingestion of unchanged content lands on the same point.
        """
        readable = f"{source_key}_chunk_{chunk_index}"
        return self._coerce_record_id(readable)

    def _coerce_record_id(self, record_id: str) -> str:
        """Map a record ID to a Qdrant-safe value when needed.

        UUID IDs, which Qdrant already accepts, are returned unchanged, so
        structured sources keyed by UUID keep their existing points. Anything
        else is mapped to a deterministic UUIDv5; for structured data the
        original business key is then only visible through ``meta_fields``,
        which is logged once per tool.
        """
        if self._provider != "qdrant" or _is_qdrant_native_id(record_id):
            return record_id
        import uuid

        namespace = uuid.uuid5(
            uuid.NAMESPACE_URL, f"holodeck/vectorstore/{self.config.name}"
        )
        return str(uuid.uuid5(namespace, record_id))

    async def _needs_reingest(self, file_path: Path) -> bool:
        """Check if file needs re-ingestion based on modification time.

        Compares file's current mtime against the stored mtime of its first
        chunk. Returns True if file should be re-ingested.

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
            # Get first chunk to check mtime (all chunks share same mtime)
            found = await self._adapter.get([self._record_id(source_key, 0)])
            if not found:
                return True  # Not in store, must ingest
            record = found[0]

            if self._is_remote:
                # For remote sources, compare content hash
                current_hash = self._compute_content_hash(file_path)
                stored_hash = text_field(record, "content_hash")
                if not stored_hash:
                    return True  # No hash stored (old record), must reingest
                return current_hash != stored_hash
            # For local sources, compare mtime (backward compat)
            current_mtime = round(file_path.stat().st_mtime, 6)
            return current_mtime > float_field(record, "mtime")

        except UnsupportedOperationError:
            raise
        except Exception as e:
            logger.debug(f"Could not retrieve record for {file_path}: {e}")
            return True  # Error = must ingest

    async def _delete_chunks_from(
        self, adapter: VectorStoreAdapter, source_key: str, start: int
    ) -> int:
        """Delete a file's chunks from index ``start`` until one is missing.

        Chunk IDs are ``{source_key}_chunk_{index}`` with no gaps, so walking
        them by ID works on every provider, including those that cannot
        filter by ``source_path``.

        Returns:
            Number of records deleted.
        """
        deleted_count = 0
        chunk_index = start
        while True:
            record_id = self._record_id(source_key, chunk_index)
            try:
                if not await adapter.get([record_id]):
                    break
                await adapter.delete([record_id])
            except UnsupportedOperationError:
                raise
            except Exception as e:
                logger.debug(f"Error deleting record {record_id}: {e}")
                break
            deleted_count += 1
            chunk_index += 1
        return deleted_count

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
        deleted_count = await self._delete_chunks_from(self._adapter, source_key, 0)

        if deleted_count > 0:
            logger.debug(f"Deleted {deleted_count} records for {file_path}")

        return deleted_count

    async def _initialize_structured(self, provider_type: str) -> None:
        """Initialize for structured data mode (CSV, JSON, JSONL).

        Loads structured data using StructuredDataLoader, generates embeddings
        for vector fields, and stores records with metadata in the vector store.

        Args:
            provider_type: LLM provider for dimension auto-detection.

        Raises:
            ConfigError: If configured fields don't exist in source data.
            FileNotFoundError: If source file doesn't exist.
        """
        from holodeck.lib.structured_loader import StructuredDataLoader

        source_path = self._resolve_source_path()

        # Validate source exists
        if not source_path.exists():
            from holodeck.config.context import agent_base_dir

            effective_base_dir = self._base_dir or agent_base_dir.get()
            raise FileNotFoundError(
                f"Source path does not exist: {source_path} "
                f"(configured source: {self.config.source}, "
                f"base_dir: {effective_base_dir})"
            )

        # Normalize vector_field to list
        vector_field = self.config.vector_field
        if isinstance(vector_field, str):
            vector_fields = [vector_field]
        else:
            vector_fields = list(vector_field) if vector_field else []

        # Create loader
        loader = StructuredDataLoader(
            source_path=str(source_path),
            id_field=self.config.id_field or "",
            vector_fields=vector_fields,
            metadata_fields=self.config.meta_fields,
            field_separator=self.config.field_separator,
            delimiter=self.config.delimiter,
        )

        # Validate schema before ingestion (raises ConfigError if fields missing)
        loader.validate_schema()

        # Set up the adapter with the structured schema (metadata fields)
        self._setup_collection(provider_type)

        # Ingest records
        await self._ingest_structured_records(loader, str(source_path))

        self.is_initialized = True
        self.last_ingest_time = datetime.now()

        logger.info(
            f"VectorStoreTool initialized (structured mode): "
            f"source={source_path}, {self.document_count} records indexed"
        )

    async def _ingest_structured_records(
        self, loader: StructuredDataLoader, source_path: str
    ) -> None:
        """Ingest structured records with embeddings.

        Args:
            loader: StructuredDataLoader instance.
            source_path: Path to source file for record metadata.
        """
        adapter = self._require_adapter()
        total_records = 0

        for batch in loader.iter_batches():
            # Extract content for embedding
            contents = [r["content"] for r in batch]

            # Generate embeddings for batch
            embeddings = await self._embed_chunks(contents)

            # Create records
            records = []
            for record_data, embedding in zip(batch, embeddings, strict=False):
                record_id = self._coerce_record_id(record_data["id"])
                if record_id != record_data["id"] and not self._warned_id_coercion:
                    self._warned_id_coercion = True
                    logger.warning(
                        f"Vectorstore tool '{self.config.name}': structured record "
                        f"IDs are not valid Qdrant point IDs and are stored as "
                        f"UUIDv5 hashes; search results carry the hash. Add the "
                        f"id column to meta_fields to keep the original key."
                    )
                records.append(
                    structured_record(
                        record_id,
                        content=record_data["content"],
                        embedding=embedding,
                        source_file=source_path,
                        metadata=record_data["metadata"],
                        metadata_field_names=self.config.meta_fields,
                    )
                )

            # Upsert batch to collection
            await adapter.ensure_collection()
            await adapter.upsert(records)

            total_records += len(records)
            logger.debug(f"Ingested batch of {len(records)} structured records")

        self.document_count = total_records

    async def initialize(
        self,
        force_ingest: bool = False,
        provider_type: str | None = None,
        progress_callback: Callable[[int, int | None], None] | None = None,
    ) -> None:
        """Initialize tool and ingest source files.

        Discovers files from the configured source, processes them into chunks,
        generates embeddings, and stores them in the vector database.
        Source path is resolved relative to base_dir if set.

        For structured data mode (when vector_field is configured), loads
        structured data from CSV/JSON/JSONL files with field mapping.

        Args:
            force_ingest: If True, re-ingest all files regardless of modification time.
            provider_type: LLM provider for dimension auto-detection
                (defaults to "openai" if not specified)
            progress_callback: Optional callback invoked after each file is
                processed (or skipped). Called as ``callback(current, total)``
                where *current* is the 1-based file index and *total* is the
                total number of discovered files.

        Raises:
            FileNotFoundError: If the source path doesn't exist.
            RuntimeError: If no supported files are found in source.
            ConfigError: If configured fields don't exist in source (structured mode).
        """
        # Default to openai if not specified
        if provider_type is None:
            provider_type = "openai"
            logger.debug(
                f"Defaulting to '{provider_type}' for dimension auto-detection"
            )

        # Branch to structured mode if vector_field is configured
        if self._is_structured_mode():
            await self._initialize_structured(provider_type)
            return

        source_path = self._resolve_source_path()

        # Validate source exists (T035)
        if not source_path.exists():
            # Get effective base_dir for error message
            from holodeck.config.context import agent_base_dir

            effective_base_dir = self._base_dir or agent_base_dir.get()
            raise FileNotFoundError(
                f"Source path does not exist: {source_path} "
                f"(configured source: {self.config.source}, "
                f"base_dir: {effective_base_dir})"
            )

        # Set up collection instance with provider type before processing files
        self._setup_collection(provider_type)

        # Discover files
        files = self._discover_files()

        if not files:
            logger.warning(
                f"No supported files found in source: {self.config.source}. "
                f"Supported extensions: {SUPPORTED_EXTENSIONS}"
            )
            # Still mark as initialized even with no files
            self.is_initialized = True
            self.document_count = 0
            self.last_ingest_time = datetime.now()
            return

        logger.info(f"Discovered {len(files)} files for ingestion")

        # Process each file with mtime checking
        total_chunks = 0
        skipped_files = 0
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

            source_file = await self._process_file(file_path)
            if source_file is None:
                processed_count += 1
                if progress_callback is not None:
                    progress_callback(processed_count, total_files)
                continue

            # Generate embeddings
            embeddings = await self._embed_chunks(source_file.chunks)

            # Store chunks
            chunks_stored = await self._store_chunks(source_file, embeddings)
            total_chunks += chunks_stored

            processed_count += 1
            if progress_callback is not None:
                progress_callback(processed_count, total_files)

        self.document_count = total_chunks
        self.is_initialized = True
        self.last_ingest_time = datetime.now()

        logger.info(
            f"VectorStoreTool initialized: {len(files)} files "
            f"({skipped_files} skipped, up-to-date), {total_chunks} chunks indexed"
        )

    async def search(self, query: str) -> str:
        """Execute semantic search and return formatted results.

        Args:
            query: Natural language search query.

        Returns:
            Formatted string with search results including scores and sources.

        Raises:
            RuntimeError: If tool not initialized.
            ValueError: If query is empty.
        """
        # Validation
        if not self.is_initialized:
            raise RuntimeError(
                "VectorStoreTool must be initialized before search. "
                "Call initialize() first."
            )

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")

        # Generate query embedding
        query_embeddings = await self._embed_chunks([query])
        query_embedding = query_embeddings[0]

        # Branch to structured search if in structured mode
        if self._is_structured_mode():
            structured_results = await self._search_structured(query_embedding)

            # Apply min_similarity_score filter
            if self.config.min_similarity_score is not None:
                structured_results = [
                    r
                    for r in structured_results
                    if r.score >= self.config.min_similarity_score
                ]

            # Apply top_k limit
            structured_results = structured_results[: self.config.top_k]

            return self._format_structured_results(structured_results, query)

        # Unstructured mode: search document chunks
        doc_results = await self._search_documents(query_embedding)

        # Apply min_similarity_score filter
        if self.config.min_similarity_score is not None:
            doc_results = [
                r for r in doc_results if r.score >= self.config.min_similarity_score
            ]

        # Apply top_k limit (T037)
        doc_results = doc_results[: self.config.top_k]

        # Format results (T034)
        return self._format_results(doc_results, query)

    async def _search_documents(
        self, query_embedding: list[float]
    ) -> list[QueryResult]:
        """Search document chunks by vector similarity.

        Args:
            query_embedding: Query embedding vector.

        Returns:
            List of QueryResults sorted by descending score.

        Raises:
            RuntimeError: If collection is not initialized.
        """
        adapter = self._require_adapter()
        hits = await adapter.vector_search(
            VectorQuery(vector=query_embedding, top_k=self.config.top_k or 5)
        )
        results = [_query_result(hit) for hit in hits]
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    @staticmethod
    def _format_results(results: list[QueryResult], query: str) -> str:
        """Format search results as a string for agent consumption.

        Args:
            results: List of QueryResults to format.
            query: Original search query (for context in empty results).

        Returns:
            Formatted string with numbered results, scores, and sources.

        Format:
            Found N result(s):

            [1] Score: 0.89 | Source: data/docs/api.md
            Content of the matched chunk...

            [2] Score: 0.76 | Source: data/docs/auth.md
            Content of another matched chunk...
        """
        if not results:
            return f"No relevant results found for query: {query}"

        lines = [f"Found {len(results)} result(s):", ""]

        for rank, result in enumerate(results, start=1):
            lines.append(
                f"[{rank}] Score: {result.score:.2f} | Source: {result.source_path}"
            )
            lines.append(result.content)
            lines.append("")

        return "\n".join(lines).rstrip()

    async def _search_structured(
        self, query_embedding: list[float]
    ) -> list[StructuredQueryResult]:
        """Search structured records and return typed results.

        Args:
            query_embedding: Query embedding vector.

        Returns:
            List of StructuredQueryResults sorted by descending score.

        Raises:
            RuntimeError: If collection is not initialized.
        """
        adapter = self._require_adapter()
        hits = await adapter.vector_search(
            VectorQuery(vector=query_embedding, top_k=self.config.top_k or 5)
        )

        results: list[StructuredQueryResult] = []
        for hit in hits:
            record = hit.record
            metadata: dict[str, Any] = {
                name: record.fields[name]
                for name in self.config.meta_fields or ()
                if name in record.fields
            }
            results.append(
                StructuredQueryResult(
                    id=record.id,
                    content=text_field(record, "content"),
                    score=clamp_similarity(hit.score),
                    source_file=text_field(record, "source_file"),
                    metadata=metadata,
                )
            )

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    @staticmethod
    def _format_structured_results(
        results: list[StructuredQueryResult], query: str
    ) -> str:
        """Format structured search results for agent consumption.

        Args:
            results: List of StructuredQueryResults to format.
            query: Original search query (for context in empty results).

        Returns:
            Formatted string with results including ID, scores, content, and metadata.
        """
        if not results:
            return f"No relevant results found for query: {query}"

        lines = [f"Found {len(results)} result(s):", ""]

        for rank, result in enumerate(results, start=1):
            lines.append(f"[{rank}] ID: {result.id} | Score: {result.score:.2f}")
            lines.append(f"Content: {result.content}")
            if result.metadata:
                meta_str = ", ".join(f"{k}: {v}" for k, v in result.metadata.items())
                lines.append(f"Metadata: {meta_str}")
            lines.append("")

        return "\n".join(lines).rstrip()
