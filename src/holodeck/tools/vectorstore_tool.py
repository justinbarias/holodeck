"""VectorStoreTool for semantic search over documents and structured data.

This module provides the VectorStoreTool class that enables agents to perform
semantic search over files and directories containing text data or structured
data (CSV, JSON, JSONL files with field mapping).

Features:
- Automatic file discovery (single files or directories)
- Support for multiple file formats (.txt, .md, .pdf, .csv, .json)
- Text chunking with configurable size and overlap
- Embedding generation via Semantic Kernel
- Vector storage in Redis or in-memory
- Modification time tracking for incremental ingestion
- Structured data ingestion with field mapping (vector_field, meta_fields, id_field)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from holodeck.lib.file_processor import FileProcessor, SourceFile
from holodeck.lib.text_chunker import TextChunker
from holodeck.lib.vector_store import (
    QueryResult,
    StructuredQueryResult,
    convert_document_to_query_result,
)

if TYPE_CHECKING:
    from holodeck.models.config import ExecutionConfig
    from holodeck.models.tool import VectorstoreTool as VectorstoreToolConfig

logger = logging.getLogger(__name__)


# Supported file extensions for ingestion
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".txt", ".md", ".pdf", ".csv", ".json"}
)


class VectorStoreTool:
    """Vectorstore tool for semantic search over unstructured data.

    This tool enables agents to perform semantic search over documents by:
    1. Discovering files from configured source (file or directory)
    2. Converting files to markdown using FileProcessor
    3. Chunking text for optimal embedding generation
    4. Generating embeddings via Semantic Kernel services
    5. Storing document chunks in a vector database
    6. Performing similarity search on queries

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

        # Persistent collection instance for vector store operations
        self._collection: Any = None
        self._provider: str = "in-memory"

        logger.debug(
            f"VectorStoreTool initialized: name={config.name}, "
            f"source={config.source}, base_dir={base_dir}, top_k={config.top_k}"
        )

    def set_embedding_service(self, service: Any) -> None:
        """Set the embedding service for generating embeddings.

        This method allows AgentFactory to inject a Semantic Kernel TextEmbedding
        service for generating real embeddings instead of placeholder zeros.

        Args:
            service: Semantic Kernel TextEmbedding service instance
                (OpenAITextEmbedding or AzureTextEmbedding).
        """
        self._embedding_service = service
        logger.debug(f"Embedding service set for tool: {self.config.name}")

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

    def _setup_collection(
        self,
        provider_type: str,
        record_class: type[Any] | None = None,
        definition: Any | None = None,
    ) -> None:
        """Set up the collection instance based on database configuration.

        Uses config.database to determine the vector store provider.
        Defaults to in-memory if no database is configured.
        Falls back to in-memory storage if database connection fails.

        Creates a persistent collection instance that is reused for both
        storing and searching documents.

        Args:
            provider_type: LLM provider type for dimension resolution
            record_class: Optional custom record class for the collection.
                If None, uses DocumentRecord for unstructured documents.
            definition: Optional VectorStoreCollectionDefinition for structured
                data with dynamic metadata fields.
        """
        # Resolve dimensions before creating collection
        if self._embedding_dimensions is None:
            self._embedding_dimensions = self._resolve_dimensions(provider_type)
        # Handle database configuration (can be DatabaseConfig, string ref, or None)
        database = self.config.database
        if isinstance(database, str):
            # Unresolved string reference - this shouldn't happen if merge_configs
            # was called, but fall back to in-memory with a warning
            logger.warning(
                f"Vectorstore tool '{self.config.name}' has unresolved database "
                f"reference '{database}'. Falling back to in-memory storage."
            )
            self._provider = "in-memory"
            connection_kwargs: dict[str, Any] = {}
        elif database is not None:
            # DatabaseConfig object - use its settings
            self._provider = database.provider
            connection_kwargs = {}
            if database.connection_string:
                connection_kwargs["connection_string"] = database.connection_string
            # Add extra fields from DatabaseConfig (extra="allow")
            if hasattr(database, "model_extra"):
                extra_fields = database.model_extra or {}
                connection_kwargs.update(extra_fields)
        else:
            # None - use in-memory
            self._provider = "in-memory"
            connection_kwargs = {}

        # Create persistent collection instance with fallback
        try:
            from holodeck.lib.vector_store import get_collection_factory

            factory = get_collection_factory(
                provider=self._provider,
                dimensions=self._embedding_dimensions,
                record_class=record_class,
                definition=definition,
                **connection_kwargs,
            )
            self._collection = factory()
            logger.info(
                f"Vector store connected: provider={self._provider}, "
                f"dimensions={self._embedding_dimensions}"
            )
        except (ImportError, ConnectionError, Exception) as e:
            # Fall back to in-memory storage for non-in-memory providers
            if self._provider != "in-memory":
                logger.warning(
                    f"Failed to connect to {self._provider}: {e}. "
                    "Falling back to in-memory storage."
                )
                self._provider = "in-memory"
                from holodeck.lib.vector_store import get_collection_factory

                factory = get_collection_factory(
                    provider="in-memory",
                    dimensions=self._embedding_dimensions,
                    record_class=record_class,
                    definition=definition,
                )
                self._collection = factory()
                logger.info("Using in-memory vector storage (fallback)")
            else:
                # Don't catch errors for in-memory provider
                raise

        logger.debug(
            f"Collection created for provider: {self._provider}, "
            f"dimensions={self._embedding_dimensions}"
        )

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

        This method handles:
        - Absolute paths: returned as-is
        - Relative paths: resolved relative to base_dir in this order:
          1. Explicit base_dir passed to constructor
          2. agent_base_dir context variable (set by CLI commands)
          3. Current working directory (fallback)

        Returns:
            Resolved absolute Path to the source.
        """
        source_path = Path(self.config.source)

        # If path is absolute, use it directly
        if source_path.is_absolute():
            return source_path

        # Resolve relative to base directory
        # Priority: explicit base_dir > context var > cwd
        base_dir = self._base_dir
        if base_dir is None:
            # Try to get from context variable
            from holodeck.config.context import agent_base_dir

            base_dir = agent_base_dir.get()

        if base_dir:
            return (Path(base_dir) / self.config.source).resolve()

        return source_path.resolve()

    def _discover_files(self) -> list[Path]:
        """Discover files to ingest from configured source.

        Recursively traverses directories and filters by supported extensions.
        Source path is resolved relative to base_dir if set.

        Returns:
            List of Path objects for files to process.

        Note:
            This method does not validate file existence - that happens
            during initialization.
        """
        source_path = self._resolve_source_path()

        if source_path.is_file():
            # Single file - check if supported
            if source_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                return [source_path]
            logger.warning(
                f"File {source_path} has unsupported extension "
                f"{source_path.suffix}. Supported: {SUPPORTED_EXTENSIONS}"
            )
            return []

        if source_path.is_dir():
            # Directory - recursively find all supported files
            discovered: list[Path] = []
            for file_path in source_path.rglob("*"):
                if file_path.is_file():
                    if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                        discovered.append(file_path)
                    else:
                        logger.debug(
                            f"Skipping unsupported file: {file_path} "
                            f"(extension: {file_path.suffix})"
                        )
            return discovered

        # Path doesn't exist - return empty list (error handled in initialize)
        return []

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

            # Map file extensions to FileInput type values
            type_mapping = {
                ".txt": "text",
                ".md": "text",
                ".pdf": "pdf",
                ".csv": "csv",
                ".json": "text",
            }
            file_type = type_mapping.get(file_path.suffix.lower(), "text")

            file_input = FileInput(
                path=str(file_path),
                url=None,
                type=file_type,
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

        Uses injected embedding service if available, otherwise returns
        placeholder embeddings (for testing without LLM).

        Validates that embedding dimensions match configuration.

        Args:
            chunks: List of text chunks to embed.

        Returns:
            List of embedding vectors (one per chunk).

        Raises:
            ValueError: If embedding dimensions don't match configuration.
        """
        if self._embedding_service is not None:
            # Use real embedding service
            try:
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

                logger.debug(
                    f"Generated {len(result)} embeddings, dim={len(result[0])}"
                )
                return result

            except Exception as e:
                logger.warning(
                    f"Embedding service failed, falling back to placeholder: {e}"
                )

        # Fallback: placeholder embeddings using configured dimensions
        if self._embedding_dimensions is None:
            self._embedding_dimensions = 1536

        placeholder: list[list[float]] = [
            [0.0] * self._embedding_dimensions for _ in chunks
        ]
        logger.debug(
            f"Generated {len(chunks)} placeholder embeddings, "
            f"dim={self._embedding_dimensions}"
        )
        return placeholder

    async def _store_chunks(
        self,
        source_file: SourceFile,
        embeddings: list[list[float]],
    ) -> int:
        """Store document chunks with embeddings in vector store.

        Uses Semantic Kernel collection abstraction for batch upsert operations.

        Args:
            source_file: SourceFile with chunks to store.
            embeddings: Embedding vectors corresponding to chunks.

        Returns:
            Number of chunks stored.

        Raises:
            RuntimeError: If collection is not initialized.
        """
        if self._collection is None:
            raise RuntimeError("Collection not initialized")

        # Import DocumentRecord creation factory
        from holodeck.lib.vector_store import create_document_record_class

        # Create DocumentRecord class with correct dimensions
        if self._embedding_dimensions is None:
            self._embedding_dimensions = 1536

        record_class = create_document_record_class(self._embedding_dimensions)

        records: list[Any] = []
        for idx, (chunk, embedding) in enumerate(
            zip(source_file.chunks, embeddings, strict=False)
        ):
            record = record_class(
                id=f"{source_file.path}_chunk_{idx}",
                source_path=str(source_file.path),
                chunk_index=idx,
                content=chunk,
                embedding=embedding,
                mtime=source_file.mtime,
                file_type=source_file.file_type,
                file_size_bytes=source_file.size_bytes,
            )
            records.append(record)

        # Batch upsert using persistent collection
        async with self._collection as collection:
            if not await collection.collection_exists():
                await collection.ensure_collection_exists()
            await collection.upsert(records)

        logger.debug(f"Stored {len(records)} chunks from {source_file.path}")
        return len(records)

    async def _needs_reingest(self, file_path: Path) -> bool:
        """Check if file needs re-ingestion based on modification time.

        Compares file's current mtime against stored DocumentRecord mtime.
        Returns True if file should be re-ingested.

        Args:
            file_path: Path to file to check.

        Returns:
            True if file needs re-ingestion (modified or not in store).
            False if file is up-to-date.
        """
        if self._collection is None:
            return True  # No collection, must ingest

        # Round to 6 decimal places to match stored precision
        current_mtime = round(file_path.stat().st_mtime, 6)
        source_path_str = str(file_path)

        # Query for existing record by ID pattern
        async with self._collection as collection:
            try:
                # Get first chunk to check mtime (all chunks share same mtime)
                record_id = f"{source_path_str}_chunk_0"
                record = await collection.get(record_id)

                if record is None:
                    return True  # Not in store, must ingest

                stored_mtime: float = float(record.mtime)
                should_reingest = current_mtime > stored_mtime
                return should_reingest

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
        if self._collection is None:
            return 0

        source_path_str = str(file_path)
        deleted_count = 0

        async with self._collection as collection:
            chunk_index = 0
            while True:
                record_id = f"{source_path_str}_chunk_{chunk_index}"
                try:
                    record = await collection.get(record_id)
                    if record is None:
                        break
                    await collection.delete(record_id)
                    deleted_count += 1
                    chunk_index += 1
                except Exception as e:
                    logger.debug(f"Error deleting record {record_id}: {e}")
                    break

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
        from holodeck.lib.vector_store import create_structured_record_class

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

        # Resolve dimensions before creating record class
        if self._embedding_dimensions is None:
            self._embedding_dimensions = self._resolve_dimensions(provider_type)

        # Create structured record class and definition with metadata fields
        record_class, definition = create_structured_record_class(
            dimensions=self._embedding_dimensions,
            metadata_field_names=self.config.meta_fields,
            collection_name=self.config.name,
        )

        # Set up collection with structured record class and definition
        self._setup_collection(
            provider_type, record_class=record_class, definition=definition
        )

        # Ingest records
        await self._ingest_structured_records(loader, str(source_path), record_class)

        self.is_initialized = True
        self.last_ingest_time = datetime.now()

        logger.info(
            f"VectorStoreTool initialized (structured mode): "
            f"source={source_path}, {self.document_count} records indexed"
        )

    async def _ingest_structured_records(
        self, loader: Any, source_path: str, record_class: type[Any]
    ) -> None:
        """Ingest structured records with embeddings.

        Args:
            loader: StructuredDataLoader instance.
            source_path: Path to source file for record metadata.
            record_class: Structured record class for creating records.
        """
        total_records = 0

        for batch in loader.iter_batches():
            # Extract content for embedding
            contents = [r["content"] for r in batch]

            # Generate embeddings for batch
            embeddings = await self._embed_chunks(contents)

            # Create records
            records: list[Any] = []
            for record_data, embedding in zip(batch, embeddings, strict=False):
                record = record_class(
                    id=record_data["id"],
                    content=record_data["content"],
                    embedding=embedding,
                    source_file=source_path,
                    **record_data["metadata"],
                )
                records.append(record)

            # Upsert batch to collection
            async with self._collection as collection:
                if not await collection.collection_exists():
                    await collection.ensure_collection_exists()
                await collection.upsert(records)

            total_records += len(records)
            logger.debug(f"Ingested batch of {len(records)} structured records")

        self.document_count = total_records

    async def initialize(
        self, force_ingest: bool = False, provider_type: str | None = None
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

        for file_path in files:
            # Check if file needs re-ingestion (unless force_ingest)
            if not force_ingest:
                needs_reingest = await self._needs_reingest(file_path)
                if not needs_reingest:
                    logger.debug(f"Skipping unchanged file: {file_path}")
                    skipped_files += 1
                    continue
            else:
                # Force ingest: delete existing records first
                await self._delete_file_records(file_path)

            source_file = await self._process_file(file_path)
            if source_file is None:
                continue

            # Generate embeddings
            embeddings = await self._embed_chunks(source_file.chunks)

            # Store chunks
            chunks_stored = await self._store_chunks(source_file, embeddings)
            total_chunks += chunks_stored

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
        """Search documents using vector store collection.

        Uses Semantic Kernel collection's native vector search capabilities.

        Args:
            query_embedding: Query embedding vector.

        Returns:
            List of QueryResults sorted by descending score.

        Raises:
            RuntimeError: If collection is not initialized.
        """
        if self._collection is None:
            raise RuntimeError("Collection not initialized")

        results: list[QueryResult] = []

        async with self._collection as collection:
            search_results = await collection.search(
                vector=query_embedding,
                top=self.config.top_k or 5,
            )

            # Process async iterable of search results
            # KernelSearchResults wraps results in .results attribute
            async for result in search_results.results:
                # Handle SK result format (may be object with attrs or tuple)
                record = result.record if hasattr(result, "record") else result[0]
                raw_score = result.score if hasattr(result, "score") else result[1]

                similarity = max(0.0, min(1.0, raw_score))

                query_result = await convert_document_to_query_result(
                    record,
                    score=similarity,
                )
                results.append(query_result)

        # Results should already be sorted by SK, but ensure ordering
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
        if self._collection is None:
            raise RuntimeError("Collection not initialized")

        results: list[StructuredQueryResult] = []

        async with self._collection as collection:
            search_results = await collection.search(
                vector=query_embedding,
                top=self.config.top_k or 5,
            )

            async for result in search_results.results:
                # Handle SK result format
                record = result.record if hasattr(result, "record") else result[0]
                raw_score = result.score if hasattr(result, "score") else result[1]

                similarity = max(0.0, min(1.0, raw_score))

                # Extract metadata fields from record
                metadata: dict[str, Any] = {}
                if self.config.meta_fields:
                    for field in self.config.meta_fields:
                        if hasattr(record, field):
                            metadata[field] = getattr(record, field)

                results.append(
                    StructuredQueryResult(
                        id=record.id,
                        content=record.content,
                        score=similarity,
                        source_file=record.source_file,
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
