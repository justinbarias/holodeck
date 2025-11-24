"""VectorStoreTool for semantic search over unstructured documents.

This module provides the VectorStoreTool class that enables agents to perform
semantic search over files and directories containing unstructured text data.

Features:
- Automatic file discovery (single files or directories)
- Support for multiple file formats (.txt, .md, .pdf, .csv, .json)
- Text chunking with configurable size and overlap
- Embedding generation via Semantic Kernel
- Vector storage in Redis or in-memory
- Modification time tracking for incremental ingestion
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from holodeck.lib.file_processor import FileProcessor, SourceFile
from holodeck.lib.text_chunker import TextChunker
from holodeck.lib.vector_store import DocumentRecord, QueryResult

if TYPE_CHECKING:
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

    def __init__(self, config: VectorstoreToolConfig) -> None:
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
        """
        self.config = config

        # State tracking
        self.is_initialized: bool = False
        self.document_count: int = 0
        self.last_ingest_time: datetime | None = None

        # Initialize components (lazy initialization for some)
        chunk_size = config.chunk_size or TextChunker.DEFAULT_CHUNK_SIZE
        chunk_overlap = config.chunk_overlap or TextChunker.DEFAULT_CHUNK_OVERLAP
        self._text_chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._file_processor: FileProcessor | None = None

        # Vector store and embedding service (initialized lazily)
        self._vector_store: Any = None
        self._embedding_service: Any = None

        # Document storage (for in-memory operations)
        self._documents: list[DocumentRecord] = []

        logger.debug(
            f"VectorStoreTool initialized: name={config.name}, "
            f"source={config.source}, top_k={config.top_k}"
        )

    def _get_file_processor(self) -> FileProcessor:
        """Get or create FileProcessor instance (lazy initialization)."""
        if self._file_processor is None:
            self._file_processor = FileProcessor()
        return self._file_processor

    def _discover_files(self) -> list[Path]:
        """Discover files to ingest from configured source.

        Recursively traverses directories and filters by supported extensions.

        Returns:
            List of Path objects for files to process.

        Note:
            This method does not validate file existence - that happens
            during initialization.
        """
        source_path = Path(self.config.source)

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
                mtime=stat.st_mtime,
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

        Args:
            chunks: List of text chunks to embed.

        Returns:
            List of embedding vectors (one per chunk).

        Note:
            Currently returns placeholder embeddings. Full implementation
            will use Semantic Kernel's TextEmbedding service.
        """
        # Placeholder: Return zero vectors
        # TODO: Implement actual embedding generation with Semantic Kernel
        embedding_dim = 1536  # text-embedding-3-small default
        embeddings = [[0.0] * embedding_dim for _ in chunks]
        logger.debug(f"Generated embeddings for {len(chunks)} chunks")
        return embeddings

    async def _store_chunks(
        self,
        source_file: SourceFile,
        embeddings: list[list[float]],
    ) -> int:
        """Store document chunks with embeddings in vector store.

        Args:
            source_file: SourceFile with chunks to store.
            embeddings: Embedding vectors corresponding to chunks.

        Returns:
            Number of chunks stored.
        """
        stored_count = 0
        for idx, (chunk, embedding) in enumerate(
            zip(source_file.chunks, embeddings, strict=False)
        ):
            record = DocumentRecord(
                id=f"{source_file.path}_chunk_{idx}",
                source_path=str(source_file.path),
                chunk_index=idx,
                content=chunk,
                embedding=embedding,
                mtime=source_file.mtime,
                file_type=source_file.file_type,
                file_size_bytes=source_file.size_bytes,
            )
            self._documents.append(record)
            stored_count += 1

        logger.debug(f"Stored {stored_count} chunks from {source_file.path}")
        return stored_count

    async def initialize(self, force_ingest: bool = False) -> None:
        """Initialize tool and ingest source files.

        Discovers files from the configured source, processes them into chunks,
        generates embeddings, and stores them in the vector database.

        Args:
            force_ingest: If True, re-ingest all files regardless of modification time.

        Raises:
            FileNotFoundError: If the source path doesn't exist.
            RuntimeError: If no supported files are found in source.
        """
        source_path = Path(self.config.source)

        # Validate source exists (T035)
        if not source_path.exists():
            raise FileNotFoundError(f"Source path does not exist: {self.config.source}")

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

        # Process each file
        total_chunks = 0
        for file_path in files:
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
            f"VectorStoreTool initialized: {len(files)} files, "
            f"{total_chunks} chunks indexed"
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

        # Perform search (simplified in-memory implementation)
        results = await self._search_documents(query_embedding)

        # Apply min_similarity_score filter
        if self.config.min_similarity_score is not None:
            results = [
                r for r in results if r.score >= self.config.min_similarity_score
            ]

        # Apply top_k limit (T037)
        results = results[: self.config.top_k]

        # Format results (T034)
        return self._format_results(results, query)

    async def _search_documents(
        self, query_embedding: list[float]
    ) -> list[QueryResult]:
        """Search documents using cosine similarity.

        Args:
            query_embedding: Query embedding vector.

        Returns:
            List of QueryResults sorted by descending score.
        """
        # Simplified cosine similarity search
        results: list[QueryResult] = []

        for doc in self._documents:
            if doc.embedding:
                # Calculate cosine similarity
                score = self._cosine_similarity(query_embedding, doc.embedding)
                results.append(
                    QueryResult(
                        content=doc.content,
                        score=max(0.0, min(1.0, score)),  # Clamp to [0, 1]
                        source_path=doc.source_path,
                        chunk_index=doc.chunk_index,
                        metadata={
                            "file_type": doc.file_type,
                            "file_size_bytes": doc.file_size_bytes,
                            "mtime": doc.mtime,
                        },
                    )
                )

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    @staticmethod
    def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec_a: First vector.
            vec_b: Second vector.

        Returns:
            Cosine similarity score (0.0 to 1.0).
        """
        if len(vec_a) != len(vec_b):
            return 0.0

        dot_product: float = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
        magnitude_a: float = sum(a * a for a in vec_a) ** 0.5
        magnitude_b: float = sum(b * b for b in vec_b) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return float(dot_product / (magnitude_a * magnitude_b))

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
