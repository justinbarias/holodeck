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

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from holodeck.lib.keyword_search import HybridSearchExecutor

if TYPE_CHECKING:
    from holodeck.lib.definition_extractor import DefinitionEntry
    from holodeck.lib.llm_context_generator import LLMContextGenerator
    from holodeck.lib.structured_chunker import (
        DocumentChunk,
        StructuredChunker,
        SubsectionPattern,
    )

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
    ) -> None:
        """Initialize the hierarchical document tool.

        Args:
            config: Tool configuration from HierarchicalDocumentToolConfig.
            base_dir: Optional base directory for resolving relative source paths.
                If None, source paths are resolved relative to current working
                directory or agent_base_dir context variable.
        """
        self.config = config
        self._base_dir = base_dir
        self._chunker: StructuredChunker | None = None
        self._searcher: Any = None  # HybridSearcher (future)
        self._context_generator: LLMContextGenerator | None = None
        self._glossary: dict[str, DefinitionEntry] | None = None
        self._chunks: list[DocumentChunk] = []
        self._initialized = False

        # Service injection attributes
        self._embedding_service: Any = None
        self._chat_service: Any = None
        self._collection: Any = None
        self._provider: str = "in-memory"
        self._embedding_dimensions: int | None = None

        # Hybrid search executor (initialized during ingestion)
        self._hybrid_executor: HybridSearchExecutor | None = None

    # set_embedding_service is inherited from EmbeddingServiceMixin

    def set_chat_service(self, service: Any) -> None:
        """Set the chat service for LLM context generation.

        This enables contextual embeddings via the LLMContextGenerator.
        When contextual_embeddings is enabled in config, this also creates
        the LLMContextGenerator instance.

        Args:
            service: Semantic Kernel ChatCompletion service instance.
        """
        self._chat_service = service

        # Create LLMContextGenerator if contextual embeddings are enabled
        if self.config.contextual_embeddings and service is not None:
            from holodeck.lib.llm_context_generator import LLMContextGenerator

            self._context_generator = LLMContextGenerator(
                chat_service=service,
                max_context_tokens=self.config.context_max_tokens,
                concurrency=self.config.context_concurrency,
            )
            logger.info(
                f"LLMContextGenerator created for {self.config.name} "
                f"(max_tokens={self.config.context_max_tokens}, "
                f"concurrency={self.config.context_concurrency})"
            )
        else:
            logger.debug("Chat service set for HierarchicalDocumentTool")

    def _setup_collection(self, provider_type: str = "openai") -> None:
        """Set up the vector store collection for chunk storage.

        Uses DatabaseConfigMixin methods for config resolution and collection creation
        with automatic fallback to in-memory storage.

        Args:
            provider_type: LLM provider type for dimension resolution
                (e.g., "openai", "azure_openai", "ollama").
        """
        from holodeck.lib.vector_store import create_hierarchical_document_record_class

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

        # Create record class with correct dimensions and tool name
        record_class = create_hierarchical_document_record_class(
            self._embedding_dimensions, self.config.name
        )

        # Use mixin to create collection with fallback
        self._collection = self._create_collection_with_fallback(
            provider=provider,
            dimensions=self._embedding_dimensions,
            connection_kwargs=connection_kwargs,
            record_class=record_class,
        )

        logger.debug(
            f"Collection setup: provider={self._provider}, "
            f"dimensions={self._embedding_dimensions}"
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
        if self._collection is None:
            return True  # No collection, must ingest

        current_mtime = file_path.stat().st_mtime
        source_path_str = str(file_path)

        async with self._collection as collection:
            try:
                # Get first available record for this file to check mtime
                # (all chunks from the same file share the same mtime)
                records = await collection.get(
                    top=1,
                    filter=lambda r: r.source_path == source_path_str,
                )

                if not records:
                    return True  # Not in store, must ingest

                stored_mtime = float(records[0].mtime)
                # Use tolerance to handle floating-point precision loss in storage
                return current_mtime - stored_mtime > 0.001

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
            try:
                # Get all records for this file
                records = await collection.get(
                    top=10000,  # Large limit to get all chunks
                    filter=lambda r: r.source_path == source_path_str,
                )

                if records:
                    # Delete all records by their IDs
                    record_ids = [r.id for r in records]
                    await collection.delete(record_ids)
                    deleted_count = len(record_ids)

            except Exception as e:
                logger.debug(f"Error deleting records for {file_path}: {e}")

        if deleted_count > 0:
            logger.debug(f"Deleted {deleted_count} records for {file_path}")

        return deleted_count

    async def _load_chunks_from_store(self) -> list[DocumentChunk]:
        """Load all document chunks from the vector store.

        Reconstructs DocumentChunk objects from stored records, enabling
        BM25 index rebuild on startup when files are unchanged (skipped
        via mtime check). This is necessary because the BM25 index is
        in-memory only and does not survive restarts.

        Returns:
            List of reconstructed DocumentChunk objects, or empty list
            if collection doesn't exist or an error occurs.
        """
        from holodeck.lib.structured_chunker import DocumentChunk

        if self._collection is None:
            return []

        try:
            async with self._collection as collection:
                if not await collection.collection_exists():
                    return []

                records = await collection.get(top=10000)

                if not records:
                    return []

                chunks: list[DocumentChunk] = []
                for record in records:
                    # Parse JSON fields with graceful fallback
                    try:
                        parent_chain = json.loads(record.parent_chain)
                    except (json.JSONDecodeError, TypeError):
                        parent_chain = []

                    try:
                        cross_references = json.loads(record.cross_references)
                    except (json.JSONDecodeError, TypeError):
                        cross_references = []

                    try:
                        subsection_ids = json.loads(record.subsection_ids)
                    except (json.JSONDecodeError, TypeError):
                        subsection_ids = []

                    # Map chunk_type string to ChunkType enum
                    try:
                        chunk_type = ChunkType(record.chunk_type)
                    except (ValueError, KeyError):
                        chunk_type = ChunkType.CONTENT

                    chunks.append(
                        DocumentChunk(
                            id=record.id,
                            source_path=record.source_path,
                            chunk_index=record.chunk_index,
                            content=record.content,
                            parent_chain=parent_chain,
                            section_id=record.section_id,
                            chunk_type=chunk_type,
                            cross_references=cross_references,
                            heading_level=0,
                            contextualized_content=record.contextualized_content or "",
                            mtime=float(record.mtime),
                            defined_term=record.defined_term or "",
                            defined_term_normalized=record.defined_term_normalized
                            or "",
                            subsection_ids=subsection_ids,
                        )
                    )

                logger.info(f"Loaded {len(chunks)} chunks from vector store")
                return chunks

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
        from holodeck.lib.file_processor import FileProcessor
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

        processor = FileProcessor()
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

    async def _ingest_documents(self, force_ingest: bool = False) -> int:
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

            # 1-2. Convert to markdown
            markdown_content = await self._convert_to_markdown(str(file_path))
            if not markdown_content.strip():
                logger.warning(f"Empty content from {file_path}, skipping")
                continue

            mtime = file_path.stat().st_mtime

            # 3. Parse structure
            chunks = self._chunker.parse(markdown_content, str(file_path), mtime)

            # 3.5 Filter out header-only chunks (no substantive content)
            chunks = [c for c in chunks if c.chunk_type != ChunkType.HEADER]

            if not chunks:
                logger.debug(
                    f"No content chunks after filtering headers from {file_path}"
                )
                continue

            # 4. Context generation (if enabled and chat service available)
            if (
                self.config.contextual_embeddings
                and self._context_generator is not None
                and self._chat_service is not None
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
                elif self._context_generator is None:
                    reason = "no context generator configured"
                else:
                    reason = "no chat service available"
                logger.info(
                    f"Skipping contextual embeddings for {file_path} ({reason})"
                )
                for chunk in chunks:
                    chunk.contextualized_content = chunk.content

            # 5-6. Embed chunks
            await self._embed_chunks(chunks)

            # 7. Store
            await self._store_chunks(chunks)
            self._chunks.extend(chunks)
            ingested_files += 1

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
            try:
                embeddings = await self._embedding_service.generate_embeddings(texts)
                for chunk, emb in zip(chunks, embeddings, strict=False):
                    chunk.embedding = list(emb)
                logger.debug(f"Generated {len(chunks)} embeddings")
                return
            except Exception as e:
                logger.warning(f"Embedding generation failed: {e}")

        # Fallback: placeholder embeddings
        dims = self._embedding_dimensions or 1536
        placeholders = generate_placeholder_embeddings(len(chunks), dims)
        for chunk, emb in zip(chunks, placeholders, strict=False):
            chunk.embedding = emb

    async def _store_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Store document chunks in the vector store.

        Uses HierarchicalDocumentRecord for storage.

        Args:
            chunks: List of DocumentChunks to store.

        Returns:
            Number of chunks stored.

        Raises:
            RuntimeError: If collection is not initialized.
        """
        if self._collection is None:
            raise RuntimeError("Collection not initialized")

        from holodeck.lib.vector_store import create_hierarchical_document_record_class

        dims = self._embedding_dimensions or 1536
        record_class = create_hierarchical_document_record_class(dims, self.config.name)

        records = []
        for chunk in chunks:
            record = record_class(
                id=chunk.id,
                source_path=chunk.source_path,
                chunk_index=chunk.chunk_index,
                section_id=chunk.section_id,
                content=chunk.content,
                embedding=chunk.embedding,
                parent_chain=json.dumps(chunk.parent_chain),
                chunk_type=chunk.chunk_type.value,
                cross_references=json.dumps(chunk.cross_references),
                contextualized_content=chunk.contextualized_content or "",
                mtime=chunk.mtime,
                file_type="",
                defined_term=chunk.defined_term or "",
                defined_term_normalized=chunk.defined_term_normalized or "",
                subsection_ids=json.dumps(chunk.subsection_ids),
            )
            records.append(record)

        async with self._collection as collection:
            if not await collection.collection_exists():
                await collection.ensure_collection_exists()
            await collection.upsert(records)

        logger.debug(f"Stored {len(records)} chunks")
        return len(records)

    async def _embed_query(self, query: str) -> list[float]:
        """Generate embedding for a query string.

        Args:
            query: Query text to embed.

        Returns:
            Query embedding vector.
        """
        if self._embedding_service is not None:
            try:
                embeddings = await self._embedding_service.generate_embeddings([query])
                return list(embeddings[0])
            except Exception as e:
                logger.warning(f"Query embedding failed: {e}")

        # Fallback: placeholder embedding
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
        results: list[SearchResult] = []

        async with self._collection as collection:
            search_results = await collection.search(
                vector=query_embedding,
                top=top_k,
            )

            async for result in search_results.results:
                # Handle SK result format (may be object with attrs or tuple)
                record = result.record if hasattr(result, "record") else result[0]
                raw_score = result.score if hasattr(result, "score") else result[1]

                # Parse parent_chain from JSON
                parent_chain_str = (
                    record.parent_chain if hasattr(record, "parent_chain") else "[]"
                )
                try:
                    parent_chain = json.loads(parent_chain_str)
                except (json.JSONDecodeError, TypeError):
                    parent_chain = []

                # Parse subsection_ids from JSON
                subsection_ids_str = (
                    record.subsection_ids if hasattr(record, "subsection_ids") else "[]"
                )
                try:
                    subsection_ids = json.loads(subsection_ids_str)
                except (json.JSONDecodeError, TypeError):
                    subsection_ids = []

                # Clamp score to [0.0, 1.0]
                fused_score = max(0.0, min(1.0, float(raw_score)))

                results.append(
                    SearchResult(
                        chunk_id=record.id,
                        content=record.content,
                        fused_score=fused_score,
                        source_path=record.source_path,
                        parent_chain=parent_chain,
                        section_id=(
                            record.section_id if hasattr(record, "section_id") else ""
                        ),
                        subsection_ids=subsection_ids,
                        semantic_score=float(raw_score),
                        keyword_score=None,  # Semantic-only mode
                        exact_match=False,
                        definitions_context=[],
                    )
                )

        # Sort by score descending
        results.sort(key=lambda r: r.fused_score, reverse=True)
        return results

    async def _build_hybrid_indices(self) -> None:
        """Build hybrid search indices (BM25 and exact match).

        Creates the HybridSearchExecutor with BM25 fallback index
        and exact match index from the stored chunks. This enables
        keyword search, exact match, and hybrid search modes.
        """
        from holodeck.lib.keyword_search import HybridSearchExecutor

        if not self._chunks:
            logger.debug("No chunks to build hybrid indices from")
            return

        # Create hybrid search executor with configured weights
        self._hybrid_executor = HybridSearchExecutor(
            self._provider,
            self._collection,
            semantic_weight=self.config.semantic_weight,
            keyword_weight=self.config.keyword_weight,
            rrf_k=self.config.rrf_k,
            keyword_index_config=self.config.keyword_index,
        )

        # Build keyword index (executor now owns chunk data)
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

        # Get fused results from hybrid executor
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
        self, force_ingest: bool = True, provider_type: str | None = None
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
        skipped_files = await self._ingest_documents(force_ingest=force_ingest)

        # Rebuild BM25 index from stored chunks when files were skipped.
        # The BM25 index is in-memory only and doesn't survive restarts.
        # For persistent providers (Qdrant, Postgres, etc.), we need to
        # reload chunks from the store to rebuild keyword search indices.
        # Skip for in-memory provider: collections don't survive restarts,
        # so _needs_reingest() returns True for all files (nothing skipped).
        if skipped_files > 0 and self._provider != "in-memory":
            stored_chunks = await self._load_chunks_from_store()
            if stored_chunks:
                self._chunks = stored_chunks
                await self._build_hybrid_indices()

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

    def to_semantic_kernel_function(self) -> Any:
        """Convert this tool to a Semantic Kernel function.

        Returns:
            Semantic Kernel function wrapper.
        """

        # This would require Semantic Kernel decorators
        # For now, return a simple wrapper
        async def sk_search_function(query: str) -> str:
            results = await self.search(query)
            if not results:
                return "No results found."
            return "\n\n".join(r.format() for r in results)

        return sk_search_function
