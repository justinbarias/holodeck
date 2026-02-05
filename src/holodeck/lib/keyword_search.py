"""Keyword-based search implementation with tiered hybrid strategy.

This module provides keyword-based full-text search using a tiered approach:
- Native hybrid search for providers that support it (azure-ai-search, weaviate, etc.)
- BM25 fallback using rank_bm25 for providers without native support

The module supports both dense (semantic) and sparse (keyword) search,
combined via Reciprocal Rank Fusion (RRF) for optimal retrieval.

Key Features:
- Tiered keyword search strategy (native hybrid vs BM25 fallback)
- Provider capability detection
- BM25Okapi implementation via rank_bm25
- Hybrid search executor with strategy routing
- OpenTelemetry instrumentation for observability
- Graceful degradation when BM25 build fails

Usage:
    from holodeck.lib.keyword_search import (
        HybridSearchExecutor,
        get_keyword_search_strategy,
    )

    # Determine strategy based on provider
    strategy = get_keyword_search_strategy("azure-ai-search")

    # Create executor and search
    executor = HybridSearchExecutor("azure-ai-search", collection)
    results = await executor.search(query, query_embedding, top_k=10)
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import TYPE_CHECKING, Any

from opentelemetry import trace

if TYPE_CHECKING:
    from holodeck.lib.structured_chunker import DocumentChunk

logger = logging.getLogger(__name__)

# OpenTelemetry tracer for search operations
tracer = trace.get_tracer("holodeck.keyword_search")


class KeywordSearchStrategy(str, Enum):
    """Keyword search strategy based on provider capabilities.

    Determines whether to use native hybrid search or BM25 fallback:
    - NATIVE_HYBRID: Provider supports hybrid_search() API directly
    - FALLBACK_BM25: Use rank_bm25 + app-level RRF fusion

    The strategy is automatically selected based on the vector store provider.
    """

    NATIVE_HYBRID = "native_hybrid"
    FALLBACK_BM25 = "fallback_bm25"


# Provider capability sets based on Semantic Kernel support
# See: https://learn.microsoft.com/en-us/semantic-kernel/concepts/vector-store-connectors/hybrid-search

NATIVE_HYBRID_PROVIDERS: set[str] = {
    "azure-ai-search",
    "weaviate",
    "qdrant",
    "mongodb",  # MongoDB Atlas via MongoDBAtlasStore
    "azure-cosmos-nosql",
}
"""Providers that support native hybrid search via collection.hybrid_search()."""

FALLBACK_BM25_PROVIDERS: set[str] = {
    "postgres",
    "pinecone",
    "chromadb",
    "faiss",
    "in-memory",
    "sql-server",
}
"""Providers that require BM25 fallback + app-level RRF.

NOTE: azure-cosmos-mongo is EXCLUDED from both sets. MongoDB vCore uses a different
API and requires MongoDBAtlasStore for native hybrid search.
"""


def get_keyword_search_strategy(provider: str) -> KeywordSearchStrategy:
    """Determine keyword search strategy based on provider capabilities.

    Args:
        provider: Vector store provider name (e.g., "azure-ai-search", "postgres")

    Returns:
        KeywordSearchStrategy indicating native hybrid or BM25 fallback

    Example:
        >>> get_keyword_search_strategy("weaviate")
        KeywordSearchStrategy.NATIVE_HYBRID
        >>> get_keyword_search_strategy("postgres")
        KeywordSearchStrategy.FALLBACK_BM25
    """
    if provider in NATIVE_HYBRID_PROVIDERS:
        return KeywordSearchStrategy.NATIVE_HYBRID
    return KeywordSearchStrategy.FALLBACK_BM25


def _tokenize(text: str) -> list[str]:
    """Tokenize text into searchable terms.

    Uses a simple regex-based tokenizer that extracts alphanumeric tokens
    and converts them to lowercase. This provides consistent tokenization
    for both indexing and querying.

    Args:
        text: Text to tokenize

    Returns:
        List of lowercase alphanumeric tokens

    Example:
        >>> _tokenize("Section 203(a)(1): Force Majeure!")
        ['section', '203', 'a', '1', 'force', 'majeure']
    """
    if not text:
        return []
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class BM25FallbackProvider:
    """BM25 sparse index implementation using rank_bm25 library.

    Provides keyword-based search for providers that don't support native
    hybrid search. Uses the BM25Okapi algorithm for term frequency scoring.

    Attributes:
        k1: BM25 term frequency saturation parameter (default 1.5)
        b: BM25 length normalization parameter (default 0.75)

    Example:
        >>> provider = BM25FallbackProvider()
        >>> provider.build([
        ...     ("doc1", "The quick brown fox"),
        ...     ("doc2", "The lazy dog"),
        ... ])
        >>> results = provider.search("brown fox", top_k=2)
        >>> print(results[0])  # (doc_id, score) tuple
        ('doc1', 1.234)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize the BM25 provider.

        Args:
            k1: BM25 k1 parameter for term frequency saturation.
                Higher values give more weight to term frequency.
            b: BM25 b parameter for document length normalization.
                Higher values penalize longer documents more.
        """
        self.k1 = k1
        self.b = b
        self._bm25: Any = None  # BM25Okapi instance
        self._doc_ids: list[str] = []

    def build(self, documents: list[tuple[str, str]]) -> None:
        """Build BM25 index from documents.

        Indexes the provided documents for keyword search. The second
        element of each tuple should be the contextualized content
        (context + original chunk) for best results.

        Args:
            documents: List of (doc_id, contextualized_content) tuples.
                The contextualized_content is tokenized and indexed.

        Raises:
            ImportError: If rank_bm25 is not installed.

        Example:
            >>> provider.build([
            ...     ("chunk1", "Context: About foxes. The quick brown fox."),
            ...     ("chunk2", "Context: About dogs. The lazy dog."),
            ... ])
        """
        from rank_bm25 import BM25Okapi

        self._doc_ids = [doc_id for doc_id, _ in documents]
        tokenized = [_tokenize(text) for _, text in documents]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)

        logger.debug(f"Built BM25 index with {len(documents)} documents")

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search indexed documents for matching results.

        Args:
            query: Search query string
            top_k: Maximum number of results to return

        Returns:
            List of (doc_id, score) tuples sorted by BM25 score descending.
            Returns empty list if index not built.

        Example:
            >>> results = provider.search("brown fox", top_k=5)
            [('doc1', 1.234), ('doc3', 0.567)]
        """
        if not self._bm25:
            return []

        with tracer.start_as_current_span(
            "bm25.search",
            attributes={
                "bm25.query_tokens": len(_tokenize(query)),
                "bm25.index_size": len(self._doc_ids),
                "bm25.top_k": top_k,
            },
        ) as span:
            query_tokens = _tokenize(query)
            scores = self._bm25.get_scores(query_tokens)

            # Get top-k indices sorted by score descending
            top_indices = scores.argsort()[-top_k:][::-1]

            results = [
                (self._doc_ids[i], float(scores[i]))
                for i in top_indices
                if scores[i] > 0
            ]

            span.set_attribute("bm25.result_count", len(results))
            return results


class HybridSearchExecutor:
    """Executes hybrid search using appropriate strategy for provider.

    Routes search requests to either native hybrid search or BM25 fallback
    based on the provider's capabilities.

    Attributes:
        provider: Vector store provider name
        collection: Semantic Kernel vector store collection
        strategy: Determined keyword search strategy

    Example:
        >>> executor = HybridSearchExecutor("weaviate", collection)
        >>> executor.build_bm25_index(documents)  # Optional for fallback
        >>> results = await executor.search(query, embedding, top_k=10)
    """

    def __init__(
        self,
        provider: str,
        collection: Any,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.3,
        rrf_k: int = 60,
    ) -> None:
        """Initialize the hybrid search executor.

        Args:
            provider: Vector store provider name (determines strategy)
            collection: Semantic Kernel vector store collection instance
            semantic_weight: Weight for semantic results in RRF fusion (default 0.5)
            keyword_weight: Weight for keyword results in RRF fusion (default 0.3)
            rrf_k: RRF ranking constant (default 60)
        """
        self.provider = provider
        self.collection = collection
        self.strategy = get_keyword_search_strategy(provider)
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k
        self._bm25_index: BM25FallbackProvider | None = None
        self._chunk_map: dict[str, DocumentChunk] = {}

        logger.debug(
            f"HybridSearchExecutor initialized: provider={provider}, "
            f"strategy={self.strategy.value}, "
            f"weights=semantic:{semantic_weight}/keyword:{keyword_weight}, "
            f"rrf_k={rrf_k}"
        )

    def build_bm25_index(self, chunks: list[DocumentChunk]) -> None:
        """Build BM25 index with graceful degradation.

        Creates the BM25 sparse index from the provided chunks and stores
        a chunk map for ID-based lookups. If BM25 build fails, logs a
        warning and continues with semantic-only search.

        Args:
            chunks: List of DocumentChunk objects. The chunk's
                contextualized_content (or content fallback) is indexed.

        Example:
            >>> executor.build_bm25_index(chunks)
        """
        # Store chunk map for ID-based lookups
        self._chunk_map = {c.id: c for c in chunks}

        # Build BM25 index with graceful degradation
        try:
            self._bm25_index = BM25FallbackProvider(k1=1.5, b=0.75)
            bm25_docs = [(c.id, c.contextualized_content or c.content) for c in chunks]
            self._bm25_index.build(bm25_docs)
            logger.debug(f"Built BM25 index with {len(chunks)} documents")
        except Exception as e:
            logger.warning(
                f"BM25 index build failed, falling back to semantic-only: {e}"
            )
            self._bm25_index = None

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """Look up a chunk by ID from the stored chunk map.

        Args:
            chunk_id: The unique identifier of the chunk.

        Returns:
            The DocumentChunk if found, or None.
        """
        return self._chunk_map.get(chunk_id)

    def keyword_search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Perform keyword-only BM25 search.

        Args:
            query: Search query string.
            top_k: Maximum number of results to return.

        Returns:
            List of (chunk_id, score) tuples sorted by BM25 score descending.
            Returns empty list if BM25 index is not built.
        """
        if self._bm25_index is None:
            return []
        return self._bm25_index.search(query, top_k)

    async def search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Execute hybrid search and return ranked results.

        Routes to native hybrid search or BM25 fallback based on provider.

        Args:
            query: Search query string
            query_embedding: Pre-computed query embedding vector
            top_k: Maximum number of results to return

        Returns:
            List of (chunk_id, score) tuples sorted by relevance.
            Scores are normalized to 0-1 range.

        Example:
            >>> results = await executor.search(
            ...     "reporting requirements",
            ...     [0.1, 0.2, ...],
            ...     top_k=10
            ... )
            [('chunk_report', 0.95), ('chunk_other', 0.72), ...]
        """
        with tracer.start_as_current_span(
            "hybrid_search.execute",
            attributes={
                "search.mode": self.strategy.value,
                "search.provider": self.provider,
                "search.top_k": top_k,
            },
        ) as span:
            # Execute hybrid search based on strategy
            if self.strategy == KeywordSearchStrategy.NATIVE_HYBRID:
                results = await self._native_hybrid_search(
                    query, query_embedding, top_k
                )
            else:
                results = await self._fallback_hybrid_search(
                    query, query_embedding, top_k
                )

            span.set_attribute("search.result_count", len(results))
            return results[:top_k]

    async def _native_hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Execute native hybrid search using SK's hybrid_search() API.

        Uses the provider's native hybrid search capability which combines
        vector similarity and full-text search in a single query.

        Args:
            query: Search query string
            query_embedding: Pre-computed query embedding vector
            top_k: Maximum number of results

        Returns:
            List of (chunk_id, score) tuples
        """
        results: list[tuple[str, float]] = []

        async with self.collection as coll:
            search_results = await coll.hybrid_search(
                query,
                vector=query_embedding,
                vector_property_name="embedding",
                additional_property_name="content",  # needs is_full_text_indexed=True
                top=top_k,
            )

            async for result in search_results.results:
                record = result.record if hasattr(result, "record") else result[0]
                score = result.score if hasattr(result, "score") else result[1]
                results.append((record.id, float(score)))

        return results

    async def _fallback_hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Execute hybrid search using BM25 fallback + RRF fusion.

        Runs vector search and BM25 search separately, then combines
        results using Reciprocal Rank Fusion.

        Args:
            query: Search query string
            query_embedding: Pre-computed query embedding vector
            top_k: Maximum number of results

        Returns:
            List of (chunk_id, score) tuples with RRF-fused scores
        """
        from holodeck.lib.hybrid_search import reciprocal_rank_fusion

        # Vector (semantic) search
        vector_results: list[tuple[str, float]] = []
        async with self.collection as coll:
            search_results = await coll.search(
                vector=query_embedding,
                top=top_k,
            )

            async for result in search_results.results:
                record = result.record if hasattr(result, "record") else result[0]
                score = result.score if hasattr(result, "score") else result[1]
                vector_results.append((record.id, float(score)))

        # BM25 (keyword) search
        bm25_results: list[tuple[str, float]] = []
        if self._bm25_index is not None:
            bm25_results = self._bm25_index.search(query, top_k=top_k)

        # Fuse results with RRF using configured weights
        if bm25_results:
            fused = reciprocal_rank_fusion(
                [vector_results, bm25_results],
                k=self.rrf_k,
                weights=[self.semantic_weight, self.keyword_weight],
            )
            return fused
        else:
            # Semantic-only fallback
            return vector_results
