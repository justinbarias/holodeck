"""Keyword-based search implementation with tiered hybrid strategy.

This module provides keyword-based full-text search using a tiered approach:
- Native hybrid search for providers that support it (azure-ai-search, weaviate, etc.)
- Configurable keyword backend (in-memory BM25 or OpenSearch) for other providers

The module supports both dense (semantic) and sparse (keyword) search,
combined via Reciprocal Rank Fusion (RRF) for optimal retrieval.

Key Features:
- Tiered keyword search strategy (native hybrid vs keyword fallback)
- Provider capability detection
- KeywordSearchProvider protocol for pluggable backends
- InMemoryBM25KeywordProvider: rank_bm25 in-process (dev/local)
- OpenSearchKeywordProvider: external OpenSearch cluster (production)
- Hybrid search executor with strategy routing and provider routing
- OpenTelemetry instrumentation for observability
- Graceful degradation when keyword index build fails

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

import asyncio
import logging
import re
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from opentelemetry import trace

if TYPE_CHECKING:
    from holodeck.lib.structured_chunker import DocumentChunk
    from holodeck.models.tool import KeywordIndexConfig

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


@runtime_checkable
class KeywordSearchProvider(Protocol):
    """Protocol for keyword search backends.

    Defines the interface that all keyword search providers must implement.
    Uses structural subtyping (Protocol) so providers satisfy the interface
    via duck typing without explicit inheritance.

    Methods:
        build: Index documents for keyword search.
        search: Search indexed documents and return ranked results.
    """

    def build(self, documents: list[tuple[str, str]]) -> None:
        """Build keyword index from documents.

        Args:
            documents: List of (doc_id, content) tuples to index.
        """
        ...

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search indexed documents for matching results.

        Args:
            query: Search query string.
            top_k: Maximum number of results to return.

        Returns:
            List of (doc_id, score) tuples sorted by score descending.
        """
        ...


class InMemoryBM25KeywordProvider:
    """In-memory BM25 keyword search provider.

    Provides keyword-based search using the BM25Okapi algorithm from
    the rank_bm25 library. Suitable for development and local workloads.

    Attributes:
        k1: BM25 term frequency saturation parameter (default 1.5)
        b: BM25 length normalization parameter (default 0.75)

    Example:
        >>> provider = InMemoryBM25KeywordProvider()
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
            "keyword.search.in_memory_bm25",
            attributes={
                "keyword.query_tokens": len(_tokenize(query)),
                "keyword.index_size": len(self._doc_ids),
                "keyword.top_k": top_k,
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

            span.set_attribute("keyword.result_count", len(results))
            return results


class OpenSearchKeywordProvider:
    """OpenSearch-backed keyword search provider for production workloads.

    Uses the opensearch-py low-level client to index and search documents
    via BM25 scoring on an external OpenSearch cluster. Implements the
    KeywordSearchProvider protocol.

    Attributes:
        endpoint: OpenSearch endpoint URL.
        index_name: Name of the OpenSearch index.
        verify_certs: Whether to verify TLS certificates.
        timeout_seconds: Connection timeout in seconds.

    Example:
        >>> provider = OpenSearchKeywordProvider(
        ...     endpoint="https://search.example.com:9200",
        ...     index_name="my-index",
        ...     username="admin",
        ...     password="secret",
        ... )
        >>> provider.build([("doc1", "The quick brown fox")])
        >>> results = provider.search("brown fox", top_k=5)
        >>> print(results[0])  # (chunk_id, score) tuple
        ('doc1', 3.456)
    """

    _INDEX_MAPPING: dict[str, Any] = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "content": {"type": "text", "analyzer": "standard"},
            }
        },
    }
    """Index mapping with single shard, keyword chunk_id, and text content."""

    def __init__(
        self,
        endpoint: str,
        index_name: str,
        username: str | None = None,
        password: str | None = None,
        api_key: str | None = None,
        verify_certs: bool = True,
        timeout_seconds: int = 10,
    ) -> None:
        """Initialize the OpenSearch keyword provider.

        Args:
            endpoint: OpenSearch endpoint URL (e.g. "https://host:9200").
            index_name: Name of the index to create/use.
            username: Basic auth username (used with password).
            password: Basic auth password (used with username).
            api_key: API key for authentication (alternative to basic auth).
            verify_certs: Whether to verify TLS certificates.
            timeout_seconds: Connection timeout in seconds.
        """
        import opensearchpy

        self.endpoint = endpoint
        self.index_name = index_name
        self.verify_certs = verify_certs
        self.timeout_seconds = timeout_seconds

        client_kwargs: dict[str, Any] = {
            "hosts": [endpoint],
            "verify_certs": verify_certs,
            "timeout": timeout_seconds,
        }

        if not verify_certs:
            client_kwargs["ssl_show_warn"] = False

        if api_key:
            client_kwargs["headers"] = {
                "Authorization": f"ApiKey {api_key}",
            }
        elif username and password:
            client_kwargs["http_auth"] = (username, password)

        self._client: opensearchpy.OpenSearch = opensearchpy.OpenSearch(**client_kwargs)

    def build(self, documents: list[tuple[str, str]]) -> None:
        """Build OpenSearch index from documents.

        Creates the index if it does not exist. If the index already
        exists, clears all existing documents before re-indexing.
        Uses bulk indexing with refresh=True for immediate searchability.

        Args:
            documents: List of (chunk_id, content) tuples to index.

        Raises:
            opensearchpy.OpenSearchException: On connection or indexing errors.
        """
        import opensearchpy.helpers as os_helpers

        with tracer.start_as_current_span(
            "opensearch.build",
            attributes={
                "opensearch.index": self.index_name,
                "opensearch.document_count": len(documents),
            },
        ) as span:
            if not self._client.indices.exists(index=self.index_name):
                self._client.indices.create(
                    index=self.index_name,
                    body=self._INDEX_MAPPING,
                )
                logger.debug(f"Created OpenSearch index '{self.index_name}'")
            else:
                self._client.delete_by_query(
                    index=self.index_name,
                    body={"query": {"match_all": {}}},
                    refresh=True,
                )
                logger.debug(
                    f"Cleared existing documents from index '{self.index_name}'"
                )

            actions = [
                {
                    "_index": self.index_name,
                    "_id": chunk_id,
                    "_source": {
                        "chunk_id": chunk_id,
                        "content": content,
                    },
                }
                for chunk_id, content in documents
            ]

            success_count, _ = os_helpers.bulk(self._client, actions, refresh=True)

            span.set_attribute("opensearch.indexed_count", success_count)
            logger.debug(
                f"Indexed {success_count}/{len(documents)} documents "
                f"into '{self.index_name}'"
            )

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search indexed documents using BM25 scoring.

        Args:
            query: Search query string.
            top_k: Maximum number of results to return.

        Returns:
            List of (chunk_id, score) tuples sorted by BM25 score descending.
            Returns empty list if the index does not exist.

        Raises:
            opensearchpy.OpenSearchException: On connection errors.
        """
        with tracer.start_as_current_span(
            "opensearch.search",
            attributes={
                "opensearch.index": self.index_name,
                "opensearch.query": query,
                "opensearch.top_k": top_k,
            },
        ) as span:
            if not self._client.indices.exists(index=self.index_name):
                span.set_attribute("opensearch.result_count", 0)
                return []

            response = self._client.search(
                index=self.index_name,
                body={
                    "query": {
                        "match": {
                            "content": {
                                "query": query,
                                "operator": "or",
                            }
                        }
                    },
                    "size": top_k,
                },
            )

            results: list[tuple[str, float]] = []
            for hit in response["hits"]["hits"]:
                chunk_id = hit["_source"]["chunk_id"]
                score = float(hit["_score"])
                results.append((chunk_id, score))

            span.set_attribute("opensearch.result_count", len(results))
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
        >>> executor.build_keyword_index(documents)  # Optional for fallback
        >>> results = await executor.search(query, embedding, top_k=10)
    """

    def __init__(
        self,
        provider: str,
        collection: Any,
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.3,
        rrf_k: int = 60,
        keyword_index_config: KeywordIndexConfig | None = None,
    ) -> None:
        """Initialize the hybrid search executor.

        Args:
            provider: Vector store provider name (determines strategy)
            collection: Semantic Kernel vector store collection instance
            semantic_weight: Weight for semantic results in RRF fusion (default 0.5)
            keyword_weight: Weight for keyword results in RRF fusion (default 0.3)
            rrf_k: RRF ranking constant (default 60)
            keyword_index_config: Keyword index backend configuration.
                If None, defaults to in-memory BM25.
        """
        self.provider = provider
        self.collection = collection
        self.strategy = get_keyword_search_strategy(provider)
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.rrf_k = rrf_k
        self.keyword_index_config = keyword_index_config
        self._keyword_index: (
            InMemoryBM25KeywordProvider | OpenSearchKeywordProvider | None
        ) = None
        self._chunk_map: dict[str, DocumentChunk] = {}

        logger.debug(
            f"HybridSearchExecutor initialized: provider={provider}, "
            f"strategy={self.strategy.value}, "
            f"weights=semantic:{semantic_weight}/keyword:{keyword_weight}, "
            f"rrf_k={rrf_k}"
        )

    async def build_keyword_index(self, chunks: list[DocumentChunk]) -> None:
        """Build keyword index with graceful degradation.

        Creates the keyword search index from the provided chunks and stores
        a chunk map for ID-based lookups. Routes to the appropriate backend
        based on keyword_index_config:
        - 'opensearch': OpenSearchKeywordProvider (I/O offloaded via asyncio.to_thread)
        - 'in-memory' or None: InMemoryBM25KeywordProvider (called directly)

        If index build fails, logs a warning and continues with
        semantic-only search.

        Args:
            chunks: List of DocumentChunk objects. The chunk's
                contextualized_content (or content fallback) is indexed.

        Example:
            >>> await executor.build_keyword_index(chunks)
        """
        # Store chunk map for ID-based lookups
        self._chunk_map = {c.id: c for c in chunks}

        docs = [(c.id, c.contextualized_content or c.content) for c in chunks]

        config = self.keyword_index_config
        provider_name = config.provider if config is not None else "in-memory"

        # Route to appropriate keyword search backend with OTel instrumentation
        with tracer.start_as_current_span(
            "keyword_index.build",
            attributes={
                "keyword_index.provider": provider_name,
                "keyword_index.document_count": len(docs),
            },
        ) as span:
            try:
                provider_inst: InMemoryBM25KeywordProvider | OpenSearchKeywordProvider
                if config is not None and config.provider == "opensearch":
                    if not config.endpoint or not config.index_name:
                        raise ValueError(
                            "endpoint and index_name are required for "
                            "opensearch keyword index provider"
                        )
                    provider_inst = OpenSearchKeywordProvider(
                        endpoint=config.endpoint,
                        index_name=config.index_name,
                        username=config.username,
                        password=config.password,
                        api_key=config.api_key,
                        verify_certs=config.verify_certs,
                        timeout_seconds=config.timeout_seconds,
                    )
                else:
                    provider_inst = InMemoryBM25KeywordProvider(k1=1.5, b=0.75)

                # Offload OpenSearch I/O to thread; call in-memory directly
                if isinstance(provider_inst, OpenSearchKeywordProvider):
                    await asyncio.to_thread(provider_inst.build, docs)
                else:
                    provider_inst.build(docs)
                self._keyword_index = provider_inst
                span.set_attribute("keyword_index.status", "success")
                logger.debug(f"Built keyword index with {len(chunks)} documents")
            except Exception as e:
                span.set_attribute("keyword_index.status", "failed")
                span.record_exception(e)
                logger.warning(
                    f"Keyword index build failed, falling back to semantic-only: {e}"
                )
                self._keyword_index = None

    def get_chunk(self, chunk_id: str) -> DocumentChunk | None:
        """Look up a chunk by ID from the stored chunk map.

        Args:
            chunk_id: The unique identifier of the chunk.

        Returns:
            The DocumentChunk if found, or None.
        """
        return self._chunk_map.get(chunk_id)

    async def keyword_search(
        self, query: str, top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Perform keyword-only search.

        Args:
            query: Search query string.
            top_k: Maximum number of results to return.

        Returns:
            List of (chunk_id, score) tuples sorted by score descending.
            Returns empty list if keyword index is not built or on search error.
        """
        if self._keyword_index is None:
            return []
        try:
            # Offload OpenSearch I/O to thread; call in-memory directly
            if isinstance(self._keyword_index, OpenSearchKeywordProvider):
                return await asyncio.to_thread(self._keyword_index.search, query, top_k)
            return self._keyword_index.search(query, top_k)
        except Exception as e:
            logger.warning(f"Keyword search failed, returning empty results: {e}")
            return []

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

        # Keyword search
        bm25_results: list[tuple[str, float]] = []
        if self._keyword_index is not None:
            try:
                bm25_results = await self.keyword_search(query, top_k=top_k)
            except Exception as e:
                logger.warning(
                    f"Keyword search failed in hybrid mode, "
                    f"continuing with semantic-only: {e}"
                )
                bm25_results = []

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
