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
- Multi-field indexing (content, parent_chain, section_id, defined_term, source_file)
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
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, runtime_checkable

from opentelemetry import trace

if TYPE_CHECKING:
    from holodeck.lib.structured_chunker import DocumentChunk
    from holodeck.models.tool import KeywordIndexConfig

logger = logging.getLogger(__name__)

# OpenTelemetry tracer for search operations
tracer = trace.get_tracer("holodeck.keyword_search")


# Tokens shorter than this are dropped from the Qdrant full-text leg. Single
# characters and one-letter tokens generate too many false matches when ORed
# across thousands of chunks; two chars is the smallest useful unit (covers
# short tickers like "K", "C" would still get caught via vector similarity).
_QDRANT_MIN_TOKEN_LEN = 2
_QDRANT_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize_qdrant_query(query: str) -> list[str]:
    """Tokenize a free-text query into terms for Qdrant ``MatchText`` clauses.

    Splits on word boundaries, lowercases, and drops tokens shorter than
    :data:`_QDRANT_MIN_TOKEN_LEN`. Duplicates are removed while preserving
    first-seen order so ``MatchText`` clauses don't blow up for repeated
    keywords. Returns an empty list for queries that contain no usable
    tokens (e.g. punctuation only).
    """
    seen: set[str] = set()
    tokens: list[str] = []
    for raw in _QDRANT_TOKEN_RE.findall(query.lower()):
        if len(raw) < _QDRANT_MIN_TOKEN_LEN or raw in seen:
            continue
        seen.add(raw)
        tokens.append(raw)
    return tokens


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


class _KeywordDocumentRequired(TypedDict):
    """Required fields for KeywordDocument."""

    id: str
    content: str


class KeywordDocument(_KeywordDocumentRequired, total=False):
    """Structured document for multi-field keyword indexing.

    Contains the content and metadata fields extracted from DocumentChunk
    that are relevant for keyword-based retrieval. Fields are used for
    multi-field indexing with per-field boosting.

    Required Attributes:
        id: Unique chunk identifier.
        content: Primary text content (contextualized_content or content fallback).

    Optional Attributes:
        parent_chain: Ancestor heading chain joined with " > "
            (e.g., "Chapter 1 > Definitions").
        section_id: Document section identifier (e.g., "1.2.3", "203(a)").
        defined_term: The term being defined (if chunk_type is definition).
        chunk_type: Classification of content type
            (content, definition, requirement, etc.).
        source_file: Source filename extracted from source_path.
    """

    parent_chain: str
    section_id: str
    defined_term: str
    chunk_type: str
    source_file: str


def _build_bm25_document(doc: KeywordDocument) -> str:
    """Build composite text for BM25 indexing with implicit field boosting.

    Concatenates multiple fields into a single text string for BM25 indexing.
    Fields with higher retrieval value are repeated to simulate boosting
    (since BM25Okapi doesn't support per-field weights natively).

    Boost levels:
        - content: 1x (base)
        - parent_chain: 2x (enables heading-based navigation queries)
        - section_id: 2x (enables section number lookups)
        - defined_term: 3x (enables definition term lookups)
        - source_file: 1x (enables filename-based filtering)

    Args:
        doc: KeywordDocument with fields to index.

    Returns:
        Composite text with repeated fields for implicit boosting.

    Example:
        >>> doc = KeywordDocument(
        ...     id="chunk1",
        ...     content="Force Majeure means any event beyond...",
        ...     parent_chain="Chapter 1 > Definitions",
        ...     section_id="1.2",
        ...     defined_term="Force Majeure",
        ...     source_file="contract.pdf",
        ... )
        >>> text = _build_bm25_document(doc)
        >>> "Force Majeure" in text
        True
        >>> text.count("Force Majeure")  # 3x boost
        3
    """
    parts: list[str] = []

    # Primary content (1x weight — base)
    content = doc.get("content", "")
    if content:
        parts.append(content)

    # Parent chain headings (2x boost — enables "Chapter 3" or "Definitions" queries)
    parent_chain = doc.get("parent_chain", "")
    if parent_chain:
        parts.extend([parent_chain, parent_chain])

    # Section ID (2x boost — enables "Section 203(a)" or "1.2.3" queries)
    section_id = doc.get("section_id", "")
    if section_id:
        parts.extend([section_id, section_id])

    # Defined term (3x boost — highest priority for definition lookups)
    defined_term = doc.get("defined_term", "")
    if defined_term:
        parts.extend([defined_term] * 3)

    # Source filename (1x — useful for multi-document corpus search)
    source_file = doc.get("source_file", "")
    if source_file:
        parts.append(source_file)

    return " ".join(parts)


def _chunk_to_keyword_document(chunk: DocumentChunk) -> KeywordDocument:
    """Convert a DocumentChunk to a KeywordDocument for indexing.

    Extracts the filename from source_path, joins parent_chain with " > ",
    and maps chunk_type enum to string.

    Args:
        chunk: DocumentChunk to convert.

    Returns:
        KeywordDocument with fields extracted from the chunk.
    """
    source_file = ""
    if chunk.source_path:
        source_file = chunk.source_path.rsplit("/", 1)[-1]

    parent_chain = ""
    if chunk.parent_chain:
        parent_chain = " > ".join(chunk.parent_chain)

    chunk_type = ""
    if chunk.chunk_type:
        chunk_type = (
            chunk.chunk_type.value
            if hasattr(chunk.chunk_type, "value")
            else str(chunk.chunk_type)
        )

    return KeywordDocument(
        id=chunk.id,
        content=chunk.contextualized_content or chunk.content,
        parent_chain=parent_chain,
        section_id=chunk.section_id,
        defined_term=chunk.defined_term,
        chunk_type=chunk_type,
        source_file=source_file,
    )


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

    def build(self, documents: list[KeywordDocument]) -> None:
        """Build keyword index from documents.

        Args:
            documents: List of KeywordDocument dicts with structured fields.
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
    """In-memory BM25 keyword search provider with multi-field indexing.

    Provides keyword-based search using the BM25Okapi algorithm from
    the rank_bm25 library. Indexes multiple document fields (content,
    parent_chain, section_id, defined_term, source_file) with implicit
    boosting via field repetition.

    Attributes:
        k1: BM25 term frequency saturation parameter (default 1.5)
        b: BM25 length normalization parameter (default 0.75)

    Example:
        >>> provider = InMemoryBM25KeywordProvider()
        >>> provider.build([
        ...     KeywordDocument(id="doc1", content="The quick brown fox"),
        ...     KeywordDocument(id="doc2", content="The lazy dog"),
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

    def build(self, documents: list[KeywordDocument]) -> None:
        """Build BM25 index from structured keyword documents.

        Builds a composite text from each KeywordDocument's fields with
        implicit boosting (defined_term 3x, parent_chain 2x, section_id 2x,
        content 1x, source_file 1x), then indexes via BM25Okapi.

        Args:
            documents: List of KeywordDocument dicts with structured fields.

        Raises:
            ImportError: If rank_bm25 is not installed.

        Example:
            >>> provider.build([
            ...     KeywordDocument(
            ...         id="chunk1",
            ...         content="Force Majeure means any event...",
            ...         parent_chain="Chapter 1 > Definitions",
            ...         section_id="1.2",
            ...         defined_term="Force Majeure",
            ...     ),
            ... ])
        """
        from rank_bm25 import BM25Okapi

        self._doc_ids = [doc["id"] for doc in documents]
        tokenized = [_tokenize(_build_bm25_document(doc)) for doc in documents]
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
    """OpenSearch-backed keyword search provider with multi-field indexing.

    Uses the opensearch-py low-level client to index and search documents
    via BM25 scoring on an external OpenSearch cluster. Implements the
    KeywordSearchProvider protocol.

    Indexes multiple fields from KeywordDocument with per-field boosting:
    - content: Primary text (standard analyzer)
    - parent_chain: Heading hierarchy (standard analyzer, 2x boost)
    - section_id: Section identifiers (simple analyzer, 2x boost)
    - defined_term: Definition terms (standard analyzer, 3x boost)
    - chunk_type: Content classification (keyword type, filterable)
    - source_file: Source filename (simple analyzer)

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
        >>> provider.build([KeywordDocument(id="doc1", content="The quick brown fox")])
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
                "parent_chain": {"type": "text", "analyzer": "standard"},
                "section_id": {"type": "text", "analyzer": "simple"},
                "defined_term": {"type": "text", "analyzer": "standard"},
                "chunk_type": {"type": "keyword"},
                "source_file": {"type": "text", "analyzer": "simple"},
            }
        },
    }
    """Multi-field index mapping for structured document search."""

    _SEARCH_FIELDS: list[str] = [
        "content",
        "parent_chain^2",
        "section_id^2",
        "defined_term^3",
        "source_file",
    ]
    """Fields to search with per-field boost factors for multi_match queries."""

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

    def build(self, documents: list[KeywordDocument]) -> None:
        """Build OpenSearch index from structured keyword documents.

        Creates the index if it does not exist. If the index already
        exists, clears all existing documents before re-indexing.
        Uses bulk indexing with refresh=True for immediate searchability.

        Each KeywordDocument's fields are indexed into separate OpenSearch
        fields with per-field boosting applied at query time via multi_match.

        Args:
            documents: List of KeywordDocument dicts with structured fields.

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
                    "_id": doc["id"],
                    "_source": {
                        "chunk_id": doc["id"],
                        "content": doc.get("content", ""),
                        "parent_chain": doc.get("parent_chain", ""),
                        "section_id": doc.get("section_id", ""),
                        "defined_term": doc.get("defined_term", ""),
                        "chunk_type": doc.get("chunk_type", ""),
                        "source_file": doc.get("source_file", ""),
                    },
                }
                for doc in documents
            ]

            success_count, _ = os_helpers.bulk(self._client, actions, refresh=True)

            span.set_attribute("opensearch.indexed_count", success_count)
            logger.debug(
                f"Indexed {success_count}/{len(documents)} documents "
                f"into '{self.index_name}'"
            )

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search indexed documents using multi-field BM25 scoring.

        Uses a multi_match query across all indexed fields with per-field
        boost factors (defined_term^3, parent_chain^2, section_id^2,
        content, source_file).

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
                        "multi_match": {
                            "query": query,
                            "fields": self._SEARCH_FIELDS,
                            "type": "best_fields",
                            "operator": "or",
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

        Each chunk is converted to a KeywordDocument with multiple fields
        (content, parent_chain, section_id, defined_term, source_file) for
        multi-field indexing with per-field boosting.

        If index build fails, logs a warning and continues with
        semantic-only search.

        Args:
            chunks: List of DocumentChunk objects. All relevant fields
                (content, parent_chain, section_id, defined_term, etc.)
                are indexed for keyword search.

        Example:
            >>> await executor.build_keyword_index(chunks)
        """
        # Store chunk map for ID-based lookups
        self._chunk_map = {c.id: c for c in chunks}

        # Native-hybrid providers (qdrant) handle the keyword leg inside the
        # vector store at query time (see _native_hybrid_search_*). The
        # in-process BM25/OpenSearch index would never be read, so skip the
        # tokenize-and-build pass entirely. _chunk_map is still useful as a
        # cache for get_chunks() lookups; it just doesn't need to be the
        # source of truth for the keyword index.
        if self.strategy == KeywordSearchStrategy.NATIVE_HYBRID:
            logger.debug(
                "Skipping in-process keyword index build for native-hybrid "
                f"provider '{self.provider}' "
                f"(strategy={self.strategy.value})"
            )
            return

        # Convert chunks to structured keyword documents
        docs = [_chunk_to_keyword_document(c) for c in chunks]

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
                        password=(
                            config.password.get_secret_value()
                            if config.password is not None
                            else None
                        ),
                        api_key=(
                            config.api_key.get_secret_value()
                            if config.api_key is not None
                            else None
                        ),
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

    def cache_chunk(self, chunk: DocumentChunk) -> None:
        """Insert a chunk into the in-memory cache by id.

        Used by the native-hybrid path (qdrant) which lazily fetches chunks
        on cache miss instead of preloading the entire corpus at startup.
        Subsequent ``get_chunk`` calls for the same id resolve locally.

        Args:
            chunk: The DocumentChunk to cache. Indexed by ``chunk.id``.
        """
        self._chunk_map[chunk.id] = chunk

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
        """Execute native hybrid search using the provider's hybrid API.

        Dispatches by provider because SK's generic ``coll.hybrid_search()``
        is broken for Qdrant: it builds the keyword leg with
        ``MatchAny(any=[kw])`` (exact-value equality) over a single ``values``
        argument that gets iterated character-by-character when a string is
        passed. The result is a keyword filter that matches zero documents
        in a free-text field, silently degrading hybrid → vector-only.
        For Qdrant we drop down to the raw client and use ``MatchText``
        (Qdrant's full-text primitive) per whitespace-tokenized query term.
        Other native-hybrid providers keep the SK path.

        Args:
            query: Search query string
            query_embedding: Pre-computed query embedding vector
            top_k: Maximum number of results

        Returns:
            List of (chunk_id, score) tuples
        """
        if self.provider == "qdrant":
            return await self._native_hybrid_search_qdrant(
                query, query_embedding, top_k
            )

        results: list[tuple[str, float]] = []

        async with self.collection as coll:
            search_results = await coll.hybrid_search(
                query,
                vector=query_embedding,
                vector_property_name="embedding",
                # `searchable_text` is the concat field carrying
                # contextualized_content + parent_chain + section_id +
                # defined_term + cross_refs + filename. Built at write
                # time by HierarchicalDocumentTool._build_searchable_text;
                # marked is_full_text_indexed=True on the record class.
                additional_property_name="searchable_text",
                top=top_k,
            )

            async for result in search_results.results:
                record = result.record if hasattr(result, "record") else result[0]
                score = result.score if hasattr(result, "score") else result[1]
                results.append((record.id, float(score)))

        return results

    async def _native_hybrid_search_qdrant(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[tuple[str, float]]:
        """Qdrant-native hybrid search bypassing SK's broken keyword leg.

        Builds a two-source RRF fusion query directly against the Qdrant
        client: one vector prefetch and one full-text prefetch where each
        token from the query is matched with ``MatchText`` (the correct
        full-text primitive) and OR-combined via ``should=``. Uses a wider
        prefetch pool than SK's hardcoded ``limit=options.top`` so the
        fusion stage actually has candidates to merge.

        If the query tokenizes to nothing (e.g. punctuation only), falls
        back to vector-only so ``FusionQuery`` still has a non-empty source.

        Args:
            query: Search query string
            query_embedding: Pre-computed query embedding vector
            top_k: Maximum number of results

        Returns:
            List of (chunk_id, score) tuples
        """
        from qdrant_client.http import models as qm

        tokens = _tokenize_qdrant_query(query)
        # Prefetch ~4x the requested top_k (min 20) so RRF has enough
        # candidates from each leg to find overlap. SK hardcodes top_k,
        # which makes hybrid degenerate to vector-only on narrow queries.
        prefetch_limit = max(top_k * 4, 20)

        prefetch: list[qm.Prefetch] = [
            qm.Prefetch(
                query=query_embedding,
                using="embedding",
                limit=prefetch_limit,
            ),
        ]
        if tokens:
            text_clauses = [
                qm.FieldCondition(
                    key="searchable_text",
                    match=qm.MatchText(text=tok),
                )
                for tok in tokens
            ]
            prefetch.append(
                qm.Prefetch(
                    filter=qm.Filter(should=text_clauses),
                    limit=prefetch_limit,
                )
            )

        async with self.collection as coll:
            client = coll.qdrant_client
            response = await client.query_points(
                collection_name=coll.collection_name,
                prefetch=prefetch,
                query=qm.FusionQuery(fusion=qm.Fusion.RRF),
                limit=top_k,
                with_payload=False,
                with_vectors=False,
            )

        return [(str(p.id), float(p.score)) for p in response.points]

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
