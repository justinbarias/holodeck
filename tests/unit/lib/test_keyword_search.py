"""Tests for keyword search module.

This module tests the tiered keyword search implementation including:
- KeywordSearchStrategy enum
- Provider capability detection
- KeywordDocument and composite text building
- KeywordSearchProvider protocol
- InMemoryBM25KeywordProvider multi-field implementation
- OpenSearchKeywordProvider multi-field implementation
- HybridSearchExecutor with provider router
- OpenTelemetry instrumentation
- Graceful degradation
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    pass


class TestKeywordSearchStrategy:
    """Test KeywordSearchStrategy enum values."""

    def test_enum_has_native_hybrid(self) -> None:
        """Test NATIVE_HYBRID enum value exists."""
        from holodeck.lib.keyword_search import KeywordSearchStrategy

        assert hasattr(KeywordSearchStrategy, "NATIVE_HYBRID")
        assert KeywordSearchStrategy.NATIVE_HYBRID.value == "native_hybrid"

    def test_enum_has_fallback_bm25(self) -> None:
        """Test FALLBACK_BM25 enum value exists."""
        from holodeck.lib.keyword_search import KeywordSearchStrategy

        assert hasattr(KeywordSearchStrategy, "FALLBACK_BM25")
        assert KeywordSearchStrategy.FALLBACK_BM25.value == "fallback_bm25"


class TestProviderSets:
    """Test provider capability sets."""

    def test_native_hybrid_providers_contains_expected(self) -> None:
        """Test NATIVE_HYBRID_PROVIDERS contains 5 expected providers."""
        from holodeck.lib.keyword_search import NATIVE_HYBRID_PROVIDERS

        expected = {
            "azure-ai-search",
            "weaviate",
            "qdrant",
            "mongodb",
            "azure-cosmos-nosql",
        }
        assert expected == NATIVE_HYBRID_PROVIDERS
        assert len(NATIVE_HYBRID_PROVIDERS) == 5

    def test_fallback_bm25_providers_contains_expected(self) -> None:
        """Test FALLBACK_BM25_PROVIDERS contains 6 expected providers."""
        from holodeck.lib.keyword_search import FALLBACK_BM25_PROVIDERS

        expected = {
            "postgres",
            "pinecone",
            "chromadb",
            "faiss",
            "in-memory",
            "sql-server",
        }
        assert expected == FALLBACK_BM25_PROVIDERS
        assert len(FALLBACK_BM25_PROVIDERS) == 6

    def test_azure_cosmos_mongo_excluded(self) -> None:
        """Test azure-cosmos-mongo is excluded from both sets (no hybrid support)."""
        from holodeck.lib.keyword_search import (
            FALLBACK_BM25_PROVIDERS,
            NATIVE_HYBRID_PROVIDERS,
        )

        assert "azure-cosmos-mongo" not in NATIVE_HYBRID_PROVIDERS
        assert "azure-cosmos-mongo" not in FALLBACK_BM25_PROVIDERS


class TestGetKeywordSearchStrategy:
    """Test strategy factory function."""

    def test_returns_native_for_azure_ai_search(self) -> None:
        """Test azure-ai-search returns NATIVE_HYBRID strategy."""
        from holodeck.lib.keyword_search import (
            KeywordSearchStrategy,
            get_keyword_search_strategy,
        )

        result = get_keyword_search_strategy("azure-ai-search")
        assert result == KeywordSearchStrategy.NATIVE_HYBRID

    def test_returns_native_for_weaviate(self) -> None:
        """Test weaviate returns NATIVE_HYBRID strategy."""
        from holodeck.lib.keyword_search import (
            KeywordSearchStrategy,
            get_keyword_search_strategy,
        )

        result = get_keyword_search_strategy("weaviate")
        assert result == KeywordSearchStrategy.NATIVE_HYBRID

    def test_returns_native_for_qdrant(self) -> None:
        """Test qdrant returns NATIVE_HYBRID strategy."""
        from holodeck.lib.keyword_search import (
            KeywordSearchStrategy,
            get_keyword_search_strategy,
        )

        result = get_keyword_search_strategy("qdrant")
        assert result == KeywordSearchStrategy.NATIVE_HYBRID

    def test_returns_native_for_mongodb(self) -> None:
        """Test mongodb returns NATIVE_HYBRID strategy."""
        from holodeck.lib.keyword_search import (
            KeywordSearchStrategy,
            get_keyword_search_strategy,
        )

        result = get_keyword_search_strategy("mongodb")
        assert result == KeywordSearchStrategy.NATIVE_HYBRID

    def test_returns_native_for_azure_cosmos_nosql(self) -> None:
        """Test azure-cosmos-nosql returns NATIVE_HYBRID strategy."""
        from holodeck.lib.keyword_search import (
            KeywordSearchStrategy,
            get_keyword_search_strategy,
        )

        result = get_keyword_search_strategy("azure-cosmos-nosql")
        assert result == KeywordSearchStrategy.NATIVE_HYBRID

    def test_returns_fallback_for_postgres(self) -> None:
        """Test postgres returns FALLBACK_BM25 strategy."""
        from holodeck.lib.keyword_search import (
            KeywordSearchStrategy,
            get_keyword_search_strategy,
        )

        result = get_keyword_search_strategy("postgres")
        assert result == KeywordSearchStrategy.FALLBACK_BM25

    def test_returns_fallback_for_in_memory(self) -> None:
        """Test in-memory returns FALLBACK_BM25 strategy."""
        from holodeck.lib.keyword_search import (
            KeywordSearchStrategy,
            get_keyword_search_strategy,
        )

        result = get_keyword_search_strategy("in-memory")
        assert result == KeywordSearchStrategy.FALLBACK_BM25

    def test_returns_fallback_for_unknown_provider(self) -> None:
        """Test unknown provider returns FALLBACK_BM25 strategy."""
        from holodeck.lib.keyword_search import (
            KeywordSearchStrategy,
            get_keyword_search_strategy,
        )

        result = get_keyword_search_strategy("unknown-provider")
        assert result == KeywordSearchStrategy.FALLBACK_BM25


class TestKeywordDocument:
    """Test KeywordDocument TypedDict structure."""

    def test_required_fields_only(self) -> None:
        """Test KeywordDocument can be created with only required fields."""
        from holodeck.lib.keyword_search import KeywordDocument

        doc = KeywordDocument(id="doc1", content="hello world")
        assert doc["id"] == "doc1"
        assert doc["content"] == "hello world"

    def test_all_fields(self) -> None:
        """Test KeywordDocument with all optional fields."""
        from holodeck.lib.keyword_search import KeywordDocument

        doc = KeywordDocument(
            id="doc1",
            content="Force Majeure means...",
            parent_chain="Chapter 1 > Definitions",
            section_id="1.2",
            defined_term="Force Majeure",
            chunk_type="definition",
            source_file="contract.pdf",
        )
        assert doc["parent_chain"] == "Chapter 1 > Definitions"
        assert doc["section_id"] == "1.2"
        assert doc["defined_term"] == "Force Majeure"
        assert doc["chunk_type"] == "definition"
        assert doc["source_file"] == "contract.pdf"


class TestBuildBm25Document:
    """Test _build_bm25_document composite text builder."""

    def test_content_only(self) -> None:
        """Test composite text with only content field."""
        from holodeck.lib.keyword_search import KeywordDocument, _build_bm25_document

        doc = KeywordDocument(id="doc1", content="hello world")
        result = _build_bm25_document(doc)
        assert result == "hello world"

    def test_parent_chain_boosted_2x(self) -> None:
        """Test parent_chain appears twice in composite text."""
        from holodeck.lib.keyword_search import KeywordDocument, _build_bm25_document

        doc = KeywordDocument(
            id="doc1",
            content="some content",
            parent_chain="Chapter 1 > Definitions",
        )
        result = _build_bm25_document(doc)
        assert result.count("Chapter 1 > Definitions") == 2

    def test_section_id_boosted_2x(self) -> None:
        """Test section_id appears twice in composite text."""
        from holodeck.lib.keyword_search import KeywordDocument, _build_bm25_document

        doc = KeywordDocument(
            id="doc1",
            content="some content",
            section_id="203(a)",
        )
        result = _build_bm25_document(doc)
        assert result.count("203(a)") == 2

    def test_defined_term_boosted_3x(self) -> None:
        """Test defined_term appears three times in composite text."""
        from holodeck.lib.keyword_search import KeywordDocument, _build_bm25_document

        doc = KeywordDocument(
            id="doc1",
            content="Force Majeure means any event beyond...",
            defined_term="Force Majeure",
        )
        result = _build_bm25_document(doc)
        # Content has "Force Majeure" once, defined_term adds 3 more
        assert result.count("Force Majeure") == 4

    def test_source_file_included(self) -> None:
        """Test source_file appears in composite text."""
        from holodeck.lib.keyword_search import KeywordDocument, _build_bm25_document

        doc = KeywordDocument(
            id="doc1",
            content="some content",
            source_file="contract.pdf",
        )
        result = _build_bm25_document(doc)
        assert "contract.pdf" in result

    def test_empty_optional_fields_excluded(self) -> None:
        """Test empty optional fields don't add extra text."""
        from holodeck.lib.keyword_search import KeywordDocument, _build_bm25_document

        doc = KeywordDocument(
            id="doc1",
            content="just content",
            parent_chain="",
            section_id="",
            defined_term="",
            source_file="",
        )
        result = _build_bm25_document(doc)
        assert result == "just content"

    def test_all_fields_combined(self) -> None:
        """Test all fields produce expected composite text."""
        from holodeck.lib.keyword_search import KeywordDocument, _build_bm25_document

        doc = KeywordDocument(
            id="doc1",
            content="Definition text",
            parent_chain="Ch1 > Defs",
            section_id="1.2",
            defined_term="Term",
            source_file="doc.pdf",
        )
        result = _build_bm25_document(doc)

        # Verify presence and boost counts
        assert "Definition text" in result
        assert result.count("Ch1 > Defs") == 2
        assert result.count("1.2") == 2
        assert result.count("Term") >= 3  # 3x from defined_term
        assert "doc.pdf" in result


class TestChunkToKeywordDocument:
    """Test _chunk_to_keyword_document converter."""

    def test_basic_conversion(self) -> None:
        """Test basic DocumentChunk to KeywordDocument conversion."""
        from holodeck.lib.keyword_search import _chunk_to_keyword_document
        from holodeck.lib.structured_chunker import DocumentChunk

        chunk = DocumentChunk(
            id="chunk1",
            source_path="/docs/contract.pdf",
            chunk_index=0,
            content="Test content",
            contextualized_content="Contextualized test content",
        )

        doc = _chunk_to_keyword_document(chunk)

        assert doc["id"] == "chunk1"
        assert doc["content"] == "Contextualized test content"
        assert doc["source_file"] == "contract.pdf"

    def test_prefers_contextualized_content(self) -> None:
        """Test prefers contextualized_content over content."""
        from holodeck.lib.keyword_search import _chunk_to_keyword_document
        from holodeck.lib.structured_chunker import DocumentChunk

        chunk = DocumentChunk(
            id="chunk1",
            source_path="/test.md",
            chunk_index=0,
            content="raw content",
            contextualized_content="better content with context",
        )

        doc = _chunk_to_keyword_document(chunk)
        assert doc["content"] == "better content with context"

    def test_falls_back_to_content(self) -> None:
        """Test falls back to content when contextualized_content is empty."""
        from holodeck.lib.keyword_search import _chunk_to_keyword_document
        from holodeck.lib.structured_chunker import DocumentChunk

        chunk = DocumentChunk(
            id="chunk1",
            source_path="/test.md",
            chunk_index=0,
            content="raw content",
        )

        doc = _chunk_to_keyword_document(chunk)
        assert doc["content"] == "raw content"

    def test_parent_chain_joined(self) -> None:
        """Test parent_chain list is joined with ' > '."""
        from holodeck.lib.keyword_search import _chunk_to_keyword_document
        from holodeck.lib.structured_chunker import DocumentChunk

        chunk = DocumentChunk(
            id="chunk1",
            source_path="/test.md",
            chunk_index=0,
            content="Test",
            parent_chain=["Chapter 1", "Section 1.1", "Definitions"],
        )

        doc = _chunk_to_keyword_document(chunk)
        assert doc["parent_chain"] == "Chapter 1 > Section 1.1 > Definitions"

    def test_section_id_preserved(self) -> None:
        """Test section_id is preserved in output."""
        from holodeck.lib.keyword_search import _chunk_to_keyword_document
        from holodeck.lib.structured_chunker import DocumentChunk

        chunk = DocumentChunk(
            id="chunk1",
            source_path="/test.md",
            chunk_index=0,
            content="Test",
            section_id="203(a)(1)",
        )

        doc = _chunk_to_keyword_document(chunk)
        assert doc["section_id"] == "203(a)(1)"

    def test_defined_term_preserved(self) -> None:
        """Test defined_term is preserved in output."""
        from holodeck.lib.keyword_search import _chunk_to_keyword_document
        from holodeck.lib.structured_chunker import DocumentChunk

        chunk = DocumentChunk(
            id="chunk1",
            source_path="/test.md",
            chunk_index=0,
            content="Test",
            defined_term="Force Majeure",
        )

        doc = _chunk_to_keyword_document(chunk)
        assert doc["defined_term"] == "Force Majeure"

    def test_chunk_type_enum_converted_to_string(self) -> None:
        """Test ChunkType enum is converted to string value."""
        from holodeck.lib.keyword_search import _chunk_to_keyword_document
        from holodeck.lib.structured_chunker import ChunkType, DocumentChunk

        chunk = DocumentChunk(
            id="chunk1",
            source_path="/test.md",
            chunk_index=0,
            content="Test",
            chunk_type=ChunkType.DEFINITION,
        )

        doc = _chunk_to_keyword_document(chunk)
        assert doc["chunk_type"] == "definition"

    def test_source_file_extracted_from_path(self) -> None:
        """Test filename is extracted from source_path."""
        from holodeck.lib.keyword_search import _chunk_to_keyword_document
        from holodeck.lib.structured_chunker import DocumentChunk

        chunk = DocumentChunk(
            id="chunk1",
            source_path="/very/deep/path/to/document.pdf",
            chunk_index=0,
            content="Test",
        )

        doc = _chunk_to_keyword_document(chunk)
        assert doc["source_file"] == "document.pdf"


class TestInMemoryBM25KeywordProvider:
    """Test BM25 fallback implementation with multi-field indexing."""

    def test_init_with_default_params(self) -> None:
        """Test InMemoryBM25KeywordProvider initializes with default k1 and b."""
        from holodeck.lib.keyword_search import InMemoryBM25KeywordProvider

        provider = InMemoryBM25KeywordProvider()
        assert provider.k1 == 1.5
        assert provider.b == 0.75

    def test_init_with_custom_params(self) -> None:
        """Test InMemoryBM25KeywordProvider accepts custom k1 and b."""
        from holodeck.lib.keyword_search import InMemoryBM25KeywordProvider

        provider = InMemoryBM25KeywordProvider(k1=2.0, b=0.5)
        assert provider.k1 == 2.0
        assert provider.b == 0.5

    def test_build_indexes_documents(self) -> None:
        """Test build() creates index from KeywordDocument list."""
        from holodeck.lib.keyword_search import (
            InMemoryBM25KeywordProvider,
            KeywordDocument,
        )

        provider = InMemoryBM25KeywordProvider()
        documents = [
            KeywordDocument(id="doc1", content="the quick brown fox"),
            KeywordDocument(id="doc2", content="jumps over the lazy dog"),
            KeywordDocument(id="doc3", content="the brown dog is quick"),
        ]
        provider.build(documents)

        assert provider._bm25 is not None
        assert provider._doc_ids == ["doc1", "doc2", "doc3"]

    def test_build_uses_composite_text(self) -> None:
        """Test build() indexes composite text with all fields."""
        from holodeck.lib.keyword_search import (
            InMemoryBM25KeywordProvider,
            KeywordDocument,
        )

        provider = InMemoryBM25KeywordProvider()
        documents = [
            KeywordDocument(
                id="doc1",
                content="Context: Section about foxes. The quick brown fox jumps.",
            ),
            KeywordDocument(
                id="doc2",
                content="Context: Section about dogs. The lazy dog sleeps here.",
            ),
            KeywordDocument(
                id="doc3",
                content="Context: Section about cats. The small cat plays nicely.",
            ),
            KeywordDocument(
                id="doc4",
                content="Context: Section about birds. The blue bird sings daily.",
            ),
            KeywordDocument(
                id="doc5",
                content="Context: Section about fish. The gold fish swims around.",
            ),
        ]
        provider.build(documents)

        results = provider.search("quick jumps", top_k=5)
        assert len(results) > 0
        assert results[0][0] == "doc1"

    def test_search_returns_ranked_results(self) -> None:
        """Test search() returns (doc_id, score) tuples sorted by score."""
        from holodeck.lib.keyword_search import (
            InMemoryBM25KeywordProvider,
            KeywordDocument,
        )

        provider = InMemoryBM25KeywordProvider()
        documents = [
            KeywordDocument(id="doc1", content="the quick brown fox"),
            KeywordDocument(id="doc2", content="jumps over the lazy dog"),
            KeywordDocument(id="doc3", content="the brown dog is quick"),
        ]
        provider.build(documents)

        results = provider.search("brown fox", top_k=3)

        assert len(results) > 0
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)
        assert all(isinstance(r[0], str) for r in results)
        assert all(isinstance(r[1], int | float) for r in results)
        assert results[0][0] == "doc1"

    def test_search_respects_top_k(self) -> None:
        """Test search() returns at most top_k results."""
        from holodeck.lib.keyword_search import (
            InMemoryBM25KeywordProvider,
            KeywordDocument,
        )

        provider = InMemoryBM25KeywordProvider()
        documents = [
            KeywordDocument(id="doc1", content="the quick brown fox"),
            KeywordDocument(id="doc2", content="jumps over the lazy dog"),
            KeywordDocument(id="doc3", content="the brown dog is quick"),
            KeywordDocument(id="doc4", content="another document here"),
            KeywordDocument(id="doc5", content="yet another one"),
        ]
        provider.build(documents)

        results = provider.search("the", top_k=2)
        assert len(results) <= 2

    def test_search_returns_empty_if_no_index(self) -> None:
        """Test search() returns empty list if index not built."""
        from holodeck.lib.keyword_search import InMemoryBM25KeywordProvider

        provider = InMemoryBM25KeywordProvider()
        results = provider.search("test query", top_k=5)
        assert results == []

    def test_defined_term_boosting_ranks_definitions_higher(self) -> None:
        """Test defined_term boosting makes definition chunks rank higher."""
        from holodeck.lib.keyword_search import (
            InMemoryBM25KeywordProvider,
            KeywordDocument,
        )

        provider = InMemoryBM25KeywordProvider()
        documents = [
            KeywordDocument(
                id="regular",
                content="Force Majeure events may delay delivery per the contract",
            ),
            KeywordDocument(
                id="definition",
                content="means any event beyond reasonable control",
                defined_term="Force Majeure",
            ),
            KeywordDocument(id="filler1", content="unrelated document about pricing"),
            KeywordDocument(id="filler2", content="another unrelated topic entirely"),
            KeywordDocument(id="filler3", content="yet more filler content here"),
        ]
        provider.build(documents)

        results = provider.search("Force Majeure", top_k=3)
        result_ids = [r[0] for r in results]

        # The definition chunk should rank first due to 3x boost
        assert result_ids[0] == "definition"

    def test_parent_chain_boosting_ranks_headings_higher(self) -> None:
        """Test parent_chain boosting helps section-based queries."""
        from holodeck.lib.keyword_search import (
            InMemoryBM25KeywordProvider,
            KeywordDocument,
        )

        provider = InMemoryBM25KeywordProvider()
        documents = [
            KeywordDocument(
                id="in_definitions",
                content="This term refers to contractual obligations",
                parent_chain="Chapter 3 > Definitions",
            ),
            KeywordDocument(
                id="mentions_definitions",
                content="Definitions are provided in the glossary section",
            ),
            KeywordDocument(id="filler1", content="unrelated pricing information"),
            KeywordDocument(id="filler2", content="delivery schedule details"),
            KeywordDocument(id="filler3", content="contact information section"),
        ]
        provider.build(documents)

        results = provider.search("Definitions", top_k=3)
        result_ids = [r[0] for r in results]

        # The chunk with "Definitions" in parent_chain (2x boost) should rank first
        assert result_ids[0] == "in_definitions"

    def test_section_id_boosting_enables_section_lookup(self) -> None:
        """Test section_id boosting enables section number queries."""
        from holodeck.lib.keyword_search import (
            InMemoryBM25KeywordProvider,
            KeywordDocument,
        )

        provider = InMemoryBM25KeywordProvider()
        documents = [
            KeywordDocument(
                id="section_203",
                content="This section describes reporting obligations",
                section_id="203",
            ),
            KeywordDocument(
                id="mentions_203",
                content="As described in section 203 of this agreement",
            ),
            KeywordDocument(id="filler1", content="unrelated filler content"),
            KeywordDocument(id="filler2", content="more unrelated content here"),
            KeywordDocument(id="filler3", content="yet another filler document"),
        ]
        provider.build(documents)

        results = provider.search("203", top_k=3)
        result_ids = [r[0] for r in results]

        # The chunk with section_id="203" (2x boost) should rank first
        assert result_ids[0] == "section_203"


class TestTokenize:
    """Test tokenization function."""

    def test_tokenize_basic(self) -> None:
        """Test basic tokenization."""
        from holodeck.lib.keyword_search import _tokenize

        result = _tokenize("Hello World")
        assert result == ["hello", "world"]

    def test_tokenize_empty_string(self) -> None:
        """Test tokenize handles empty string."""
        from holodeck.lib.keyword_search import _tokenize

        result = _tokenize("")
        assert result == []

    def test_tokenize_special_chars(self) -> None:
        """Test tokenize removes special characters."""
        from holodeck.lib.keyword_search import _tokenize

        result = _tokenize("Section 203(a)(1): Force Majeure!")
        assert "section" in result
        assert "203" in result
        assert "a" in result
        assert "1" in result
        assert "force" in result
        assert "majeure" in result
        # Special chars not in tokens
        assert "(" not in result
        assert ")" not in result
        assert ":" not in result
        assert "!" not in result

    def test_tokenize_lowercase(self) -> None:
        """Test tokenize converts to lowercase."""
        from holodeck.lib.keyword_search import _tokenize

        result = _tokenize("UPPERCASE lowercase MixedCase")
        assert result == ["uppercase", "lowercase", "mixedcase"]

    def test_tokenize_numbers(self) -> None:
        """Test tokenize keeps numbers."""
        from holodeck.lib.keyword_search import _tokenize

        result = _tokenize("Section 123 and 456")
        assert "123" in result
        assert "456" in result


class TestHybridSearchExecutor:
    """Test hybrid search executor."""

    def test_init_sets_strategy_for_native_provider(self) -> None:
        """Test executor uses NATIVE_HYBRID for supported providers."""
        from holodeck.lib.keyword_search import (
            HybridSearchExecutor,
            KeywordSearchStrategy,
        )

        mock_collection = MagicMock()
        executor = HybridSearchExecutor("azure-ai-search", mock_collection)

        assert executor.strategy == KeywordSearchStrategy.NATIVE_HYBRID

    def test_init_sets_strategy_for_fallback_provider(self) -> None:
        """Test executor uses FALLBACK_BM25 for unsupported providers."""
        from holodeck.lib.keyword_search import (
            HybridSearchExecutor,
            KeywordSearchStrategy,
        )

        mock_collection = MagicMock()
        executor = HybridSearchExecutor("in-memory", mock_collection)

        assert executor.strategy == KeywordSearchStrategy.FALLBACK_BM25

    @pytest.mark.asyncio
    async def test_build_keyword_index_creates_index(self) -> None:
        """Test build_keyword_index() creates keyword index and chunk map."""
        from holodeck.lib.keyword_search import HybridSearchExecutor
        from holodeck.lib.structured_chunker import DocumentChunk

        mock_collection = MagicMock()
        executor = HybridSearchExecutor("in-memory", mock_collection)

        chunks = [
            DocumentChunk(
                id="chunk1",
                source_path="/test.md",
                chunk_index=0,
                content="Content about reporting requirements",
                contextualized_content="Content about reporting requirements",
            ),
            DocumentChunk(
                id="chunk2",
                source_path="/test.md",
                chunk_index=1,
                content="Content about compliance",
                contextualized_content="Content about compliance",
            ),
        ]
        await executor.build_keyword_index(chunks)

        assert executor._keyword_index is not None
        assert executor.get_chunk("chunk1") is chunks[0]
        assert executor.get_chunk("chunk2") is chunks[1]
        assert executor.get_chunk("nonexistent") is None

    @pytest.mark.asyncio
    async def test_build_keyword_index_with_structured_fields(self) -> None:
        """Test build_keyword_index indexes parent_chain, section_id, etc."""
        from holodeck.lib.keyword_search import HybridSearchExecutor
        from holodeck.lib.structured_chunker import ChunkType, DocumentChunk

        mock_collection = MagicMock()
        executor = HybridSearchExecutor("in-memory", mock_collection)

        chunks = [
            DocumentChunk(
                id="def_chunk",
                source_path="/contract.pdf",
                chunk_index=0,
                content="means any event beyond reasonable control",
                parent_chain=["Chapter 1", "Definitions"],
                section_id="1.2",
                chunk_type=ChunkType.DEFINITION,
                defined_term="Force Majeure",
            ),
            DocumentChunk(
                id="regular_chunk",
                source_path="/contract.pdf",
                chunk_index=1,
                content="The parties agree to the following terms",
            ),
            DocumentChunk(
                id="filler1",
                source_path="/other.pdf",
                chunk_index=0,
                content="Completely unrelated pricing document",
            ),
            DocumentChunk(
                id="filler2",
                source_path="/other.pdf",
                chunk_index=1,
                content="Another filler document about logistics",
            ),
            DocumentChunk(
                id="filler3",
                source_path="/other.pdf",
                chunk_index=2,
                content="Supply chain management details",
            ),
        ]
        await executor.build_keyword_index(chunks)

        # Search for defined term should find the definition chunk
        results = await executor.keyword_search("Force Majeure", top_k=3)
        assert len(results) > 0
        assert results[0][0] == "def_chunk"

    @pytest.mark.asyncio
    async def test_search_routes_to_native_hybrid(self) -> None:
        """Test search() routes to native hybrid_search for supported providers."""
        from holodeck.lib.keyword_search import HybridSearchExecutor

        # Mock collection with async context manager
        mock_result = MagicMock()
        mock_result.record = MagicMock()
        mock_result.record.id = "chunk1"
        mock_result.score = 0.9

        async def mock_results():
            yield mock_result

        mock_search_results = MagicMock()
        mock_search_results.results = mock_results()

        mock_collection = MagicMock()
        mock_coll_instance = AsyncMock()
        mock_coll_instance.hybrid_search = AsyncMock(return_value=mock_search_results)
        mock_collection.__aenter__ = AsyncMock(return_value=mock_coll_instance)
        mock_collection.__aexit__ = AsyncMock(return_value=None)

        executor = HybridSearchExecutor("weaviate", mock_collection)

        await executor.search("test query", [0.1, 0.2, 0.3], top_k=5)

        # Should have called hybrid_search
        mock_coll_instance.hybrid_search.assert_called_once()

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_bm25_failure(self) -> None:
        """Test falls back to semantic-only when BM25 build fails."""
        from holodeck.lib.keyword_search import HybridSearchExecutor

        mock_collection = MagicMock()
        mock_coll_instance = AsyncMock()

        async def mock_results():
            yield MagicMock(record=MagicMock(id="chunk1"), score=0.9)

        mock_search_results = MagicMock()
        mock_search_results.results = mock_results()
        mock_coll_instance.search = AsyncMock(return_value=mock_search_results)

        mock_collection.__aenter__ = AsyncMock(return_value=mock_coll_instance)
        mock_collection.__aexit__ = AsyncMock(return_value=None)

        executor = HybridSearchExecutor("in-memory", mock_collection)

        # Simulate keyword index not being available
        executor._keyword_index = None

        # Search should still work (semantic only)
        results = await executor.search("test query", [0.1, 0.2, 0.3], top_k=5)

        # Should return results from semantic search
        assert len(results) > 0


class TestOpenTelemetry:
    """Test OpenTelemetry instrumentation."""

    def test_bm25_search_emits_span(self) -> None:
        """Test InMemoryBM25KeywordProvider.search() emits OpenTelemetry span."""
        from holodeck.lib.keyword_search import (
            InMemoryBM25KeywordProvider,
            KeywordDocument,
        )

        provider = InMemoryBM25KeywordProvider()
        documents = [KeywordDocument(id="doc1", content="test content")]
        provider.build(documents)

        with patch("holodeck.lib.keyword_search.tracer") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
                return_value=mock_span
            )
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
                return_value=None
            )

            provider.search("test", top_k=5)

            # Verify span was started with correct name
            mock_tracer.start_as_current_span.assert_called_once()
            call_args = mock_tracer.start_as_current_span.call_args
            assert "keyword.search.in_memory_bm25" in str(call_args)

    @pytest.mark.asyncio
    async def test_hybrid_executor_emits_span(self) -> None:
        """Test HybridSearchExecutor.search() emits OpenTelemetry span."""
        from holodeck.lib.keyword_search import HybridSearchExecutor

        mock_collection = MagicMock()
        mock_coll_instance = AsyncMock()

        async def mock_results():
            yield MagicMock(record=MagicMock(id="chunk1"), score=0.9)

        mock_search_results = MagicMock()
        mock_search_results.results = mock_results()
        mock_coll_instance.search = AsyncMock(return_value=mock_search_results)

        mock_collection.__aenter__ = AsyncMock(return_value=mock_coll_instance)
        mock_collection.__aexit__ = AsyncMock(return_value=None)

        executor = HybridSearchExecutor("in-memory", mock_collection)

        with patch("holodeck.lib.keyword_search.tracer") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
                return_value=mock_span
            )
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
                return_value=None
            )

            await executor.search("test", [0.1, 0.2, 0.3], top_k=5)

            # Verify span was started
            assert mock_tracer.start_as_current_span.called


class TestReciprocalRankFusion:
    """Test RRF fusion function."""

    def test_basic_merge_two_lists(self) -> None:
        """Test basic RRF merge of two ranked lists."""
        from holodeck.lib.hybrid_search import reciprocal_rank_fusion

        list1 = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        list2 = [("b", 0.95), ("a", 0.85), ("d", 0.75)]

        result = reciprocal_rank_fusion([list1, list2], k=60)

        # Should contain all unique docs
        doc_ids = [r[0] for r in result]
        assert "a" in doc_ids
        assert "b" in doc_ids
        assert "c" in doc_ids
        assert "d" in doc_ids

    def test_rrf_with_k60_default(self) -> None:
        """Test RRF uses k=60 by default and normalizes scores."""
        from holodeck.lib.hybrid_search import reciprocal_rank_fusion

        list1 = [("a", 0.9)]  # rank 1

        # With k=60, raw score = 1/(60+1) = 0.0164...
        # But scores are normalized: 0.0164 / 0.0164 = 1.0
        result = reciprocal_rank_fusion([list1], k=60)

        assert len(result) == 1
        # Single item in single list normalizes to 1.0
        assert result[0][1] == 1.0

    def test_weighted_rrf(self) -> None:
        """Test weighted RRF with custom weights affects ranking."""
        from holodeck.lib.hybrid_search import reciprocal_rank_fusion

        # Two docs with different ranking in each list
        list1 = [("a", 0.9), ("b", 0.8)]
        list2 = [("b", 0.95), ("a", 0.85)]

        # Equal weights - both have same RRF contribution
        result_equal = reciprocal_rank_fusion([list1, list2], weights=[1.0, 1.0])

        # Heavy weight on list1 where "a" is ranked higher
        result_weighted = reciprocal_rank_fusion([list1, list2], weights=[10.0, 1.0])

        # With equal weights, "b" may win (rank 1 in list2)
        # With list1 heavily weighted, "a" should win (rank 1 in list1)
        # At minimum, the relative scores should differ
        score_a_equal = next(s for d, s in result_equal if d == "a")
        score_b_equal = next(s for d, s in result_equal if d == "b")
        score_a_weighted = next(s for d, s in result_weighted if d == "a")
        score_b_weighted = next(s for d, s in result_weighted if d == "b")

        # In weighted, a should have higher relative advantage
        ratio_equal = score_a_equal / score_b_equal if score_b_equal else float("inf")
        ratio_weighted = (
            score_a_weighted / score_b_weighted if score_b_weighted else float("inf")
        )

        assert ratio_weighted > ratio_equal

    def test_handles_imbalanced_result_sets(self) -> None:
        """Test RRF handles lists with different lengths."""
        from holodeck.lib.hybrid_search import reciprocal_rank_fusion

        list1 = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        list2 = [("a", 0.95)]  # Only one result

        result = reciprocal_rank_fusion([list1, list2])

        # Should still work
        assert len(result) == 3
        doc_ids = [r[0] for r in result]
        assert "a" in doc_ids
        assert "b" in doc_ids
        assert "c" in doc_ids

    def test_handles_empty_result_list(self) -> None:
        """Test RRF handles empty result lists."""
        from holodeck.lib.hybrid_search import reciprocal_rank_fusion

        list1 = [("a", 0.9)]
        list2: list[tuple[str, float]] = []

        result = reciprocal_rank_fusion([list1, list2])

        assert len(result) == 1
        assert result[0][0] == "a"

    def test_normalizes_scores_to_0_1(self) -> None:
        """Test RRF scores are normalized to 0-1 range."""
        from holodeck.lib.hybrid_search import reciprocal_rank_fusion

        list1 = [("a", 0.9), ("b", 0.8)]
        list2 = [("a", 0.95), ("b", 0.85)]

        result = reciprocal_rank_fusion([list1, list2])

        # All scores should be in 0-1 range
        for _, score in result:
            assert 0.0 <= score <= 1.0


class TestOpenSearchKeywordProvider:
    """Test OpenSearch keyword search provider."""

    def _make_provider(self, **kwargs: Any) -> Any:
        """Create an OpenSearchKeywordProvider with mocked OpenSearch client."""
        with patch("opensearchpy.OpenSearch") as mock_cls:
            mock_cls.return_value = MagicMock()
            from holodeck.lib.keyword_search import OpenSearchKeywordProvider

            defaults: dict[str, Any] = {
                "endpoint": "https://search.example.com:9200",
                "index_name": "test-index",
            }
            defaults.update(kwargs)
            provider = OpenSearchKeywordProvider(**defaults)
        return provider

    def test_init_with_basic_auth(self) -> None:
        """Test constructor passes http_auth for basic auth credentials."""
        with patch("opensearchpy.OpenSearch") as mock_os_cls:
            from holodeck.lib.keyword_search import OpenSearchKeywordProvider

            OpenSearchKeywordProvider(
                endpoint="https://host:9200",
                index_name="idx",
                username="admin",
                password="secret",  # noqa: S106
            )

            call_kwargs = mock_os_cls.call_args[1]
            assert call_kwargs["http_auth"] == ("admin", "secret")

    def test_init_with_api_key(self) -> None:
        """Test constructor passes Authorization header for API key auth."""
        with patch("opensearchpy.OpenSearch") as mock_os_cls:
            from holodeck.lib.keyword_search import OpenSearchKeywordProvider

            OpenSearchKeywordProvider(
                endpoint="https://host:9200",
                index_name="idx",
                api_key="my-api-key",
            )

            call_kwargs = mock_os_cls.call_args[1]
            assert call_kwargs["headers"] == {"Authorization": "ApiKey my-api-key"}

    def test_init_verify_certs_false(self) -> None:
        """Test constructor sets ssl_show_warn=False when verify_certs is False."""
        with patch("opensearchpy.OpenSearch") as mock_os_cls:
            from holodeck.lib.keyword_search import OpenSearchKeywordProvider

            OpenSearchKeywordProvider(
                endpoint="https://host:9200",
                index_name="idx",
                verify_certs=False,
            )

            call_kwargs = mock_os_cls.call_args[1]
            assert call_kwargs["verify_certs"] is False
            assert call_kwargs["ssl_show_warn"] is False

    def test_build_creates_index_if_not_exists(self) -> None:
        """Test build() creates the index when it doesn't exist."""
        from holodeck.lib.keyword_search import KeywordDocument

        provider = self._make_provider()
        provider._client.indices.exists.return_value = False

        with patch("opensearchpy.helpers.bulk", return_value=(2, [])):
            provider.build(
                [
                    KeywordDocument(id="doc1", content="content1"),
                    KeywordDocument(id="doc2", content="content2"),
                ]
            )

        provider._client.indices.create.assert_called_once_with(
            index="test-index",
            body=provider._INDEX_MAPPING,
        )

    def test_build_clears_existing_docs(self) -> None:
        """Test build() clears existing documents when index already exists."""
        from holodeck.lib.keyword_search import KeywordDocument

        provider = self._make_provider()
        provider._client.indices.exists.return_value = True

        with patch("opensearchpy.helpers.bulk", return_value=(1, [])):
            provider.build([KeywordDocument(id="doc1", content="content1")])

        provider._client.delete_by_query.assert_called_once_with(
            index="test-index",
            body={"query": {"match_all": {}}},
            refresh=True,
        )

    def test_build_bulk_indexes_multi_field_documents(self) -> None:
        """Test build() sends correct multi-field bulk actions."""
        from holodeck.lib.keyword_search import KeywordDocument

        provider = self._make_provider()
        provider._client.indices.exists.return_value = False

        with patch("opensearchpy.helpers.bulk", return_value=(2, [])) as mock_bulk:
            provider.build(
                [
                    KeywordDocument(
                        id="doc1",
                        content="hello world",
                        parent_chain="Chapter 1 > Intro",
                        section_id="1.1",
                        defined_term="Greeting",
                        chunk_type="definition",
                        source_file="doc.pdf",
                    ),
                    KeywordDocument(id="doc2", content="foo bar"),
                ]
            )

            actions = mock_bulk.call_args[0][1]
            assert len(actions) == 2
            # Check first doc has all fields
            assert actions[0]["_id"] == "doc1"
            assert actions[0]["_source"]["chunk_id"] == "doc1"
            assert actions[0]["_source"]["content"] == "hello world"
            assert actions[0]["_source"]["parent_chain"] == "Chapter 1 > Intro"
            assert actions[0]["_source"]["section_id"] == "1.1"
            assert actions[0]["_source"]["defined_term"] == "Greeting"
            assert actions[0]["_source"]["chunk_type"] == "definition"
            assert actions[0]["_source"]["source_file"] == "doc.pdf"
            # Check second doc has defaults for missing fields
            assert actions[1]["_id"] == "doc2"
            assert actions[1]["_source"]["content"] == "foo bar"
            assert actions[1]["_source"]["parent_chain"] == ""
            assert actions[1]["_source"]["defined_term"] == ""

    def test_build_propagates_exceptions(self) -> None:
        """Test build() propagates exceptions from bulk indexing."""
        from holodeck.lib.keyword_search import KeywordDocument

        provider = self._make_provider()
        provider._client.indices.exists.return_value = False

        with (
            patch(
                "opensearchpy.helpers.bulk",
                side_effect=RuntimeError("bulk failed"),
            ),
            pytest.raises(RuntimeError, match="bulk failed"),
        ):
            provider.build([KeywordDocument(id="doc1", content="content")])

    def test_search_returns_ranked_results(self) -> None:
        """Test search() returns (chunk_id, score) tuples from hits."""
        provider = self._make_provider()
        provider._client.indices.exists.return_value = True
        provider._client.search.return_value = {
            "hits": {
                "hits": [
                    {"_source": {"chunk_id": "doc1"}, "_score": 5.0},
                    {"_source": {"chunk_id": "doc2"}, "_score": 3.2},
                ]
            }
        }

        results = provider.search("test query", top_k=5)

        assert results == [("doc1", 5.0), ("doc2", 3.2)]

    def test_search_returns_empty_if_index_missing(self) -> None:
        """Test search() returns [] when the index does not exist."""
        provider = self._make_provider()
        provider._client.indices.exists.return_value = False

        results = provider.search("test query")

        assert results == []

    def test_search_uses_multi_match_query(self) -> None:
        """Test search() sends correct multi_match query DSL."""
        provider = self._make_provider()
        provider._client.indices.exists.return_value = True
        provider._client.search.return_value = {"hits": {"hits": []}}

        provider.search("my query", top_k=7)

        provider._client.search.assert_called_once_with(
            index="test-index",
            body={
                "query": {
                    "multi_match": {
                        "query": "my query",
                        "fields": [
                            "content",
                            "parent_chain^2",
                            "section_id^2",
                            "defined_term^3",
                            "source_file",
                        ],
                        "type": "best_fields",
                        "operator": "or",
                    }
                },
                "size": 7,
            },
        )

    def test_index_mapping_has_multi_field_properties(self) -> None:
        """Test index mapping includes all expected field properties."""
        provider = self._make_provider()
        props = provider._INDEX_MAPPING["mappings"]["properties"]

        assert "chunk_id" in props
        assert props["chunk_id"]["type"] == "keyword"
        assert "content" in props
        assert props["content"]["type"] == "text"
        assert "parent_chain" in props
        assert props["parent_chain"]["type"] == "text"
        assert "section_id" in props
        assert props["section_id"]["type"] == "text"
        assert props["section_id"]["analyzer"] == "simple"
        assert "defined_term" in props
        assert props["defined_term"]["type"] == "text"
        assert "chunk_type" in props
        assert props["chunk_type"]["type"] == "keyword"
        assert "source_file" in props
        assert props["source_file"]["type"] == "text"

    def test_search_propagates_connection_errors(self) -> None:
        """Test search() propagates connection errors from the client."""
        provider = self._make_provider()
        provider._client.indices.exists.return_value = True
        provider._client.search.side_effect = ConnectionError("timeout")

        with pytest.raises(ConnectionError, match="timeout"):
            provider.search("query")

    def test_build_emits_otel_span(self) -> None:
        """Test build() emits an OpenTelemetry span with correct attributes."""
        from holodeck.lib.keyword_search import KeywordDocument

        provider = self._make_provider()
        provider._client.indices.exists.return_value = False

        with (
            patch("opensearchpy.helpers.bulk", return_value=(1, [])),
            patch("holodeck.lib.keyword_search.tracer") as mock_tracer,
        ):
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
                return_value=mock_span
            )
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
                return_value=None
            )

            provider.build([KeywordDocument(id="doc1", content="content")])

            mock_tracer.start_as_current_span.assert_called_once()
            call_args = mock_tracer.start_as_current_span.call_args
            assert call_args[0][0] == "opensearch.build"
            attrs = call_args[1]["attributes"]
            assert attrs["opensearch.document_count"] == 1
            mock_span.set_attribute.assert_called_with("opensearch.indexed_count", 1)

    def test_search_emits_otel_span(self) -> None:
        """Test search() emits an OpenTelemetry span with correct attributes."""
        provider = self._make_provider()
        provider._client.indices.exists.return_value = True
        provider._client.search.return_value = {
            "hits": {
                "hits": [
                    {"_source": {"chunk_id": "doc1"}, "_score": 2.0},
                ]
            }
        }

        with patch("holodeck.lib.keyword_search.tracer") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
                return_value=mock_span
            )
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
                return_value=None
            )

            provider.search("test", top_k=3)

            mock_tracer.start_as_current_span.assert_called_once()
            call_args = mock_tracer.start_as_current_span.call_args
            assert call_args[0][0] == "opensearch.search"
            attrs = call_args[1]["attributes"]
            assert attrs["opensearch.query"] == "test"
            assert attrs["opensearch.top_k"] == 3
            mock_span.set_attribute.assert_called_with("opensearch.result_count", 1)


class TestKeywordSearchProviderProtocol:
    """Test KeywordSearchProvider protocol satisfaction."""

    def test_in_memory_bm25_satisfies_protocol(self) -> None:
        """Test InMemoryBM25KeywordProvider satisfies KeywordSearchProvider."""
        from holodeck.lib.keyword_search import (
            InMemoryBM25KeywordProvider,
            KeywordSearchProvider,
        )

        provider = InMemoryBM25KeywordProvider()
        assert isinstance(provider, KeywordSearchProvider)

    def test_opensearch_satisfies_protocol(self) -> None:
        """Test OpenSearchKeywordProvider satisfies KeywordSearchProvider."""
        from holodeck.lib.keyword_search import (
            KeywordSearchProvider,
            OpenSearchKeywordProvider,
        )

        with patch("opensearchpy.OpenSearch"):
            provider = OpenSearchKeywordProvider(
                endpoint="https://host:9200",
                index_name="idx",
            )
        assert isinstance(provider, KeywordSearchProvider)

    def test_protocol_is_runtime_checkable(self) -> None:
        """Test KeywordSearchProvider is runtime_checkable."""
        from holodeck.lib.keyword_search import KeywordSearchProvider

        # A non-conforming object should not satisfy the protocol
        assert not isinstance("not a provider", KeywordSearchProvider)


class TestProviderRouter:
    """Test provider routing in HybridSearchExecutor.build_keyword_index."""

    @pytest.mark.asyncio
    async def test_default_config_uses_in_memory(self) -> None:
        """Test None config defaults to InMemoryBM25KeywordProvider."""
        from holodeck.lib.keyword_search import (
            HybridSearchExecutor,
            InMemoryBM25KeywordProvider,
        )
        from holodeck.lib.structured_chunker import DocumentChunk

        mock_collection = MagicMock()
        executor = HybridSearchExecutor("in-memory", mock_collection)

        chunks = [
            DocumentChunk(
                id="chunk1",
                source_path="/test.md",
                chunk_index=0,
                content="Test content",
                contextualized_content="Test content",
            ),
        ]
        await executor.build_keyword_index(chunks)

        assert isinstance(executor._keyword_index, InMemoryBM25KeywordProvider)

    @pytest.mark.asyncio
    async def test_explicit_in_memory_config_uses_in_memory(self) -> None:
        """Test explicit in-memory config uses InMemoryBM25KeywordProvider."""
        from holodeck.lib.keyword_search import (
            HybridSearchExecutor,
            InMemoryBM25KeywordProvider,
        )
        from holodeck.lib.structured_chunker import DocumentChunk
        from holodeck.models.tool import KeywordIndexConfig

        config = KeywordIndexConfig(provider="in-memory")
        mock_collection = MagicMock()
        executor = HybridSearchExecutor(
            "in-memory", mock_collection, keyword_index_config=config
        )

        chunks = [
            DocumentChunk(
                id="chunk1",
                source_path="/test.md",
                chunk_index=0,
                content="Test content",
                contextualized_content="Test content",
            ),
        ]
        await executor.build_keyword_index(chunks)

        assert isinstance(executor._keyword_index, InMemoryBM25KeywordProvider)

    @pytest.mark.asyncio
    async def test_opensearch_config_creates_opensearch_provider(self) -> None:
        """Test opensearch config creates OpenSearchKeywordProvider."""
        from holodeck.lib.keyword_search import (
            HybridSearchExecutor,
            OpenSearchKeywordProvider,
        )
        from holodeck.lib.structured_chunker import DocumentChunk
        from holodeck.models.tool import KeywordIndexConfig

        config = KeywordIndexConfig(
            provider="opensearch",
            endpoint="https://search.example.com:9200",
            index_name="test-index",
        )
        mock_collection = MagicMock()

        with patch("opensearchpy.OpenSearch") as mock_os_cls:
            mock_client = MagicMock()
            mock_os_cls.return_value = mock_client
            mock_client.indices.exists.return_value = False

            with patch("opensearchpy.helpers.bulk", return_value=(1, [])):
                executor = HybridSearchExecutor(
                    "in-memory", mock_collection, keyword_index_config=config
                )
                chunks = [
                    DocumentChunk(
                        id="chunk1",
                        source_path="/test.md",
                        chunk_index=0,
                        content="Test content",
                        contextualized_content="Test content",
                    ),
                ]
                await executor.build_keyword_index(chunks)

        assert isinstance(executor._keyword_index, OpenSearchKeywordProvider)


class TestKeywordIndexBuildOTel:
    """Test OTel instrumentation on build_keyword_index."""

    @pytest.mark.asyncio
    async def test_build_emits_span_on_success(self) -> None:
        """Test build_keyword_index emits span with correct attrs on success."""
        from holodeck.lib.keyword_search import HybridSearchExecutor
        from holodeck.lib.structured_chunker import DocumentChunk

        mock_collection = MagicMock()
        executor = HybridSearchExecutor("in-memory", mock_collection)

        chunks = [
            DocumentChunk(
                id="chunk1",
                source_path="/test.md",
                chunk_index=0,
                content="Test content",
                contextualized_content="Test content",
            ),
        ]

        with patch("holodeck.lib.keyword_search.tracer") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
                return_value=mock_span
            )
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
                return_value=None
            )

            await executor.build_keyword_index(chunks)

            # Verify span name and attributes
            call_args = mock_tracer.start_as_current_span.call_args
            assert call_args[0][0] == "keyword_index.build"
            attrs = call_args[1]["attributes"]
            assert attrs["keyword_index.provider"] == "in-memory"
            assert attrs["keyword_index.document_count"] == 1

            # Verify success status
            mock_span.set_attribute.assert_any_call("keyword_index.status", "success")

    @pytest.mark.asyncio
    async def test_build_records_exception_on_failure(self) -> None:
        """Test build_keyword_index records exception in span on failure."""
        from holodeck.lib.keyword_search import HybridSearchExecutor
        from holodeck.lib.structured_chunker import DocumentChunk
        from holodeck.models.tool import KeywordIndexConfig

        config = KeywordIndexConfig(
            provider="opensearch",
            endpoint="https://search.example.com:9200",
            index_name="test-index",
        )
        mock_collection = MagicMock()

        with patch("holodeck.lib.keyword_search.tracer") as mock_tracer:
            mock_span = MagicMock()
            mock_tracer.start_as_current_span.return_value.__enter__ = MagicMock(
                return_value=mock_span
            )
            mock_tracer.start_as_current_span.return_value.__exit__ = MagicMock(
                return_value=None
            )

            # Make OpenSearch constructor raise
            with patch(
                "opensearchpy.OpenSearch",
                side_effect=ConnectionError("connect failed"),
            ):
                executor = HybridSearchExecutor(
                    "in-memory", mock_collection, keyword_index_config=config
                )
                chunks = [
                    DocumentChunk(
                        id="chunk1",
                        source_path="/test.md",
                        chunk_index=0,
                        content="Test",
                        contextualized_content="Test",
                    ),
                ]
                await executor.build_keyword_index(chunks)

            # Verify failure status and exception recording
            mock_span.set_attribute.assert_any_call("keyword_index.status", "failed")
            mock_span.record_exception.assert_called_once()
            assert executor._keyword_index is None


class TestOpenSearchGracefulDegradation:
    """Test graceful degradation when OpenSearch operations fail."""

    @pytest.mark.asyncio
    async def test_keyword_search_returns_empty_on_provider_error(self) -> None:
        """Test keyword_search() returns [] when provider search raises."""
        from holodeck.lib.keyword_search import HybridSearchExecutor

        mock_collection = MagicMock()
        executor = HybridSearchExecutor("in-memory", mock_collection)

        # Set up a mock keyword index that raises on search
        mock_index = MagicMock()
        mock_index.search.side_effect = RuntimeError("search failed")
        executor._keyword_index = mock_index

        results = await executor.keyword_search("test query", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_fallback_hybrid_returns_semantic_only_on_keyword_failure(
        self,
    ) -> None:
        """Test _fallback_hybrid_search returns semantic-only on keyword error."""
        from holodeck.lib.keyword_search import HybridSearchExecutor

        mock_collection = MagicMock()
        mock_coll_instance = AsyncMock()

        async def mock_results():
            yield MagicMock(record=MagicMock(id="chunk1"), score=0.9)

        mock_search_results = MagicMock()
        mock_search_results.results = mock_results()
        mock_coll_instance.search = AsyncMock(return_value=mock_search_results)

        mock_collection.__aenter__ = AsyncMock(return_value=mock_coll_instance)
        mock_collection.__aexit__ = AsyncMock(return_value=None)

        executor = HybridSearchExecutor("in-memory", mock_collection)

        # Set up a mock keyword index that raises on search
        mock_index = MagicMock()
        mock_index.search.side_effect = RuntimeError("search failed")
        executor._keyword_index = mock_index

        results = await executor.search("test query", [0.1, 0.2, 0.3], top_k=5)

        # Should still return semantic results
        assert len(results) > 0
        assert results[0][0] == "chunk1"

    @pytest.mark.asyncio
    async def test_opensearch_build_failure_sets_keyword_index_none(self) -> None:
        """Test OpenSearch build failure sets _keyword_index to None."""
        from holodeck.lib.keyword_search import HybridSearchExecutor
        from holodeck.lib.structured_chunker import DocumentChunk
        from holodeck.models.tool import KeywordIndexConfig

        config = KeywordIndexConfig(
            provider="opensearch",
            endpoint="https://search.example.com:9200",
            index_name="test-index",
        )
        mock_collection = MagicMock()

        with patch(
            "opensearchpy.OpenSearch",
            side_effect=ConnectionError("connect failed"),
        ):
            executor = HybridSearchExecutor(
                "in-memory", mock_collection, keyword_index_config=config
            )
            chunks = [
                DocumentChunk(
                    id="chunk1",
                    source_path="/test.md",
                    chunk_index=0,
                    content="Test",
                    contextualized_content="Test",
                ),
            ]
            await executor.build_keyword_index(chunks)

        assert executor._keyword_index is None
