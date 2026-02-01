"""Tests for keyword search module.

This module tests the tiered keyword search implementation including:
- KeywordSearchStrategy enum
- Provider capability detection
- BM25 fallback implementation
- Exact match index
- Hybrid search executor
- OpenTelemetry instrumentation
"""

from __future__ import annotations

from typing import TYPE_CHECKING
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


class TestBM25FallbackProvider:
    """Test BM25 fallback implementation."""

    def test_init_with_default_params(self) -> None:
        """Test BM25FallbackProvider initializes with default k1 and b."""
        from holodeck.lib.keyword_search import BM25FallbackProvider

        provider = BM25FallbackProvider()
        assert provider.k1 == 1.5
        assert provider.b == 0.75

    def test_init_with_custom_params(self) -> None:
        """Test BM25FallbackProvider accepts custom k1 and b."""
        from holodeck.lib.keyword_search import BM25FallbackProvider

        provider = BM25FallbackProvider(k1=2.0, b=0.5)
        assert provider.k1 == 2.0
        assert provider.b == 0.5

    def test_build_indexes_documents(self) -> None:
        """Test build() creates index from documents."""
        from holodeck.lib.keyword_search import BM25FallbackProvider

        provider = BM25FallbackProvider()
        documents = [
            ("doc1", "the quick brown fox"),
            ("doc2", "jumps over the lazy dog"),
            ("doc3", "the brown dog is quick"),
        ]
        provider.build(documents)

        assert provider._bm25 is not None
        assert provider._doc_ids == ["doc1", "doc2", "doc3"]

    def test_build_uses_contextualized_content(self) -> None:
        """Test build() indexes the contextualized_content field."""
        from holodeck.lib.keyword_search import BM25FallbackProvider

        provider = BM25FallbackProvider()
        # Second element is contextualized_content (not raw content)
        # Use more documents to make BM25 scoring work properly
        documents = [
            ("doc1", "Context: Section about foxes. The quick brown fox jumps."),
            ("doc2", "Context: Section about dogs. The lazy dog sleeps here."),
            ("doc3", "Context: Section about cats. The small cat plays nicely."),
            ("doc4", "Context: Section about birds. The blue bird sings daily."),
            ("doc5", "Context: Section about fish. The gold fish swims around."),
        ]
        provider.build(documents)

        # Search for terms that appear across documents to get proper BM25 scoring
        # "quick" appears only in doc1
        results = provider.search("quick jumps", top_k=5)
        assert len(results) > 0
        # First result should be doc1 which contains both "quick" and "jumps"
        assert results[0][0] == "doc1"

    def test_search_returns_ranked_results(self) -> None:
        """Test search() returns (doc_id, score) tuples sorted by score."""
        from holodeck.lib.keyword_search import BM25FallbackProvider

        provider = BM25FallbackProvider()
        documents = [
            ("doc1", "the quick brown fox"),
            ("doc2", "jumps over the lazy dog"),
            ("doc3", "the brown dog is quick"),
        ]
        provider.build(documents)

        results = provider.search("brown fox", top_k=3)

        # Should return list of (doc_id, score) tuples
        assert len(results) > 0
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)
        assert all(isinstance(r[0], str) for r in results)
        assert all(isinstance(r[1], int | float) for r in results)

        # First result should be doc1 (best match for "brown fox")
        assert results[0][0] == "doc1"

    def test_search_respects_top_k(self) -> None:
        """Test search() returns at most top_k results."""
        from holodeck.lib.keyword_search import BM25FallbackProvider

        provider = BM25FallbackProvider()
        documents = [
            ("doc1", "the quick brown fox"),
            ("doc2", "jumps over the lazy dog"),
            ("doc3", "the brown dog is quick"),
            ("doc4", "another document here"),
            ("doc5", "yet another one"),
        ]
        provider.build(documents)

        results = provider.search("the", top_k=2)
        assert len(results) <= 2

    def test_search_returns_empty_if_no_index(self) -> None:
        """Test search() returns empty list if index not built."""
        from holodeck.lib.keyword_search import BM25FallbackProvider

        provider = BM25FallbackProvider()
        results = provider.search("test query", top_k=5)
        assert results == []


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


class TestExactMatchIndex:
    """Test exact match index for section IDs and phrases."""

    def test_build_creates_section_mapping(self) -> None:
        """Test build() creates section_id to chunk_ids mapping."""
        from holodeck.lib.hybrid_search import ExactMatchIndex

        index = ExactMatchIndex()
        chunks = [
            ("chunk1", "Section 1.1", "Content of section 1.1"),
            ("chunk2", "Section 1.2", "Content of section 1.2"),
            ("chunk3", "Section 1.1", "More content from 1.1"),  # Same section
        ]
        index.build(chunks)

        # Section 1.1 should map to both chunk1 and chunk3
        assert "chunk1" in index.search_section("Section 1.1")
        assert "chunk3" in index.search_section("Section 1.1")
        assert "chunk2" in index.search_section("Section 1.2")

    def test_search_section_exact_match(self) -> None:
        """Test search_section() finds exact section ID."""
        from holodeck.lib.hybrid_search import ExactMatchIndex

        index = ExactMatchIndex()
        chunks = [
            ("chunk1", "203(a)(1)", "Content here"),
            ("chunk2", "203(a)(2)", "Other content"),
        ]
        index.build(chunks)

        result = index.search_section("203(a)(1)")
        assert "chunk1" in result
        assert "chunk2" not in result

    def test_search_phrase_finds_quoted_phrase(self) -> None:
        """Test search_phrase() finds exact phrase in content."""
        from holodeck.lib.hybrid_search import ExactMatchIndex

        index = ExactMatchIndex()
        chunks = [
            ("chunk1", "1.1", "The term Force Majeure means any event..."),
            ("chunk2", "1.2", "Other content without the phrase"),
            ("chunk3", "1.3", "Also contains Force Majeure here"),
        ]
        index.build(chunks)

        result = index.search_phrase("Force Majeure", top_k=10)
        assert "chunk1" in result
        assert "chunk3" in result
        assert "chunk2" not in result


class TestSectionIdPatterns:
    """Test section ID pattern detection."""

    def test_detects_section_with_number(self) -> None:
        """Test detects 'Section 203' pattern."""
        from holodeck.lib.hybrid_search import is_exact_match_query

        assert is_exact_match_query("Section 203")

    def test_detects_section_with_hierarchy(self) -> None:
        """Test detects 'Section 203(a)(1)' pattern."""
        from holodeck.lib.hybrid_search import is_exact_match_query

        assert is_exact_match_query("Section 203(a)(1)")
        assert is_exact_match_query("203(a)(1)")

    def test_detects_section_symbol(self) -> None:
        """Test detects '§4.2' pattern."""
        from holodeck.lib.hybrid_search import is_exact_match_query

        assert is_exact_match_query("§4.2")

    def test_detects_numbered_section(self) -> None:
        """Test detects '1.2.3' pattern."""
        from holodeck.lib.hybrid_search import is_exact_match_query

        assert is_exact_match_query("1.2.3")

    def test_detects_quoted_phrase(self) -> None:
        """Test detects quoted exact phrase."""
        from holodeck.lib.hybrid_search import is_exact_match_query

        assert is_exact_match_query('"Force Majeure"')
        assert is_exact_match_query('"reasonable best efforts"')

    def test_rejects_natural_language_query(self) -> None:
        """Test rejects natural language queries."""
        from holodeck.lib.hybrid_search import is_exact_match_query

        assert not is_exact_match_query("What are the reporting requirements?")
        assert not is_exact_match_query("reporting requirements")

    def test_extract_exact_query_removes_quotes(self) -> None:
        """Test extract_exact_query() removes outer quotes."""
        from holodeck.lib.hybrid_search import extract_exact_query

        assert extract_exact_query('"Force Majeure"') == "Force Majeure"
        assert extract_exact_query("Section 203") == "Section 203"


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

    def test_build_bm25_index_creates_index(self) -> None:
        """Test build_bm25_index() creates BM25 index."""
        from holodeck.lib.keyword_search import HybridSearchExecutor

        mock_collection = MagicMock()
        executor = HybridSearchExecutor("in-memory", mock_collection)

        documents = [
            ("chunk1", "sec1", "Content about reporting requirements"),
            ("chunk2", "sec2", "Content about compliance"),
        ]
        executor.build_bm25_index(documents)

        assert executor._bm25_index is not None

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
    async def test_exact_match_boosting_sc002(self) -> None:
        """Test exact matches get boosted to ensure top-result position (SC-002)."""
        from holodeck.lib.keyword_search import HybridSearchExecutor

        mock_collection = MagicMock()
        mock_coll_instance = AsyncMock()

        # Setup semantic search results (Section 203 NOT at top)
        async def mock_semantic_results():
            for item in [
                MagicMock(record=MagicMock(id="other_chunk"), score=0.95),
                MagicMock(record=MagicMock(id="section_203_chunk"), score=0.80),
            ]:
                yield item

        mock_search_results = MagicMock()
        mock_search_results.results = mock_semantic_results()
        mock_coll_instance.search = AsyncMock(return_value=mock_search_results)

        mock_collection.__aenter__ = AsyncMock(return_value=mock_coll_instance)
        mock_collection.__aexit__ = AsyncMock(return_value=None)

        executor = HybridSearchExecutor("in-memory", mock_collection)

        # Build BM25 and exact match indices
        documents = [
            ("section_203_chunk", "Section 203(a)(1)", "Content of Section 203(a)(1)"),
            ("other_chunk", "1.0", "Some other content"),
        ]
        executor.build_bm25_index(documents)

        # Query for exact section ID
        results = await executor.search("Section 203(a)(1)", [0.1, 0.2, 0.3], top_k=10)

        # The exact match should be at position 1 (index 0)
        assert len(results) > 0
        assert results[0][0] == "section_203_chunk"

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

        # Try to build with invalid data that causes BM25 to fail
        with patch.object(executor, "_bm25_index", None):
            # This simulates BM25 not being available
            executor._bm25_index = None

        # Search should still work (semantic only)
        results = await executor.search("test query", [0.1, 0.2, 0.3], top_k=5)

        # Should return results from semantic search
        assert len(results) > 0


class TestOpenTelemetry:
    """Test OpenTelemetry instrumentation."""

    def test_bm25_search_emits_span(self) -> None:
        """Test BM25FallbackProvider.search() emits OpenTelemetry span."""
        from holodeck.lib.keyword_search import BM25FallbackProvider

        provider = BM25FallbackProvider()
        documents = [("doc1", "test content")]
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
            assert "bm25.search" in str(call_args)

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
