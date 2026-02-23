"""Tests for ClaudeSDKContextGenerator — Claude Agent SDK contextual embeddings.

This module contains comprehensive tests for the ClaudeSDKContextGenerator class.
Tests cover configuration, document truncation, prompt construction, JSON parsing,
batch processing, concurrency, and protocol conformance.
"""

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from holodeck.lib.backends.base import ContextGenerator
from holodeck.lib.claude_context_generator import (
    ClaudeContextConfig,
    ClaudeSDKContextGenerator,
)
from holodeck.lib.structured_chunker import ChunkType, DocumentChunk


# Helper to create test DocumentChunks
def create_test_chunk(
    chunk_id: str = "test_chunk_0",
    content: str = "Test chunk content.",
    parent_chain: list[str] | None = None,
    chunk_index: int = 0,
) -> DocumentChunk:
    """Create a DocumentChunk for testing."""
    return DocumentChunk(
        id=chunk_id,
        source_path="/test/doc.md",
        chunk_index=chunk_index,
        content=content,
        parent_chain=parent_chain or [],
        section_id=f"sec_{chunk_index}",
        chunk_type=ChunkType.CONTENT,
    )


class TestClaudeContextConfig:
    """Tests for ClaudeContextConfig dataclass."""

    def test_default_values(self) -> None:
        """Test ClaudeContextConfig default values."""
        config = ClaudeContextConfig()
        assert config.model == "claude-haiku-4-5-20251001"
        assert config.batch_size == 10
        assert config.concurrency == 5
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_document_tokens == 8000

    def test_custom_values(self) -> None:
        """Test ClaudeContextConfig with custom values."""
        config = ClaudeContextConfig(
            model="claude-sonnet-4-5-20250514",
            batch_size=5,
            concurrency=3,
            max_retries=5,
            base_delay=0.5,
            max_document_tokens=4000,
        )
        assert config.model == "claude-sonnet-4-5-20250514"
        assert config.batch_size == 5
        assert config.concurrency == 3
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_document_tokens == 4000


class TestClaudeSDKContextGeneratorInit:
    """Tests for ClaudeSDKContextGenerator initialization."""

    def test_default_config(self) -> None:
        """Test initialization with default config."""
        gen = ClaudeSDKContextGenerator()
        assert gen._config.model == "claude-haiku-4-5-20251001"
        assert gen._config.batch_size == 10

    def test_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = ClaudeContextConfig(model="custom-model", batch_size=5)
        gen = ClaudeSDKContextGenerator(config=config)
        assert gen._config.model == "custom-model"
        assert gen._config.batch_size == 5

    def test_max_context_tokens(self) -> None:
        """Test custom max_context_tokens."""
        gen = ClaudeSDKContextGenerator(max_context_tokens=200)
        assert gen._max_context_tokens == 200

    def test_default_max_context_tokens(self) -> None:
        """Test default max_context_tokens is 100."""
        gen = ClaudeSDKContextGenerator()
        assert gen._max_context_tokens == 100


class TestDocumentTruncation:
    """Tests for _truncate_document method."""

    def test_short_document_passthrough(self) -> None:
        """Test that short documents are returned unchanged."""
        gen = ClaudeSDKContextGenerator()
        doc = "This is a short document."
        assert gen._truncate_document(doc) == doc

    def test_long_document_truncated(self) -> None:
        """Test that long documents are truncated."""
        config = ClaudeContextConfig(max_document_tokens=100)
        gen = ClaudeSDKContextGenerator(config=config)
        long_doc = "Word " * 10000
        result = gen._truncate_document(long_doc)
        assert len(result) < len(long_doc)
        assert "[... document truncated ...]" in result

    def test_preserves_beginning_and_end(self) -> None:
        """Test that truncation preserves document beginning and end."""
        config = ClaudeContextConfig(max_document_tokens=200)
        gen = ClaudeSDKContextGenerator(config=config)
        long_doc = "BEGINNING_MARKER " + ("middle " * 5000) + " END_MARKER"
        result = gen._truncate_document(long_doc)
        assert "BEGINNING_MARKER" in result
        assert "END_MARKER" in result


class TestBatchPromptConstruction:
    """Tests for _build_batch_prompt method."""

    def test_contains_document(self) -> None:
        """Test that batch prompt contains the document text."""
        gen = ClaudeSDKContextGenerator()
        chunks = [create_test_chunk(content=f"Chunk {i}") for i in range(3)]
        prompt = gen._build_batch_prompt(chunks, "The full document text.")
        assert "The full document text." in prompt

    def test_all_chunks_numbered(self) -> None:
        """Test that all chunks appear numbered in the prompt."""
        gen = ClaudeSDKContextGenerator()
        chunks = [create_test_chunk(content=f"Content {i}") for i in range(3)]
        prompt = gen._build_batch_prompt(chunks, "Document")
        assert "Chunk 1:" in prompt
        assert "Chunk 2:" in prompt
        assert "Chunk 3:" in prompt
        assert "Content 0" in prompt
        assert "Content 1" in prompt
        assert "Content 2" in prompt

    def test_requests_json_array(self) -> None:
        """Test that prompt requests JSON array output."""
        gen = ClaudeSDKContextGenerator()
        chunks = [create_test_chunk()]
        prompt = gen._build_batch_prompt(chunks, "Document")
        assert "JSON array" in prompt

    def test_document_delimiters(self) -> None:
        """Test that prompt has document delimiters."""
        gen = ClaudeSDKContextGenerator()
        chunks = [create_test_chunk()]
        prompt = gen._build_batch_prompt(chunks, "Document")
        assert "!---DOCUMENT START---!" in prompt
        assert "!---DOCUMENT END---!" in prompt


class TestSinglePromptConstruction:
    """Tests for _build_single_prompt method."""

    def test_contains_document_and_chunk(self) -> None:
        """Test that single prompt contains document and chunk."""
        gen = ClaudeSDKContextGenerator()
        prompt = gen._build_single_prompt("My chunk", "My document")
        assert "My document" in prompt
        assert "My chunk" in prompt

    def test_has_delimiters(self) -> None:
        """Test that single prompt has document and chunk delimiters."""
        gen = ClaudeSDKContextGenerator()
        prompt = gen._build_single_prompt("Chunk", "Document")
        assert "!---DOCUMENT START---!" in prompt
        assert "!---DOCUMENT END---!" in prompt
        assert "!---CHUNK START---!" in prompt
        assert "!---CHUNK END---!" in prompt

    def test_requests_succinct_context(self) -> None:
        """Test that prompt requests succinct context."""
        gen = ClaudeSDKContextGenerator()
        prompt = gen._build_single_prompt("Chunk", "Document")
        assert "succinct" in prompt.lower()


class TestBatchResponseParsing:
    """Tests for _parse_batch_response method."""

    def test_valid_json_array(self) -> None:
        """Test parsing a valid JSON array."""
        gen = ClaudeSDKContextGenerator()
        response = '["Context 1", "Context 2", "Context 3"]'
        result = gen._parse_batch_response(response, 3)
        assert result == ["Context 1", "Context 2", "Context 3"]

    def test_markdown_fenced_json(self) -> None:
        """Test parsing JSON inside markdown code fences."""
        gen = ClaudeSDKContextGenerator()
        response = '```json\n["Context 1", "Context 2"]\n```'
        result = gen._parse_batch_response(response, 2)
        assert result == ["Context 1", "Context 2"]

    def test_wrong_count_returns_none(self) -> None:
        """Test that wrong item count returns None."""
        gen = ClaudeSDKContextGenerator()
        response = '["Context 1", "Context 2"]'
        result = gen._parse_batch_response(response, 3)
        assert result is None

    def test_invalid_json_returns_none(self) -> None:
        """Test that invalid JSON returns None."""
        gen = ClaudeSDKContextGenerator()
        result = gen._parse_batch_response("not json at all", 2)
        assert result is None

    def test_non_array_returns_none(self) -> None:
        """Test that non-array JSON returns None."""
        gen = ClaudeSDKContextGenerator()
        result = gen._parse_batch_response('{"key": "value"}', 1)
        assert result is None

    def test_coerces_items_to_strings(self) -> None:
        """Test that non-string items are coerced to strings."""
        gen = ClaudeSDKContextGenerator()
        response = '["text", 42, true]'
        result = gen._parse_batch_response(response, 3)
        assert result == ["text", "42", "True"]

    def test_plain_code_fence(self) -> None:
        """Test parsing JSON inside plain (no language) code fences."""
        gen = ClaudeSDKContextGenerator()
        response = '```\n["A", "B"]\n```'
        result = gen._parse_batch_response(response, 2)
        assert result == ["A", "B"]


class TestQueryClaude:
    """Tests for _query_claude method."""

    @pytest.mark.asyncio
    async def test_returns_text(self) -> None:
        """Test that _query_claude returns text from the SDK response."""
        gen = ClaudeSDKContextGenerator()

        # Create mock message with TextBlock
        mock_text_block = MagicMock()
        mock_text_block.__class__ = type("TextBlock", (), {})
        mock_text_block.text = "Generated context"

        mock_msg = MagicMock()
        mock_msg.__class__ = type("AssistantMessage", (), {})
        mock_msg.content = [mock_text_block]

        async def mock_query(prompt, options):
            yield mock_msg

        with patch("claude_agent_sdk.query", side_effect=mock_query):
            result = await gen._query_claude("Test prompt")

        assert result == "Generated context"

    @pytest.mark.asyncio
    async def test_empty_response(self) -> None:
        """Test handling of empty SDK response."""
        gen = ClaudeSDKContextGenerator()

        async def mock_query(prompt, options):
            return
            yield  # Make it an async generator  # pragma: no cover

        with patch("claude_agent_sdk.query", side_effect=mock_query):
            result = await gen._query_claude("Test prompt")

        assert result == ""

    @pytest.mark.asyncio
    async def test_strips_whitespace(self) -> None:
        """Test that response whitespace is stripped."""
        gen = ClaudeSDKContextGenerator()

        mock_text_block = MagicMock()
        mock_text_block.__class__ = type("TextBlock", (), {})
        mock_text_block.text = "  Context with spaces  \n"

        mock_msg = MagicMock()
        mock_msg.__class__ = type("AssistantMessage", (), {})
        mock_msg.content = [mock_text_block]

        async def mock_query(prompt, options):
            yield mock_msg

        with patch("claude_agent_sdk.query", side_effect=mock_query):
            result = await gen._query_claude("Test")

        assert result == "Context with spaces"


class TestProcessBatch:
    """Tests for _process_batch method."""

    @pytest.mark.asyncio
    async def test_batch_success(self) -> None:
        """Test successful batch processing with valid JSON response."""
        config = ClaudeContextConfig(base_delay=0.01)
        gen = ClaudeSDKContextGenerator(config=config)
        chunks = [
            create_test_chunk(content="Chunk A", chunk_index=0),
            create_test_chunk(content="Chunk B", chunk_index=1),
        ]

        with patch.object(
            gen,
            "_query_claude",
            new_callable=AsyncMock,
            return_value='["Context for A", "Context for B"]',
        ):
            results = await gen._process_batch(chunks, "Document text")

        assert len(results) == 2
        assert "Context for A" in results[0]
        assert "Chunk A" in results[0]
        assert "Context for B" in results[1]
        assert "Chunk B" in results[1]

    @pytest.mark.asyncio
    async def test_fallback_on_invalid_json(self) -> None:
        """Test fallback to single prompts when batch JSON parsing fails."""
        config = ClaudeContextConfig(base_delay=0.01)
        gen = ClaudeSDKContextGenerator(config=config)
        chunks = [
            create_test_chunk(content="Chunk A", chunk_index=0),
            create_test_chunk(content="Chunk B", chunk_index=1),
        ]

        call_count = 0

        async def mock_query(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call is batch — return invalid JSON
                return "Not valid JSON"
            # Subsequent calls are single-chunk fallbacks
            return f"Single context {call_count}"

        with patch.object(gen, "_query_claude", side_effect=mock_query):
            results = await gen._process_batch(chunks, "Document text")

        assert len(results) == 2
        # Should have fallen back to single prompts (1 batch + 2 singles = 3)
        assert call_count == 3
        assert "Single context" in results[0]
        assert "Single context" in results[1]

    @pytest.mark.asyncio
    async def test_fallback_on_exception(self) -> None:
        """Test fallback to single prompts when batch query raises exception."""
        config = ClaudeContextConfig(base_delay=0.01)
        gen = ClaudeSDKContextGenerator(config=config)
        chunks = [create_test_chunk(content="Chunk A", chunk_index=0)]

        call_count = 0

        async def mock_query(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Network error")
            return "Fallback context"

        with patch.object(gen, "_query_claude", side_effect=mock_query):
            results = await gen._process_batch(chunks, "Document text")

        assert len(results) == 1
        assert "Fallback context" in results[0]

    @pytest.mark.asyncio
    async def test_graceful_degradation(self) -> None:
        """Test that on all failures, bare chunk content is returned."""
        config = ClaudeContextConfig(base_delay=0.01, max_retries=1)
        gen = ClaudeSDKContextGenerator(config=config)
        chunks = [create_test_chunk(content="Original content", chunk_index=0)]

        async def mock_query(prompt: str) -> str:
            raise Exception("Total failure")

        with patch.object(gen, "_query_claude", side_effect=mock_query):
            results = await gen._process_batch(chunks, "Document text")

        assert len(results) == 1
        assert results[0] == "Original content"


class TestGenerateSingleContext:
    """Tests for _generate_single_context method."""

    @pytest.mark.asyncio
    async def test_returns_context(self) -> None:
        """Test that single context generation returns text."""
        config = ClaudeContextConfig(base_delay=0.01)
        gen = ClaudeSDKContextGenerator(config=config)

        with patch.object(
            gen,
            "_query_claude",
            new_callable=AsyncMock,
            return_value="Generated context",
        ):
            result = await gen._generate_single_context("Chunk text", "Doc text")

        assert result == "Generated context"

    @pytest.mark.asyncio
    async def test_retries_on_failure(self) -> None:
        """Test retry logic for single context generation."""
        config = ClaudeContextConfig(base_delay=0.01, max_retries=3)
        gen = ClaudeSDKContextGenerator(config=config)

        call_count = 0

        async def mock_query(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Transient error")
            return "Success after retry"

        with patch.object(gen, "_query_claude", side_effect=mock_query):
            result = await gen._generate_single_context("Chunk", "Doc")

        assert result == "Success after retry"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_returns_empty_on_exhausted_retries(self) -> None:
        """Test that empty string is returned after all retries fail."""
        config = ClaudeContextConfig(base_delay=0.01, max_retries=2)
        gen = ClaudeSDKContextGenerator(config=config)

        with patch.object(
            gen,
            "_query_claude",
            new_callable=AsyncMock,
            side_effect=Exception("Persistent failure"),
        ):
            result = await gen._generate_single_context("Chunk", "Doc")

        assert result == ""


class TestContextualizeBatch:
    """Tests for contextualize_batch public API."""

    @pytest.mark.asyncio
    async def test_empty_input(self) -> None:
        """Test that empty input returns empty list."""
        gen = ClaudeSDKContextGenerator()
        result = await gen.contextualize_batch([], "Document")
        assert result == []

    @pytest.mark.asyncio
    async def test_single_batch(self) -> None:
        """Test processing chunks that fit in a single batch."""
        config = ClaudeContextConfig(batch_size=10, base_delay=0.01)
        gen = ClaudeSDKContextGenerator(config=config)
        chunks = [
            create_test_chunk(content=f"Chunk {i}", chunk_index=i) for i in range(3)
        ]

        with patch.object(
            gen,
            "_query_claude",
            new_callable=AsyncMock,
            return_value='["Ctx 0", "Ctx 1", "Ctx 2"]',
        ):
            results = await gen.contextualize_batch(chunks, "Document text")

        assert len(results) == 3
        assert "Ctx 0" in results[0]
        assert "Chunk 0" in results[0]

    @pytest.mark.asyncio
    async def test_multiple_batches_merged_in_order(self) -> None:
        """Test that results from multiple batches are merged in order."""
        config = ClaudeContextConfig(batch_size=2, base_delay=0.01)
        gen = ClaudeSDKContextGenerator(config=config)
        chunks = [
            create_test_chunk(content=f"Chunk {i}", chunk_index=i) for i in range(5)
        ]

        call_count = 0

        async def mock_query(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            # Determine batch size from prompt
            if "5 chunks" in prompt or "Chunk 4" in prompt:
                # Last batch has 1 chunk
                return '["Ctx 4"]'
            if "Chunk 2" in prompt and "Chunk 3" in prompt:
                return '["Ctx 2", "Ctx 3"]'
            return '["Ctx 0", "Ctx 1"]'

        with patch.object(gen, "_query_claude", side_effect=mock_query):
            results = await gen.contextualize_batch(chunks, "Document text")

        assert len(results) == 5
        # Verify ordering
        for i in range(5):
            assert f"Chunk {i}" in results[i]

    @pytest.mark.asyncio
    async def test_concurrency_override(self) -> None:
        """Test that concurrency parameter overrides config."""
        config = ClaudeContextConfig(batch_size=1, concurrency=10, base_delay=0.01)
        gen = ClaudeSDKContextGenerator(config=config)
        chunks = [
            create_test_chunk(content=f"Chunk {i}", chunk_index=i) for i in range(5)
        ]

        max_concurrent = 0
        current_concurrent = 0

        async def tracking_process_batch(
            batch_chunks: list[DocumentChunk], doc_text: str
        ) -> list[str]:
            nonlocal max_concurrent, current_concurrent
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
            await asyncio.sleep(0.02)
            current_concurrent -= 1
            return [f"Ctx\n\n{c.content}" for c in batch_chunks]

        with patch.object(gen, "_process_batch", side_effect=tracking_process_batch):
            results = await gen.contextualize_batch(
                chunks, "Document text", concurrency=2
            )

        assert len(results) == 5
        assert max_concurrent <= 2


class TestProtocolConformance:
    """ClaudeSDKContextGenerator conforms to the ContextGenerator protocol."""

    def test_is_context_generator(self) -> None:
        """Test that ClaudeSDKContextGenerator is recognized as a ContextGenerator."""
        gen = ClaudeSDKContextGenerator()
        assert isinstance(gen, ContextGenerator)

    def test_has_contextualize_batch_method(self) -> None:
        """Test that contextualize_batch method exists and is async."""
        gen = ClaudeSDKContextGenerator()
        assert hasattr(gen, "contextualize_batch")
        assert inspect.iscoroutinefunction(gen.contextualize_batch)
