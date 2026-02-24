"""Tests for LLMContextGenerator - contextual embeddings generation.

This module contains comprehensive tests for the LLMContextGenerator class,
following TDD methodology. Tests cover prompt template structure, context
generation, chunk contextualization, batch processing with concurrency,
and error handling with exponential backoff.
"""

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest

from holodeck.lib.backends.base import ContextGenerator
from holodeck.lib.llm_context_generator import (
    CONTEXT_PROMPT_TEMPLATE,
    LLMContextGenerator,
    RetryConfig,
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


class TestContextPromptTemplate:
    """Tests for CONTEXT_PROMPT_TEMPLATE constant."""

    def test_template_contains_document_placeholder(self) -> None:
        """Test that template has {document} placeholder."""
        assert "{document}" in CONTEXT_PROMPT_TEMPLATE

    def test_template_contains_chunk_placeholder(self) -> None:
        """Test that template has {chunk} placeholder."""
        assert "{chunk}" in CONTEXT_PROMPT_TEMPLATE

    def test_template_has_document_xml_tags(self) -> None:
        """Test that template contains document delimiter markers."""
        assert "!---DOCUMENT START---!" in CONTEXT_PROMPT_TEMPLATE
        assert "!---DOCUMENT END---!" in CONTEXT_PROMPT_TEMPLATE

    def test_template_has_chunk_xml_tags(self) -> None:
        """Test that template contains chunk delimiter markers."""
        assert "!---CHUNK START---!" in CONTEXT_PROMPT_TEMPLATE
        assert "!---CHUNK END---!" in CONTEXT_PROMPT_TEMPLATE

    def test_template_requests_succinct_context(self) -> None:
        """Test that template requests succinct context output."""
        assert "succinct" in CONTEXT_PROMPT_TEMPLATE.lower()

    def test_template_is_for_search_retrieval(self) -> None:
        """Test that template mentions search retrieval purpose."""
        assert "search" in CONTEXT_PROMPT_TEMPLATE.lower()
        assert "retrieval" in CONTEXT_PROMPT_TEMPLATE.lower()


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_default_values(self) -> None:
        """Test RetryConfig default values match spec."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.exponential_base == 2.0
        assert config.max_delay == 10.0

    def test_custom_values(self) -> None:
        """Test RetryConfig with custom values."""
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            exponential_base=3.0,
            max_delay=30.0,
        )
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.exponential_base == 3.0
        assert config.max_delay == 30.0


class TestLLMContextGeneratorInit:
    """Tests for LLMContextGenerator initialization."""

    def test_init_with_chat_service(self) -> None:
        """Test initialization with a chat service."""
        mock_service = MagicMock()
        generator = LLMContextGenerator(chat_service=mock_service)
        assert generator._chat_service is mock_service

    def test_init_default_max_context_tokens(self) -> None:
        """Test default max_context_tokens is 100."""
        mock_service = MagicMock()
        generator = LLMContextGenerator(chat_service=mock_service)
        assert generator._max_context_tokens == 100

    def test_init_custom_max_context_tokens(self) -> None:
        """Test custom max_context_tokens."""
        mock_service = MagicMock()
        generator = LLMContextGenerator(
            chat_service=mock_service,
            max_context_tokens=150,
        )
        assert generator._max_context_tokens == 150

    def test_init_default_concurrency(self) -> None:
        """Test default concurrency is 10."""
        mock_service = MagicMock()
        generator = LLMContextGenerator(chat_service=mock_service)
        assert generator._concurrency == 10

    def test_init_custom_retry_config(self) -> None:
        """Test initialization with custom retry config."""
        mock_service = MagicMock()
        retry_config = RetryConfig(max_retries=5)
        generator = LLMContextGenerator(
            chat_service=mock_service,
            retry_config=retry_config,
        )
        assert generator._retry_config.max_retries == 5


class TestGenerateContext:
    """Tests for generate_context() async method."""

    @pytest.mark.asyncio
    async def test_returns_context_string(self) -> None:
        """Test that generate_context returns a string."""
        mock_service = MagicMock()
        mock_result = MagicMock()
        mock_result.content = "This chunk describes the authentication flow."
        mock_service.get_chat_message_contents = AsyncMock(return_value=[mock_result])

        generator = LLMContextGenerator(chat_service=mock_service)
        context = await generator.generate_context(
            chunk_text="Users must authenticate before accessing the system.",
            document_text="# Security Policy\n\nThis document outlines...",
        )

        assert isinstance(context, str)
        assert len(context) > 0

    @pytest.mark.asyncio
    async def test_calls_llm_with_formatted_prompt(self) -> None:
        """Test that LLM is called with properly formatted prompt."""
        mock_service = MagicMock()
        mock_result = MagicMock()
        mock_result.content = "Context here"
        mock_service.get_chat_message_contents = AsyncMock(return_value=[mock_result])

        generator = LLMContextGenerator(chat_service=mock_service)
        await generator.generate_context(
            chunk_text="Sample chunk",
            document_text="Sample document",
        )

        # Verify LLM was called
        mock_service.get_chat_message_contents.assert_called_once()

        # Get the chat history that was passed
        call_args = mock_service.get_chat_message_contents.call_args
        chat_history = call_args.kwargs.get("chat_history")
        assert chat_history is not None

    @pytest.mark.asyncio
    async def test_strips_whitespace_from_response(self) -> None:
        """Test that response whitespace is stripped."""
        mock_service = MagicMock()
        mock_result = MagicMock()
        mock_result.content = "  Context with whitespace  \n\n"
        mock_service.get_chat_message_contents = AsyncMock(return_value=[mock_result])

        generator = LLMContextGenerator(chat_service=mock_service)
        context = await generator.generate_context(
            chunk_text="Chunk",
            document_text="Document",
        )

        assert context == "Context with whitespace"
        assert not context.startswith(" ")
        assert not context.endswith(" ")

    @pytest.mark.asyncio
    async def test_handles_empty_response(self) -> None:
        """Test handling of empty LLM response."""
        mock_service = MagicMock()
        mock_result = MagicMock()
        mock_result.content = ""
        mock_service.get_chat_message_contents = AsyncMock(return_value=[mock_result])

        generator = LLMContextGenerator(chat_service=mock_service)
        context = await generator.generate_context(
            chunk_text="Chunk",
            document_text="Document",
        )

        assert context == ""


class TestContextualizeChunk:
    """Tests for contextualize_chunk() method."""

    @pytest.mark.asyncio
    async def test_prepends_context_to_content(self) -> None:
        """Test that context is prepended to chunk content."""
        mock_service = MagicMock()
        mock_result = MagicMock()
        mock_result.content = "This chunk is about authentication."
        mock_service.get_chat_message_contents = AsyncMock(return_value=[mock_result])

        generator = LLMContextGenerator(chat_service=mock_service)
        chunk = create_test_chunk(content="Users must log in.")

        result = await generator.contextualize_chunk(
            chunk=chunk,
            document_text="# Security Guide\n\nFull document here...",
        )

        assert "This chunk is about authentication." in result
        assert "Users must log in." in result

    @pytest.mark.asyncio
    async def test_format_is_context_newlines_content(self) -> None:
        """Test that format is '{context}\\n\\n{content}'."""
        mock_service = MagicMock()
        mock_result = MagicMock()
        mock_result.content = "Context text"
        mock_service.get_chat_message_contents = AsyncMock(return_value=[mock_result])

        generator = LLMContextGenerator(chat_service=mock_service)
        chunk = create_test_chunk(content="Chunk content")

        result = await generator.contextualize_chunk(
            chunk=chunk,
            document_text="Document",
        )

        assert result == "Context text\n\nChunk content"

    @pytest.mark.asyncio
    async def test_returns_original_content_on_empty_context(self) -> None:
        """Test that empty context still includes chunk content."""
        mock_service = MagicMock()
        mock_result = MagicMock()
        mock_result.content = ""
        mock_service.get_chat_message_contents = AsyncMock(return_value=[mock_result])

        generator = LLMContextGenerator(chat_service=mock_service)
        chunk = create_test_chunk(content="Original content")

        result = await generator.contextualize_chunk(
            chunk=chunk,
            document_text="Document",
        )

        # With empty context, should just return original content
        assert "Original content" in result


class TestContextualizeBatch:
    """Tests for contextualize_batch() method with concurrency control."""

    @pytest.mark.asyncio
    async def test_processes_all_chunks(self) -> None:
        """Test that all chunks are processed."""
        mock_service = MagicMock()
        call_count = 0

        async def mock_get_messages(**kwargs):
            nonlocal call_count
            call_count += 1
            mock_result = MagicMock()
            mock_result.content = f"Context {call_count}"
            return [mock_result]

        mock_service.get_chat_message_contents = mock_get_messages

        generator = LLMContextGenerator(chat_service=mock_service)
        chunks = [create_test_chunk(chunk_index=i) for i in range(5)]

        results = await generator.contextualize_batch(
            chunks=chunks,
            document_text="Document",
        )

        assert len(results) == 5
        assert call_count == 5

    @pytest.mark.asyncio
    async def test_respects_concurrency_limit(self) -> None:
        """Test that concurrency limit is respected."""
        mock_service = MagicMock()
        concurrent_calls = 0
        max_concurrent = 0

        async def mock_get_messages(**kwargs):
            nonlocal concurrent_calls, max_concurrent
            concurrent_calls += 1
            max_concurrent = max(max_concurrent, concurrent_calls)
            await asyncio.sleep(0.05)  # Small delay to test concurrency
            concurrent_calls -= 1
            mock_result = MagicMock()
            mock_result.content = "Context"
            return [mock_result]

        mock_service.get_chat_message_contents = mock_get_messages

        generator = LLMContextGenerator(chat_service=mock_service)
        generator._concurrency = 3  # Limit to 3 concurrent calls
        chunks = [create_test_chunk(chunk_index=i) for i in range(10)]

        await generator.contextualize_batch(
            chunks=chunks,
            document_text="Document",
            concurrency=3,
        )

        # Max concurrent should not exceed the limit
        assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_maintains_chunk_order(self) -> None:
        """Test that result order matches input chunk order."""
        mock_service = MagicMock()
        call_order = []

        async def mock_get_messages(**kwargs):
            # Extract chunk index from chat history
            chat_history = kwargs.get("chat_history")
            if chat_history and chat_history.messages:
                msg = str(chat_history.messages[0].content)
                # Add small random delay to shuffle completion order
                await asyncio.sleep(0.01 * (hash(msg) % 5))
            call_order.append(len(call_order))
            mock_result = MagicMock()
            mock_result.content = f"Context {len(call_order)}"
            return [mock_result]

        mock_service.get_chat_message_contents = mock_get_messages

        generator = LLMContextGenerator(chat_service=mock_service)
        chunks = [
            create_test_chunk(content=f"Chunk {i}", chunk_index=i) for i in range(5)
        ]

        results = await generator.contextualize_batch(
            chunks=chunks,
            document_text="Document",
        )

        # Results should be in order matching input chunks
        assert len(results) == 5
        for i, result in enumerate(results):
            assert f"Chunk {i}" in result

    @pytest.mark.asyncio
    async def test_empty_chunks_list(self) -> None:
        """Test handling of empty chunks list."""
        mock_service = MagicMock()
        generator = LLMContextGenerator(chat_service=mock_service)

        results = await generator.contextualize_batch(
            chunks=[],
            document_text="Document",
        )

        assert results == []
        mock_service.get_chat_message_contents.assert_not_called()


class TestExponentialBackoff:
    """Tests for retry logic with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retries_on_failure(self) -> None:
        """Test that generate_context retries on transient failure."""
        mock_service = MagicMock()
        call_count = 0

        async def mock_get_messages(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Transient error")
            mock_result = MagicMock()
            mock_result.content = "Success"
            return [mock_result]

        mock_service.get_chat_message_contents = mock_get_messages

        generator = LLMContextGenerator(
            chat_service=mock_service,
            retry_config=RetryConfig(base_delay=0.01),  # Fast retries for test
        )

        result = await generator.generate_context(
            chunk_text="Chunk",
            document_text="Document",
        )

        assert result == "Success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_delays_increase_exponentially(self) -> None:
        """Test that retry delays increase exponentially."""
        mock_service = MagicMock()
        call_times = []

        async def mock_get_messages(**kwargs):
            call_times.append(asyncio.get_event_loop().time())
            if len(call_times) < 3:
                raise Exception("Error")
            mock_result = MagicMock()
            mock_result.content = "Success"
            return [mock_result]

        mock_service.get_chat_message_contents = mock_get_messages

        generator = LLMContextGenerator(
            chat_service=mock_service,
            retry_config=RetryConfig(
                base_delay=0.1,
                exponential_base=2.0,
            ),
        )

        await generator.generate_context(
            chunk_text="Chunk",
            document_text="Document",
        )

        # Check that delays approximately follow exponential pattern
        if len(call_times) >= 3:
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]
            # Second delay should be roughly 2x the first (with some tolerance)
            # delay1 ~= 0.1, delay2 ~= 0.2
            assert delay2 > delay1 * 1.5  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_max_three_attempts(self) -> None:
        """Test that maximum 3 retry attempts are made by default."""
        mock_service = MagicMock()
        call_count = 0

        async def mock_get_messages(**kwargs):
            nonlocal call_count
            call_count += 1
            raise Exception("Persistent error")

        mock_service.get_chat_message_contents = mock_get_messages

        generator = LLMContextGenerator(
            chat_service=mock_service,
            retry_config=RetryConfig(base_delay=0.01),
        )

        # Should return empty string after exhausting retries
        result = await generator.generate_context(
            chunk_text="Chunk",
            document_text="Document",
        )

        # Default max_retries is 3
        assert call_count == 3
        assert result == ""  # Fallback to empty on failure

    @pytest.mark.asyncio
    async def test_fallback_to_no_context_on_final_failure(self) -> None:
        """Test graceful degradation on final retry failure."""
        mock_service = MagicMock()
        mock_service.get_chat_message_contents = AsyncMock(
            side_effect=Exception("Persistent error")
        )

        generator = LLMContextGenerator(
            chat_service=mock_service,
            retry_config=RetryConfig(base_delay=0.01, max_retries=2),
        )
        chunk = create_test_chunk(content="Original chunk content")

        result = await generator.contextualize_chunk(
            chunk=chunk,
            document_text="Document",
        )

        # Should fall back to original content without context
        assert result == "Original chunk content"


class TestAdaptiveConcurrency:
    """Tests for adaptive concurrency on rate limiting."""

    @pytest.mark.asyncio
    async def test_halves_concurrency_on_rate_limit(self) -> None:
        """Test that concurrency is halved on HTTP 429 error."""
        mock_service = MagicMock()
        rate_limit_count = 0
        initial_concurrency = 10

        async def mock_get_messages(**kwargs):
            nonlocal rate_limit_count
            if rate_limit_count < 2:
                rate_limit_count += 1
                error = Exception("Rate limit exceeded")
                error.status_code = 429  # type: ignore[attr-defined]
                raise error
            mock_result = MagicMock()
            mock_result.content = "Success"
            return [mock_result]

        mock_service.get_chat_message_contents = mock_get_messages

        generator = LLMContextGenerator(
            chat_service=mock_service,
            retry_config=RetryConfig(base_delay=0.01),
        )
        generator._concurrency = initial_concurrency

        await generator.generate_context(
            chunk_text="Chunk",
            document_text="Document",
        )

        # Concurrency should be reduced after rate limit
        assert generator._concurrency < initial_concurrency

    @pytest.mark.asyncio
    async def test_respects_retry_after_header(self) -> None:
        """Test that Retry-After header is respected when present."""
        mock_service = MagicMock()
        call_times = []

        async def mock_get_messages(**kwargs):
            call_times.append(asyncio.get_event_loop().time())
            if len(call_times) == 1:
                error = Exception("Rate limit")
                error.status_code = 429  # type: ignore[attr-defined]
                error.headers = {"Retry-After": "0.1"}  # type: ignore[attr-defined]
                raise error
            mock_result = MagicMock()
            mock_result.content = "Success"
            return [mock_result]

        mock_service.get_chat_message_contents = mock_get_messages

        generator = LLMContextGenerator(
            chat_service=mock_service,
            retry_config=RetryConfig(base_delay=0.01),
        )

        await generator.generate_context(
            chunk_text="Chunk",
            document_text="Document",
        )

        # Check that delay was at least the Retry-After value
        if len(call_times) >= 2:
            delay = call_times[1] - call_times[0]
            assert delay >= 0.09  # Allow small tolerance


class TestDocumentTruncation:
    """Tests for document truncation when exceeding context window."""

    @pytest.mark.asyncio
    async def test_truncates_long_document(self) -> None:
        """Test that very long documents are truncated."""
        mock_service = MagicMock()
        received_prompt = None

        async def mock_get_messages(**kwargs):
            nonlocal received_prompt
            chat_history = kwargs.get("chat_history")
            if chat_history and chat_history.messages:
                received_prompt = str(chat_history.messages[0].content)
            mock_result = MagicMock()
            mock_result.content = "Context"
            return [mock_result]

        mock_service.get_chat_message_contents = mock_get_messages

        generator = LLMContextGenerator(
            chat_service=mock_service,
            max_document_tokens=100,  # Small limit for testing
        )

        # Create a very long document
        long_document = "Word " * 10000

        await generator.generate_context(
            chunk_text="Short chunk",
            document_text=long_document,
        )

        # The prompt should be truncated
        assert received_prompt is not None
        assert len(received_prompt) < len(long_document)

    @pytest.mark.asyncio
    async def test_preserves_beginning_and_end_on_truncation(self) -> None:
        """Test that truncation preserves document beginning and end."""
        mock_service = MagicMock()
        received_prompt = None

        async def mock_get_messages(**kwargs):
            nonlocal received_prompt
            chat_history = kwargs.get("chat_history")
            if chat_history and chat_history.messages:
                received_prompt = str(chat_history.messages[0].content)
            mock_result = MagicMock()
            mock_result.content = "Context"
            return [mock_result]

        mock_service.get_chat_message_contents = mock_get_messages

        generator = LLMContextGenerator(
            chat_service=mock_service,
            max_document_tokens=200,  # Small limit
        )

        # Document with distinct beginning and end
        long_document = "BEGINNING_MARKER " + ("middle content " * 1000) + " END_MARKER"

        await generator.generate_context(
            chunk_text="Chunk",
            document_text=long_document,
        )

        # Both beginning and end should be preserved
        assert received_prompt is not None
        assert "BEGINNING_MARKER" in received_prompt
        assert "END_MARKER" in received_prompt


class TestProtocolConformance:
    """Tests that LLMContextGenerator conforms to the ContextGenerator protocol."""

    def test_is_context_generator(self) -> None:
        """Test that LLMContextGenerator is recognized as a ContextGenerator."""
        mock_service = MagicMock()
        gen = LLMContextGenerator(chat_service=mock_service)
        assert isinstance(gen, ContextGenerator)

    def test_has_contextualize_batch_method(self) -> None:
        """Test that contextualize_batch method exists and is async."""
        mock_service = MagicMock()
        gen = LLMContextGenerator(chat_service=mock_service)
        assert hasattr(gen, "contextualize_batch")
        assert inspect.iscoroutinefunction(gen.contextualize_batch)
