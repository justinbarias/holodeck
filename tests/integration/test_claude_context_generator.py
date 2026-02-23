"""Integration tests: ClaudeSDKContextGenerator → Claude Agent SDK → Claude API.

These tests exercise the real end-to-end context generation path with no mocks.
Requires a valid ``CLAUDE_CODE_OAUTH_TOKEN`` in ``tests/integration/.env``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from holodeck.lib.claude_context_generator import (
    ClaudeContextConfig,
    ClaudeSDKContextGenerator,
)
from holodeck.lib.structured_chunker import ChunkType, DocumentChunk

# ---------------------------------------------------------------------------
# Environment & skip logic
# ---------------------------------------------------------------------------

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

SKIP_LLM_TESTS = os.getenv("SKIP_LLM_INTEGRATION_TESTS", "false").lower() == "true"
CLAUDE_CODE_OAUTH_TOKEN = os.getenv("CLAUDE_CODE_OAUTH_TOKEN")

skip_if_no_claude_oauth = pytest.mark.skipif(
    SKIP_LLM_TESTS or not CLAUDE_CODE_OAUTH_TOKEN,
    reason="CLAUDE_CODE_OAUTH_TOKEN not configured or LLM tests disabled",
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _unset_claudecode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset CLAUDECODE so the Agent SDK subprocess doesn't reject nested sessions."""
    monkeypatch.delenv("CLAUDECODE", raising=False)


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

SAMPLE_DOCUMENT = """\
# Company Return Policy

## Overview
Our company offers a comprehensive return policy designed to ensure
customer satisfaction. All purchases are eligible for return within
30 days of the original purchase date.

## Eligibility
Products must be in their original packaging and unused condition.
Electronics must include all original accessories and manuals.
Perishable goods are not eligible for return.

## Process
To initiate a return, customers should contact our support team
via email or phone. A return merchandise authorization (RMA) number
will be provided within 24 hours. Items must be shipped back within
7 business days of receiving the RMA.

## Refunds
Refunds are processed within 5-7 business days after we receive
the returned item. Original shipping costs are non-refundable.
Store credit is available as an alternative to monetary refund.
"""


def _make_chunk(
    content: str,
    chunk_index: int = 0,
    chunk_id: str | None = None,
) -> DocumentChunk:
    """Create a DocumentChunk for integration testing."""
    return DocumentChunk(
        id=chunk_id or f"policy_chunk_{chunk_index}",
        source_path="/docs/return_policy.md",
        chunk_index=chunk_index,
        content=content,
        parent_chain=["Company Return Policy"],
        section_id=f"sec_{chunk_index}",
        chunk_type=ChunkType.CONTENT,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
class TestClaudeContextGeneratorIntegration:
    """End-to-end integration tests for ClaudeSDKContextGenerator."""

    @skip_if_no_claude_oauth
    @pytest.mark.asyncio
    async def test_query_claude_returns_non_empty_text(self) -> None:
        """_query_claude returns a non-empty string from the real API."""
        gen = ClaudeSDKContextGenerator()
        result = await gen._query_claude("Respond with the single word: hello")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "hello" in result.lower()

    @skip_if_no_claude_oauth
    @pytest.mark.asyncio
    async def test_single_context_generation(self) -> None:
        """_generate_single_context produces relevant context for a chunk."""
        gen = ClaudeSDKContextGenerator()
        chunk_text = (
            "Products must be in their original packaging and unused condition. "
            "Electronics must include all original accessories and manuals."
        )

        context = await gen._generate_single_context(chunk_text, SAMPLE_DOCUMENT)

        assert isinstance(context, str)
        assert len(context) > 0
        # Context should reference the document topic
        context_lower = context.lower()
        assert any(
            keyword in context_lower
            for keyword in ("return", "policy", "eligib", "product", "condition")
        ), f"Context should reference document topic, got: {context}"

    @skip_if_no_claude_oauth
    @pytest.mark.asyncio
    async def test_contextualize_batch_single_chunk(self) -> None:
        """contextualize_batch with one chunk returns contextualized content."""
        gen = ClaudeSDKContextGenerator()
        chunk = _make_chunk(
            "Refunds are processed within 5-7 business days after we receive "
            "the returned item."
        )

        results = await gen.contextualize_batch([chunk], SAMPLE_DOCUMENT)

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, str)
        # Result should contain the original chunk content
        assert chunk.content in result
        # Result should be longer than the original (context was prepended)
        assert len(result) > len(chunk.content)

    @skip_if_no_claude_oauth
    @pytest.mark.asyncio
    async def test_contextualize_batch_multiple_chunks(self) -> None:
        """contextualize_batch with multiple chunks returns correct count."""
        config = ClaudeContextConfig(batch_size=3)
        gen = ClaudeSDKContextGenerator(config=config)
        chunks = [
            _make_chunk(
                "All purchases are eligible for return within 30 days.",
                chunk_index=0,
            ),
            _make_chunk(
                "Perishable goods are not eligible for return.",
                chunk_index=1,
            ),
            _make_chunk(
                "Store credit is available as an alternative to monetary refund.",
                chunk_index=2,
            ),
        ]

        results = await gen.contextualize_batch(chunks, SAMPLE_DOCUMENT)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert isinstance(result, str)
            # Each result should contain its original chunk content
            assert (
                chunks[i].content in result
            ), f"Result {i} should contain original chunk content"

    @skip_if_no_claude_oauth
    @pytest.mark.asyncio
    async def test_contextualize_batch_preserves_order(self) -> None:
        """Results from contextualize_batch match input chunk order."""
        config = ClaudeContextConfig(batch_size=2)
        gen = ClaudeSDKContextGenerator(config=config)
        chunks = [
            _make_chunk("CHUNK_ALPHA: Overview section content.", chunk_index=0),
            _make_chunk("CHUNK_BETA: Eligibility section content.", chunk_index=1),
            _make_chunk("CHUNK_GAMMA: Refund section content.", chunk_index=2),
        ]

        results = await gen.contextualize_batch(chunks, SAMPLE_DOCUMENT)

        assert len(results) == 3
        assert "CHUNK_ALPHA" in results[0]
        assert "CHUNK_BETA" in results[1]
        assert "CHUNK_GAMMA" in results[2]

    @skip_if_no_claude_oauth
    @pytest.mark.asyncio
    async def test_contextualize_batch_empty_input(self) -> None:
        """contextualize_batch with empty list returns empty list (no API call)."""
        gen = ClaudeSDKContextGenerator()
        results = await gen.contextualize_batch([], SAMPLE_DOCUMENT)
        assert results == []

    @skip_if_no_claude_oauth
    @pytest.mark.asyncio
    async def test_batch_prompt_returns_parseable_json(self) -> None:
        """Batch prompt elicits a response that parses as a JSON array."""
        gen = ClaudeSDKContextGenerator()
        chunks = [
            _make_chunk("The return window is 30 days.", chunk_index=0),
            _make_chunk("Electronics must include all accessories.", chunk_index=1),
        ]

        prompt = gen._build_batch_prompt(chunks, SAMPLE_DOCUMENT)
        response = await gen._query_claude(prompt)

        parsed = gen._parse_batch_response(response, 2)
        assert (
            parsed is not None
        ), f"Expected parseable JSON array of 2 items, got: {response!r}"
        assert len(parsed) == 2
        assert all(isinstance(s, str) and len(s) > 0 for s in parsed)
