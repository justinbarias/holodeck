"""Claude Agent SDK Context Generator for contextual embeddings.

This module implements the Anthropic contextual retrieval approach using the
Claude Agent SDK ``query()`` function instead of Semantic Kernel.  It enables
Claude backend users (who authenticate via ``CLAUDE_CODE_OAUTH_TOKEN``) to get
contextual embeddings without separate API key configuration.

Key Features:
- Batch prompt strategy: sends N numbered chunks per prompt, requests JSON array
- Automatic fallback to single-chunk prompts when batch JSON parsing fails
- Exponential backoff retry logic with configurable parameters
- Document truncation preserving beginning and end
- Concurrency control via asyncio.Semaphore

References:
- https://www.anthropic.com/news/contextual-retrieval
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import tiktoken

if TYPE_CHECKING:
    from holodeck.lib.backends.base import ContextGenerator

    # Protocol conformance — verified at type-check time only
    _: ContextGenerator = cast("ClaudeSDKContextGenerator", None)

from holodeck.lib.structured_chunker import DocumentChunk

logger = logging.getLogger(__name__)

# Regex for stripping markdown code fences from LLM responses
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?\s*```$", re.DOTALL)


@dataclass
class ClaudeContextConfig:
    """Configuration for ClaudeSDKContextGenerator.

    Attributes:
        model: Claude model ID for context generation (default: Haiku for cost).
        batch_size: Number of chunks per batch prompt.
        concurrency: Maximum concurrent ``query()`` calls.
        max_retries: Retry attempts per query.
        base_delay: Initial retry delay in seconds.
        max_document_tokens: Document truncation limit in tokens.
    """

    model: str = "claude-haiku-4-5"
    batch_size: int = 10
    concurrency: int = 5
    max_retries: int = 3
    base_delay: float = 1.0
    max_document_tokens: int = 8000


class ClaudeSDKContextGenerator:
    """Generate contextual embeddings using the Claude Agent SDK.

    Conforms to the ``ContextGenerator`` protocol defined in
    ``holodeck.lib.backends.base``.

    Uses the Claude Agent SDK ``query()`` function to call a cheap/fast model
    (Haiku by default) for generating situating context for document chunks.
    Supports batched prompts (multiple chunks per call) with JSON parsing and
    automatic fallback to single-chunk prompts on failure.

    Attributes:
        _config: Configuration for the generator.
        _max_context_tokens: Maximum tokens for generated context.
        _encoder: Tiktoken encoder for token counting.
    """

    _SYSTEM_PROMPT = (
        "You are a document analysis assistant. You produce short, succinct "
        "context descriptions that situate document chunks within their source "
        "document. Always respond with valid JSON when asked for JSON output."
    )

    def __init__(
        self,
        config: ClaudeContextConfig | None = None,
        max_context_tokens: int = 100,
    ) -> None:
        """Initialize the Claude SDK Context Generator.

        Args:
            config: Configuration for the generator. Uses defaults if not provided.
            max_context_tokens: Maximum tokens for generated context (default: 100).
        """
        self._config = config or ClaudeContextConfig()
        self._max_context_tokens = max_context_tokens
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def _truncate_document(self, document: str) -> str:
        """Truncate document to fit within max_document_tokens.

        Preserves beginning and end of document when truncation is needed,
        as these sections often contain important context (title, conclusion).

        Args:
            document: Full document text.

        Returns:
            Truncated document if needed, otherwise original.
        """
        tokens = self._encoder.encode(document)
        if len(tokens) <= self._config.max_document_tokens:
            return document

        half_tokens = self._config.max_document_tokens // 2
        beginning: str = self._encoder.decode(tokens[:half_tokens])
        end: str = self._encoder.decode(tokens[-half_tokens:])

        return beginning + "\n\n[... document truncated ...]\n\n" + end

    def _build_batch_prompt(
        self, chunks: list[DocumentChunk], document_text: str
    ) -> str:
        """Build a batch prompt requesting JSON array of context strings.

        Args:
            chunks: List of chunks to contextualize in one call.
            document_text: Full document text (will be truncated if needed).

        Returns:
            Formatted prompt string.
        """
        truncated_doc = self._truncate_document(document_text)
        numbered_chunks = "\n".join(
            f"Chunk {i + 1}:\n{chunk.content}" for i, chunk in enumerate(chunks)
        )
        return (
            f"!---DOCUMENT START---!\n{truncated_doc}\n!---DOCUMENT END---!\n\n"
            f"Below are {len(chunks)} chunks from the above document.\n\n"
            f"{numbered_chunks}\n\n"
            f"For each chunk, provide a short succinct context (50-100 tokens) "
            f"to situate it within the overall document for the purposes of "
            f"improving search retrieval.\n\n"
            f"Respond with a JSON array of exactly {len(chunks)} strings, "
            f"one context per chunk, in the same order. "
            f"Output ONLY the JSON array, no other text."
        )

    def _build_single_prompt(self, chunk_text: str, document_text: str) -> str:
        """Build a single-chunk context prompt (fallback).

        Args:
            chunk_text: The chunk content to contextualize.
            document_text: Full document text (will be truncated if needed).

        Returns:
            Formatted prompt string.
        """
        truncated_doc = self._truncate_document(document_text)
        return (
            f"!---DOCUMENT START---!\n{truncated_doc}\n!---DOCUMENT END---!\n"
            f"Here is the chunk we want to situate within the whole document\n"
            f"!---CHUNK START---!\n{chunk_text}\n!---CHUNK END---!\n"
            f"Please give a short succinct context to situate this chunk within "
            f"the overall document for the purposes of improving search retrieval "
            f"of the chunk. Answer only with the succinct context and nothing else."
        )

    def _parse_batch_response(
        self, response: str, expected_count: int
    ) -> list[str] | None:
        """Parse a batch response expecting a JSON array of context strings.

        Handles markdown code fences and validates structure/count.

        Args:
            response: Raw LLM response text.
            expected_count: Expected number of items in the array.

        Returns:
            List of context strings if valid, None on any parsing failure.
        """
        text = response.strip()

        # Strip markdown code fences
        fence_match = _CODE_FENCE_RE.match(text)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

        if not isinstance(parsed, list):
            return None

        if len(parsed) != expected_count:
            return None

        # Coerce all items to strings
        return [str(item) for item in parsed]

    async def _query_claude(self, prompt: str) -> str:
        """Execute a single query against the Claude Agent SDK.

        Uses the ``_wrap_prompt()`` pattern from ``claude_backend.py`` to keep
        stdin open for bidirectional communication.

        Args:
            prompt: The prompt text to send.

        Returns:
            The text response from Claude, stripped of whitespace.
        """
        from claude_agent_sdk import ClaudeAgentOptions, query

        async def _wrap_prompt(message: str) -> AsyncGenerator[dict[str, Any], None]:
            yield {
                "type": "user",
                "session_id": "",
                "message": {"role": "user", "content": message},
                "parent_tool_use_id": None,
            }

        options = ClaudeAgentOptions(
            model=self._config.model,
            system_prompt=self._SYSTEM_PROMPT,
            permission_mode="bypassPermissions",
            max_turns=1,
        )

        text_parts: list[str] = []
        prompt_iter = _wrap_prompt(prompt)
        async for msg in query(prompt=prompt_iter, options=options):
            msg_type = msg.__class__.__name__
            if msg_type == "AssistantMessage":
                for block in cast(Any, msg).content:
                    if block.__class__.__name__ == "TextBlock":
                        text_parts.append(block.text)

        return "".join(text_parts).strip()

    async def _generate_single_context(
        self, chunk_text: str, document_text: str
    ) -> str:
        """Generate context for a single chunk with retry logic.

        Args:
            chunk_text: The chunk content.
            document_text: The full document text.

        Returns:
            Generated context string, or empty string on failure.
        """
        prompt = self._build_single_prompt(chunk_text, document_text)

        for attempt in range(self._config.max_retries):
            try:
                return await self._query_claude(prompt)
            except Exception as e:
                is_last = attempt == self._config.max_retries - 1
                if is_last:
                    logger.warning(
                        "Single context generation failed after %d attempts: %s",
                        self._config.max_retries,
                        e,
                    )
                    return ""

                delay = self._config.base_delay * (2**attempt)
                logger.debug(
                    "Retry attempt %d/%d after %.2fs delay",
                    attempt + 1,
                    self._config.max_retries,
                    delay,
                )
                await asyncio.sleep(delay)

        return ""  # pragma: no cover

    async def _process_batch(
        self, chunks: list[DocumentChunk], document_text: str
    ) -> list[str]:
        """Process a batch of chunks — try batch prompt, fall back to singles.

        Args:
            chunks: List of chunks for this batch.
            document_text: Full document text.

        Returns:
            List of contextualized content strings (context + chunk content).
        """
        # Try batch prompt first
        try:
            batch_prompt = self._build_batch_prompt(chunks, document_text)
            batch_response = await self._query_claude(batch_prompt)
            contexts = self._parse_batch_response(batch_response, len(chunks))

            if contexts is not None:
                results: list[str] = []
                for ctx, chunk in zip(contexts, chunks, strict=True):
                    if ctx:
                        results.append(f"{ctx}\n\n{chunk.content}")
                    else:
                        results.append(chunk.content)
                return results

            logger.debug(
                "Batch JSON parsing failed, falling back to single prompts "
                "for %d chunks",
                len(chunks),
            )
        except Exception as e:
            logger.debug(
                "Batch query failed (%s), falling back to single prompts "
                "for %d chunks",
                e,
                len(chunks),
            )

        # Fallback: process each chunk individually
        results = []
        for chunk in chunks:
            context = await self._generate_single_context(chunk.content, document_text)
            if context:
                results.append(f"{context}\n\n{chunk.content}")
            else:
                results.append(chunk.content)
        return results

    async def contextualize_batch(
        self,
        chunks: list[DocumentChunk],
        document_text: str,
        concurrency: int | None = None,
    ) -> list[str]:
        """Generate contextual descriptions for a batch of chunks.

        Splits chunks into sub-batches of ``config.batch_size``, processes them
        concurrently (bounded by semaphore), and returns results in order.

        Args:
            chunks: Document chunks to contextualize.
            document_text: Full text of the source document.
            concurrency: Maximum concurrent LLM calls. Uses config default if None.

        Returns:
            A list of contextualized content strings, one per chunk.
        """
        if not chunks:
            return []

        effective_concurrency = concurrency or self._config.concurrency
        semaphore = asyncio.Semaphore(effective_concurrency)
        batch_size = self._config.batch_size

        # Split into sub-batches
        sub_batches = [
            chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)
        ]

        async def process_with_semaphore(
            batch: list[DocumentChunk],
        ) -> list[str]:
            async with semaphore:
                return await self._process_batch(batch, document_text)

        # Process all sub-batches concurrently
        batch_results = await asyncio.gather(
            *(process_with_semaphore(batch) for batch in sub_batches)
        )

        # Flatten results in order
        return [item for sublist in batch_results for item in sublist]
