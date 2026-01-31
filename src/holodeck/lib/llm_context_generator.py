"""LLM Context Generator for contextual embeddings.

This module implements the Anthropic contextual retrieval approach, which
generates short context snippets for document chunks to improve semantic
search retrieval by 35-49%.

Key Features:
- Prompt template following Anthropic's recommended format
- Async context generation with configurable LLM service
- Batch processing with semaphore-based concurrency control
- Exponential backoff retry logic for resilience
- Adaptive concurrency on rate limiting (HTTP 429)
- Document truncation for context window management

References:
- https://www.anthropic.com/news/contextual-retrieval
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import tiktoken

if TYPE_CHECKING:
    from semantic_kernel.connectors.ai.chat_completion_client_base import (
        ChatCompletionClientBase,
    )
    from semantic_kernel.connectors.ai.prompt_execution_settings import (
        PromptExecutionSettings,
    )

from holodeck.lib.structured_chunker import DocumentChunk

logger = logging.getLogger(__name__)

# Anthropic's recommended prompt template for contextual embeddings
CONTEXT_PROMPT_TEMPLATE = """!---DOCUMENT START---!
{document}
!---DOCUMENT END---!
Here is the chunk we want to situate within the whole document
!---CHUNK START---!
{chunk}
!---CHUNK END---!
Please give a short succinct context to situate this chunk within the overall \
document for the purposes of improving search retrieval of the chunk. \
Answer only with the succinct context and nothing else."""


@dataclass
class RetryConfig:
    """Configuration for exponential backoff retry logic.

    Attributes:
        max_retries: Maximum number of retry attempts (default: 3).
        base_delay: Initial delay in seconds before first retry (default: 1.0).
        exponential_base: Multiplier for exponential backoff (default: 2.0).
        max_delay: Maximum delay cap in seconds (default: 10.0).

    Example:
        With defaults, delays are: 1s, 2s, 4s (capped at max_delay if exceeded).
    """

    max_retries: int = 3
    base_delay: float = 1.0
    exponential_base: float = 2.0
    max_delay: float = 10.0


class LLMContextGenerator:
    """Generate contextual embeddings using LLM (Anthropic approach).

    This class generates short context snippets for document chunks to improve
    semantic search retrieval. It follows Anthropic's contextual retrieval
    approach which prepends situational context to each chunk before embedding.

    Attributes:
        _chat_service: Semantic Kernel chat completion service.
        _max_context_tokens: Maximum tokens for generated context (default: 100).
        _max_document_tokens: Maximum tokens for document in prompt (default: 8000).
        _concurrency: Current concurrency limit for batch processing.
        _retry_config: Configuration for exponential backoff retries.

    Example:
        >>> from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
        >>> chat_service = OpenAIChatCompletion(ai_model_id="gpt-4o-mini")
        >>> generator = LLMContextGenerator(chat_service=chat_service)
        >>> context = await generator.generate_context(chunk_text, document_text)
    """

    DEFAULT_MAX_CONTEXT_TOKENS = 100
    DEFAULT_MAX_DOCUMENT_TOKENS = 8000
    DEFAULT_CONCURRENCY = 10
    MIN_CONCURRENCY = 1

    def __init__(
        self,
        chat_service: "ChatCompletionClientBase",
        execution_settings: "PromptExecutionSettings | None" = None,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        max_document_tokens: int = DEFAULT_MAX_DOCUMENT_TOKENS,
        retry_config: RetryConfig | None = None,
    ) -> None:
        """Initialize the LLM Context Generator.

        Args:
            chat_service: Semantic Kernel chat completion service instance.
            execution_settings: Optional prompt execution settings. If not provided,
                defaults will be used with max_tokens set to max_context_tokens.
            max_context_tokens: Maximum tokens for generated context (default: 100).
            max_document_tokens: Maximum tokens for document truncation (default: 8000).
            retry_config: Configuration for retry logic. Uses defaults if not provided.
        """
        self._chat_service = chat_service
        self._execution_settings = execution_settings
        self._max_context_tokens = max_context_tokens
        self._max_document_tokens = max_document_tokens
        self._concurrency = self.DEFAULT_CONCURRENCY
        self._retry_config = retry_config or RetryConfig()
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens for.

        Returns:
            Token count.
        """
        if not text:
            return 0
        return len(self._encoder.encode(text))

    def _truncate_document(self, document: str) -> str:
        """Truncate document to fit within max_document_tokens.

        Preserves beginning and end of document when truncation is needed,
        as these sections often contain important context (title, conclusion).

        Args:
            document: Full document text.

        Returns:
            Truncated document if needed, otherwise original.
        """
        token_count = self._count_tokens(document)
        if token_count <= self._max_document_tokens:
            return document

        # Split tokens between beginning and end
        tokens = self._encoder.encode(document)
        half_tokens = self._max_document_tokens // 2

        # Take first half from beginning, second half from end
        beginning_tokens = tokens[:half_tokens]
        end_tokens = tokens[-half_tokens:]

        # Decode back to text
        beginning: str = self._encoder.decode(beginning_tokens)
        end: str = self._encoder.decode(end_tokens)

        truncation_marker = "\n\n[... document truncated ...]\n\n"
        return beginning + truncation_marker + end

    def _format_prompt(self, chunk_text: str, document_text: str) -> str:
        """Format the context generation prompt.

        Args:
            chunk_text: The chunk content to contextualize.
            document_text: The full document (may be truncated).

        Returns:
            Formatted prompt string.
        """
        truncated_doc = self._truncate_document(document_text)
        return CONTEXT_PROMPT_TEMPLATE.format(document=truncated_doc, chunk=chunk_text)

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM service with the given prompt.

        Args:
            prompt: The prompt to send to the LLM.

        Returns:
            The LLM's response text, stripped of whitespace.

        Raises:
            Exception: If the LLM call fails.
        """
        # Import here to avoid circular imports and allow mocking
        from semantic_kernel.connectors.ai.open_ai import (
            OpenAIChatPromptExecutionSettings,
        )
        from semantic_kernel.contents import ChatHistory

        chat_history = ChatHistory()
        chat_history.add_user_message(prompt)

        # Call with or without execution settings
        if self._execution_settings is not None:
            result = await self._chat_service.get_chat_message_contents(
                chat_history=chat_history,
                settings=self._execution_settings,
            )
        else:
            # When no settings provided, use service defaults
            # Type ignore needed as SK types don't allow None settings
            result = await self._chat_service.get_chat_message_contents(
                chat_history=chat_history,
                settings=OpenAIChatPromptExecutionSettings(),
            )

        if result and len(result) > 0:
            content = result[0].content
            return str(content).strip() if content else ""
        return ""

    def _get_retry_delay(self, attempt: int, error: Exception | None = None) -> float:
        """Calculate retry delay with exponential backoff.

        Args:
            attempt: Current attempt number (0-indexed).
            error: The exception that triggered the retry (may have Retry-After).

        Returns:
            Delay in seconds before next retry.
        """
        # Check for Retry-After header on rate limit errors
        if error is not None:
            retry_after = getattr(
                getattr(error, "headers", None), "get", lambda x: None
            )("Retry-After")
            if retry_after is None and hasattr(error, "headers"):
                headers = getattr(error, "headers", {})
                if isinstance(headers, dict):
                    retry_after = headers.get("Retry-After")

            if retry_after is not None:
                try:
                    return float(retry_after)
                except (ValueError, TypeError):
                    pass

        # Calculate exponential backoff
        delay = self._retry_config.base_delay * (
            self._retry_config.exponential_base**attempt
        )
        return min(delay, self._retry_config.max_delay)

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit (HTTP 429) error.

        Args:
            error: The exception to check.

        Returns:
            True if this is a rate limit error.
        """
        status_code = getattr(error, "status_code", None)
        return status_code == 429

    def _reduce_concurrency(self) -> None:
        """Reduce concurrency after rate limiting."""
        new_concurrency = max(self._concurrency // 2, self.MIN_CONCURRENCY)
        if new_concurrency < self._concurrency:
            logger.warning(
                f"Rate limited - reducing concurrency "
                f"from {self._concurrency} to {new_concurrency}"
            )
            self._concurrency = new_concurrency

    async def generate_context(self, chunk_text: str, document_text: str) -> str:
        """Generate context for a single chunk.

        Uses the LLM to generate a short (50-100 token) context snippet that
        situates the chunk within the broader document. Includes retry logic
        with exponential backoff for resilience.

        Args:
            chunk_text: The chunk content to generate context for.
            document_text: The full document text for context.

        Returns:
            Generated context string, or empty string on failure.

        Example:
            >>> context = await generator.generate_context(
            ...     "Force Majeure means any event...",
            ...     full_policy_document
            ... )
            >>> print(context)
            "This is the definition section of the insurance policy."
        """
        prompt = self._format_prompt(chunk_text, document_text)

        for attempt in range(self._retry_config.max_retries):
            try:
                return await self._call_llm(prompt)
            except Exception as e:
                is_last_attempt = attempt == self._retry_config.max_retries - 1

                if is_last_attempt:
                    max_retries = self._retry_config.max_retries
                    logger.warning(
                        f"Context generation failed after {max_retries} attempts: {e}"
                    )
                    return ""  # Graceful degradation

                # Handle rate limiting
                if self._is_rate_limit_error(e):
                    self._reduce_concurrency()

                delay = self._get_retry_delay(attempt, e)
                logger.debug(
                    f"Retry attempt {attempt + 1}/{self._retry_config.max_retries} "
                    f"after {delay:.2f}s delay"
                )
                await asyncio.sleep(delay)

        return ""  # Should not reach here, but for safety

    async def contextualize_chunk(
        self, chunk: DocumentChunk, document_text: str
    ) -> str:
        """Contextualize a single document chunk.

        Generates context for the chunk and prepends it to the chunk content
        in the format: "{context}\\n\\n{chunk.content}".

        Args:
            chunk: The DocumentChunk to contextualize.
            document_text: The full document text for context.

        Returns:
            Contextualized content string. On failure, returns original content.

        Example:
            >>> result = await generator.contextualize_chunk(chunk, document)
            >>> print(result)
            "This chunk defines force majeure terms.

            Force Majeure means any event beyond reasonable control..."
        """
        context = await self.generate_context(chunk.content, document_text)

        if context:
            return f"{context}\n\n{chunk.content}"
        else:
            # Graceful degradation - return original content
            return chunk.content

    async def contextualize_batch(
        self,
        chunks: list[DocumentChunk],
        document_text: str,
        concurrency: int | None = None,
    ) -> list[str]:
        """Batch process multiple chunks with concurrency control.

        Processes all chunks concurrently using a semaphore to limit the number
        of simultaneous LLM calls. Results maintain the same order as input chunks.

        Args:
            chunks: List of DocumentChunks to contextualize.
            document_text: The full document text for context.
            concurrency: Optional concurrency override. Uses instance default if None.

        Returns:
            List of contextualized content strings in same order as input chunks.

        Example:
            >>> results = await generator.contextualize_batch(chunks, document)
            >>> assert len(results) == len(chunks)
        """
        if not chunks:
            return []

        effective_concurrency = concurrency or self._concurrency
        semaphore = asyncio.Semaphore(effective_concurrency)

        async def process_chunk(chunk: DocumentChunk) -> str:
            async with semaphore:
                return await self.contextualize_chunk(chunk, document_text)

        # Use gather to maintain order
        tasks = [process_chunk(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)

        return list(results)
