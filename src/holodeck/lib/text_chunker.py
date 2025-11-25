"""Text chunking utilities for document processing.

This module provides text splitting capabilities for preparing documents
for embedding generation. Uses Semantic Kernel's text splitting functions
for intelligent chunking based on semantic boundaries.
"""

from __future__ import annotations

import logging

from semantic_kernel.text import split_plaintext_paragraph

logger = logging.getLogger(__name__)


class TextChunker:
    """Wrapper for text chunking using Semantic Kernel.

    Splits text into chunks of approximately equal size with token-based sizing.
    Uses Semantic Kernel's split functions for consistent chunk boundaries.

    Attributes:
        chunk_size: Target number of tokens per chunk
        chunk_overlap: Overlapping tokens (note: not fully supported by Semantic Kernel)
    """

    DEFAULT_CHUNK_SIZE = 512
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        separator_list: list[str] | None = None,
    ) -> None:
        """Initialize text chunker.

        Args:
            chunk_size: Target number of tokens per chunk (default: 512)
            chunk_overlap: Number of overlapping tokens between chunks (default: 50)
                Note: Semantic Kernel's split functions don't support overlap,
                so this is stored for API compatibility
            separator_list: Custom list of separators (ignored, kept for compatibility)

        Raises:
            ValueError: If chunk_size <= 0 or chunk_overlap < 0
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator_list = separator_list or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks using Semantic Kernel's split functions.

        Uses split_plaintext_paragraph to chunk text at paragraph boundaries
        when possible, falling back to line-based splitting for consistency.

        Args:
            text: Text to split into chunks

        Returns:
            List of text chunks, each approximately chunk_size tokens

        Raises:
            ValueError: If text is empty
            RuntimeError: If chunking operation fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            # Split text into lines first
            lines = text.split("\n")

            # Use Semantic Kernel's paragraph splitter
            # It handles token counting automatically
            chunks_result = split_plaintext_paragraph(lines, max_tokens=self.chunk_size)

            # Filter out empty chunks
            chunks: list[str] = [chunk for chunk in chunks_result if chunk.strip()]
            return chunks
        except Exception as e:
            logger.error(f"Error during text chunking: {e}")
            raise RuntimeError(f"Failed to chunk text: {e}") from e
