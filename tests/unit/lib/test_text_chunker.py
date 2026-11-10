"""Tests for TextChunker text splitting utility."""

import re

import pytest
import tiktoken

from holodeck.lib.text_chunker import TextChunker

_ENC = tiktoken.get_encoding("cl100k_base")


def _tokens(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


class TestTextChunkerInitialization:
    """Tests for TextChunker initialization and configuration."""

    def test_chunker_default_initialization(self) -> None:
        """Test TextChunker with default parameters."""
        chunker = TextChunker()
        assert chunker.chunk_size == 512
        assert chunker.chunk_overlap == 50
        assert chunker.separator_list == ["\n\n", "\n", " ", ""]

    def test_chunker_custom_chunk_size(self) -> None:
        """Test TextChunker with custom chunk_size."""
        chunker = TextChunker(chunk_size=256)
        assert chunker.chunk_size == 256
        assert chunker.chunk_overlap == 50  # Default

    def test_chunker_custom_chunk_overlap(self) -> None:
        """Test TextChunker with custom chunk_overlap."""
        chunker = TextChunker(chunk_overlap=100)
        assert chunker.chunk_overlap == 100
        assert chunker.chunk_size == 512  # Default

    def test_chunker_custom_separator_list_rejected(self) -> None:
        """Non-default separators cannot be honored, so they are rejected."""
        with pytest.raises(ValueError, match="separator_list"):
            TextChunker(separator_list=["\n", " ", ""])

    def test_chunker_default_separator_list_accepted(self) -> None:
        """The default separator list is accepted explicitly."""
        chunker = TextChunker(separator_list=["\n\n", "\n", " ", ""])
        assert chunker.separator_list == ["\n\n", "\n", " ", ""]

    def test_chunker_all_custom_parameters(self) -> None:
        """Test TextChunker with all custom parameters."""
        chunker = TextChunker(chunk_size=1024, chunk_overlap=200)
        assert chunker.chunk_size == 1024
        assert chunker.chunk_overlap == 200

    def test_chunker_invalid_chunk_size_zero(self) -> None:
        """Test that chunk_size cannot be zero."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextChunker(chunk_size=0)

    def test_chunker_invalid_chunk_size_negative(self) -> None:
        """Test that chunk_size cannot be negative."""
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            TextChunker(chunk_size=-1)

    def test_chunker_invalid_chunk_overlap_negative(self) -> None:
        """Test that chunk_overlap cannot be negative."""
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            TextChunker(chunk_overlap=-1)

    @pytest.mark.parametrize("size,overlap", [(10, 10), (10, 50)])
    def test_chunker_overlap_not_smaller_than_size_rejected(
        self, size: int, overlap: int
    ) -> None:
        """chunk_overlap must be smaller than chunk_size."""
        with pytest.raises(ValueError, match="chunk_overlap"):
            TextChunker(chunk_size=size, chunk_overlap=overlap)

    def test_chunker_chunk_overlap_zero_allowed(self) -> None:
        """Test that chunk_overlap can be zero."""
        chunker = TextChunker(chunk_overlap=0)
        assert chunker.chunk_overlap == 0


class TestTextChunkerSplitting:
    """Tests for TextChunker.split_text() method."""

    def test_split_single_paragraph(self) -> None:
        """Test splitting a single paragraph."""
        chunker = TextChunker(chunk_size=512)
        text = "This is a test paragraph with some content that should be split."
        chunks = chunker.split_text(text)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk) > 0

    def test_split_multiple_paragraphs(self) -> None:
        """Test splitting text with multiple paragraphs."""
        chunker = TextChunker(chunk_size=512)
        text = """First paragraph with some content.

Second paragraph with more content.

Third paragraph with additional content."""
        chunks = chunker.split_text(text)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_split_preserves_content(self) -> None:
        """Test that all content is preserved after splitting."""
        chunker = TextChunker(chunk_size=512)
        text = "The quick brown fox jumps over the lazy dog."
        chunks = chunker.split_text(text)
        joined = " ".join(chunks)
        # Content should be preserved (allowing for whitespace normalization)
        assert "quick" in joined
        assert "brown" in joined
        assert "fox" in joined

    def test_split_with_small_chunk_size(self) -> None:
        """Test splitting with very small chunk_size."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=2)
        text = "This is a test with multiple words to split into small chunks."
        chunks = chunker.split_text(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.strip()  # No empty chunks

    def test_split_with_large_chunk_size(self) -> None:
        """Test splitting with very large chunk_size."""
        chunker = TextChunker(chunk_size=10000)
        text = "Short text."
        chunks = chunker.split_text(text)
        assert len(chunks) >= 1

    def test_split_text_with_newlines(self) -> None:
        """Test splitting text with multiple newlines."""
        chunker = TextChunker(chunk_size=512)
        text = """Line 1
Line 2

Line 3
Line 4"""
        chunks = chunker.split_text(text)
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.strip()

    def test_split_text_returns_list_of_strings(self) -> None:
        """Test that split_text returns a list of strings."""
        chunker = TextChunker()
        text = "This is test content."
        result = chunker.split_text(text)
        assert isinstance(result, list)
        assert all(isinstance(chunk, str) for chunk in result)

    def test_split_filters_empty_chunks(self) -> None:
        """Test that empty chunks are filtered out."""
        chunker = TextChunker(chunk_size=512)
        text = "Text with content."
        chunks = chunker.split_text(text)
        # All chunks should be non-empty after stripping
        assert all(chunk.strip() for chunk in chunks)

    def test_split_long_document(self) -> None:
        """Test splitting a longer document."""
        chunker = TextChunker(
            chunk_size=50, chunk_overlap=5
        )  # Smaller chunk size to force splitting
        # Create a moderately long document
        text = "\n\n".join(
            [f"Paragraph {i}: " + "This is sample text. " * 20 for i in range(5)]
        )
        chunks = chunker.split_text(text)
        assert len(chunks) > 1
        assert all(chunk.strip() for chunk in chunks)

    def test_split_unicode_text(self) -> None:
        """Test splitting text with unicode characters."""
        chunker = TextChunker(chunk_size=512)
        text = "Hello 世界. This is unicode: café, naïve, 日本語."
        chunks = chunker.split_text(text)
        assert len(chunks) > 0
        assert all(chunk.strip() for chunk in chunks)

    def test_split_text_with_special_characters(self) -> None:
        """Test splitting text with special characters."""
        chunker = TextChunker(chunk_size=512)
        text = "Special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/~`"
        chunks = chunker.split_text(text)
        assert len(chunks) > 0

    def test_split_text_with_only_whitespace_raises_error(self) -> None:
        """Test that text with only whitespace raises ValueError."""
        chunker = TextChunker()
        with pytest.raises(ValueError, match="Text cannot be empty"):
            chunker.split_text("   \n\n   ")

    def test_split_empty_text_raises_error(self) -> None:
        """Test that empty text raises ValueError."""
        chunker = TextChunker()
        with pytest.raises(ValueError, match="Text cannot be empty"):
            chunker.split_text("")

    def test_split_none_text_raises_error(self) -> None:
        """Test that None text raises an appropriate error."""
        chunker = TextChunker()
        with pytest.raises((ValueError, AttributeError, TypeError)):
            chunker.split_text(None)  # type: ignore

    def test_split_text_with_tabs_and_spaces(self) -> None:
        """Test splitting text with tabs and spaces."""
        chunker = TextChunker(chunk_size=512)
        text = "Text\twith\ttabs\tand    spaces\nand newlines."
        chunks = chunker.split_text(text)
        assert len(chunks) > 0
        assert all(chunk.strip() for chunk in chunks)


class TestTextChunkerEdgeCases:
    """Tests for edge cases and error handling in TextChunker."""

    def test_chunker_with_very_small_chunk_size_one(self) -> None:
        """Test chunker with chunk_size of 1."""
        chunker = TextChunker(chunk_size=1, chunk_overlap=0)
        text = "Test text with multiple words."
        chunks = chunker.split_text(text)
        assert len(chunks) > 0

    def test_chunker_overlap_larger_than_chunk_size(self) -> None:
        """Overlap larger than chunk_size is rejected."""
        with pytest.raises(ValueError, match="chunk_overlap"):
            TextChunker(chunk_size=10, chunk_overlap=50)

    def test_split_single_word(self) -> None:
        """Test splitting a single word."""
        chunker = TextChunker(chunk_size=512)
        text = "Word"
        chunks = chunker.split_text(text)
        assert len(chunks) >= 1

    def test_split_repeated_content(self) -> None:
        """Test splitting text with repeated content."""
        chunker = TextChunker(chunk_size=512)
        text = "repeat " * 100
        chunks = chunker.split_text(text)
        assert len(chunks) > 0

    def test_split_very_long_single_word(self) -> None:
        """Test splitting text with a very long single word."""
        chunker = TextChunker(chunk_size=512)
        long_word = "a" * 1000
        text = long_word
        chunks = chunker.split_text(text)
        assert len(chunks) >= 1

    def test_multiple_consecutive_newlines(self) -> None:
        """Test text with multiple consecutive newlines."""
        chunker = TextChunker(chunk_size=512)
        text = "Text\n\n\n\nMore text\n\n\nEven more"
        chunks = chunker.split_text(text)
        assert len(chunks) > 0
        assert all(chunk.strip() for chunk in chunks)

    def test_split_with_different_chunk_sizes(self) -> None:
        """Test that different chunk sizes produce different results."""
        text = """This is a longer text with multiple sentences.
        It should be chunked differently based on the chunk size.
        Smaller chunk sizes should produce more chunks."""

        chunker_small = TextChunker(chunk_size=50, chunk_overlap=5)
        chunker_large = TextChunker(chunk_size=500)

        chunks_small = chunker_small.split_text(text)
        chunks_large = chunker_large.split_text(text)

        # Larger chunk size should generally produce fewer or equal chunks
        assert len(chunks_small) >= len(chunks_large)


class TestTextChunkerConfiguration:
    """Tests for TextChunker configuration and attribute access."""

    def test_chunker_attributes_accessible(self) -> None:
        """Test that all chunker attributes are accessible."""
        chunker = TextChunker(chunk_size=256, chunk_overlap=75)
        assert hasattr(chunker, "chunk_size")
        assert hasattr(chunker, "chunk_overlap")
        assert hasattr(chunker, "separator_list")
        assert chunker.chunk_size == 256
        assert chunker.chunk_overlap == 75

    def test_chunker_default_constants(self) -> None:
        """Test that default constants are defined correctly."""
        assert TextChunker.DEFAULT_CHUNK_SIZE == 512
        assert TextChunker.DEFAULT_CHUNK_OVERLAP == 50

    def test_chunker_uses_defaults_when_none_provided(self) -> None:
        """Test that chunker uses defaults when parameters not provided."""
        chunker = TextChunker()
        assert chunker.chunk_size == TextChunker.DEFAULT_CHUNK_SIZE
        assert chunker.chunk_overlap == TextChunker.DEFAULT_CHUNK_OVERLAP

    def test_chunker_multiple_instances_independent(self) -> None:
        """Test that multiple TextChunker instances are independent."""
        chunker1 = TextChunker(chunk_size=256)
        chunker2 = TextChunker(chunk_size=1024)
        assert chunker1.chunk_size == 256
        assert chunker2.chunk_size == 1024
        chunker1.chunk_size = 512
        assert chunker2.chunk_size == 1024  # Should not be affected


class TestTextChunkerTokenSplitting:
    """Behavior of the HoloDeck token-based splitter."""

    LONG = " ".join(f"word{i}" for i in range(600))

    @pytest.mark.parametrize("size,overlap", [(16, 0), (32, 8), (64, 10), (100, 99)])
    def test_every_chunk_within_token_limit(self, size: int, overlap: int) -> None:
        text = "\n\n".join(
            [f"Paragraph {i}. " + "A sentence here. " * 30 for i in range(6)]
            + [self.LONG, "x" * 500]
        )
        chunks = TextChunker(chunk_size=size, chunk_overlap=overlap).split_text(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk == chunk.strip() and chunk
            assert _tokens(chunk) <= size

    def test_overlap_repeated_inside_long_unit(self) -> None:
        overlap = 6
        chunks = TextChunker(chunk_size=40, chunk_overlap=overlap).split_text(self.LONG)
        assert len(chunks) > 2
        for prev, nxt in zip(chunks, chunks[1:], strict=False):
            shared = None
            for k in range(min(len(prev), len(nxt)), 0, -1):
                if prev.endswith(nxt[:k]):
                    shared = nxt[:k]
                    break
            assert shared is not None
            assert 0 < _tokens(shared) <= overlap
            assert _tokens(shared) >= overlap - 1

    def test_zero_overlap_has_no_repetition(self) -> None:
        chunks = TextChunker(chunk_size=40, chunk_overlap=0).split_text(self.LONG)
        assert " ".join(chunks).split() == self.LONG.split()

    def test_paragraph_boundaries_preferred(self) -> None:
        paras = [
            (f"Paragraph {i} " + "alpha beta gamma. " * 8).strip() for i in range(6)
        ]
        size = max(_tokens(p) for p in paras) * 2 + 4
        chunks = TextChunker(chunk_size=size, chunk_overlap=0).split_text(
            "\n\n".join(paras)
        )
        assert len(chunks) > 1
        for chunk in chunks:
            for part in chunk.split("\n\n"):
                assert part in paras

    def test_sentence_boundary_used_before_words(self) -> None:
        sentences = [f"Sentence number {i} is here." for i in range(12)]
        chunks = TextChunker(chunk_size=20, chunk_overlap=0).split_text(
            " ".join(sentences)
        )
        for chunk in chunks:
            assert chunk.endswith(".")
            assert chunk.startswith("Sentence")

    def test_unicode_content_fully_covered(self) -> None:
        text = ("日本語のテキスト。café naïve résumé 😀 " * 40).strip()
        chunks = TextChunker(chunk_size=24, chunk_overlap=4).split_text(text)
        assert all(_tokens(c) <= 24 for c in chunks)
        assert "\ufffd" not in "".join(chunks)
        covered = re.sub(r"\s+", "", "".join(chunks))
        # Overlap only adds repeated text; every original char appears in order.
        it = iter(covered)
        assert all(ch in it for ch in re.sub(r"\s+", "", text))

    def test_hard_split_without_overlap_reconstructs_text(self) -> None:
        text = "é" * 300
        chunks = TextChunker(chunk_size=20, chunk_overlap=0).split_text(text)
        assert "".join(chunks) == text
        assert all(_tokens(c) <= 20 for c in chunks)

    def test_special_token_text_does_not_raise(self) -> None:
        chunks = TextChunker(chunk_size=8, chunk_overlap=0).split_text(
            "a <|endoftext|> b"
        )
        assert chunks

    def test_deterministic(self) -> None:
        chunker = TextChunker(chunk_size=32, chunk_overlap=5)
        assert chunker.split_text(self.LONG) == chunker.split_text(self.LONG)

    def test_text_within_limit_is_single_chunk(self) -> None:
        assert TextChunker(chunk_size=50, chunk_overlap=5).split_text(
            "  Hello world.  "
        ) == ["Hello world."]
