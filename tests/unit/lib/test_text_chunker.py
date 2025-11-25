"""Tests for TextChunker text splitting utility."""

import pytest

from holodeck.lib.text_chunker import TextChunker


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

    def test_chunker_custom_separator_list(self) -> None:
        """Test TextChunker with custom separator_list."""
        custom_separators = ["\n", " ", ""]
        chunker = TextChunker(separator_list=custom_separators)
        assert chunker.separator_list == custom_separators

    def test_chunker_all_custom_parameters(self) -> None:
        """Test TextChunker with all custom parameters."""
        custom_separators = ["\n\n\n", "\n\n"]
        chunker = TextChunker(
            chunk_size=1024,
            chunk_overlap=200,
            separator_list=custom_separators,
        )
        assert chunker.chunk_size == 1024
        assert chunker.chunk_overlap == 200
        assert chunker.separator_list == custom_separators

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
        chunker = TextChunker(chunk_size=10)
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
        chunker = TextChunker(chunk_size=50)  # Smaller chunk size to force splitting
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
        chunker = TextChunker(chunk_size=1)
        text = "Test text with multiple words."
        chunks = chunker.split_text(text)
        assert len(chunks) > 0

    def test_chunker_overlap_larger_than_chunk_size(self) -> None:
        """Test chunker where overlap is larger than chunk_size (allowed)."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=50)
        assert chunker.chunk_overlap == 50
        text = "This is a test."
        chunks = chunker.split_text(text)
        assert len(chunks) > 0

    def test_chunker_separator_list_not_used(self) -> None:
        """Test that separator_list is stored but not used by Semantic Kernel."""
        custom_separators = ["|", "-", "_"]
        chunker = TextChunker(separator_list=custom_separators)
        text = "First|Second-Third_Fourth"
        chunks = chunker.split_text(text)
        # Should still work even though our custom separators aren't used
        # (Semantic Kernel uses its own splitting logic)
        assert len(chunks) > 0

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

        chunker_small = TextChunker(chunk_size=50)
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
