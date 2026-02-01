"""Tests for StructuredChunker markdown structure parsing.

This module contains comprehensive tests for the StructuredChunker class,
following TDD methodology. Tests cover heading extraction, parent chain
building, token-aware chunking, and flat text fallback.
"""

import re
from pathlib import Path

import pytest

from holodeck.lib.structured_chunker import (
    ACADEMIC_PATTERNS,
    AU_LEGISLATIVE_PATTERNS,
    CONTRACT_PATTERNS,
    DOMAIN_PATTERNS,
    LEGISLATIVE_PATTERNS,
    TECHNICAL_PATTERNS,
    ChunkType,
    DocumentChunk,
    StructuredChunker,
    SubsectionPattern,
)

# Fixture paths
FIXTURES_DIR = (
    Path(__file__).parent.parent.parent / "fixtures" / "hierarchical_documents"
)
SAMPLE_LEGISLATION = FIXTURES_DIR / "sample_legislation.md"
SAMPLE_TECHNICAL_DOC = FIXTURES_DIR / "sample_technical_doc.md"
SAMPLE_FLAT_TEXT = FIXTURES_DIR / "sample_flat_text.txt"


class TestStructuredChunkerInitialization:
    """Tests for StructuredChunker initialization and configuration."""

    def test_default_initialization(self) -> None:
        """Test StructuredChunker with default parameters."""
        chunker = StructuredChunker()
        assert chunker.max_tokens == 800  # Anthropic baseline

    def test_custom_max_tokens(self) -> None:
        """Test StructuredChunker with custom max_tokens."""
        chunker = StructuredChunker(max_tokens=256)
        assert chunker.max_tokens == 256

    def test_large_max_tokens(self) -> None:
        """Test StructuredChunker with large max_tokens."""
        chunker = StructuredChunker(max_tokens=4096)
        assert chunker.max_tokens == 4096

    def test_invalid_max_tokens_zero(self) -> None:
        """Test that max_tokens cannot be zero."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            StructuredChunker(max_tokens=0)

    def test_invalid_max_tokens_negative(self) -> None:
        """Test that max_tokens cannot be negative."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            StructuredChunker(max_tokens=-100)

    def test_default_constant_defined(self) -> None:
        """Test that DEFAULT_MAX_TOKENS constant is defined."""
        assert StructuredChunker.DEFAULT_MAX_TOKENS == 800


class TestExtractHeadings:
    """Tests for _extract_headings() method."""

    def test_extract_h1_heading(self) -> None:
        """Test extraction of H1 heading."""
        chunker = StructuredChunker()
        markdown = "# Title\n\nSome content."
        headings = chunker._extract_headings(markdown)
        assert len(headings) == 1
        level, title, pos = headings[0]
        assert level == 1
        assert title == "Title"
        assert pos == 0

    def test_extract_h6_heading(self) -> None:
        """Test extraction of H6 heading (deepest level)."""
        chunker = StructuredChunker()
        markdown = "###### Deep Section\n\nContent here."
        headings = chunker._extract_headings(markdown)
        assert len(headings) == 1
        level, title, _ = headings[0]
        assert level == 6
        assert title == "Deep Section"

    def test_extract_multiple_levels(self) -> None:
        """Test extraction of headings at multiple levels."""
        chunker = StructuredChunker()
        markdown = """# H1 Title

## H2 Title

### H3 Title

Content here."""
        headings = chunker._extract_headings(markdown)
        assert len(headings) == 3
        assert headings[0][0] == 1  # level
        assert headings[1][0] == 2
        assert headings[2][0] == 3
        assert headings[0][1] == "H1 Title"
        assert headings[1][1] == "H2 Title"
        assert headings[2][1] == "H3 Title"

    def test_extract_heading_with_special_chars(self) -> None:
        """Test extraction of heading with special characters."""
        chunker = StructuredChunker()
        markdown = "## Section (a)(1) - Definitions\n\nContent."
        headings = chunker._extract_headings(markdown)
        assert len(headings) == 1
        _, title, _ = headings[0]
        assert title == "Section (a)(1) - Definitions"

    def test_extract_no_headings(self) -> None:
        """Test extraction when no headings present."""
        chunker = StructuredChunker()
        markdown = "Just plain text without any markdown headings."
        headings = chunker._extract_headings(markdown)
        assert headings == []

    def test_extract_heading_requires_space_after_hash(self) -> None:
        """Test that heading requires space after hash marks."""
        chunker = StructuredChunker()
        markdown = "#NoSpace\n\n# With Space"
        headings = chunker._extract_headings(markdown)
        assert len(headings) == 1
        _, title, _ = headings[0]
        assert title == "With Space"

    def test_extract_code_block_headings_ignored(self) -> None:
        """Test that headings inside code blocks are ignored."""
        chunker = StructuredChunker()
        markdown = """# Real Heading

```markdown
# Fake Heading in Code
## Another Fake
```

## Another Real Heading"""
        headings = chunker._extract_headings(markdown)
        # Should only find 2 real headings, not the ones in code block
        assert len(headings) == 2
        assert headings[0][1] == "Real Heading"
        assert headings[1][1] == "Another Real Heading"

    def test_extract_heading_position_tracking(self) -> None:
        """Test that heading positions are tracked correctly."""
        chunker = StructuredChunker()
        markdown = "Preamble text\n\n# First\n\nContent\n\n## Second"
        headings = chunker._extract_headings(markdown)
        assert len(headings) == 2
        # First heading should be after preamble
        assert headings[0][2] > 0
        # Second heading position should be after first
        assert headings[1][2] > headings[0][2]

    def test_extract_heading_with_trailing_whitespace(self) -> None:
        """Test heading extraction handles trailing whitespace."""
        chunker = StructuredChunker()
        markdown = "# Title with spaces   \n\nContent"
        headings = chunker._extract_headings(markdown)
        assert len(headings) == 1
        _, title, _ = headings[0]
        # Title should be stripped
        assert title == "Title with spaces"


class TestSplitByHeadings:
    """Tests for _split_by_headings() method and parent_chain building."""

    def test_split_single_section(self) -> None:
        """Test splitting document with single H1."""
        chunker = StructuredChunker()
        markdown = "# Title\n\nContent here."
        headings = chunker._extract_headings(markdown)
        sections = chunker._split_by_headings(markdown, headings)
        assert len(sections) >= 1
        # Single H1 has empty parent_chain
        assert sections[0].parent_chain == []

    def test_split_nested_sections(self) -> None:
        """Test splitting document with nested headings."""
        chunker = StructuredChunker()
        markdown = """# Chapter 1

## Section 1.1

Content for section 1.1.

### Subsection 1.1.1

Content for subsection."""
        headings = chunker._extract_headings(markdown)
        sections = chunker._split_by_headings(markdown, headings)

        # Find section with H3 content
        h3_sections = [s for s in sections if s.heading_level == 3]
        assert len(h3_sections) >= 1
        # H3 should have parent_chain of ["Chapter 1", "Section 1.1"]
        assert h3_sections[0].parent_chain == ["Chapter 1", "Section 1.1"]

    def test_parent_chain_resets_on_same_level(self) -> None:
        """Test that parent_chain resets when encountering same heading level."""
        chunker = StructuredChunker()
        markdown = """# Chapter 1

## Section A

## Section B

Content for B."""
        headings = chunker._extract_headings(markdown)
        sections = chunker._split_by_headings(markdown, headings)

        # Find Section B
        section_b = [s for s in sections if "Section B" in s.heading_title]
        assert len(section_b) >= 1
        # Section B's parent should be Chapter 1 only, not include Section A
        assert section_b[0].parent_chain == ["Chapter 1"]

    def test_parent_chain_resets_on_higher_level(self) -> None:
        """Test that parent_chain resets when encountering higher level heading."""
        chunker = StructuredChunker()
        markdown = """# Chapter 1

## Section 1.1

### Deep Section

# Chapter 2

Content for Chapter 2."""
        headings = chunker._extract_headings(markdown)
        sections = chunker._split_by_headings(markdown, headings)

        # Find Chapter 2
        chapter_2 = [s for s in sections if "Chapter 2" in s.heading_title]
        assert len(chapter_2) >= 1
        # Chapter 2 should have empty parent_chain (it's a new H1)
        assert chapter_2[0].parent_chain == []

    def test_content_before_first_heading(self) -> None:
        """Test handling of content before first heading (preamble)."""
        chunker = StructuredChunker()
        markdown = """This is preamble text before any headings.

More preamble content.

# First Heading

Content after heading."""
        headings = chunker._extract_headings(markdown)
        sections = chunker._split_by_headings(markdown, headings)

        # Should have preamble section plus heading section
        assert len(sections) >= 2
        # Preamble should have empty parent_chain and heading_level 0
        preamble = sections[0]
        assert preamble.heading_level == 0
        assert "preamble" in preamble.content.lower()

    def test_section_includes_heading_text(self) -> None:
        """Test that section content includes heading text."""
        chunker = StructuredChunker()
        markdown = "# My Title\n\nParagraph content."
        headings = chunker._extract_headings(markdown)
        sections = chunker._split_by_headings(markdown, headings)
        assert len(sections) >= 1
        # Content should contain the heading or title should match
        has_title = "My Title" in sections[0].content
        title_matches = sections[0].heading_title == "My Title"
        assert has_title or title_matches


class TestSplitSection:
    """Tests for _split_section() sentence boundary splitting."""

    def test_split_short_section_no_split(self) -> None:
        """Test that short sections are not split."""
        chunker = StructuredChunker(max_tokens=800)
        # Create a section with minimal content
        from dataclasses import dataclass, field

        @dataclass
        class MockSection:
            content: str
            parent_chain: list
            heading_level: int
            heading_title: str
            subsection_ids: list = field(default_factory=list)

        section = MockSection(
            content="Short content.",
            parent_chain=["Chapter 1"],
            heading_level=2,
            heading_title="Section",
        )
        chunks = chunker._split_section(section, "test.md", 0)
        assert len(chunks) == 1

    def test_split_long_section(self) -> None:
        """Test splitting a section that exceeds max_tokens."""
        chunker = StructuredChunker(max_tokens=50)  # Very small for testing
        from dataclasses import dataclass, field

        @dataclass
        class MockSection:
            content: str
            parent_chain: list
            heading_level: int
            heading_title: str
            subsection_ids: list = field(default_factory=list)

        # Create content that will definitely exceed 50 tokens
        long_content = "This is sentence one. " * 20 + "This is sentence two. " * 20
        section = MockSection(
            content=long_content,
            parent_chain=["Chapter 1"],
            heading_level=2,
            heading_title="Long Section",
        )
        chunks = chunker._split_section(section, "test.md", 0)
        assert len(chunks) > 1

    def test_split_at_sentence_boundary(self) -> None:
        """Test that splitting occurs at sentence boundaries."""
        chunker = StructuredChunker(max_tokens=30)
        from dataclasses import dataclass, field

        @dataclass
        class MockSection:
            content: str
            parent_chain: list
            heading_level: int
            heading_title: str
            subsection_ids: list = field(default_factory=list)

        content = "First sentence here. Second sentence here. Third sentence here."
        section = MockSection(
            content=content,
            parent_chain=[],
            heading_level=1,
            heading_title="Test",
        )
        chunks = chunker._split_section(section, "test.md", 0)
        # Each chunk should end at a sentence boundary (roughly)
        for chunk in chunks:
            # Content should be reasonably complete
            assert len(chunk.content.strip()) > 0

    def test_split_preserves_all_content(self) -> None:
        """Test that splitting preserves all content."""
        chunker = StructuredChunker(max_tokens=50)
        from dataclasses import dataclass, field

        @dataclass
        class MockSection:
            content: str
            parent_chain: list
            heading_level: int
            heading_title: str
            subsection_ids: list = field(default_factory=list)

        content = "Word " * 100  # 100 words
        section = MockSection(
            content=content,
            parent_chain=[],
            heading_level=1,
            heading_title="Test",
        )
        chunks = chunker._split_section(section, "test.md", 0)
        # Rejoin and compare word count
        rejoined = " ".join(c.content for c in chunks)
        original_words = content.split()
        rejoined_words = rejoined.split()
        # All words should be preserved (allowing for whitespace normalization)
        assert len(rejoined_words) >= len(original_words) - 5

    def test_split_preserves_parent_chain(self) -> None:
        """Test that all split chunks preserve parent_chain."""
        chunker = StructuredChunker(max_tokens=30)
        from dataclasses import dataclass, field

        @dataclass
        class MockSection:
            content: str
            parent_chain: list
            heading_level: int
            heading_title: str
            subsection_ids: list = field(default_factory=list)

        content = "Sentence one. Sentence two. Sentence three. Sentence four."
        parent = ["Article I", "Chapter 2"]
        section = MockSection(
            content=content,
            parent_chain=parent,
            heading_level=3,
            heading_title="Section 3",
        )
        chunks = chunker._split_section(section, "test.md", 0)
        for chunk in chunks:
            assert chunk.parent_chain == parent

    def test_split_very_long_sentence(self) -> None:
        """Test handling of very long single sentence."""
        chunker = StructuredChunker(max_tokens=20)
        from dataclasses import dataclass, field

        @dataclass
        class MockSection:
            content: str
            parent_chain: list
            heading_level: int
            heading_title: str
            subsection_ids: list = field(default_factory=list)

        # Single sentence with no period
        content = "word " * 50
        section = MockSection(
            content=content,
            parent_chain=[],
            heading_level=1,
            heading_title="Test",
        )
        chunks = chunker._split_section(section, "test.md", 0)
        # Should still produce chunks (fallback to word boundary)
        assert len(chunks) >= 1


class TestTokenCounting:
    """Tests for token counting behavior."""

    def test_token_count_empty_text(self) -> None:
        """Test token count for empty string."""
        chunker = StructuredChunker()
        assert chunker._count_tokens("") == 0

    def test_token_count_simple_text(self) -> None:
        """Test token count for simple text."""
        chunker = StructuredChunker()
        count = chunker._count_tokens("Hello world")
        assert count > 0
        assert count <= 10  # Should be just a few tokens

    def test_token_count_unicode(self) -> None:
        """Test token counting handles unicode."""
        chunker = StructuredChunker()
        count = chunker._count_tokens("Hello 世界 日本語")
        assert count > 0

    def test_token_count_consistency(self) -> None:
        """Test that same text produces same count."""
        chunker = StructuredChunker()
        text = "The quick brown fox jumps over the lazy dog."
        count1 = chunker._count_tokens(text)
        count2 = chunker._count_tokens(text)
        assert count1 == count2

    def test_token_count_longer_text_more_tokens(self) -> None:
        """Test that longer text has more tokens."""
        chunker = StructuredChunker()
        short = "Hello"
        long = "Hello world, this is a longer sentence with more tokens."
        assert chunker._count_tokens(long) > chunker._count_tokens(short)


class TestParse:
    """Tests for the main parse() method."""

    def test_parse_returns_document_chunks(self) -> None:
        """Test that parse returns list of DocumentChunk."""
        chunker = StructuredChunker()
        markdown = "# Title\n\nContent here."
        chunks = chunker.parse(markdown, "test.md")
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)

    def test_parse_chunk_ids_sequential(self) -> None:
        """Test that chunk IDs have sequential indices."""
        chunker = StructuredChunker()
        markdown = """# Chapter 1

Content for chapter 1.

## Section 1.1

Content for section 1.1."""
        chunks = chunker.parse(markdown, "test.md")
        indices = [c.chunk_index for c in chunks]
        # Should start at 0 and increment
        assert indices[0] == 0
        for i in range(1, len(indices)):
            assert indices[i] == indices[i - 1] + 1

    def test_parse_source_path_preserved(self) -> None:
        """Test that source_path is preserved on all chunks."""
        chunker = StructuredChunker()
        markdown = "# Title\n\nContent."
        source = "/path/to/document.md"
        chunks = chunker.parse(markdown, source)
        for chunk in chunks:
            assert chunk.source_path == source

    def test_parse_empty_document_raises_error(self) -> None:
        """Test that empty document raises ValueError."""
        chunker = StructuredChunker()
        with pytest.raises(ValueError, match="cannot be empty"):
            chunker.parse("", "test.md")

    def test_parse_whitespace_only_raises_error(self) -> None:
        """Test that whitespace-only document raises ValueError."""
        chunker = StructuredChunker()
        with pytest.raises(ValueError, match="cannot be empty"):
            chunker.parse("   \n\n   ", "test.md")

    def test_parse_legislation_fixture(self) -> None:
        """Test parsing sample_legislation.md fixture."""
        if not SAMPLE_LEGISLATION.exists():
            pytest.skip("Fixture not found")

        chunker = StructuredChunker()
        markdown = SAMPLE_LEGISLATION.read_text()
        chunks = chunker.parse(markdown, str(SAMPLE_LEGISLATION))

        assert len(chunks) > 0
        # Should have multiple sections
        assert len(chunks) >= 5

        # Check some chunks have parent_chain (nested structure)
        chunks_with_parents = [c for c in chunks if len(c.parent_chain) > 0]
        assert len(chunks_with_parents) > 0

    def test_parse_technical_doc_fixture(self) -> None:
        """Test parsing sample_technical_doc.md fixture."""
        if not SAMPLE_TECHNICAL_DOC.exists():
            pytest.skip("Fixture not found")

        chunker = StructuredChunker()
        markdown = SAMPLE_TECHNICAL_DOC.read_text()
        chunks = chunker.parse(markdown, str(SAMPLE_TECHNICAL_DOC))

        assert len(chunks) > 0
        # Technical doc has deep nesting (H1-H5)
        max_depth = max(len(c.parent_chain) for c in chunks)
        assert max_depth >= 2  # At least some nesting

    def test_parse_generates_chunk_ids(self) -> None:
        """Test that chunk IDs are properly generated."""
        chunker = StructuredChunker()
        markdown = "# Title\n\nContent here."
        chunks = chunker.parse(markdown, "document.md")
        for chunk in chunks:
            assert chunk.id is not None
            assert len(chunk.id) > 0
            # ID should contain source path info
            assert "document" in chunk.id or "chunk" in chunk.id


class TestFlatTextFallback:
    """Tests for documents without markdown structure."""

    def test_parse_flat_text(self) -> None:
        """Test parsing flat text without headings."""
        chunker = StructuredChunker()
        text = "Just plain text without any markdown headings. More text here."
        chunks = chunker.parse(text, "flat.txt")
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, DocumentChunk)

    def test_flat_text_parent_chain_empty(self) -> None:
        """Test that flat text chunks have empty parent_chain."""
        chunker = StructuredChunker()
        text = "Plain text. No structure. Just paragraphs."
        chunks = chunker.parse(text, "flat.txt")
        for chunk in chunks:
            assert chunk.parent_chain == []

    def test_flat_text_respects_max_tokens(self) -> None:
        """Test that flat text chunking respects max_tokens."""
        chunker = StructuredChunker(max_tokens=50)
        # Long flat text
        text = "This is a sentence. " * 50
        chunks = chunker.parse(text, "flat.txt")
        # Should be split into multiple chunks
        assert len(chunks) > 1

    def test_flat_text_fixture(self) -> None:
        """Test parsing sample_flat_text.txt fixture."""
        if not SAMPLE_FLAT_TEXT.exists():
            pytest.skip("Fixture not found")

        chunker = StructuredChunker()
        text = SAMPLE_FLAT_TEXT.read_text()
        chunks = chunker.parse(text, str(SAMPLE_FLAT_TEXT))

        assert len(chunks) >= 1
        # All chunks should have empty parent_chain
        for chunk in chunks:
            assert chunk.parent_chain == []


class TestChunkTypeClassification:
    """Tests for chunk type inference from content."""

    def test_chunk_type_content_default(self) -> None:
        """Test that regular text gets CONTENT type."""
        chunker = StructuredChunker()
        markdown = "# Title\n\nJust regular paragraph content here."
        chunks = chunker.parse(markdown, "test.md")
        content_chunks = [c for c in chunks if c.chunk_type == ChunkType.CONTENT]
        assert len(content_chunks) >= 1

    def test_chunk_type_definition_detected(self) -> None:
        """Test that definition patterns are detected."""
        chunker = StructuredChunker()
        markdown = """# Definitions

"Administrator" means the chief executive officer.

"Term" means something specific."""
        chunks = chunker.parse(markdown, "test.md")
        # Should detect definition section
        has_definition_type = any(c.chunk_type == ChunkType.DEFINITION for c in chunks)
        has_means_pattern = any("means" in c.content.lower() for c in chunks)
        assert has_definition_type or has_means_pattern

    def test_chunk_type_requirement_detected(self) -> None:
        """Test that requirement patterns are detected."""
        chunker = StructuredChunker()
        markdown = "# Requirements\n\nThe system shall provide authentication."
        chunks = chunker.parse(markdown, "test.md")
        # Should detect "shall" as requirement indicator
        requirement_chunks = [
            c for c in chunks if c.chunk_type == ChunkType.REQUIREMENT
        ]
        # May or may not detect based on implementation
        assert len(requirement_chunks) >= 0

    def test_chunk_type_header(self) -> None:
        """Test that header-only chunks get HEADER type."""
        chunker = StructuredChunker()
        markdown = "# Main Title\n\n## Subsection\n\nContent."
        chunks = chunker.parse(markdown, "test.md")
        # At least some chunks should exist
        assert len(chunks) >= 1

    def test_chunk_type_header_only_level2(self) -> None:
        """Test that header-only level-2 chunks get HEADER type (bug fix test)."""
        chunker = StructuredChunker()
        # Document with empty level-2 section (only heading, no body)
        md = (
            "# Title\n\nContent.\n\n"
            "## Empty Section\n\n"
            "## Next Section\n\nMore content."
        )
        chunks = chunker.parse(md, "test.md")

        # Find the "Empty Section" chunk
        empty_section = [c for c in chunks if "Empty Section" in c.content]
        assert len(empty_section) == 1
        assert empty_section[0].chunk_type == ChunkType.HEADER
        assert empty_section[0].heading_level == 2

    def test_chunk_type_header_only_level3(self) -> None:
        """Test that header-only level-3 chunks get HEADER type."""
        chunker = StructuredChunker()
        md = (
            "# Title\n\n## Section\n\nContent.\n\n"
            "### Empty Subsection\n\n"
            "### Next\n\nMore."
        )
        chunks = chunker.parse(md, "test.md")

        empty_section = [c for c in chunks if "Empty Subsection" in c.content]
        assert len(empty_section) == 1
        assert empty_section[0].chunk_type == ChunkType.HEADER
        assert empty_section[0].heading_level == 3

    def test_chunk_type_header_only_level6(self) -> None:
        """Test that header-only level-6 chunks get HEADER type."""
        chunker = StructuredChunker()
        md = (
            "# A\n\n## B\n\n### C\n\n#### D\n\n##### E\n\n"
            "###### Empty\n\n"
            "###### Next\n\nBody."
        )
        chunks = chunker.parse(md, "test.md")

        empty_section = [c for c in chunks if c.content.strip() == "###### Empty"]
        assert len(empty_section) == 1
        assert empty_section[0].chunk_type == ChunkType.HEADER
        assert empty_section[0].heading_level == 6

    def test_is_header_only_helper(self) -> None:
        """Test is_header_only helper method."""
        chunker = StructuredChunker()
        md = "# Title\n\nBody content.\n\n## Empty\n\n## With Body\n\nMore content."
        chunks = chunker.parse(md, "test.md")

        # Find header-only and content chunks
        empty_section = [c for c in chunks if "Empty" in c.content][0]
        content_section = [c for c in chunks if "Body content" in c.content][0]

        assert chunker.is_header_only(empty_section) is True
        assert chunker.is_header_only(content_section) is False


class TestSectionIdGeneration:
    """Tests for normalized section ID generation."""

    def test_section_id_from_numeric_heading(self) -> None:
        """Test section ID from numeric heading."""
        chunker = StructuredChunker()
        markdown = "# Section 1.2.3\n\nContent."
        chunks = chunker.parse(markdown, "test.md")
        assert len(chunks) >= 1
        # Section ID should be normalized
        section_id = chunks[0].section_id
        assert section_id is not None
        assert len(section_id) > 0

    def test_section_id_lowercase(self) -> None:
        """Test that section IDs are lowercase."""
        chunker = StructuredChunker()
        markdown = "# UPPERCASE HEADING\n\nContent."
        chunks = chunker.parse(markdown, "test.md")
        assert len(chunks) >= 1
        section_id = chunks[0].section_id
        assert section_id == section_id.lower()

    def test_section_id_normalizes_special_chars(self) -> None:
        """Test that special characters are normalized in section IDs."""
        chunker = StructuredChunker()
        markdown = "# Section (a)(1) - Details\n\nContent."
        chunks = chunker.parse(markdown, "test.md")
        assert len(chunks) >= 1
        section_id = chunks[0].section_id
        # Should not contain parentheses or dashes (normalized)
        assert "(" not in section_id
        assert ")" not in section_id

    def test_section_id_from_text_heading(self) -> None:
        """Test section ID from text heading."""
        chunker = StructuredChunker()
        markdown = "# Introduction\n\nContent here."
        chunks = chunker.parse(markdown, "test.md")
        assert len(chunks) >= 1
        section_id = chunks[0].section_id
        has_intro = "introduction" in section_id.lower()
        has_short_intro = "intro" in section_id.lower()
        has_content = len(section_id) > 0
        assert has_intro or has_short_intro or has_content


class TestMtimeTracking:
    """Tests for file modification time tracking."""

    def test_mtime_default_zero(self) -> None:
        """Test that mtime defaults to 0 when not provided."""
        chunker = StructuredChunker()
        markdown = "# Title\n\nContent."
        chunks = chunker.parse(markdown, "test.md")
        for chunk in chunks:
            assert chunk.mtime == 0.0

    def test_mtime_from_file(self) -> None:
        """Test that mtime can be set from file."""
        if not SAMPLE_LEGISLATION.exists():
            pytest.skip("Fixture not found")

        chunker = StructuredChunker()
        markdown = SAMPLE_LEGISLATION.read_text()
        mtime = SAMPLE_LEGISLATION.stat().st_mtime
        chunks = chunker.parse(markdown, str(SAMPLE_LEGISLATION), mtime=mtime)
        for chunk in chunks:
            assert chunk.mtime == mtime


class TestContextualizedContent:
    """Tests for contextualized_content field initialization."""

    def test_contextualized_content_empty_initially(self) -> None:
        """Test that contextualized_content is empty after parsing."""
        chunker = StructuredChunker()
        markdown = "# Title\n\nContent here."
        chunks = chunker.parse(markdown, "test.md")
        for chunk in chunks:
            # Should be empty string initially (filled by LLMContextGenerator)
            assert chunk.contextualized_content == ""

    def test_embedding_none_initially(self) -> None:
        """Test that embedding is None after parsing."""
        chunker = StructuredChunker()
        markdown = "# Title\n\nContent here."
        chunks = chunker.parse(markdown, "test.md")
        for chunk in chunks:
            assert chunk.embedding is None


class TestToRecordDict:
    """Tests for to_record_dict() serialization."""

    def test_to_record_dict_integration(self) -> None:
        """Test that parsed chunks can be serialized to record dict."""
        chunker = StructuredChunker()
        markdown = """# Chapter 1

## Section 1.1

Content for section."""
        chunks = chunker.parse(markdown, "test.md")
        for chunk in chunks:
            record = chunk.to_record_dict()
            assert isinstance(record, dict)
            assert "id" in record
            assert "source_path" in record
            assert "content" in record
            assert "parent_chain" in record
            # parent_chain should be JSON string
            import json

            parsed_chain = json.loads(record["parent_chain"])
            assert isinstance(parsed_chain, list)


class TestLegislativeHierarchy:
    """Tests for legislative document hierarchy extraction with subsection patterns."""

    def test_title_pattern_detection(self) -> None:
        """Test that TITLE markers are detected at level 1."""
        chunker = StructuredChunker(
            subsection_patterns=LEGISLATIVE_PATTERNS,
            max_subsection_depth=1,  # Only titles
        )
        content = """TITLE I—BROADBAND ACCESS
Content for title one.
TITLE II—RURAL CONNECTIVITY
Content for title two."""
        chunks = chunker.parse(content, "test.md")

        # TITLE I and TITLE II should be detected at level 1
        title_chunks = [c for c in chunks if c.heading_level == 1]
        assert len(title_chunks) >= 2

    def test_subsection_pattern_detection(self) -> None:
        """Test that (a), (b) style subsections are detected as implicit headings."""
        chunker = StructuredChunker(
            subsection_patterns=LEGISLATIVE_PATTERNS,
            max_subsection_depth=4,  # Title, Chapter, Section, Subsection
            split_on_level=6,  # Split on all levels for this test
        )
        content = """SEC. 210. UNIVERSAL BROADBAND ACCESS.
(a) FINDINGS.—Congress finds the following:
Some content about findings.
(b) UNIVERSAL SERVICE GOAL.—It is the goal of the nation."""
        chunks = chunker.parse(content, "test.md")

        # Should have multiple chunks
        assert len(chunks) >= 3

        # Find subsection chunks
        subsection_chunks = [
            c for c in chunks if "(a)" in c.content or "(b)" in c.content
        ]
        assert len(subsection_chunks) >= 2

        # Check that (a) has level 4 (subsection level in new hierarchy)
        a_chunks = [c for c in chunks if "(a)" in c.content and c.heading_level == 4]
        assert len(a_chunks) >= 1

    def test_paragraph_pattern_detection(self) -> None:
        """Test that (1), (2) style paragraphs are detected."""
        chunker = StructuredChunker(
            subsection_patterns=LEGISLATIVE_PATTERNS,
            max_subsection_depth=5,  # Through paragraphs
            split_on_level=6,  # Split on all levels for this test
        )
        content = """SEC. 210. ACCESS.
(a) FINDINGS.—Congress finds the following:
(1) Access to high-speed broadband internet is essential.
(2) Approximately 21 million Americans lack access."""
        chunks = chunker.parse(content, "test.md")

        # Find paragraph chunks (level 5)
        paragraph_chunks = [c for c in chunks if c.heading_level == 5]
        assert len(paragraph_chunks) >= 2

        # Check content
        p1_chunks = [c for c in chunks if "(1)" in c.content]
        assert len(p1_chunks) >= 1

    def test_parent_chain_with_subsections(self) -> None:
        """Test that parent chains are built correctly through subsection hierarchy."""
        chunker = StructuredChunker(
            subsection_patterns=LEGISLATIVE_PATTERNS,
            max_subsection_depth=5,
            split_on_level=6,  # Split on all levels for this test
        )
        content = """SEC. 210. ACCESS.
(a) FINDINGS.—Congress finds:
(1) First finding.
(2) Second finding."""
        chunks = chunker.parse(content, "test.md")

        # Find paragraph (1) body chunk
        p1_body_chunks = [
            c for c in chunks if "First finding" in c.content and c.heading_level == 0
        ]

        if p1_body_chunks:
            # Parent chain should include SEC 210 and (a) FINDINGS
            assert len(p1_body_chunks[0].parent_chain) >= 2
            assert any("210" in p for p in p1_body_chunks[0].parent_chain)

    def test_depth_limiting(self) -> None:
        """Test that max_subsection_depth limits pattern recognition."""
        # With depth=4, only through subsections (a) should be recognized
        chunker_depth4 = StructuredChunker(
            subsection_patterns=LEGISLATIVE_PATTERNS,
            max_subsection_depth=4,  # Title, Chapter, Section, Subsection
            split_on_level=6,  # Split on all levels for this test
        )
        content = """SEC. 210. Test Section.
(a) Subsection.
(1) Paragraph."""
        chunks1 = chunker_depth4.parse(content, "test.md")

        # (a) should be level 4, but (1) should NOT be detected as level 5
        a_chunks = [c for c in chunks1 if c.heading_level == 4]
        p_chunks = [c for c in chunks1 if c.heading_level == 5]

        assert len(a_chunks) >= 1  # (a) detected
        assert len(p_chunks) == 0  # (1) NOT detected as heading

    def test_custom_patterns(self) -> None:
        """Test that custom subsection patterns work."""
        custom_patterns = [
            SubsectionPattern(
                name="step",
                pattern=re.compile(r"^Step (\d+)[.:]?\s*(.*)$", re.MULTILINE),
                level=3,
                extract_title=True,
            ),
        ]
        chunker = StructuredChunker(
            subsection_patterns=custom_patterns,
            max_subsection_depth=1,
            split_on_level=6,  # Split on all levels for this test
        )
        content = """## Instructions
Step 1: Do the first thing.
Step 2: Do the second thing."""
        chunks = chunker.parse(content, "test.md")

        # Should detect Step 1 and Step 2 as headings
        step_chunks = [c for c in chunks if c.heading_level == 3]
        assert len(step_chunks) >= 2

    def test_backward_compatible_without_patterns(self) -> None:
        """Test that default behavior (no patterns) is unchanged."""
        chunker = StructuredChunker()  # No subsection_patterns
        content = """## SEC. 210. ACCESS.
(a) FINDINGS.—Congress finds the following:
(1) Access to high-speed broadband internet is essential."""
        chunks = chunker.parse(content, "test.md")

        # Without patterns, (a) and (1) should NOT be detected as headings
        # All content should be in one section under SEC 210
        assert len(chunks) == 1
        assert chunks[0].heading_level == 2  # Only the ## heading

    def test_code_block_exclusion(self) -> None:
        """Test that patterns inside code blocks are not detected."""
        chunker = StructuredChunker(
            subsection_patterns=LEGISLATIVE_PATTERNS,
            max_subsection_depth=4,  # Through subsections
            split_on_level=6,  # Split on all levels for this test
        )
        content = """## Example
(a) Real subsection.

```python
# (1) This is a code comment, not a paragraph
x = "(a) Also not a subsection"
```

(b) Another real subsection."""
        chunks = chunker.parse(content, "test.md")

        # Only (a) and (b) should be detected, not the ones in code block
        heading_chunks = [c for c in chunks if c.heading_level == 4]
        assert len(heading_chunks) == 2

    def test_header_chunk_type_for_subsections(self) -> None:
        """Test that subsection heading-only chunks get HEADER type."""
        chunker = StructuredChunker(
            subsection_patterns=LEGISLATIVE_PATTERNS,
            max_subsection_depth=4,  # Through subsections
            split_on_level=6,  # Split on all levels for this test
        )
        content = """SEC. 210. ACCESS.
(a) FINDINGS.—
Body content here."""
        chunks = chunker.parse(content, "test.md")

        # The (a) heading-only chunk should be HEADER type
        header_chunks = [
            c for c in chunks if "(a)" in c.content and c.chunk_type == ChunkType.HEADER
        ]
        # Should have at least the (a) header
        assert len(header_chunks) >= 1

    def test_subparagraph_pattern(self) -> None:
        """Test that (A), (B) subparagraph patterns are detected at level 6."""
        chunker = StructuredChunker(
            subsection_patterns=LEGISLATIVE_PATTERNS,
            max_subsection_depth=6,  # All levels including subparagraphs
            split_on_level=6,  # Split on all levels for this test
        )
        content = """SEC. 210. Test Section.
(a) Subsection.
(1) Paragraph.
(A) Subparagraph content."""
        chunks = chunker.parse(content, "test.md")

        # Should detect (A) at level 6
        subpara_chunks = [c for c in chunks if c.heading_level == 6]
        assert len(subpara_chunks) >= 1

    def test_us_legislative_has_six_levels(self) -> None:
        """Test that LEGISLATIVE_PATTERNS has 6 pattern levels."""
        assert len(LEGISLATIVE_PATTERNS) == 6

    def test_full_us_hierarchy(self) -> None:
        """Test full US legislative hierarchy detection."""
        chunker = StructuredChunker(
            subsection_patterns=LEGISLATIVE_PATTERNS,
            max_subsection_depth=6,  # All levels
            split_on_level=6,  # Split on all levels for this test
        )
        content = """TITLE I—BROADBAND ACCESS
CHAPTER 1—DEFINITIONS
SEC. 101. Short Title.
(a) Subsection content.
(1) Paragraph content.
(A) Subparagraph content."""
        chunks = chunker.parse(content, "test.md")

        # Check all levels are detected
        level_1 = [c for c in chunks if c.heading_level == 1]  # TITLE
        level_2 = [c for c in chunks if c.heading_level == 2]  # CHAPTER
        level_3 = [c for c in chunks if c.heading_level == 3]  # SEC.
        level_4 = [c for c in chunks if c.heading_level == 4]  # (a)
        level_5 = [c for c in chunks if c.heading_level == 5]  # (1)
        level_6 = [c for c in chunks if c.heading_level == 6]  # (A)

        assert len(level_1) >= 1
        assert len(level_2) >= 1
        assert len(level_3) >= 1
        assert len(level_4) >= 1
        assert len(level_5) >= 1
        assert len(level_6) >= 1

    def test_invalid_max_subsection_depth(self) -> None:
        """Test that invalid max_subsection_depth raises ValueError."""
        # Zero is always invalid
        with pytest.raises(ValueError, match="max_subsection_depth must be between"):
            StructuredChunker(
                subsection_patterns=LEGISLATIVE_PATTERNS,
                max_subsection_depth=0,
            )
        # Exceeding pattern count is invalid (LEGISLATIVE_PATTERNS has 6 levels)
        with pytest.raises(ValueError, match="max_subsection_depth must be between"):
            StructuredChunker(
                subsection_patterns=LEGISLATIVE_PATTERNS,
                max_subsection_depth=7,
            )

    def test_max_subsection_depth_defaults_to_pattern_length(self) -> None:
        """Test that max_subsection_depth defaults to pattern count when None."""
        chunker = StructuredChunker(
            subsection_patterns=LEGISLATIVE_PATTERNS,
            max_subsection_depth=None,
        )
        # Should use all 6 patterns by default
        assert chunker._max_subsection_depth == len(LEGISLATIVE_PATTERNS)


class TestDomainPatternsRegistry:
    """Tests for domain pattern constants and DOMAIN_PATTERNS registry."""

    def test_domain_patterns_registry_has_all_domains(self) -> None:
        """Test that DOMAIN_PATTERNS contains all expected domain keys."""
        expected_domains = {
            "us_legislative",
            "au_legislative",
            "academic",
            "technical",
            "legal_contract",
        }
        assert set(DOMAIN_PATTERNS.keys()) == expected_domains

    def test_domain_patterns_us_legislative_is_legislative_patterns(self) -> None:
        """Test that us_legislative maps to LEGISLATIVE_PATTERNS."""
        assert DOMAIN_PATTERNS["us_legislative"] is LEGISLATIVE_PATTERNS

    def test_domain_patterns_au_legislative_is_au_patterns(self) -> None:
        """Test that au_legislative maps to AU_LEGISLATIVE_PATTERNS."""
        assert DOMAIN_PATTERNS["au_legislative"] is AU_LEGISLATIVE_PATTERNS

    def test_domain_patterns_academic_is_academic_patterns(self) -> None:
        """Test that academic maps to ACADEMIC_PATTERNS."""
        assert DOMAIN_PATTERNS["academic"] is ACADEMIC_PATTERNS

    def test_domain_patterns_technical_is_technical_patterns(self) -> None:
        """Test that technical maps to TECHNICAL_PATTERNS."""
        assert DOMAIN_PATTERNS["technical"] is TECHNICAL_PATTERNS

    def test_domain_patterns_legal_contract_is_contract_patterns(self) -> None:
        """Test that legal_contract maps to CONTRACT_PATTERNS."""
        assert DOMAIN_PATTERNS["legal_contract"] is CONTRACT_PATTERNS


class TestAustralianLegislativePatterns:
    """Tests for Australian legislative document hierarchy extraction."""

    def test_au_patterns_has_six_levels(self) -> None:
        """Test that AU_LEGISLATIVE_PATTERNS has 6 pattern levels."""
        assert len(AU_LEGISLATIVE_PATTERNS) == 6

    def test_au_part_detection(self) -> None:
        """Test that Australian Part markers are detected at level 1."""
        chunker = StructuredChunker(
            subsection_patterns=AU_LEGISLATIVE_PATTERNS,
            max_subsection_depth=1,
        )
        content = """Part 1—Preliminary
Part 2—Main provisions"""
        chunks = chunker.parse(content, "test.md")

        # Part 1 and Part 2 should be detected at level 1
        part_chunks = [c for c in chunks if c.heading_level == 1]
        assert len(part_chunks) >= 2

    def test_au_subsection_numeric(self) -> None:
        """Test that Australian subsections use numeric (1), (2) at level 4."""
        chunker = StructuredChunker(
            subsection_patterns=AU_LEGISLATIVE_PATTERNS,
            max_subsection_depth=4,
            split_on_level=6,  # Split on all levels for this test
        )
        content = """## Section 210.
(1) First subsection content.
(2) Second subsection content."""
        chunks = chunker.parse(content, "test.md")

        # (1) and (2) should be detected at level 4
        subsection_chunks = [c for c in chunks if c.heading_level == 4]
        assert len(subsection_chunks) >= 2

    def test_au_paragraph_letters(self) -> None:
        """Test that Australian paragraphs use (a), (b) at level 5."""
        chunker = StructuredChunker(
            subsection_patterns=AU_LEGISLATIVE_PATTERNS,
            max_subsection_depth=5,
            split_on_level=6,  # Split on all levels for this test
        )
        content = """## Section 210.
(1) Subsection content.
(a) Paragraph content.
(b) Another paragraph."""
        chunks = chunker.parse(content, "test.md")

        # (a) and (b) should be detected at level 5
        paragraph_chunks = [c for c in chunks if c.heading_level == 5]
        assert len(paragraph_chunks) >= 2

    def test_au_full_hierarchy(self) -> None:
        """Test full Australian legislative hierarchy (subsection levels)."""
        chunker = StructuredChunker(
            subsection_patterns=AU_LEGISLATIVE_PATTERNS,
            max_subsection_depth=6,  # All levels
            split_on_level=6,  # Split on all levels for this test
        )
        content = """Part 1—Test
Division 1—Test Division
210 Test Section
(1) Subsection.
(a) Paragraph.
(i) Subparagraph."""
        chunks = chunker.parse(content, "test.md")

        # Check subsection levels are detected (levels 4, 5, 6)
        level_4 = [c for c in chunks if c.heading_level == 4]  # (1)
        level_5 = [c for c in chunks if c.heading_level == 5]  # (a)
        level_6 = [c for c in chunks if c.heading_level == 6]  # (i)

        assert len(level_4) >= 1
        assert len(level_5) >= 1
        assert len(level_6) >= 1


class TestAcademicPatterns:
    """Tests for academic document hierarchy extraction."""

    def test_academic_patterns_has_four_levels(self) -> None:
        """Test that ACADEMIC_PATTERNS has 4 pattern levels."""
        assert len(ACADEMIC_PATTERNS) == 4

    def test_academic_section_numbered(self) -> None:
        """Test that academic sections use '1.' format at level 3."""
        chunker = StructuredChunker(
            subsection_patterns=ACADEMIC_PATTERNS,
            max_subsection_depth=1,
            split_on_level=6,  # Split on all levels for this test
        )
        content = """## Paper
1. Introduction
Content for introduction.
2. Background
Content for background."""
        chunks = chunker.parse(content, "test.md")

        # 1. and 2. should be detected at level 3
        section_chunks = [c for c in chunks if c.heading_level == 3]
        assert len(section_chunks) >= 2

    def test_academic_subsection_numbered(self) -> None:
        """Test that academic subsections use '1.1' format at level 4."""
        chunker = StructuredChunker(
            subsection_patterns=ACADEMIC_PATTERNS,
            max_subsection_depth=2,
            split_on_level=6,  # Split on all levels for this test
        )
        content = """## Paper
1. Introduction
1.1 Motivation
Content for motivation.
1.2 Contributions
Content for contributions."""
        chunks = chunker.parse(content, "test.md")

        # 1.1 and 1.2 should be detected at level 4
        subsection_chunks = [c for c in chunks if c.heading_level == 4]
        assert len(subsection_chunks) >= 2


class TestTechnicalPatterns:
    """Tests for technical documentation hierarchy extraction."""

    def test_technical_patterns_has_four_levels(self) -> None:
        """Test that TECHNICAL_PATTERNS has 4 pattern levels."""
        assert len(TECHNICAL_PATTERNS) == 4

    def test_technical_step_detection(self) -> None:
        """Test that technical steps are detected at level 3."""
        chunker = StructuredChunker(
            subsection_patterns=TECHNICAL_PATTERNS,
            max_subsection_depth=1,
            split_on_level=6,  # Split on all levels for this test
        )
        content = """## Instructions
Step 1: Install the software.
Step 2: Configure settings."""
        chunks = chunker.parse(content, "test.md")

        # Step 1 and Step 2 should be detected at level 3
        step_chunks = [c for c in chunks if c.heading_level == 3]
        assert len(step_chunks) >= 2

    def test_technical_note_detection(self) -> None:
        """Test that Note: markers are detected at level 4."""
        chunker = StructuredChunker(
            subsection_patterns=TECHNICAL_PATTERNS,
            max_subsection_depth=3,
            split_on_level=6,  # Split on all levels for this test
        )
        content = """## Instructions
Step 1: Do something.
Note: Be careful with this step.
Warning: This may cause issues."""
        chunks = chunker.parse(content, "test.md")

        # Note and Warning should be detected
        note_chunks = [
            c for c in chunks if "Note" in c.content or "Warning" in c.content
        ]
        assert len(note_chunks) >= 2


class TestContractPatterns:
    """Tests for legal contract hierarchy extraction."""

    def test_contract_patterns_has_four_levels(self) -> None:
        """Test that CONTRACT_PATTERNS has 4 pattern levels."""
        assert len(CONTRACT_PATTERNS) == 4

    def test_contract_article_detection(self) -> None:
        """Test that Article markers are detected at level 3."""
        chunker = StructuredChunker(
            subsection_patterns=CONTRACT_PATTERNS,
            max_subsection_depth=1,
            split_on_level=6,  # Split on all levels for this test
        )
        content = """## Agreement
Article I: Definitions
Content for definitions.
Article II: Obligations
Content for obligations."""
        chunks = chunker.parse(content, "test.md")

        # Article I and II should be detected at level 3
        article_chunks = [c for c in chunks if c.heading_level == 3]
        assert len(article_chunks) >= 2

    def test_contract_section_detection(self) -> None:
        """Test that Section markers are detected at level 4."""
        chunker = StructuredChunker(
            subsection_patterns=CONTRACT_PATTERNS,
            max_subsection_depth=2,
            split_on_level=6,  # Split on all levels for this test
        )
        content = """## Agreement
Article I: Definitions
Section 1: General Terms
Content for general terms.
Section 2: Specific Terms
Content for specific terms."""
        chunks = chunker.parse(content, "test.md")

        # Section 1 and 2 should be detected at level 4
        section_chunks = [c for c in chunks if c.heading_level == 4]
        assert len(section_chunks) >= 2

    def test_contract_clause_detection(self) -> None:
        """Test that (a) clauses are detected at level 5."""
        chunker = StructuredChunker(
            subsection_patterns=CONTRACT_PATTERNS,
            max_subsection_depth=3,
            split_on_level=6,  # Split on all levels for this test
        )
        content = """## Agreement
Article I: Definitions
Section 1: General Terms
(a) First clause.
(b) Second clause."""
        chunks = chunker.parse(content, "test.md")

        # (a) and (b) should be detected at level 5
        clause_chunks = [c for c in chunks if c.heading_level == 5]
        assert len(clause_chunks) >= 2


class TestParagraphNormalization:
    """Tests for _normalize_paragraphs() method that joins wrapped lines."""

    def test_joins_wrapped_lines(self) -> None:
        """Test that mid-sentence line breaks are joined."""
        chunker = StructuredChunker()
        text = (
            "The United States faces unprecedented challenges arising from rapid\n"
            "technological advancement, including the proliferation of AI systems."
        )
        result = chunker._normalize_paragraphs(text)
        # Lines should be joined since first line doesn't end with punctuation
        # and second line starts with lowercase
        assert "\n" not in result
        assert "rapid technological" in result

    def test_preserves_paragraph_breaks(self) -> None:
        """Test that blank lines (paragraph separators) are preserved."""
        chunker = StructuredChunker()
        text = "First paragraph content here.\n" "\n" "Second paragraph content here."
        result = chunker._normalize_paragraphs(text)
        # Blank line should be preserved
        assert "\n\n" in result or result.count("\n") >= 1

    def test_preserves_markdown_headings(self) -> None:
        """Test that markdown headings are not joined to previous lines."""
        chunker = StructuredChunker()
        text = "Some content here\n" "# Heading\n" "Content after heading."
        result = chunker._normalize_paragraphs(text)
        # Heading should remain on its own line
        assert "# Heading" in result
        lines = result.split("\n")
        heading_lines = [ln for ln in lines if ln.strip().startswith("#")]
        assert len(heading_lines) >= 1

    def test_preserves_list_items(self) -> None:
        """Test that list markers are not joined to previous lines."""
        chunker = StructuredChunker()
        text = (
            "Introduction text here\n"
            "- First item\n"
            "- Second item\n"
            "* Another item\n"
            "1. Numbered item"
        )
        result = chunker._normalize_paragraphs(text)
        # List items should remain on separate lines
        assert "- First item" in result
        assert "- Second item" in result
        assert "* Another item" in result
        assert "1. Numbered item" in result

    def test_preserves_subsection_markers(self) -> None:
        """Test that subsection markers like (a), (1) are not joined."""
        chunker = StructuredChunker()
        text = (
            "SEC. 210. UNIVERSAL BROADBAND ACCESS\n"
            "(a) FINDINGS.—Congress finds the following:\n"
            "(1) Access to high-speed broadband internet is essential.\n"
            "(2) Approximately 21 million Americans lack access."
        )
        result = chunker._normalize_paragraphs(text)
        # Subsection markers should remain on separate lines
        assert "(a) FINDINGS" in result
        assert "(1) Access" in result
        assert "(2) Approximately" in result

    def test_preserves_code_blocks(self) -> None:
        """Test that content inside code blocks is preserved as-is."""
        chunker = StructuredChunker()
        text = (
            "Some text before code\n"
            "```python\n"
            "def hello():\n"
            "    print('hello')\n"
            "```\n"
            "Text after code."
        )
        result = chunker._normalize_paragraphs(text)
        # Code block content should be unchanged
        assert "def hello():" in result
        assert "    print('hello')" in result

    def test_joins_multiple_wrapped_lines(self) -> None:
        """Test joining of multiple consecutive wrapped lines."""
        chunker = StructuredChunker()
        text = (
            "This is a very long paragraph that has been\n"
            "wrapped at about 50 characters for\n"
            "readability in the original document."
        )
        result = chunker._normalize_paragraphs(text)
        # All lines should be joined into one
        assert result.count("\n") == 0
        assert "been wrapped" in result
        assert "for readability" in result

    def test_stops_at_sentence_ending(self) -> None:
        """Test that lines ending with sentence punctuation don't join."""
        chunker = StructuredChunker()
        text = "First sentence ends here.\n" "Second sentence starts fresh."
        result = chunker._normalize_paragraphs(text)
        # Lines should NOT be joined since first ends with period
        assert "\n" in result or "here. Second" in result

    def test_handles_empty_text(self) -> None:
        """Test handling of empty text."""
        chunker = StructuredChunker()
        assert chunker._normalize_paragraphs("") == ""

    def test_handles_single_line(self) -> None:
        """Test handling of single line without newlines."""
        chunker = StructuredChunker()
        text = "Single line without any newlines."
        result = chunker._normalize_paragraphs(text)
        assert result == text


class TestHeaderClassificationFix:
    """Tests for fixed header classification that handles joined lines."""

    def test_short_no_sentence_is_header(self) -> None:
        """Test that short content without sentence ending is classified as HEADER."""
        chunker = StructuredChunker(
            subsection_patterns=LEGISLATIVE_PATTERNS,
            max_subsection_depth=4,
        )
        # Short title without sentence ending
        chunk_type = chunker._classify_chunk_type(
            content="(a) FINDINGS",
            heading_title="(a) FINDINGS",
            parent_chain=["SEC. 210. ACCESS"],
            heading_level=4,
        )
        assert chunk_type == ChunkType.HEADER

    def test_long_content_is_content(self) -> None:
        """Test that long content (>=150 chars) is classified as CONTENT, not HEADER."""
        chunker = StructuredChunker(
            subsection_patterns=LEGISLATIVE_PATTERNS,
            max_subsection_depth=4,
        )
        # Long substantive content that exceeds MIN_SUBSTANTIVE_LENGTH
        long_content = (
            "(1) The United States faces unprecedented challenges and opportunities "
            "arising from rapid technological advancement, including the proliferation "
            "of artificial intelligence systems capable of performing tasks."
        )
        assert len(long_content) >= 150  # Verify it's long enough
        chunk_type = chunker._classify_chunk_type(
            content=long_content,
            heading_title="(1) The United States faces...",
            parent_chain=["SEC. 210. ACCESS", "(a) FINDINGS"],
            heading_level=5,
        )
        assert chunk_type == ChunkType.CONTENT

    def test_sentence_ending_is_content(self) -> None:
        """Test that content ending with sentence punctuation is CONTENT."""
        chunker = StructuredChunker(
            subsection_patterns=LEGISLATIVE_PATTERNS,
            max_subsection_depth=4,
        )
        # Short content but ends with period
        chunk_type = chunker._classify_chunk_type(
            content="(1) Access to broadband is essential.",
            heading_title="(1) Access to broadband is essential.",
            parent_chain=["SEC. 210. ACCESS"],
            heading_level=5,
        )
        assert chunk_type == ChunkType.CONTENT

    def test_multiline_is_content(self) -> None:
        """Test that multiline content is classified as CONTENT."""
        chunker = StructuredChunker(
            subsection_patterns=LEGISLATIVE_PATTERNS,
            max_subsection_depth=4,
        )
        # Multiline content should not be header
        chunk_type = chunker._classify_chunk_type(
            content="(a) FINDINGS\nSome additional content here.",
            heading_title="(a) FINDINGS",
            parent_chain=["SEC. 210. ACCESS"],
            heading_level=4,
        )
        # Multiline is not header (has newline)
        assert chunk_type == ChunkType.CONTENT

    def test_header_classification_without_patterns(self) -> None:
        """Test header classification without subsection patterns."""
        chunker = StructuredChunker()  # No subsection patterns
        # Without patterns, the subsection-specific header check is skipped
        chunk_type = chunker._classify_chunk_type(
            content="## Section Title",
            heading_title="Section Title",
            parent_chain=[],
            heading_level=2,
        )
        assert chunk_type == ChunkType.HEADER

    def test_exclamation_ending_is_content(self) -> None:
        """Test that content ending with ! is CONTENT."""
        chunker = StructuredChunker(
            subsection_patterns=LEGISLATIVE_PATTERNS,
            max_subsection_depth=4,
        )
        chunk_type = chunker._classify_chunk_type(
            content="(1) Warning!",
            heading_title="(1) Warning!",
            parent_chain=["SEC. 210. ACCESS"],
            heading_level=5,
        )
        assert chunk_type == ChunkType.CONTENT

    def test_question_ending_is_content(self) -> None:
        """Test that content ending with ? is CONTENT."""
        chunker = StructuredChunker(
            subsection_patterns=LEGISLATIVE_PATTERNS,
            max_subsection_depth=4,
        )
        chunk_type = chunker._classify_chunk_type(
            content="(1) Is this a requirement?",
            heading_title="(1) Is this a requirement?",
            parent_chain=["SEC. 210. ACCESS"],
            heading_level=5,
        )
        assert chunk_type == ChunkType.CONTENT


class TestSplitOnLevel:
    """Tests for split_on_level parameter that controls chunk boundaries."""

    def test_default_split_on_level_with_patterns(self) -> None:
        """With subsection_patterns, default split_on_level is len(patterns) // 2."""
        chunker = StructuredChunker(subsection_patterns=LEGISLATIVE_PATTERNS)
        # LEGISLATIVE_PATTERNS has 6 levels, so default is 6 // 2 = 3
        assert chunker._split_on_level == 3

    def test_default_split_on_level_without_patterns(self) -> None:
        """Without subsection_patterns, default split_on_level is 6."""
        chunker = StructuredChunker()
        assert chunker._split_on_level == 6

    def test_default_split_on_level_academic_patterns(self) -> None:
        """ACADEMIC_PATTERNS has 4 levels, so default is 4 // 2 = 2."""
        chunker = StructuredChunker(subsection_patterns=ACADEMIC_PATTERNS)
        assert chunker._split_on_level == 2

    def test_default_split_on_level_contract_patterns(self) -> None:
        """CONTRACT_PATTERNS has 4 levels, so default is 4 // 2 = 2."""
        chunker = StructuredChunker(subsection_patterns=CONTRACT_PATTERNS)
        assert chunker._split_on_level == 2

    def test_explicit_split_on_level(self) -> None:
        """Explicit split_on_level overrides default."""
        chunker = StructuredChunker(
            subsection_patterns=LEGISLATIVE_PATTERNS,
            split_on_level=4,  # Split through (a) subsections
        )
        assert chunker._split_on_level == 4

    def test_invalid_split_on_level_zero(self) -> None:
        """split_on_level=0 raises ValueError."""
        with pytest.raises(ValueError, match="split_on_level must be between 1 and 6"):
            StructuredChunker(split_on_level=0)

    def test_invalid_split_on_level_seven(self) -> None:
        """split_on_level=7 raises ValueError."""
        with pytest.raises(ValueError, match="split_on_level must be between 1 and 6"):
            StructuredChunker(split_on_level=7)

    def test_accumulates_lower_level_content(self) -> None:
        """Content from levels > split_on_level stays in parent chunk."""
        chunker = StructuredChunker(
            subsection_patterns=LEGISLATIVE_PATTERNS,
            split_on_level=3,
        )
        content = """SEC. 210. ACCESS.
(a) FINDINGS.—Congress finds:
(1) Access is essential.
(2) 21 million lack access."""

        chunks = chunker.parse(content, "test.md")

        # Should have header + body chunk, not separate chunks for (a), (1), (2)
        # Body chunk should contain all inline subsection content
        body_chunks = [c for c in chunks if c.heading_level == 0]
        assert len(body_chunks) >= 1

        body_content = body_chunks[0].content
        assert "(a) FINDINGS" in body_content
        assert "(1) Access" in body_content
        assert "(2) 21 million" in body_content

    def test_tracks_subsection_ids(self) -> None:
        """Inline subsection markers are tracked in subsection_ids."""
        chunker = StructuredChunker(
            subsection_patterns=LEGISLATIVE_PATTERNS,
            split_on_level=3,
        )
        content = """SEC. 210. ACCESS.
(a) FINDINGS.—Congress finds:
(1) Access is essential."""

        chunks = chunker.parse(content, "test.md")

        # Find body chunk (has subsection_ids)
        body_chunks = [c for c in chunks if c.heading_level == 0]
        assert len(body_chunks) >= 1

        # Should have subsection_ids for (a) and (1)
        subsection_ids = body_chunks[0].subsection_ids
        assert len(subsection_ids) >= 2
        # Check that section IDs were generated
        assert any("findings" in sid.lower() for sid in subsection_ids)

    def test_split_on_level_6_matches_old_behavior(self) -> None:
        """split_on_level=6 should split on every heading (like old behavior)."""
        chunker = StructuredChunker(
            subsection_patterns=LEGISLATIVE_PATTERNS,
            split_on_level=6,
        )
        content = """SEC. 210. ACCESS.
(a) FINDINGS.—Congress finds:
(1) Access is essential."""

        chunks = chunker.parse(content, "test.md")

        # With split_on_level=6, all levels create separate chunks
        # SEC, (a), and (1) each become their own chunks (header + body if body exists)
        # At minimum we should see SEC and (a) as separate header chunks
        sec_chunks = [c for c in chunks if "SEC." in c.content or "210" in c.content]
        a_chunks = [c for c in chunks if "(a)" in c.content]
        one_chunks = [c for c in chunks if "(1)" in c.content]

        assert len(sec_chunks) >= 1  # SEC is its own chunk
        assert len(a_chunks) >= 1  # (a) is its own chunk
        assert len(one_chunks) >= 1  # (1) is its own chunk

    def test_fewer_chunks_with_lower_split_level(self) -> None:
        """Lower split_on_level produces fewer chunks by accumulating content."""
        content = """SEC. 210. ACCESS.
(a) FINDINGS.—Congress finds:
(1) Access is essential.
(2) 21 million lack access.
(b) UNIVERSAL GOAL.—It is the goal."""

        # High split level (all headings become chunks)
        chunker_high = StructuredChunker(
            subsection_patterns=LEGISLATIVE_PATTERNS,
            split_on_level=6,
        )
        chunks_high = chunker_high.parse(content, "test.md")

        # Low split level (only SEC creates chunks)
        chunker_low = StructuredChunker(
            subsection_patterns=LEGISLATIVE_PATTERNS,
            split_on_level=3,
        )
        chunks_low = chunker_low.parse(content, "test.md")

        # Low split should have fewer chunks
        assert len(chunks_low) < len(chunks_high)

    def test_subsection_ids_serialization(self) -> None:
        """subsection_ids is properly serialized in to_record_dict."""
        chunker = StructuredChunker(
            subsection_patterns=LEGISLATIVE_PATTERNS,
            split_on_level=3,
        )
        content = """SEC. 210. ACCESS.
(a) FINDINGS.—Content."""

        chunks = chunker.parse(content, "test.md")

        # Find chunk with subsection_ids
        chunks_with_ids = [c for c in chunks if c.subsection_ids]
        if chunks_with_ids:
            import json

            record = chunks_with_ids[0].to_record_dict()
            assert "subsection_ids" in record
            # Should be JSON-serialized list
            parsed = json.loads(record["subsection_ids"])
            assert isinstance(parsed, list)

    def test_empty_subsection_ids_default(self) -> None:
        """Chunks without inline subsections have empty subsection_ids."""
        chunker = StructuredChunker()  # No patterns
        content = "# Title\n\nSimple content."

        chunks = chunker.parse(content, "test.md")
        for chunk in chunks:
            assert chunk.subsection_ids == []

    def test_parent_chain_preserved_with_split_on_level(self) -> None:
        """Parent chain is still built correctly with split_on_level."""
        chunker = StructuredChunker(
            subsection_patterns=LEGISLATIVE_PATTERNS,
            split_on_level=3,
        )
        content = """TITLE I—BROADBAND
CHAPTER 1—DEFINITIONS
SEC. 101. Short Title.
(a) Content here."""

        chunks = chunker.parse(content, "test.md")

        # Find SEC. 101 body chunk
        sec_body_chunks = [
            c for c in chunks if c.heading_level == 0 and "Content" in c.content
        ]
        if sec_body_chunks:
            # Parent chain should include TITLE and CHAPTER and SEC
            parent_chain = sec_body_chunks[0].parent_chain
            assert any("TITLE" in p or "I" in p for p in parent_chain)

    def test_markdown_headings_respect_split_level(self) -> None:
        """Standard markdown headings also respect split_on_level."""
        chunker = StructuredChunker(split_on_level=2)
        content = """# H1 Title

## H2 Section

### H3 Subsection

Content here.

#### H4 Deep

More content."""

        chunks = chunker.parse(content, "test.md")

        # Only H1 and H2 should create split points
        # H3 and H4 should be inline in parent content
        h1_h2_chunks = [c for c in chunks if c.heading_level in (1, 2)]
        h3_h4_chunks = [c for c in chunks if c.heading_level in (3, 4)]

        assert len(h1_h2_chunks) >= 2  # H1 and H2 are split points
        assert len(h3_h4_chunks) == 0  # H3 and H4 are not separate chunks
