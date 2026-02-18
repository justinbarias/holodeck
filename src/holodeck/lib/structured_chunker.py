"""Structured document chunking with hierarchy preservation.

This module provides data structures for representing document chunks that
preserve hierarchical structure, cross-references, and content classification.
Designed for use with the HierarchicalDocumentTool for advanced document retrieval.

Key Features:
- Structure-aware markdown parsing with heading hierarchy extraction
- Parent chain building for document navigation context
- Token-aware chunking with sentence boundary splitting
- Chunk type classification (content, definition, requirement, reference, header)
- Flat text fallback for unstructured documents
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import tiktoken


@dataclass
class SubsectionPattern:
    """Configurable pattern for implicit subsection detection in documents.

    Used to recognize legislative-style numbering schemes (e.g., (a), (1), (A))
    as implicit headings that should create separate chunks with proper hierarchy.

    Attributes:
        name: Human-readable name for the pattern (e.g., "subsection", "paragraph")
        pattern: Compiled regex to match the marker. Should have groups for:
            - Group 1: The marker itself (e.g., "a", "1", "A")
            - Group 2: Optional text after the marker (title/content)
        level: Heading level to assign (3-6, since 1-2 are reserved for markdown)
        extract_title: Whether to include text after marker as the heading title

    Example:
        >>> pattern = SubsectionPattern(
        ...     name="subsection",
        ...     pattern=re.compile(r"^\\(([a-z])\\)\\s*(.*)$", re.MULTILINE),
        ...     level=3,
        ...     extract_title=True,
        ... )
    """

    name: str
    pattern: re.Pattern[str]
    level: int
    extract_title: bool = True


# US Legislative patterns - full hierarchy from Title to subparagraph
# Hierarchy: Title > Chapter > Section > (a) subsection > (1) para > (A) subpara
LEGISLATIVE_PATTERNS: list[SubsectionPattern] = [
    SubsectionPattern(
        name="title",
        pattern=re.compile(
            r"^TITLE\s+([IVXLCDM]+|\d+)[.:\-—]?\s*(.*)$", re.MULTILINE | re.IGNORECASE
        ),
        level=1,
        extract_title=True,
    ),
    SubsectionPattern(
        name="chapter",
        pattern=re.compile(
            r"^CHAPTER\s+(\d+)[.:\-—]?\s*(.*)$", re.MULTILINE | re.IGNORECASE
        ),
        level=2,
        extract_title=True,
    ),
    SubsectionPattern(
        name="section",
        pattern=re.compile(
            r"^SEC\.?\s+(\d+)[.:\-—]?\s*(.*)$", re.MULTILINE | re.IGNORECASE
        ),
        level=3,
        extract_title=True,
    ),
    SubsectionPattern(
        name="subsection",
        pattern=re.compile(r"^\(([a-z])\)\s*(.*)$", re.MULTILINE),
        level=4,
        extract_title=True,
    ),
    SubsectionPattern(
        name="paragraph",
        pattern=re.compile(r"^\((\d+)\)\s*(.*)$", re.MULTILINE),
        level=5,
        extract_title=True,
    ),
    SubsectionPattern(
        name="subparagraph",
        pattern=re.compile(r"^\(([A-Z])\)\s*(.*)$", re.MULTILINE),
        level=6,
        extract_title=True,
    ),
]

# Australian legislative patterns - full hierarchy from Part to sub-subparagraph
# Hierarchy: Part > Division > Section > (1) subsection > (a) paragraph > (i) subpara
# Note: Australian uses numeric subsections first, then letters (opposite of US)
AU_LEGISLATIVE_PATTERNS: list[SubsectionPattern] = [
    SubsectionPattern(
        name="part",
        pattern=re.compile(
            r"^Part\s+(\d+)[.:\-—]?\s*(.*)$", re.MULTILINE | re.IGNORECASE
        ),
        level=1,
        extract_title=True,
    ),
    SubsectionPattern(
        name="division",
        pattern=re.compile(
            r"^Division\s+(\d+)[.:\-—]?\s*(.*)$", re.MULTILINE | re.IGNORECASE
        ),
        level=2,
        extract_title=True,
    ),
    SubsectionPattern(
        name="section",
        pattern=re.compile(
            r"^(?:Section\s+)?(\d+)[.:\-—]?\s+(.+)$", re.MULTILINE | re.IGNORECASE
        ),
        level=3,
        extract_title=True,
    ),
    SubsectionPattern(
        name="subsection",
        pattern=re.compile(r"^\((\d+)\)\s*(.*)$", re.MULTILINE),
        level=4,
        extract_title=True,
    ),
    SubsectionPattern(
        name="paragraph",
        pattern=re.compile(r"^\(([a-z])\)\s*(.*)$", re.MULTILINE),
        level=5,
        extract_title=True,
    ),
    SubsectionPattern(
        name="subparagraph",
        pattern=re.compile(
            r"^\((i{1,3}|iv|v|vi{0,3})\)\s*(.*)$", re.MULTILINE | re.IGNORECASE
        ),
        level=6,
        extract_title=True,
    ),
]

# Academic paper style: numbered sections and subsections
# Format: 1. Section, 1.1 Subsection, 1.1.1 Subsubsection
ACADEMIC_PATTERNS: list[SubsectionPattern] = [
    SubsectionPattern(
        name="section",
        pattern=re.compile(r"^(\d+)\.\s+(.*)$", re.MULTILINE),
        level=3,
        extract_title=True,
    ),
    SubsectionPattern(
        name="subsection",
        pattern=re.compile(r"^(\d+\.\d+)\s+(.*)$", re.MULTILINE),
        level=4,
        extract_title=True,
    ),
    SubsectionPattern(
        name="subsubsection",
        pattern=re.compile(r"^(\d+\.\d+\.\d+)\s+(.*)$", re.MULTILINE),
        level=5,
        extract_title=True,
    ),
    SubsectionPattern(
        name="paragraph",
        pattern=re.compile(r"^(\d+\.\d+\.\d+\.\d+)\s+(.*)$", re.MULTILINE),
        level=6,
        extract_title=True,
    ),
]

# Technical documentation style: steps, substeps, notes
# Format: Step 1, 1.1, Note:, Warning:, Caution:
TECHNICAL_PATTERNS: list[SubsectionPattern] = [
    SubsectionPattern(
        name="step",
        pattern=re.compile(r"^Step\s+(\d+)[.:]?\s*(.*)$", re.MULTILINE | re.IGNORECASE),
        level=3,
        extract_title=True,
    ),
    SubsectionPattern(
        name="substep",
        pattern=re.compile(r"^(\d+\.\d+)[.:]?\s*(.*)$", re.MULTILINE),
        level=4,
        extract_title=True,
    ),
    SubsectionPattern(
        name="note",
        pattern=re.compile(r"^(Note)[.:]?\s*(.*)$", re.MULTILINE | re.IGNORECASE),
        level=4,
        extract_title=True,
    ),
    SubsectionPattern(
        name="warning",
        pattern=re.compile(
            r"^(Warning|Caution)[.:]?\s*(.*)$", re.MULTILINE | re.IGNORECASE
        ),
        level=4,
        extract_title=True,
    ),
]

# Legal contract style: articles, sections, clauses
# Format: Article I, Section 1, (a) clause
CONTRACT_PATTERNS: list[SubsectionPattern] = [
    SubsectionPattern(
        name="article",
        pattern=re.compile(
            r"^Article\s+([IVXLCDM]+|\d+)[.:]?\s*(.*)$", re.MULTILINE | re.IGNORECASE
        ),
        level=3,
        extract_title=True,
    ),
    SubsectionPattern(
        name="section",
        pattern=re.compile(
            r"^Section\s+(\d+(?:\.\d+)?)[.:]?\s*(.*)$", re.MULTILINE | re.IGNORECASE
        ),
        level=4,
        extract_title=True,
    ),
    SubsectionPattern(
        name="clause",
        pattern=re.compile(r"^\(([a-z])\)\s*(.*)$", re.MULTILINE),
        level=5,
        extract_title=True,
    ),
    SubsectionPattern(
        name="subclause",
        pattern=re.compile(r"^\((\d+)\)\s*(.*)$", re.MULTILINE),
        level=6,
        extract_title=True,
    ),
]

# Domain pattern registry for easy lookup by domain name
DOMAIN_PATTERNS: dict[str, list[SubsectionPattern]] = {
    "us_legislative": LEGISLATIVE_PATTERNS,
    "au_legislative": AU_LEGISLATIVE_PATTERNS,
    "academic": ACADEMIC_PATTERNS,
    "technical": TECHNICAL_PATTERNS,
    "legal_contract": CONTRACT_PATTERNS,
}


class ChunkType(str, Enum):
    """Classification of chunk content type.

    Used to categorize document chunks for specialized handling during
    search and retrieval operations.

    Attributes:
        CONTENT: Regular document content (paragraphs, lists, etc.)
        DEFINITION: Term definitions (e.g., glossary entries, defined terms)
        REQUIREMENT: Requirements or obligations (e.g., "shall", "must")
        REFERENCE: Cross-references or citations to other sections
        HEADER: Section headers or titles
    """

    CONTENT = "content"
    DEFINITION = "definition"
    REQUIREMENT = "requirement"
    REFERENCE = "reference"
    HEADER = "header"


@dataclass
class DocumentChunk:
    """A parsed section of a document with structure metadata.

    Represents a single chunk of a document that preserves its position
    in the document hierarchy, cross-references to other sections,
    and optional definition information.

    Attributes:
        id: Unique identifier for this chunk (typically source_path + chunk_index)
        source_path: Path to the source document file
        chunk_index: Zero-based index of this chunk within the document
        content: The actual text content of the chunk
        parent_chain: List of ancestor heading texts from root to immediate parent
        section_id: Document section identifier (e.g., "1.2.3", "A.1")
        chunk_type: Classification of the content type
        cross_references: List of section IDs referenced by this chunk
        heading_level: Heading level if this is a header chunk (1-6, 0 for non-headers)
        embedding: Optional vector embedding for semantic search
        contextualized_content: Content with added context for better retrieval
        mtime: File modification time (Unix timestamp) for change detection
        defined_term: The term being defined (if chunk_type is DEFINITION)
        defined_term_normalized: Lowercase normalized term for case-insensitive lookup

    Example:
        >>> chunk = DocumentChunk(
        ...     id="policy_md_chunk_5",
        ...     source_path="/docs/policy.md",
        ...     chunk_index=5,
        ...     content="Force Majeure means any event beyond...",
        ...     parent_chain=["Chapter 1", "Definitions"],
        ...     section_id="1.2",
        ...     chunk_type=ChunkType.DEFINITION,
        ...     defined_term="Force Majeure",
        ...     defined_term_normalized="force majeure",
        ... )
    """

    id: str
    source_path: str
    chunk_index: int
    content: str
    parent_chain: list[str] = field(default_factory=list)
    section_id: str = ""
    chunk_type: ChunkType = ChunkType.CONTENT
    cross_references: list[str] = field(default_factory=list)
    heading_level: int = 0
    embedding: list[float] | None = None
    contextualized_content: str = ""
    mtime: float = 0.0
    defined_term: str = ""
    defined_term_normalized: str = ""
    subsection_ids: list[str] = field(default_factory=list)

    def to_record_dict(self) -> dict[str, Any]:
        """Convert to dict for vector store record creation.

        Serializes list fields as JSON strings for storage in vector databases
        that expect flat field structures.

        Returns:
            Dictionary with all fields serialized for vector store insertion.
            List fields (parent_chain, cross_references) are JSON-encoded.

        Example:
            >>> chunk = DocumentChunk(
            ...     id="doc_0",
            ...     source_path="/doc.md",
            ...     chunk_index=0,
            ...     content="Hello",
            ...     parent_chain=["Ch1", "Sec1"],
            ... )
            >>> record = chunk.to_record_dict()
            >>> record["parent_chain"]
            '["Ch1", "Sec1"]'
        """
        import json

        return {
            "id": self.id,
            "source_path": self.source_path,
            "chunk_index": self.chunk_index,
            "content": self.content,
            "embedding": self.embedding,
            "parent_chain": json.dumps(self.parent_chain),
            "section_id": self.section_id,
            "chunk_type": self.chunk_type.value,
            "cross_references": json.dumps(self.cross_references),
            "contextualized_content": self.contextualized_content,
            "mtime": self.mtime,
            "defined_term": self.defined_term,
            "defined_term_normalized": self.defined_term_normalized,
            "subsection_ids": json.dumps(self.subsection_ids),
        }


@dataclass
class _Section:
    """Internal representation of a document section during parsing.

    Attributes:
        content: The text content of the section including heading.
        parent_chain: List of ancestor heading titles.
        heading_level: The heading level (1-6, 0 for preamble/body).
        heading_title: The heading text (empty for preamble).
        subsection_ids: List of inline subsection IDs tracked but not split.
    """

    content: str
    parent_chain: list[str]
    heading_level: int
    heading_title: str
    subsection_ids: list[str] = field(default_factory=list)


class StructuredChunker:
    """Structure-aware markdown chunker with hierarchy preservation.

    Parses markdown documents into chunks while preserving:
    - Parent chain (heading hierarchy for navigation context)
    - Section IDs (normalized identifiers)
    - Chunk type classification (content, definition, requirement, etc.)
    - Token-bounded sections with sentence-aware splitting

    The chunker follows Anthropic's contextual retrieval baseline with a
    default max_tokens of 800 per chunk.

    Attributes:
        max_tokens: Maximum tokens per chunk (default 800).

    Example:
        >>> chunker = StructuredChunker(max_tokens=800)
        >>> chunks = chunker.parse(markdown_content, "document.md")
        >>> for chunk in chunks:
        ...     print(f"{chunk.section_id}: {chunk.parent_chain}")
    """

    DEFAULT_MAX_TOKENS = 800  # Anthropic contextual retrieval baseline

    # Regex patterns
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    SENTENCE_ENDINGS = re.compile(r"(?<=[.!?])\s+")

    # Patterns for chunk type classification
    DEFINITION_KEYWORDS = {"definitions", "glossary", "terms", "interpretation"}
    DEFINITION_PATTERN = re.compile(r'"([^"]+)"\s+(means|shall mean)', re.IGNORECASE)
    REQUIREMENT_KEYWORDS = {"shall", "must", "required", "mandatory"}

    # Section ID normalization pattern
    SECTION_ID_CLEANUP = re.compile(r"[^a-z0-9]+")

    # Constants for header classification fix
    MIN_SUBSTANTIVE_LENGTH = 150  # Content >= this is not header-only
    SENTENCE_ENDING_PATTERN = re.compile(r"[.!?]\s*$")

    # Patterns for paragraph normalization (lines that should NOT be joined)
    # Matches markdown headings, list items, subsection markers, code fences
    BLOCK_START_PATTERN = re.compile(
        r"^(?:"
        r"#{1,6}\s|"  # Markdown headings
        r"[-*+]\s|"  # Unordered list items
        r"\d+\.\s|"  # Ordered list items (1. 2. etc.)
        r"\([a-zA-Z0-9]+\)\s|"  # Subsection markers like (a), (1), (A)
        r"```|"  # Code fence
        r">\s"  # Block quote
        r")"
    )

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        subsection_patterns: list[SubsectionPattern] | None = None,
        max_subsection_depth: int | None = None,
        split_on_level: int | None = None,
    ) -> None:
        """Initialize the structured chunker.

        Args:
            max_tokens: Maximum tokens per chunk. Defaults to 800.
            subsection_patterns: Optional list of SubsectionPattern for detecting
                implicit headings (e.g., legislative numbering like (a), (1)).
                When provided, enables enhanced hierarchy extraction.
            max_subsection_depth: Maximum number of subsection levels to recognize.
                If None (default), uses all patterns in subsection_patterns.
                Must be between 1 and len(subsection_patterns) if specified.
            split_on_level: Heading levels <= this value create new chunks.
                Levels > this value are accumulated into parent chunks with their
                markers tracked in subsection_ids. If None (default):
                - With subsection_patterns: defaults to len(patterns) // 2
                - Without patterns: defaults to 6 (split on all markdown levels)

        Raises:
            ValueError: If max_tokens is not positive.
            ValueError: If max_subsection_depth is invalid for the given patterns.
            ValueError: If split_on_level is invalid.
        """
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        # Determine effective max depth
        if subsection_patterns is not None:
            max_allowed = len(subsection_patterns)
            if max_subsection_depth is None:
                max_subsection_depth = max_allowed  # Default to all patterns
            elif not 1 <= max_subsection_depth <= max_allowed:
                raise ValueError(
                    f"max_subsection_depth must be between 1 and {max_allowed} "
                    f"for the given patterns"
                )
        else:
            # No patterns, depth doesn't matter
            max_subsection_depth = max_subsection_depth or 0

        # Determine effective split_on_level
        if split_on_level is not None:
            if split_on_level < 1 or split_on_level > 6:
                raise ValueError("split_on_level must be between 1 and 6")
            effective_split_on_level = split_on_level
        elif subsection_patterns is not None:
            # Default to half of pattern count (rounded down)
            effective_split_on_level = len(subsection_patterns) // 2
        else:
            # No patterns: split on all markdown heading levels
            effective_split_on_level = 6

        self._max_tokens = max_tokens
        self._subsection_patterns = subsection_patterns
        self._max_subsection_depth = max_subsection_depth
        self._split_on_level = effective_split_on_level
        self._encoder = self._initialize_tokenizer()

    @property
    def max_tokens(self) -> int:
        """Get the maximum tokens per chunk."""
        return self._max_tokens

    def _initialize_tokenizer(self) -> tiktoken.Encoding:
        """Initialize tiktoken encoder.

        Returns:
            tiktoken.Encoding instance for cl100k_base.
        """
        return tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken.

        Uses tiktoken's cl100k_base encoder for accurate token counting.

        Args:
            text: Text to count tokens for.

        Returns:
            Token count.
        """
        if not text:
            return 0
        return len(self._encoder.encode(text))

    def _normalize_paragraphs(self, text: str) -> str:
        """Join wrapped lines into proper paragraphs.

        PDF extraction often preserves column-width line breaks (~100 chars)
        that split sentences mid-flow. This method joins such lines while
        preserving intentional paragraph breaks and structural elements.

        Preserves:
        - Blank lines (paragraph separators)
        - Markdown headings (# lines)
        - List items (- * 1.)
        - Subsection markers ((a), (1), (A))
        - Code blocks (```)
        - Block quotes (>)

        Args:
            text: Text with potentially wrapped lines.

        Returns:
            Text with wrapped lines joined into proper paragraphs.

        Example:
            Input:
                "The United States faces challenges arising from rapid
                technological advancement, including AI proliferation."

            Output:
                "The United States faces challenges arising from rapid \
technological advancement, including AI proliferation."
        """
        if not text:
            return text

        lines = text.split("\n")
        normalized: list[str] = []
        i = 0
        in_code_block = False

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Track code block state
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                normalized.append(line)
                i += 1
                continue

            # Inside code blocks, preserve all lines as-is
            if in_code_block:
                normalized.append(line)
                i += 1
                continue

            # Empty lines are paragraph breaks - preserve them
            if not stripped:
                normalized.append(line)
                i += 1
                continue

            # Lines starting with block elements should not be joined to previous
            if self.BLOCK_START_PATTERN.match(stripped):
                normalized.append(line)
                i += 1
                continue

            # Try to join with following lines if:
            # 1. Current line doesn't end with sentence punctuation
            # 2. Next line exists and starts with lowercase letter
            # 3. Next line is not a block element
            while i + 1 < len(lines):
                next_line = lines[i + 1]
                next_stripped = next_line.strip()

                # Stop if next line is empty (paragraph break)
                if not next_stripped:
                    break

                # Stop if next line is a block element
                if self.BLOCK_START_PATTERN.match(next_stripped):
                    break

                # Stop if current line ends with sentence punctuation
                if line.rstrip().endswith((".", ":", ";", "!", "?")):
                    break

                # Stop if next line doesn't start with lowercase
                # (indicates new sentence or heading)
                first_char = next_stripped[0:1]
                if not first_char.islower():
                    break

                # Join the lines
                i += 1
                line = line.rstrip() + " " + next_stripped

            normalized.append(line)
            i += 1

        return "\n".join(normalized)

    def _get_code_block_ranges(self, text: str) -> list[tuple[int, int]]:
        """Get the start and end positions of all code blocks.

        Args:
            text: Text to scan for code blocks.

        Returns:
            List of (start, end) position tuples for each code block.
        """
        ranges: list[tuple[int, int]] = []
        for match in self.CODE_BLOCK_PATTERN.finditer(text):
            ranges.append((match.start(), match.end()))
        return ranges

    def _is_in_code_block(
        self, position: int, code_block_ranges: list[tuple[int, int]]
    ) -> bool:
        """Check if a position is inside a code block.

        Args:
            position: Character position to check.
            code_block_ranges: List of (start, end) tuples from _get_code_block_ranges.

        Returns:
            True if position is inside any code block.
        """
        return any(start <= position < end for start, end in code_block_ranges)

    def _extract_headings(self, markdown: str) -> list[tuple[int, str, int]]:
        """Extract headings from markdown with level, title, and position.

        Ignores headings inside code blocks. When subsection_patterns is configured,
        also detects implicit headings from legislative-style numbering.

        Args:
            markdown: Markdown content to parse.

        Returns:
            List of (level, title, position) tuples, sorted by position.
        """
        # Mask code blocks to avoid extracting headings from them
        masked = self.CODE_BLOCK_PATTERN.sub(lambda m: " " * len(m.group(0)), markdown)
        code_block_ranges = self._get_code_block_ranges(markdown)

        headings: list[tuple[int, str, int]] = []

        # Build pattern list once for reuse (if configured)
        patterns_to_check: list[SubsectionPattern] = []
        if self._subsection_patterns:
            patterns_to_check = self._subsection_patterns[: self._max_subsection_depth]

        # Extract markdown headings
        for match in self.HEADING_PATTERN.finditer(masked):
            level = len(match.group(1))
            title = match.group(2).strip()
            position = match.start()

            # When subsection patterns are active, check if the heading title
            # matches a pattern and use the pattern's level instead of the
            # markdown # count. This corrects font-size-based PDF extraction
            # where e.g. CHAPTER and TITLE both get # (h1) due to similar
            # font sizes, but the pattern knows CHAPTER should be level 2.
            if patterns_to_check:
                for pattern_config in patterns_to_check:
                    if pattern_config.pattern.match(title):
                        level = pattern_config.level
                        break

            headings.append((level, title, position))

        # Extract subsection patterns if configured
        if patterns_to_check:
            for pattern_config in patterns_to_check:
                for match in pattern_config.pattern.finditer(markdown):
                    position = match.start()

                    # Skip if inside code block
                    if self._is_in_code_block(position, code_block_ranges):
                        continue

                    # Build the title from the match
                    marker = match.group(1)  # e.g., "a", "1", "A"
                    text_after = ""
                    if match.lastindex is not None and match.lastindex >= 2:
                        text_after = match.group(2).strip()

                    if pattern_config.extract_title and text_after:
                        # Include first line of text after marker as title
                        # Get text up to first newline
                        first_line = text_after.split("\n")[0].strip()
                        title = f"({marker}) {first_line}"
                    else:
                        title = f"({marker})"

                    headings.append((pattern_config.level, title, position))

        # Sort by position to maintain document order
        headings.sort(key=lambda h: h[2])

        return headings

    def _split_by_headings(
        self, markdown: str, headings: list[tuple[int, str, int]]
    ) -> list[_Section]:
        """Split markdown into sections based on headings.

        Builds parent_chain by tracking heading levels in a stack.
        Only creates new sections for headings with level <= split_on_level.
        Lower-level headings are accumulated into parent sections with their
        markers tracked in subsection_ids.

        Args:
            markdown: Original markdown content.
            headings: List of (level, title, position) tuples.

        Returns:
            List of _Section objects with parent chains and subsection_ids.
        """
        sections: list[_Section] = []
        parent_stack: list[tuple[int, str]] = []  # (level, title) pairs

        # Separate split-point headings from inline headings
        split_headings = [
            (lvl, title, pos)
            for lvl, title, pos in headings
            if lvl <= self._split_on_level
        ]

        # Handle content before first heading (preamble)
        first_pos = headings[0][2] if headings else len(markdown)
        if first_pos > 0:
            preamble_content = markdown[:first_pos].strip()
            if preamble_content:
                sections.append(
                    _Section(
                        content=preamble_content,
                        parent_chain=[],
                        heading_level=0,
                        heading_title="",
                    )
                )

        # If no split headings but we have inline headings, treat as one section
        if not split_headings and headings:
            full_content = markdown.strip()
            # Normalize paragraphs
            full_content = self._normalize_paragraphs(full_content)
            # Track all headings as subsection_ids (using level for prefix)
            # No parent context available for inline-only headings
            subsection_ids = [
                self._generate_section_id(title, 0, lvl, [])
                for lvl, title, _ in headings
            ]
            sections.append(
                _Section(
                    content=full_content,
                    parent_chain=[],
                    heading_level=0,
                    heading_title="",
                    subsection_ids=subsection_ids,
                )
            )
            return sections

        for i, (level, title, pos) in enumerate(split_headings):
            # Pop stack until we find a parent (level < current)
            while parent_stack and parent_stack[-1][0] >= level:
                parent_stack.pop()

            # Build parent_chain from current stack
            parent_chain = [t for _, t in parent_stack]

            # Find end position (next split heading or document end)
            end_pos = (
                split_headings[i + 1][2]
                if i + 1 < len(split_headings)
                else len(markdown)
            )

            # Extract full content including all lower-level headings
            full_content = markdown[pos:end_pos].strip()

            # Find all inline subsection markers within this range
            inline_subsections = [
                (lvl, sub_title, sub_pos)
                for lvl, sub_title, sub_pos in headings
                if pos < sub_pos < end_pos and lvl > self._split_on_level
            ]

            # Generate subsection_ids for inline markers (using level for prefix)
            # Include parent_chain + current title as hierarchy context
            subsection_ids = [
                self._generate_section_id(sub_title, 0, lvl, parent_chain + [title])
                for lvl, sub_title, _ in inline_subsections
            ]

            # When subsection patterns are active and this is a split-level heading,
            # separate heading from body for proper hierarchy
            if self._subsection_patterns and level <= self._split_on_level:
                lines = full_content.split("\n", 1)
                heading_line = lines[0].strip()
                body_content = lines[1].strip() if len(lines) > 1 else ""

                # Normalize paragraph wrapping in body content
                if body_content:
                    body_content = self._normalize_paragraphs(body_content)

                # Emit heading-only chunk first (no subsection_ids on header)
                sections.append(
                    _Section(
                        content=heading_line,
                        parent_chain=list(parent_chain),
                        heading_level=level,
                        heading_title=title,
                    )
                )

                # Emit body content with subsection_ids if present
                if body_content:
                    body_parent_chain = list(parent_chain) + [title]
                    sections.append(
                        _Section(
                            content=body_content,
                            parent_chain=body_parent_chain,
                            heading_level=0,
                            heading_title="",
                            subsection_ids=subsection_ids,
                        )
                    )
            else:
                # Default behavior: emit full content with subsection_ids
                sections.append(
                    _Section(
                        content=full_content,
                        parent_chain=list(parent_chain),
                        heading_level=level,
                        heading_title=title,
                        subsection_ids=subsection_ids,
                    )
                )

            # Push current heading onto stack
            parent_stack.append((level, title))

        return sections

    def _split_section(
        self, section: _Section, source_path: str, start_index: int
    ) -> list[DocumentChunk]:
        """Split a section into chunks respecting max_tokens.

        Splits at sentence boundaries when section exceeds max_tokens.
        The first chunk receives all subsection_ids from the section.

        Args:
            section: Section to split.
            source_path: Source file path for chunk IDs.
            start_index: Starting chunk index.

        Returns:
            List of DocumentChunk objects.
        """
        token_count = self._count_tokens(section.content)

        # If section fits in one chunk, return as-is
        if token_count <= self._max_tokens:
            return [
                self._create_chunk(
                    content=section.content,
                    parent_chain=section.parent_chain,
                    heading_level=section.heading_level,
                    heading_title=section.heading_title,
                    source_path=source_path,
                    chunk_index=start_index,
                    subsection_ids=section.subsection_ids,
                )
            ]

        # Split at sentence boundaries
        chunks: list[DocumentChunk] = []
        sentences = self.SENTENCE_ENDINGS.split(section.content)

        current_content: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)

            # If single sentence exceeds max, we still need to include it
            # (will be handled as oversized chunk)
            if current_tokens + sentence_tokens > self._max_tokens and current_content:
                # Emit current chunk (first chunk gets subsection_ids)
                is_first_chunk = len(chunks) == 0
                chunks.append(
                    self._create_chunk(
                        content=" ".join(current_content),
                        parent_chain=section.parent_chain,
                        heading_level=section.heading_level,
                        heading_title=section.heading_title,
                        source_path=source_path,
                        chunk_index=start_index + len(chunks),
                        subsection_ids=section.subsection_ids if is_first_chunk else [],
                    )
                )
                current_content = [sentence]
                current_tokens = sentence_tokens
            else:
                current_content.append(sentence)
                current_tokens += sentence_tokens

        # Emit final chunk
        if current_content:
            is_first_chunk = len(chunks) == 0
            chunks.append(
                self._create_chunk(
                    content=" ".join(current_content),
                    parent_chain=section.parent_chain,
                    heading_level=section.heading_level,
                    heading_title=section.heading_title,
                    source_path=source_path,
                    chunk_index=start_index + len(chunks),
                    subsection_ids=section.subsection_ids if is_first_chunk else [],
                )
            )

        return chunks

    def _create_chunk(
        self,
        content: str,
        parent_chain: list[str],
        heading_level: int,
        heading_title: str,
        source_path: str,
        chunk_index: int,
        mtime: float = 0.0,
        subsection_ids: list[str] | None = None,
    ) -> DocumentChunk:
        """Create a DocumentChunk with all metadata.

        Args:
            content: Chunk text content.
            parent_chain: List of ancestor headings.
            heading_level: Heading level (0 for body).
            heading_title: Heading text.
            source_path: Source file path.
            chunk_index: Sequential chunk index.
            mtime: File modification time.
            subsection_ids: List of inline subsection IDs tracked in this chunk.

        Returns:
            Fully populated DocumentChunk.
        """
        # Generate chunk ID
        source_name = source_path.replace("/", "_").replace("\\", "_")
        chunk_id = f"{source_name}_chunk_{chunk_index}"

        # Generate section ID
        section_id = self._generate_section_id(
            heading_title, chunk_index, heading_level, parent_chain
        )

        # Classify chunk type
        chunk_type = self._classify_chunk_type(
            content, heading_title, parent_chain, heading_level
        )

        # Extract defined term if definition
        defined_term = ""
        defined_term_normalized = ""
        if chunk_type == ChunkType.DEFINITION:
            defined_term, defined_term_normalized = self._extract_defined_term(content)

        return DocumentChunk(
            id=chunk_id,
            source_path=source_path,
            chunk_index=chunk_index,
            content=content,
            parent_chain=list(parent_chain),  # Copy to avoid mutation
            section_id=section_id,
            chunk_type=chunk_type,
            cross_references=[],  # TODO: Extract cross-references
            heading_level=heading_level,
            embedding=None,
            contextualized_content="",  # Filled by LLMContextGenerator
            mtime=mtime,
            defined_term=defined_term,
            defined_term_normalized=defined_term_normalized,
            subsection_ids=list(subsection_ids) if subsection_ids else [],
        )

    def _get_level_prefix(self, level: int) -> str:
        """Get the appropriate ID prefix for a heading level.

        Uses pattern names if subsection_patterns is configured, otherwise
        falls back to generic prefixes based on level.

        Args:
            level: The heading level (1-6).

        Returns:
            Prefix string for the section ID (e.g., "title", "sec", "subsec").
        """
        if self._subsection_patterns:
            # Find the pattern matching this level
            for pattern in self._subsection_patterns:
                if pattern.level == level:
                    return pattern.name
        # Fallback prefixes for markdown headings
        return {1: "h1", 2: "h2", 3: "h3", 4: "h4", 5: "h5", 6: "h6"}.get(level, "sec")

    def _generate_section_id(
        self,
        heading_title: str,
        chunk_index: int,
        level: int = 0,
        parent_chain: list[str] | None = None,
    ) -> str:
        """Generate a normalized, fully-qualified section ID from heading.

        Builds a hierarchical ID by normalizing each ancestor in parent_chain
        and joining them with the current heading title.

        Args:
            heading_title: The heading text.
            chunk_index: Chunk index for fallback.
            level: Optional heading level for appropriate prefix.
            parent_chain: List of ancestor heading titles for hierarchy.

        Returns:
            Normalized section ID (lowercase, underscores).
        """
        if not heading_title:
            return f"chunk_{chunk_index}"

        def _normalize(text: str) -> str:
            """Normalize a single heading title to an ID segment."""
            seg = text.lower()
            seg = self.SECTION_ID_CLEANUP.sub("_", seg)
            return re.sub(r"_+", "_", seg).strip("_")

        # Build hierarchical segments from parent_chain + current heading
        segments: list[str] = []
        if parent_chain:
            for ancestor in parent_chain:
                norm = _normalize(ancestor)
                if norm:
                    segments.append(norm)

        current = _normalize(heading_title)
        if current:
            segments.append(current)

        if not segments:
            return f"chunk_{chunk_index}"

        # Use level-appropriate prefix if level is provided
        if level > 0:
            prefix = self._get_level_prefix(level)
            return f"{prefix}_{'_'.join(segments)}"

        return f"sec_{'_'.join(segments)}"

    def _classify_chunk_type(
        self,
        content: str,
        heading_title: str,
        parent_chain: list[str],
        heading_level: int = 0,
    ) -> ChunkType:
        """Classify the type of chunk based on content and context.

        Classification priority:
        1. HEADER - if content is only the heading (no body)
        2. DEFINITION - if in definitions section or has definition patterns
        3. REQUIREMENT - if body text (not heading) contains requirement language
        4. CONTENT - default for regular content

        Args:
            content: Chunk content.
            heading_title: Heading text.
            parent_chain: Ancestor headings.
            heading_level: Heading level (1-6, 0 for non-headers).

        Returns:
            Classified ChunkType.
        """
        heading_lower = heading_title.lower()

        # Header-only chunks (check if content is just the heading)
        # This must come FIRST to avoid misclassifying headers as requirements
        if heading_title and heading_level > 0:
            content_stripped = content.strip()
            # Check for markdown heading format
            expected_header = f"{'#' * heading_level} {heading_title}".strip()
            if content_stripped == expected_header:
                return ChunkType.HEADER
            # Check for subsection pattern format (e.g., "(a) TITLE...")
            # These are single-line heading chunks with no body, but we must
            # also verify it's not substantive content that got line-joined.
            # After paragraph normalization, long single-line content with
            # sentence endings is substantive content, not a header.
            if (
                heading_level >= 3
                and self._subsection_patterns
                and "\n" not in content_stripped
                and len(content_stripped) < self.MIN_SUBSTANTIVE_LENGTH
                and not self.SENTENCE_ENDING_PATTERN.search(content_stripped)
            ):
                # Short single-line content without sentence ending = header-only
                return ChunkType.HEADER

        # Check if in definitions section
        all_headings = [heading_lower] + [p.lower() for p in parent_chain]
        in_definitions = any(
            kw in h for h in all_headings for kw in self.DEFINITION_KEYWORDS
        )

        # Check for definition patterns
        if in_definitions or self.DEFINITION_PATTERN.search(content):
            return ChunkType.DEFINITION

        # Extract body text (content without the heading line) for requirement check
        # This prevents sections from being classified as REQUIREMENT just because
        # the heading contains requirement keywords
        body_text = content
        if heading_title and heading_level > 0:
            # Remove the heading line to check only body content
            lines = content.split("\n")
            expected_heading_prefix = "#" * heading_level + " "
            body_lines = [
                line
                for line in lines
                if not line.strip().startswith(expected_heading_prefix)
            ]
            body_text = "\n".join(body_lines)

        body_lower = body_text.lower()

        # Check for requirement patterns in body text only
        # This avoids classifying entire sections as REQUIREMENT when they're
        # structured content that happens to contain requirement language
        has_requirement_keyword = any(
            kw in body_lower for kw in self.REQUIREMENT_KEYWORDS
        )
        is_requirement_statement = "shall" in body_lower or "must" in body_lower

        # Only classify as REQUIREMENT if:
        # 1. Body contains requirement keywords (shall/must), AND
        # 2. It's body-only content (no heading) OR short focused text (<500 chars)
        # This avoids classifying full legislative sections as REQUIREMENT
        is_short_or_no_heading = heading_level == 0 or len(body_text.strip()) < 500
        is_requirement = (
            has_requirement_keyword
            and is_requirement_statement
            and is_short_or_no_heading
        )
        if is_requirement:
            return ChunkType.REQUIREMENT

        return ChunkType.CONTENT

    def _extract_defined_term(self, content: str) -> tuple[str, str]:
        """Extract defined term from definition content.

        Args:
            content: Content potentially containing a definition.

        Returns:
            Tuple of (term, normalized_term).
        """
        match = self.DEFINITION_PATTERN.search(content)
        if match:
            term = match.group(1)
            return term, term.lower().replace(" ", "_")
        return "", ""

    def is_header_only(self, chunk: DocumentChunk) -> bool:
        """Check if chunk contains only a heading with no body content.

        Args:
            chunk: DocumentChunk to check.

        Returns:
            True if chunk is header-only (no substantive content).
        """
        return chunk.chunk_type == ChunkType.HEADER

    def _parse_flat_text(
        self, text: str, source_path: str, mtime: float = 0.0
    ) -> list[DocumentChunk]:
        """Parse flat text (no headings) into token-bounded chunks.

        Args:
            text: Plain text content.
            source_path: Source file path.
            mtime: File modification time.

        Returns:
            List of DocumentChunk objects.
        """
        # Split at sentence boundaries
        sentences = self.SENTENCE_ENDINGS.split(text)

        chunks: list[DocumentChunk] = []
        current_content: list[str] = []
        current_tokens = 0
        chunk_index = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_tokens = self._count_tokens(sentence)

            if current_tokens + sentence_tokens > self._max_tokens and current_content:
                # Emit current chunk
                chunks.append(
                    self._create_chunk(
                        content=" ".join(current_content),
                        parent_chain=[],
                        heading_level=0,
                        heading_title="",
                        source_path=source_path,
                        chunk_index=chunk_index,
                        mtime=mtime,
                    )
                )
                chunk_index += 1
                current_content = [sentence]
                current_tokens = sentence_tokens
            else:
                current_content.append(sentence)
                current_tokens += sentence_tokens

        # Emit final chunk
        if current_content:
            chunks.append(
                self._create_chunk(
                    content=" ".join(current_content),
                    parent_chain=[],
                    heading_level=0,
                    heading_title="",
                    source_path=source_path,
                    chunk_index=chunk_index,
                    mtime=mtime,
                )
            )

        return chunks

    def parse(
        self, markdown: str, source_path: str = "", mtime: float = 0.0
    ) -> list[DocumentChunk]:
        """Parse markdown into structure-aware chunks.

        Main entry point for document processing. Extracts headings,
        builds parent chains, and splits content into token-bounded
        chunks while preserving hierarchical context.

        Args:
            markdown: The markdown content to parse.
            source_path: Optional path to source file (for metadata).
            mtime: Optional file modification time (Unix timestamp).

        Returns:
            List of DocumentChunk objects with hierarchy preserved.

        Raises:
            ValueError: If markdown is empty or whitespace-only.

        Example:
            >>> chunker = StructuredChunker()
            >>> chunks = chunker.parse("# Title\\n\\nContent", "doc.md")
            >>> print(chunks[0].section_id)
            'sec_title'
        """
        if not markdown or not markdown.strip():
            raise ValueError("Markdown content cannot be empty")

        # Extract headings
        headings = self._extract_headings(markdown)

        # If no headings, fall back to flat text chunking
        if not headings:
            return self._parse_flat_text(markdown, source_path, mtime)

        # Split by headings into sections
        sections = self._split_by_headings(markdown, headings)

        # Process each section (split if needed) and collect chunks
        chunks: list[DocumentChunk] = []
        chunk_index = 0

        for section in sections:
            section_chunks = self._split_section(section, source_path, chunk_index)
            # Update mtime on all chunks
            for chunk in section_chunks:
                chunk.mtime = mtime
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        return chunks
