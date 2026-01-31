"""Hybrid search combining semantic and keyword search.

This module provides data structures for representing search results from
hybrid search operations that combine semantic (vector) similarity with
keyword (full-text) matching.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from holodeck.lib.definition_extractor import DefinitionEntry


@dataclass
class SearchResult:
    """A single result from hybrid search.

    Represents a search result that may include scores from both semantic
    (vector) and keyword (full-text) search, along with document structure
    metadata and related definitions.

    Attributes:
        chunk_id: Unique identifier of the matched chunk
        content: The text content of the matched chunk
        fused_score: Combined score from semantic and keyword search (0.0-1.0)
        source_path: Path to the source document file
        parent_chain: List of ancestor headings from root to immediate parent
        section_id: Document section identifier (e.g., "1.2.3")
        subsection_ids: List of inline subsection IDs contained in this chunk
        semantic_score: Score from semantic/vector similarity search (optional)
        keyword_score: Score from keyword/full-text search (optional)
        exact_match: Whether this result contains an exact phrase match
        definitions_context: Related definitions for terms found in the content

    Example:
        >>> result = SearchResult(
        ...     chunk_id="policy_md_chunk_5",
        ...     content="Force Majeure means any event...",
        ...     fused_score=0.92,
        ...     source_path="/docs/policy.md",
        ...     parent_chain=["Chapter 1", "Definitions"],
        ...     section_id="1.2",
        ...     subsection_ids=["subsec_a_findings", "para_1_access"],
        ...     semantic_score=0.88,
        ...     keyword_score=0.95,
        ...     exact_match=True,
        ... )
        >>> print(result.format())
    """

    chunk_id: str
    content: str
    fused_score: float
    source_path: str
    parent_chain: list[str]
    section_id: str
    subsection_ids: list[str] = field(default_factory=list)
    semantic_score: float | None = None
    keyword_score: float | None = None
    exact_match: bool = False
    definitions_context: list["DefinitionEntry"] = field(default_factory=list)

    def format(self) -> str:
        """Format result for agent consumption.

        Produces a human-readable representation of the search result
        suitable for inclusion in agent context or display to users.

        Returns:
            Formatted string with score, source, location, content,
            and any relevant definitions.

        Example:
            >>> result = SearchResult(
            ...     chunk_id="doc_0",
            ...     content="Hello world",
            ...     fused_score=0.85,
            ...     source_path="/doc.md",
            ...     parent_chain=["Chapter 1"],
            ...     section_id="1.1",
            ... )
            >>> print(result.format())
            Score: 0.850 | Source: /doc.md
            Location: Chapter 1
            Section: 1.1
            <BLANKLINE>
            Hello world
        """
        location = " > ".join(self.parent_chain) if self.parent_chain else "Root"
        lines = [
            f"Score: {self.fused_score:.3f} | Source: {self.source_path}",
            f"Location: {location}",
        ]
        if self.section_id:
            lines.append(f"Section: {self.section_id}")
        lines.append("")
        lines.append(self.content)
        if self.definitions_context:
            lines.append("")
            lines.append("Relevant definitions:")
            for defn in self.definitions_context:
                text = defn.definition_text
                if len(text) > 100:
                    text = text[:97] + "..."
                lines.append(f"  - {defn.term}: {text}")
        return "\n".join(lines)
