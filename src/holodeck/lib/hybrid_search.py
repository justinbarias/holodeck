"""Hybrid search combining semantic and keyword search.

This module provides data structures and algorithms for hybrid search operations
that combine semantic (vector) similarity with keyword (full-text) matching.

Key Features:
- SearchResult dataclass for unified result representation
- Reciprocal Rank Fusion (RRF) for merging ranked result lists
- Exact match index for section IDs and quoted phrases
- Pattern detection for identifying exact match queries

Usage:
    from holodeck.lib.hybrid_search import (
        SearchResult,
        reciprocal_rank_fusion,
        ExactMatchIndex,
        is_exact_match_query,
    )

    # Merge results from multiple search modalities
    fused = reciprocal_rank_fusion([semantic_results, keyword_results], k=60)

    # Check if query is an exact match query
    if is_exact_match_query("Section 203(a)(1)"):
        # Route to exact match search
        ...
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from holodeck.lib.definition_extractor import DefinitionEntry


# Regex patterns for exact match query detection

# Section ID patterns: "Section 203", "203(a)(1)", "§4.2", "1.2.3"
SECTION_ID_PATTERN = re.compile(
    r"^(?:Section\s+)?(\d+(?:\.\d+)*(?:\([a-zA-Z0-9]+\))*)$",
    re.IGNORECASE,
)

# Section symbol pattern: "§4.2", "§ 4.2.1"
SECTION_SYMBOL_PATTERN = re.compile(
    r"^§\s*(\d+(?:\.\d+)*)$",
    re.IGNORECASE,
)

# Numbered section pattern: "1.2.3", "4.5"
NUMBERED_SECTION_PATTERN = re.compile(
    r"^(\d+(?:\.\d+)+)$",
)

# Quoted exact phrase pattern: "Force Majeure", "reasonable best efforts"
EXACT_PHRASE_PATTERN = re.compile(r'^"([^"]+)"$')


def is_exact_match_query(query: str) -> bool:
    """Check if query is a section ID or quoted exact phrase.

    Detects queries that should be routed to exact match search:
    - Section references: "Section 203", "203(a)(1)", "§4.2"
    - Numbered sections: "1.2.3"
    - Quoted phrases: "Force Majeure"

    Args:
        query: Search query string

    Returns:
        True if query matches exact match patterns, False otherwise

    Example:
        >>> is_exact_match_query("Section 203(a)(1)")
        True
        >>> is_exact_match_query('"Force Majeure"')
        True
        >>> is_exact_match_query("What are the reporting requirements?")
        False
    """
    query = query.strip()

    # Check all patterns: quoted phrase, section symbol, numbered section, section ID
    return bool(
        EXACT_PHRASE_PATTERN.match(query)
        or SECTION_SYMBOL_PATTERN.match(query)
        or NUMBERED_SECTION_PATTERN.match(query)
        or SECTION_ID_PATTERN.match(query)
    )


def extract_exact_query(query: str) -> str:
    """Extract the exact term from a query.

    Removes outer quotes from quoted phrases, otherwise returns
    the query as-is (for section ID queries).

    Args:
        query: Query string (may be quoted or section ID)

    Returns:
        Extracted exact term without outer quotes

    Example:
        >>> extract_exact_query('"Force Majeure"')
        'Force Majeure'
        >>> extract_exact_query("Section 203")
        'Section 203'
    """
    query = query.strip()

    # Remove quotes from exact phrase
    match = EXACT_PHRASE_PATTERN.match(query)
    if match:
        return match.group(1)

    return query


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF combines results from different retrieval systems by scoring
    each document based on its rank in each list:
        score(d) = Σ weight_i / (k + rank_i(d))

    This approach is robust to different score distributions across
    retrieval systems and doesn't require score calibration.

    Args:
        ranked_lists: List of ranked result lists, each containing
            (doc_id, score) tuples sorted by relevance descending.
        k: RRF constant (default 60). Higher values give more weight
            to lower-ranked results, reducing the impact of rank position.
        weights: Optional weights for each list (default equal weights).
            Use to prioritize certain retrieval modalities.

    Returns:
        Merged list of (doc_id, score) tuples sorted by RRF score.
        Scores are normalized to 0-1 range based on maximum possible score.

    Example:
        >>> semantic = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
        >>> keyword = [("b", 0.95), ("a", 0.85), ("d", 0.75)]
        >>> fused = reciprocal_rank_fusion([semantic, keyword], k=60)
        >>> print(fused[0])  # Most relevant document
        ('b', 0.032...)
    """
    if not ranked_lists:
        return []

    # Default to equal weights
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]

    # Calculate RRF scores
    scores: dict[str, float] = {}
    for weight, ranked_list in zip(weights, ranked_lists, strict=False):
        for rank, (doc_id, _) in enumerate(ranked_list, start=1):
            if doc_id not in scores:
                scores[doc_id] = 0.0
            scores[doc_id] += weight / (k + rank)

    # Sort by fused score descending
    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Normalize scores to 0-1 range
    # Maximum possible score is when doc is rank 1 in all lists with max weights
    max_possible_score = sum(w / (k + 1) for w in weights) if weights else 1.0
    if max_possible_score > 0:
        normalized = [
            (doc_id, score / max_possible_score) for doc_id, score in sorted_results
        ]
        return normalized

    return sorted_results


class ExactMatchIndex:
    """Index for fast exact section ID and phrase lookup.

    Maintains mappings for:
    - Section ID → chunk IDs (for "Section 203(a)(1)" queries)
    - Content → chunk IDs (for exact phrase search)

    This enables O(1) lookup for section ID queries and efficient
    phrase matching for quoted search terms.

    Example:
        >>> index = ExactMatchIndex()
        >>> index.build([
        ...     ("chunk1", "Section 1.1", "Content about policies"),
        ...     ("chunk2", "Section 1.2", "Content about compliance"),
        ... ])
        >>> index.search_section("Section 1.1")
        ['chunk1']
    """

    def __init__(self) -> None:
        """Initialize empty exact match index."""
        self._section_to_chunks: dict[str, list[str]] = {}
        self._chunks_content: dict[str, str] = {}

    def build(self, chunks: list[tuple[str, str, str]]) -> None:
        """Build exact match index from chunks.

        Args:
            chunks: List of (chunk_id, section_id, content) tuples.
                section_id is used for section lookup.
                content is used for phrase search.

        Example:
            >>> index.build([
            ...     ("chunk1", "203(a)(1)", "Force Majeure means..."),
            ...     ("chunk2", "203(a)(2)", "Other content here..."),
            ... ])
        """
        self._section_to_chunks.clear()
        self._chunks_content.clear()

        for chunk_id, section_id, content in chunks:
            # Index by section ID
            if section_id:
                if section_id not in self._section_to_chunks:
                    self._section_to_chunks[section_id] = []
                self._section_to_chunks[section_id].append(chunk_id)

            # Store content for phrase search
            self._chunks_content[chunk_id] = content

    def search_section(self, section_id: str) -> list[str]:
        """Find chunks matching exact section ID.

        Args:
            section_id: Section identifier (e.g., "203(a)(1)", "Section 1.2")

        Returns:
            List of chunk IDs matching the section ID

        Example:
            >>> index.search_section("203(a)(1)")
            ['chunk1']
        """
        # Try exact match first
        if section_id in self._section_to_chunks:
            return self._section_to_chunks[section_id]

        # Try with "Section " prefix removed/added
        normalized = section_id.replace("Section ", "").strip()
        if normalized in self._section_to_chunks:
            return self._section_to_chunks[normalized]

        prefixed = f"Section {normalized}"
        if prefixed in self._section_to_chunks:
            return self._section_to_chunks[prefixed]

        return []

    def search_phrase(self, phrase: str, top_k: int = 10) -> list[str]:
        """Find chunks containing exact phrase.

        Performs case-insensitive substring matching on chunk content.

        Args:
            phrase: Exact phrase to search for
            top_k: Maximum number of results to return

        Returns:
            List of chunk IDs containing the phrase

        Example:
            >>> index.search_phrase("Force Majeure")
            ['chunk1', 'chunk3']
        """
        phrase_lower = phrase.lower()
        matches: list[str] = []

        for chunk_id, content in self._chunks_content.items():
            if phrase_lower in content.lower():
                matches.append(chunk_id)
                if len(matches) >= top_k:
                    break

        return matches


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
    definitions_context: list[DefinitionEntry] = field(default_factory=list)

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
