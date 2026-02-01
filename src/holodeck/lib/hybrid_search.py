"""Hybrid search combining semantic and keyword search.

This module provides data structures and algorithms for hybrid search operations
that combine semantic (vector) similarity with keyword (full-text) matching.

Key Features:
- SearchResult dataclass for unified result representation
- Reciprocal Rank Fusion (RRF) for merging ranked result lists

Usage:
    from holodeck.lib.hybrid_search import (
        SearchResult,
        reciprocal_rank_fusion,
    )

    # Merge results from multiple search modalities
    fused = reciprocal_rank_fusion([semantic_results, keyword_results], k=60)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from holodeck.lib.definition_extractor import DefinitionEntry


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
