"""Definition extraction from documents.

This module provides data structures for representing extracted definitions
from documents. Definitions are key terms and their explanations that can
be used to enhance search results with contextual information.
"""

from dataclasses import dataclass, field


@dataclass
class DefinitionEntry:
    """An extracted definition from a document.

    Represents a term definition extracted from a document, including the
    term itself, its definition text, and metadata about where it was found.

    Attributes:
        id: Unique identifier for this definition entry
        source_path: Path to the source document containing the definition
        term: The term being defined (original casing)
        term_normalized: Lowercase normalized term for case-insensitive lookup
        definition_text: The full definition text explaining the term
        source_section: Section ID or heading where the definition was found
        exceptions: List of exceptions or exclusions to the definition

    Example:
        >>> entry = DefinitionEntry(
        ...     id="policy_md_def_force_majeure",
        ...     source_path="/docs/policy.md",
        ...     term="Force Majeure",
        ...     term_normalized="force majeure",
        ...     definition_text="Any event beyond the reasonable control...",
        ...     source_section="1.2 Definitions",
        ...     exceptions=["acts of negligence", "breach of contract"],
        ... )
    """

    id: str
    source_path: str
    term: str
    term_normalized: str
    definition_text: str
    source_section: str
    exceptions: list[str] = field(default_factory=list)
