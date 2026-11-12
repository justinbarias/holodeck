"""Which entities a timeline reference names (spec 043, RET-08).

One pure rule, used by every backend so they resolve a reference the same
way:

1. An entity matches exactly when its id equals the reference, or its name
   equals the reference ignoring case. If any entity matches exactly, only
   those entities are used.
2. Otherwise an entity is used when its whole name appears in the reference
   at word boundaries, ignoring case: the character before and the
   character after the occurrence are not letters or digits. The start and
   the end of the reference count as boundaries.

So "history of s. 10" names "s. 10" and not "s. 1", and "the rate over
time" names "rate" and not "rat".
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EntityCandidate:
    """An entity a backend may resolve a reference to."""

    entity_id: str
    name: str


def _is_word_character(character: str) -> bool:
    return character.isalnum()


def name_in_reference(name: str, reference: str) -> bool:
    """True when the whole ``name`` appears in ``reference`` at word boundaries.

    Case is ignored. An empty name never matches.
    """
    needle = name.lower()
    haystack = reference.lower()
    if not needle:
        return False
    start = haystack.find(needle)
    while start != -1:
        end = start + len(needle)
        before_ok = start == 0 or not _is_word_character(haystack[start - 1])
        after_ok = end == len(haystack) or not _is_word_character(haystack[end])
        if before_ok and after_ok:
            return True
        start = haystack.find(needle, start + 1)
    return False


def is_exact_match(candidate: EntityCandidate, reference: str) -> bool:
    """True when the id equals ``reference`` or the name equals it ignoring case."""
    return (
        candidate.entity_id == reference or candidate.name.lower() == reference.lower()
    )


def resolve_entities(
    reference: str, candidates: Iterable[EntityCandidate]
) -> list[str]:
    """The ids of the entities ``reference`` names, in candidate order, once each.

    Exact matches win. Only when there is none are the entities whose whole
    name appears in ``reference`` at word boundaries used. An empty list
    means no entity is named.
    """
    pool = list(candidates)
    exact = [c.entity_id for c in pool if is_exact_match(c, reference)]
    if exact:
        return list(dict.fromkeys(exact))
    contained = [c.entity_id for c in pool if name_in_reference(c.name, reference)]
    return list(dict.fromkeys(contained))
