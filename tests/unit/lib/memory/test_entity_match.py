"""How a timeline reference names entities (spec 043, RET-08).

``CASES`` is shared with the fake and Graphiti backend tests, so all three
resolve every reference the same way.
"""

from __future__ import annotations

import pytest

from holodeck.lib.memory.entity_match import (
    EntityCandidate,
    name_in_reference,
    resolve_entities,
)

pytestmark = [pytest.mark.unit]

# (reference, entity names, names the reference resolves to)
CASES: list[tuple[str, tuple[str, ...], list[str]]] = [
    ("history of s. 10", ("s. 1", "s. 10"), ["s. 10"]),
    ("history of s. 1", ("s. 1", "s. 10"), ["s. 1"]),
    ("history of Part III", ("Part I", "Part II", "Part III"), ["Part III"]),
    ("UK child benefit", ("UK",), ["UK"]),
    (
        "Family Benefit base rate",
        ("Family Benefit", "Family Benefit Act"),
        ["Family Benefit"],
    ),
    ("the rate over time", ("rate", "rat"), ["rate"]),
    ("rate a supplement", ("Rate A", "Rate A supplement"), ["Rate A supplement"]),
    ("RATE A", ("Rate A", "Rate A supplement"), ["Rate A"]),
    ("Rate Aardvark", ("Rate A",), []),
    ("rates of s. 1, s. 10", ("s. 1", "s. 10"), ["s. 1", "s. 10"]),
    ("nothing named", ("Rate A",), []),
]


def ids_for(names: tuple[str, ...]) -> list[EntityCandidate]:
    return [EntityCandidate(entity_id=f"id-{name}", name=name) for name in names]


@pytest.mark.parametrize(("reference", "names", "expected"), CASES)
def test_ret_08_reference_resolves_to_whole_names_at_word_boundaries(
    reference: str, names: tuple[str, ...], expected: list[str]
) -> None:
    resolved = resolve_entities(reference, ids_for(names))

    assert resolved == [f"id-{name}" for name in expected]


def test_ret_08_an_id_equal_to_the_reference_is_an_exact_match() -> None:
    candidates = [
        EntityCandidate(entity_id="uuid-1", name="Rate A"),
        EntityCandidate(entity_id="uuid-2", name="uuid-1 notes"),
    ]

    assert resolve_entities("uuid-1", candidates) == ["uuid-1"]


@pytest.mark.parametrize(
    ("name", "reference", "expected"),
    [
        ("UK", "UK", True),
        ("UK", "the UK.", True),
        ("UK", "UKraine", False),
        ("s. 1", "s. 1(a)", True),
        ("s. 1", "s. 1a", False),
        ("s. 1", "ss. 1", False),
        ("rat", "rate rat", True),
        ("", "anything", False),
        ("Café", "le café noir", True),
        ("café", "cafés", False),
    ],
)
def test_ret_08_name_in_reference_needs_a_boundary_on_both_sides(
    name: str, reference: str, expected: bool
) -> None:
    assert name_in_reference(name, reference) is expected
