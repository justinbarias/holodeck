"""The closure check's rules, without a graph (spec 043, design seam 3).

The backend tests in ``test_graphiti_backend_write.py`` run the same rules
through ``ingest`` on a fake driver.
"""

from __future__ import annotations

import pytest

from holodeck.lib.memory.backends.graphiti_closure import (
    MIN_LINE_MATCH,
    SUPPORT_MARGIN,
    EdgeShape,
    SectionKey,
    could_replace,
    decide_closure,
    decide_text_closure,
    fold_relation,
    heading_key,
    same_section,
    section_lines,
    tokens,
)

pytestmark = [pytest.mark.unit]

PATH = ("Part 2—Interpretation", "Division 1—Definitions")


def test_tokens_are_long_words_and_numbers_with_separators() -> None:
    found = tokens("The rate is $2,332.35 for 182 days under s. 61AA of the Act.")

    assert found == frozenset(
        {"the", "rate", "2,332.35", "for", "182", "days", "under", "61aa", "act"}
    )


@pytest.mark.parametrize(
    ("old", "new", "same"),
    [
        (SectionKey("s3", "3", PATH), SectionKey("s3", "3", PATH), True),
        # A renamed heading keeps its number and parents.
        (
            SectionKey("old57gj", "57GJ", PATH),
            SectionKey("new57gj", "57GJ", PATH),
            True,
        ),
        (SectionKey("s3", "3", PATH), SectionKey("s3a", "3A", PATH), False),
        (SectionKey("s3", "3", PATH), SectionKey("s3aa", "3AA", PATH), False),
        (
            SectionKey("s57", "57GJ", PATH),
            SectionKey("t57", "57GJ", ("Part 9",)),
            False,
        ),
        # No number: only the id counts.
        (SectionKey("a", "", PATH), SectionKey("b", "", PATH), False),
        # A positional id names a chunk, not a section: equal ids with other
        # numbers or other parents are other sections.
        (
            SectionKey("chunk_12", "3", PATH),
            SectionKey("chunk_12", "4", PATH),
            False,
        ),
        (
            SectionKey("chunk_12", "3", PATH),
            SectionKey("chunk_12", "3", ("Part 9",)),
            False,
        ),
        (SectionKey("chunk_12", "", ()), SectionKey("chunk_12", "", ()), False),
        # Different positions, the same number and parents: the same section.
        (SectionKey("chunk_12", "3", PATH), SectionKey("chunk_40", "3", PATH), True),
        # "chunk_12a" is a heading id, not a position.
        (SectionKey("chunk_12a", "", ()), SectionKey("chunk_12a", "", ()), True),
    ],
)
def test_same_section(old: SectionKey, new: SectionKey, same: bool) -> None:
    assert same_section(old, new) is same


def test_lines_drop_headings_and_table_rules_and_keep_table_rows() -> None:
    text = (
        "#### 3 Rates\n\n(1) The rate is:\n\n| Item | Rate |\n|---|---|\n"
        "| 1 | $2,332.35 |\n(2) It is indexed.  It   applies yearly."
    )

    assert section_lines([text]) == [
        "(1) The rate is:",
        "| Item | Rate |",
        "| 1 | $2,332.35 |",
        "(2) It is indexed.",
        "It applies yearly.",
    ]


def test_a_changed_table_row_keeps_a_fact_about_it_closed() -> None:
    old = "| Item | Rate |\n| 1 | $2,332.35 a year |\n| 2 | $182 a fortnight |"
    new = "| Item | Rate |\n| 1 | $2,400.00 a year |\n| 2 | $182 a fortnight |"
    section = SectionKey("s3", "3", PATH)

    def decide(fact: str) -> bool:
        return decide_closure(
            fact=fact,
            old_section=section,
            old_texts=[old],
            new_section=section,
            new_texts=[new],
        ).reopen

    assert decide("The item 1 rate is $2,332.35 a year") is False
    assert decide("The item 2 rate is $182 a fortnight") is True


def test_a_fact_with_no_word_in_common_keeps_its_closure() -> None:
    section = SectionKey("s3", "3", PATH)

    decision = decide_closure(
        fact="Zebras graze",
        old_section=section,
        old_texts=["(1) The rate is 10."],
        new_section=section,
        new_texts=["(1) The rate is 12."],
    )

    assert (decision.reopen, decision.reason) == (False, "no_matching_line")


def test_an_old_section_with_no_text_keeps_the_closure() -> None:
    section = SectionKey("s3", "3", PATH)

    decision = decide_closure(
        fact="The rate is 10",
        old_section=section,
        old_texts=["#### 3 Rates"],
        new_section=section,
        new_texts=["(1) The rate is 12."],
    )

    assert (decision.reopen, decision.reason) == (False, "no_old_text")


def test_an_unchanged_section_reopens_a_fact_that_matches_no_line() -> None:
    # Rule (b) alone: the fact shares no word with the text, so rule (c)
    # could not reopen it; the section is unchanged, so it is reopened.
    section = SectionKey("s3", "3", PATH)
    text = "#### 3 Rates\n(1) The rate is 10."

    decision = decide_closure(
        fact="Zebras graze",
        old_section=section,
        old_texts=[text],
        new_section=section,
        new_texts=["(1) The rate\n  is 10.", "(2) A new subsection."],
    )

    assert (decision.reopen, decision.reason) == (True, "section_unchanged")


def test_an_unknown_old_section_is_not_taken_for_another_section() -> None:
    decision = decide_closure(
        fact="The rate is 10",
        old_section=None,
        old_texts=["(1) The rate is 10."],
        new_section=SectionKey("s9", "9", PATH),
        new_texts=["(1) The rate is 12."],
    )

    assert (decision.reopen, decision.reason) == (False, "line_changed")


# --- the replacement guard (getzep/graphiti#1729) ---------------------------

USES_AB = EdgeShape("A", "B", "USES")


@pytest.mark.parametrize(
    ("new", "replaces"),
    [
        # B is the target of one and the source of the other.
        (EdgeShape("B", "C", "USES"), False),
        (EdgeShape("A", "B", "OWNS"), True),
        (EdgeShape("B", "A", "USED_BY"), True),
        (EdgeShape("A", "C", "uses"), True),
        (EdgeShape("A", "C", "Uses "), True),
        (EdgeShape("D", "B", "u-s_e s"), True),
        (EdgeShape("A", "C", "OWNS"), False),
        (EdgeShape("C", "A", "USES"), False),
        (EdgeShape("C", "D", "USES"), False),
    ],
)
def test_could_replace(new: EdgeShape, replaces: bool) -> None:
    assert could_replace(USES_AB, new) is replaces


def test_fold_relation_drops_case_and_separators() -> None:
    assert fold_relation("Has_Rate-2 ") == fold_relation("HASRATE2") == "hasrate2"


def test_other_section_is_decided_before_no_replacement() -> None:
    old = SectionKey("s3", "3", PATH)
    other = SectionKey("s4", "4", PATH)
    same = SectionKey("s3", "3", PATH)
    text = ["#### 3 Definitions\n(1) The rate is $10."]

    elsewhere = decide_closure(
        fact="The rate is $10",
        old_section=old,
        old_texts=text,
        new_section=other,
        new_texts=["The rate is $20."],
        replaced=False,
    )
    unreplaced = decide_closure(
        fact="The rate is $10",
        old_section=old,
        old_texts=text,
        new_section=same,
        new_texts=["The rate is $20."],
        replaced=False,
    )
    replaced = decide_closure(
        fact="The rate is $10",
        old_section=old,
        old_texts=text,
        new_section=same,
        new_texts=["The rate is $20."],
    )

    assert (elsewhere.reopen, elsewhere.reason) == (True, "other_section")
    assert (unreplaced.reopen, unreplaced.reason) == (True, "no_replacement")
    assert (replaced.reopen, replaced.reason) == (False, "line_changed")


# --- parent headings compare by their designators ---------------------------

PART_8 = "Part 8—Security"
DIV_7_OLD = "Division 7—Loss of family assistance for individuals on security grounds"
DIV_7_NEW = "Division 7—Loss of family assistance for individuals"


def test_a_section_renamed_under_a_renamed_division_is_the_same_section() -> None:
    # The live 57GJ case: the section and its Division 7 were both renamed.
    old = SectionKey(
        "h4_part_8_division_7_57gj_security_notice", "57GJ", (PART_8, DIV_7_OLD)
    )
    new = SectionKey(
        "h4_part_8_division_7_57gj_benefit_restriction", "57GJ", (PART_8, DIV_7_NEW)
    )

    assert same_section(old, new)


def test_another_division_number_is_another_section() -> None:
    old = SectionKey("a", "57GJ", (PART_8, DIV_7_OLD))
    new = SectionKey("b", "57GJ", (PART_8, "Division 8—Loss of family assistance"))

    assert not same_section(old, new)


@pytest.mark.parametrize(
    ("heading", "key"),
    [
        (DIV_7_OLD, "division 7"),
        ("Part 3A—Rates", "part 3A"),
        ("Subdivision AA—Rates", "subdivision AA"),
        ("SCHEDULE 2 Rates", "schedule 2"),
        ("Chapter 4", "chapter 4"),
        # No designator: the text, with whitespace normalised.
        ("Part of  the Act", "Part of the Act"),
        ("Interpretation", "Interpretation"),
    ],
)
def test_heading_key(heading: str, key: str) -> None:
    assert heading_key(heading) == key


def test_headings_without_a_designator_compare_by_text() -> None:
    old = SectionKey("a", "3", ("Preliminary", "Definitions"))
    renamed = SectionKey("b", "3", ("Preliminary", "Interpretation"))
    equal = SectionKey("c", "3", ("Preliminary", "Definitions"))

    assert not same_section(old, renamed)
    assert same_section(old, equal)


# --- decide_text_closure -----------------------------------------------------

FACT = "alpha bravo charlie delta"


def test_text_closure_thresholds_are_the_measured_starting_values() -> None:
    assert (MIN_LINE_MATCH, SUPPORT_MARGIN) == (0.2, 0.1)


def test_text_closure_a_line_at_the_minimum_match_counts_as_the_source() -> None:
    # 1 shared token of 5: exactly 0.2.
    decision = decide_text_closure(
        fact="alpha bravo charlie",
        old_lines=["alpha echo foxtrot."],
        new_text="zulu yankee.",
    )

    assert decision.old_best == 0.2
    assert (decision.close, decision.reason) == (True, "line_changed")


def test_text_closure_a_line_below_the_minimum_match_is_not_the_source() -> None:
    # 1 shared token of 6: under 0.2.
    decision = decide_text_closure(
        fact="alpha bravo charlie",
        old_lines=["alpha echo foxtrot golf."],
        new_text="zulu yankee.",
    )

    assert (decision.close, decision.reason) == (False, "no_matching_line")


def test_text_closure_new_text_at_exactly_the_margin_keeps_the_fact() -> None:
    # old_best 4/8 = 0.5; new_best 2/5 = 0.4 = 0.5 - 0.1: still supported.
    decision = decide_text_closure(
        fact=FACT,
        old_lines=["alpha bravo charlie delta echo foxtrot golf hotel."],
        new_text="alpha bravo india.",
    )

    assert (decision.old_best, decision.new_best) == (0.5, 0.4)
    assert (decision.close, decision.reason) == (False, "still_supported")


def test_text_closure_new_text_just_past_the_margin_closes_the_fact() -> None:
    # old_best 0.5; new_best 3/8 = 0.375 < 0.4.
    decision = decide_text_closure(
        fact=FACT,
        old_lines=["alpha bravo charlie delta echo foxtrot golf hotel."],
        new_text="alpha bravo charlie india juliet kilo lima.",
    )

    assert decision.new_best == 0.375
    assert (decision.close, decision.reason) == (True, "line_changed")


def test_text_closure_a_best_line_still_in_the_new_text_keeps_the_fact() -> None:
    kept = "alpha bravo echo."
    changed = "charlie delta echo."
    decision = decide_text_closure(
        fact=FACT, old_lines=[kept, changed], new_text=f"{kept}\nzulu."
    )

    # A tie between a kept and a changed best line keeps the fact open.
    assert (decision.close, decision.reason) == (False, "line_kept")


def test_text_closure_no_old_lines_closes_nothing() -> None:
    decision = decide_text_closure(fact=FACT, old_lines=[], new_text="alpha.")

    assert (decision.close, decision.reason) == (False, "no_old_text")


def test_text_closure_a_removed_clause_the_fact_never_named_is_within_the_margin() -> (
    None
):
    # Old 6/9 = 0.667; new 5/8 = 0.625, within 0.1: the fact stays open.
    decision = decide_text_closure(
        fact="Subsection (4) does not apply to the waiting period",
        old_lines=["(c) subsection (4) does not apply to the individual;"],
        new_text="(c) subsection (4) does not apply;",
    )

    assert (decision.close, decision.reason) == (False, "still_supported")


def test_text_closure_an_added_qualifier_that_changes_the_words_closes() -> None:
    # "subsections ... do" no longer share "subsection" and "does": 3/11
    # against 5/8. The fact is still true; lexical matching closes it.
    decision = decide_text_closure(
        fact="Subsection (4) does not apply to the waiting period",
        old_lines=["(c) subsection (4) does not apply;"],
        new_text="(c) subsections (4) and (4A) do not apply;",
    )

    assert (decision.old_best, decision.new_best) == (0.625, pytest.approx(3 / 11))
    assert (decision.close, decision.reason) == (True, "line_changed")


def test_a_later_source_and_another_document_are_decided_first() -> None:
    section = SectionKey("s3", "3", PATH)
    text = ["#### 3 Definitions\n(1) The rate is $10."]

    def decide(**flags: bool) -> tuple[bool, str]:
        decision = decide_closure(
            fact="The rate is $10",
            old_section=section,
            old_texts=text,
            new_section=section,
            new_texts=["The rate is $20."],
            **flags,
        )
        return decision.reopen, decision.reason

    assert decide(restated_later=True, other_document=True) == (
        True,
        "restated_later",
    )
    assert decide(other_document=True, replaced=False) == (True, "other_document")
    assert decide() == (False, "line_changed")
