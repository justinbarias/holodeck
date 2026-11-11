"""The closure check: is a fact Graphiti closed still stated by the source?

Graphiti closes an older fact when its model marks it contradicted by a new
one. The live ingest of two compilations of one Act (2026-11-12) showed the
model closing facts whose source text had not changed: about 41 of 63
closures were false. A replay of labelled pairs gave the same accuracy with
the old and the stricter prompt (45 of 67), so wording is not the fix. The
backend runs this check after each episode instead. It is deterministic and
reads only text that is already in the graph.

For a fact F that Graphiti closed while writing episode E, with O the
earliest episode other than E that states F:

l. An episode dated after E states F: reopen. E is an older version
   ingested after a newer one, and F cannot end before a later source
   still states it.
d. O is in another document lineage than E: reopen. When both documents
   have a front-matter ``series_id``, different ids are different
   documents; otherwise a different source tool is.
a. O's section is not E's section: reopen. Sections are equal when their
   ids are equal, or (a renamed heading) when their numbers and parent
   headings are equal. A positional id (``chunk_<n>``) is no section id:
   only the number and parent headings compare. Parent headings with a
   designator ("Part 3", "Division 7") compare by the designator only, so a
   renamed parent is the same parent. An unknown old section proves nothing: go on.
r. No open edge that E wrote could replace F: reopen. An edge could replace
   F when it joins the same two entities (either direction), or shares one
   endpoint in the same position and has the same relation name, folded
   (the guard of getzep/graphiti PR #1729). Graphiti picks the facts to close
   from a search over the whole namespace, so without this a fact about
   other entities can be closed.
b. Every line of O's section text is in E's section text: reopen.
c. The old line most like F (token-set Jaccard) is not in E's section text:
   F stays closed. Otherwise reopen.

Known limits: a section split into two numbers ("3" into "3" and "3A") counts
as two sections, so a real change there is reopened; a heading renamed
together with its number, or under a parent heading renamed without a
designator, counts as another section; a replacement edge that the model
gives other entities or another relation name than the closed fact does not
count, so that closure is undone;
a later chunk of E's section that is not yet in the graph is not part of the
new text, so a fact whose line moved there stays closed.

Text closing (``decide_text_closure``) is the other direction: it closes an
open fact whose source line changed and that Graphiti never proposed to
close. For a fact F stated by an older version of new episode E's section
(every earlier date, in the same document lineage), stated by no chunk of
E's section and by no episode dated after E:

1. Take the old line(s) most like F (token-set Jaccard, ``old_best``).
   ``old_best`` below ``MIN_LINE_MATCH``: keep F open (no line states it).
2. One of those lines is in E's section text: keep F open.
3. ``new_best``, F's best Jaccard against the new lines, is not below
   ``old_best - SUPPORT_MARGIN``: keep F open (the new text still states it).
4. For a best old line L, the anchor is the content words (``content_tokens``:
   no stopwords, singular forms) that F and L share; skip L when it has
   fewer than ``MIN_ANCHOR``. One new line holds every anchored word and
   adds no ``RESTRICTIVE`` word ("not", "only", "unless", ...) the anchor
   lacks: keep F open (``line_reworded``). "subsection (4) does not apply"
   reworded to "subsections (4) and (4A) do not apply" stays open; a changed
   number, a dropped or added "not", or another name closes.
5. Otherwise close F.

The matching is lexical: a changed line that F's wording still matches
about as well as the old line is kept open by the margin, and a fact
paraphrased far from its line is never closed. Step 4 can keep a fact
whose meaning changed without losing an anchored word or adding a
restrictive one ("applies to children" to "applies to children over 16").
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from holodeck.lib.memory.chunk_source import is_positional_section_id

ClosureReason = Literal[
    "restated_later",
    "other_document",
    "other_section",
    "no_replacement",
    "section_unchanged",
    "line_unchanged",
    "line_changed",
    "no_matching_line",
    "no_old_text",
]

TextClosureReason = Literal[
    "no_old_text",
    "no_matching_line",
    "line_kept",
    "still_supported",
    "line_reworded",
    "line_changed",
]

# The lowest Jaccard at which an old line counts as the fact's source line.
MIN_LINE_MATCH = 0.2
# How much worse than its old line the new text must match a fact to close it.
SUPPORT_MARGIN = 0.1

# Words that carry no content for the reworded-line test (``content_tokens``).
_STOPWORDS = frozenset(
    {
        "the", "and", "does", "did", "are", "was", "were", "has", "have", "had",
        "for", "that", "this", "with", "any", "all", "its", "from", "into",
        "under", "may", "must", "shall", "will", "been", "being",
    }
)  # fmt: skip
# Words that narrow or negate a rule. A new line that adds one the fact's
# anchored words lack does not restate the fact.
RESTRICTIVE = frozenset(
    {
        "not", "never", "nil", "none", "cannot", "without", "except", "unless",
        "only", "excluding",
    }
)  # fmt: skip
# Fewest anchored words (fact and old line share them) the reworded-line
# test needs; with fewer it proves nothing.
MIN_ANCHOR = 2

_WHITESPACE = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_HEADING_LINE = re.compile(r"^#{1,6}(\s|$)")
_ALPHANUMERIC = re.compile(r"[0-9A-Za-z]")
# The designator that opens a parent heading: "Part 3", "Division 7",
# "Subdivision A", "Schedule 2", "Chapter 4". The token is a number with
# letters after it, or one to three capital letters ("Part of" has none);
# it must end the word. Only the word ignores case.
_DESIGNATOR = re.compile(
    r"^\s*((?i:chapter|part|division|subdivision|schedule))\s+"
    r"([0-9]+[A-Za-z]*|[A-Z]{1,3})(?![0-9A-Za-z])"
)
_NOT_ALPHANUMERIC = re.compile(r"[^0-9a-z]")
# Numbers with their separators ("2,332.35", "182", "61aa") and words of
# three or more letters. Short words ("a", "of", "to") carry no topic.
_TOKEN = re.compile(r"\d+(?:[.,]\d+)*[a-z]*|[a-z]{3,}")


@dataclass(frozen=True, slots=True)
class SectionKey:
    """Where a chunk sits: its section id, number and parent headings."""

    section_id: str
    number: str = ""
    path: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ClosureDecision:
    """Whether to reopen one closed fact, and the rule that decided."""

    reopen: bool
    reason: ClosureReason


@dataclass(frozen=True, slots=True)
class TextClosureDecision:
    """Whether to close one open fact whose source line changed, and why.

    ``old_best`` and ``new_best`` are the fact's best Jaccard against the
    old and the new section lines (0 when there are none).
    """

    close: bool
    reason: TextClosureReason
    old_best: float
    new_best: float


@dataclass(frozen=True, slots=True)
class EdgeShape:
    """The endpoints and relation name of one fact edge."""

    source_uuid: str
    target_uuid: str
    name: str


def heading_key(heading: str) -> str:
    """What a parent heading is compared by: its designator, or its text.

    "Division 7—Loss of family assistance" gives "division 7"; a heading
    with no designator gives its whitespace-normalised text.
    """
    found = _DESIGNATOR.match(heading)
    if found is None:
        return _WHITESPACE.sub(" ", heading).strip()
    return f"{found.group(1).lower()} {found.group(2).upper()}"


def same_section(old: SectionKey, new: SectionKey) -> bool:
    """True when two chunks are in the same section of two versions.

    Equal ids, or equal non-empty numbers under the same parent headings (a
    renamed heading keeps its number). "3" and "3A" are different numbers.
    Parent headings are the same when their ``heading_key`` values are. A
    positional id (``chunk_<n>``, ``is_positional_section_id``) is a chunk
    position, not a section: equal positional ids prove nothing, and the
    number and parent headings decide.
    """
    if old.section_id == new.section_id and not is_positional_section_id(
        old.section_id
    ):
        return True
    if not old.number or old.number != new.number:
        return False
    return [heading_key(h) for h in old.path] == [heading_key(h) for h in new.path]


def fold_relation(name: str) -> str:
    """A relation name in lower case with every non-alphanumeric removed."""
    return _NOT_ALPHANUMERIC.sub("", name.lower())


def could_replace(old: EdgeShape, new: EdgeShape) -> bool:
    """True when ``new`` may be the fact that replaces ``old``.

    The same two entities in either direction; or the same source, or the
    same target, with the same folded relation name (getzep/graphiti#1729).
    ``A -uses-> B`` is not replaced by ``B -uses-> C``: B is the target of
    one and the source of the other.
    """
    if {old.source_uuid, old.target_uuid} == {new.source_uuid, new.target_uuid}:
        return True
    same_name = fold_relation(old.name) == fold_relation(new.name)
    if not same_name:
        return False
    return old.source_uuid == new.source_uuid or old.target_uuid == new.target_uuid


def section_lines(texts: Sequence[str]) -> list[str]:
    """Split chunk texts into whitespace-normalised lines, in order.

    A line ends at a line break or a sentence end (the chunker joins the
    sentences of a split section with spaces, so a line break is not kept
    across chunks). Markdown heading lines and lines with no letter or digit
    (a table rule) are left out. A table row is one line.
    """
    lines: list[str] = []
    for text in texts:
        for raw in text.splitlines():
            stripped = raw.strip()
            if not stripped or _HEADING_LINE.match(stripped):
                continue
            for piece in _SENTENCE_END.split(stripped):
                line = _WHITESPACE.sub(" ", piece).strip()
                if line and _ALPHANUMERIC.search(line):
                    lines.append(line)
    return lines


def tokens(text: str) -> frozenset[str]:
    """Lower-case words of three or more letters, and numbers."""
    return frozenset(_TOKEN.findall(text.lower()))


def _singular(token: str) -> str:
    """A word with a plural ending removed; a number unchanged.

    "applies" gives "apply", "subsections" gives "subsection"; "class",
    "status" and "basis" are kept.
    """
    if token[0].isdigit():
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if (
        len(token) >= 4
        and token.endswith("s")
        and not token.endswith(("ss", "us", "is"))
    ):
        return token[:-1]
    return token


def content_tokens(text: str) -> frozenset[str]:
    """``tokens`` without stopwords, words in singular form.

    Used only by the reworded-line test of ``decide_text_closure``; the
    Jaccard scores keep using ``tokens``.
    """
    return frozenset(
        _singular(token)
        for token in _TOKEN.findall(text.lower())
        if token not in _STOPWORDS
    )


def _reworded(fact: str, best_lines: Sequence[str], new_lines: Sequence[str]) -> bool:
    """True when one new line restates the fact's best old line in new words.

    The anchor is the content words the fact shares with an old line (at
    least ``MIN_ANCHOR``). One new line must hold every anchored word (a
    changed number, a dropped "not" or another name fails) and add no
    ``RESTRICTIVE`` word the anchor lacks.
    """
    wanted = content_tokens(fact)
    new_sets = [content_tokens(line) for line in new_lines]
    for line in best_lines:
        anchor = wanted & content_tokens(line)
        if len(anchor) < MIN_ANCHOR:
            continue
        for found in new_sets:
            if anchor <= found and not (found & RESTRICTIVE) - anchor:
                return True
    return False


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Token-set Jaccard similarity; 0 when both sets are empty."""
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _in_text(line: str, joined: str) -> bool:
    """True when ``line`` occurs in ``joined`` between spaces or its ends."""
    return f" {line} " in joined


def decide_closure(
    *,
    fact: str,
    old_section: SectionKey | None,
    old_texts: Sequence[str],
    new_section: SectionKey,
    new_texts: Sequence[str],
    replaced: bool = True,
    restated_later: bool = False,
    other_document: bool = False,
) -> ClosureDecision:
    """Decide whether a fact Graphiti closed should be reopened.

    Args:
        fact: The closed fact's text.
        old_section: The section of the episode the fact came from, or None
            when that episode predates section metadata.
        old_texts: That episode's text, and other old chunks of its section.
        new_section: The section of the episode that closed the fact.
        new_texts: Every chunk text of that section in the new document.
        replaced: Whether an open edge of the closing episode could replace
            the fact (``could_replace``). True skips rule (r).
        restated_later: Whether an episode dated after the closing episode
            states the fact (rule l).
        other_document: Whether the fact's origin is in another document
            lineage than the closing episode (rule d).
    """
    if restated_later:
        return ClosureDecision(reopen=True, reason="restated_later")
    if other_document:
        return ClosureDecision(reopen=True, reason="other_document")
    if old_section is not None and not same_section(old_section, new_section):
        return ClosureDecision(reopen=True, reason="other_section")
    if not replaced:
        return ClosureDecision(reopen=True, reason="no_replacement")
    old_lines = section_lines(old_texts)
    if not old_lines:
        return ClosureDecision(reopen=False, reason="no_old_text")
    joined = f" {' '.join(section_lines(new_texts))} "
    present = [_in_text(line, joined) for line in old_lines]
    if all(present):
        return ClosureDecision(reopen=True, reason="section_unchanged")
    wanted = tokens(fact)
    scores = [jaccard(wanted, tokens(line)) for line in old_lines]
    best = max(scores)
    if best == 0:
        return ClosureDecision(reopen=False, reason="no_matching_line")
    # A tie between a kept and a changed line is not resolved: the fact may
    # be about the changed one, so Graphiti's closure stands.
    if any(
        score == best and not kept for score, kept in zip(scores, present, strict=True)
    ):
        return ClosureDecision(reopen=False, reason="line_changed")
    return ClosureDecision(reopen=True, reason="line_unchanged")


def _best(wanted: frozenset[str], lines: Sequence[str]) -> float:
    """The highest Jaccard of ``wanted`` against ``lines``; 0 for no lines."""
    return max((jaccard(wanted, tokens(line)) for line in lines), default=0.0)


def decide_text_closure(
    *, fact: str, old_lines: Sequence[str], new_text: str
) -> TextClosureDecision:
    """Decide whether an open fact should close because its line changed.

    Args:
        fact: The open fact's text.
        old_lines: The lines of the old version of the section
            (``section_lines``).
        new_text: The new version of the section; chunk texts joined with
            line breaks.
    """
    wanted = tokens(fact)
    new_lines = section_lines([new_text])
    new_best = _best(wanted, new_lines)
    if not old_lines:
        return TextClosureDecision(False, "no_old_text", 0.0, new_best)
    scores = [jaccard(wanted, tokens(line)) for line in old_lines]
    old_best = max(scores)
    if old_best < MIN_LINE_MATCH:
        return TextClosureDecision(False, "no_matching_line", old_best, new_best)
    joined = f" {' '.join(new_lines)} "
    best_lines = [
        line for line, score in zip(old_lines, scores, strict=True) if score == old_best
    ]
    if any(_in_text(line, joined) for line in best_lines):
        return TextClosureDecision(False, "line_kept", old_best, new_best)
    if new_best >= old_best - SUPPORT_MARGIN:
        return TextClosureDecision(False, "still_supported", old_best, new_best)
    if _reworded(fact, best_lines, new_lines):
        return TextClosureDecision(False, "line_reworded", old_best, new_best)
    return TextClosureDecision(True, "line_changed", old_best, new_best)
