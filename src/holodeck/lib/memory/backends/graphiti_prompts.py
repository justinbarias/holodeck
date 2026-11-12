"""HoloDeck's override of one Graphiti prompt: the contradiction decision.

Graphiti's own ``dedupe_edges.resolve_edge`` prompt is written for
conversations. It tells the model not to treat a newer fact as a replacement
unless the text says so. A dated provision replaces the earlier value without
saying so, and the default left superseded rates open (graphiti-evidence.md,
supersession probe). This prompt keeps Graphiti's inputs, outputs and
duplicate rules, and changes the contradiction rules only.

A contradiction needs one single-valued attribute of one subject with two
different values, one in each fact. The live ingest of two compilations of
one Act (2026-11-12) showed the earlier wording closing facts whose text had
not changed: a rule restated with other words and other entities, and an
unrelated definition from another section. So a restatement, a paraphrase,
the same value, or a fact about another provision is not a contradiction,
and different start dates count only after two different values are found.
``scripts/memory_resolve_edge_replay.py`` replays labelled cases against a
live model. The replay scored 45 of 67 with this wording and with the
earlier one, so the wording does not stop false closures; the backend's
closure check (``graphiti_closure``) reopens them after each episode.

Graphiti does the date arithmetic in code after this decision: a contradicted
older fact is closed at the new fact's start date. The prompt decides only
which facts conflict. Graphiti closes a contradicted fact only when both facts
have a start date and the older one starts strictly earlier; otherwise both
stay open.

The prompt is a copy to keep in step with Graphiti. Check it against
``graphiti_core/prompts/dedupe_edges.py`` on every Graphiti upgrade; a unit
test fails when Graphiti's own prompt changes.
"""

# ruff: noqa: E501 - the prompt text is kept exactly as it was validated live.

from __future__ import annotations

from collections.abc import Mapping

from graphiti_core.prompts.lib import ChatPromptLibrary, create_prompt_library
from graphiti_core.prompts.models import ChatPrompt, SystemMessage, UserMessage

RESOLVE_EDGE_SYSTEM = (
    "You are a fact deduplication assistant for dated provisions: rules, "
    "rates and responsibilities that take effect from a date and are replaced "
    "by later ones. You decide which facts are duplicates of a new fact and "
    "which facts the new fact replaces."
)

RESOLVE_EDGE_RULES = """
You will receive TWO lists of facts with CONTINUOUS idx numbering across both lists.
EXISTING FACTS are indexed first, followed by FACT INVALIDATION CANDIDATES.

IMPORTANT constraints:
- duplicate_facts: ONLY idx values from EXISTING FACTS (NEVER include FACT INVALIDATION CANDIDATES)
- contradicted_facts: idx values from EITHER list (EXISTING FACTS or FACT INVALIDATION CANDIDATES)
- The idx values are continuous across both lists (INVALIDATION CANDIDATES start where EXISTING FACTS end)

1. DUPLICATE DETECTION:
   - If the NEW FACT represents identical factual information as any fact in EXISTING FACTS, return those idx values in duplicate_facts.
   - NEVER mark facts as duplicates if they have key differences, particularly around numeric values, dates, or key qualifiers.
   - If no duplicates, return an empty list for duplicate_facts.

2. CONTRADICTION DETECTION:
   - Some attributes have ONE value at a time: a rate, amount, percentage, threshold, limit, period, or
     the meaning of a defined term. Mark a fact as contradicted ONLY when you can name ONE such attribute
     of ONE subject, and two DIFFERENT values of it: one stated in that fact and one stated in the NEW
     FACT. One of them replaced the other.
   - If the two values are the same (the same period, the same rate, the same meaning), the fact is NOT
     contradicted, however differently the two facts are worded and whatever their dates.
   - A fact that restates, paraphrases or summarises the same rule, or describes it from another angle, is
     NOT contradicted, even when it names different entities or adds detail the other fact leaves out.
   - Facts about different provisions, sections or defined terms are NOT contradicted. A definition of one
     term never contradicts a rule in another section or the definition of another term.
   - When you HAVE found two different values of one single-valued attribute, the text does NOT have to
     say "replaces", "no longer" or "instead", and different start dates on the two facts do NOT make them
     compatible: they show that the value changed.
   - Mark the fact as contradicted whichever of the two has the earlier date.
   - A role, power, function or responsibility can be held by several bodies or persons at once. Mark an
     earlier holder as contradicted ONLY when the role has one holder by its nature (the one body that
     administers a payment, the one person who holds an office) or the NEW FACT says the earlier holder
     stopped (transferred, replaced, no longer, instead of). A power that another body "may" also
     exercise does NOT contradict the first holder.
   - Do NOT mark a fact as contradicted when the two facts are about different subjects, different
     attributes of one subject, or different categories of one attribute (for example the rate for a
     single person and the rate for a couple, or a base rate and a supplement).
   - Do NOT mark a fact as contradicted when both facts can be true at the same time.
   - Return all contradicted idx values in contradicted_facts.
   - If no contradictions, return an empty list for contradicted_facts.

<EXAMPLE>
EXISTING FACT: idx=0, "From 1 July 2020 the base rate of Carer Payment is 100 dollars per week"
NEW FACT: "From 1 July 2020 the base rate of Carer Payment is 100 dollars per week"
Result: duplicate_facts=[0], contradicted_facts=[] (identical factual information)

INVALIDATION CANDIDATE: idx=1, "From 1 July 2020 the base rate of Carer Payment is 100 dollars per week"
NEW FACT: "From 1 July 2022 the base rate of Carer Payment is 120 dollars per week"
Result: duplicate_facts=[], contradicted_facts=[1] (same subject and attribute, different value: the rate changed)

INVALIDATION CANDIDATE: idx=2, "Carer Payment is administered by the Payments Office"
NEW FACT: "From 1 July 2024 the Care Agency administers Carer Payment"
Result: duplicate_facts=[], contradicted_facts=[2] (one body administers the payment at a time)

INVALIDATION CANDIDATE: idx=3, "From 1 July 2020 the couple rate of Carer Payment is 150 dollars per week"
NEW FACT: "From 1 July 2022 the base rate of Carer Payment is 120 dollars per week"
Result: duplicate_facts=[], contradicted_facts=[] (different categories of rate: both can be true)
INVALIDATION CANDIDATE: idx=4, "From 1 July 2020 the Minister may appoint inspectors under section 10"
NEW FACT: "From 1 July 2024 the Commissioner may appoint inspectors under section 10"
Result: duplicate_facts=[], contradicted_facts=[] (a power two bodies can hold at once; nothing says the Minister's power ended)

INVALIDATION CANDIDATE: idx=5, "A Class 300 (Overseas Carer) visa is excluded from the permanent visas that make a person subject to the newly arrived resident's waiting period for Carer Payment"
NEW FACT: "From 1 July 2024 a Class 300 (Overseas Carer) visa is identified by the Migration Regulations and is an exception to the newly arrived resident's waiting period for Carer Payment"
Result: duplicate_facts=[], contradicted_facts=[] (the same rule from another angle, naming other entities: no attribute has two different values)

INVALIDATION CANDIDATE: idx=6, "Section 12 applies to a person who receives a Carer Payment advance"
NEW FACT: "From 1 July 2024 care leave pay has the same meaning as in the Care Leave Act 2010"
Result: duplicate_facts=[], contradicted_facts=[] (a definition of one term and a rule in another section: different subjects)

INVALIDATION CANDIDATE: idx=7, "Benefits Secretary means the Secretary of the Department administered by the Minister for Social Services"
NEW FACT: "From 1 July 2024 Benefits Secretary means the Chief Executive Officer of the Care Agency"
Result: duplicate_facts=[], contradicted_facts=[7] (one defined term, two different meanings: the definition changed)
</EXAMPLE>
"""


def resolve_edge(context: Mapping[str, object]) -> ChatPrompt:
    """Build the contradiction prompt from Graphiti's ``resolve_edge`` context."""
    user = f"""
<EXISTING FACTS>
{context['existing_edges']}
</EXISTING FACTS>

<FACT INVALIDATION CANDIDATES>
{context['edge_invalidation_candidates']}
</FACT INVALIDATION CANDIDATES>

<NEW FACT>
{context['new_edge']}
</NEW FACT>
{RESOLVE_EDGE_RULES}"""
    return ChatPrompt(
        system=SystemMessage(content=RESOLVE_EDGE_SYSTEM),
        user=UserMessage(content=user),
    )


def holodeck_prompt_library() -> ChatPromptLibrary:
    """Graphiti's prompts with HoloDeck's contradiction prompt in place."""
    return create_prompt_library({"dedupe_edges": {"resolve_edge": resolve_edge}})
