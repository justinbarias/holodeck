"""HoloDeck's override of one Graphiti prompt: the contradiction decision.

Graphiti's own ``dedupe_edges.resolve_edge`` prompt is written for
conversations. It tells the model not to treat a newer fact as a replacement
unless the text says so. A dated provision replaces the earlier value without
saying so, and the default left superseded rates open (graphiti-evidence.md,
supersession probe). This prompt keeps Graphiti's inputs, outputs and
duplicate rules, and changes the contradiction rules only.

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
   - Some attributes have ONE value at a time: a rate, amount, percentage, threshold, limit or period.
     Mark a fact as contradicted when it and the NEW FACT state DIFFERENT values of the SAME such
     attribute of the SAME subject. One of them replaced the other.
   - For these attributes the text does NOT have to say "replaces", "no longer" or "instead". Different
     start dates on the two facts do NOT make them compatible: they show that the value changed.
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
