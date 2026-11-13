---
version: 1
slug: "sample-copilotkit-src-app-page-tsx"
primary_target: "sample/copilotkit/src/app/page.tsx"
related_targets: ["src/holodeck/dashboard/app.py"]
---

## Scope

Mode: Operate. Two surfaces, one world: the CopilotKit reference chat (`sample/copilotkit`, primary) and the `holodeck test view` Dash dashboard (`src/holodeck/dashboard`). Audience: agent builders and domain reviewers (PRODUCT.md). Task: ask an agent and trust its evidence; find regressions across runs and prompt versions. Constraints: the chat stays generic for any agent; saved results never look live; pass/fail never by colour alone. Reference mockup: `.impeccable/mocks/prompt-book.html` (user-approved over the compiled-register alternate).

## Direction contract

THESIS: Every run is a rehearsal and the screen is the stage manager's prompt book. Tool calls are numbered cues in a ruled margin, prompt versions are coloured revision pages. It refuses the category default of a neutral sidebar chat with grey tool chips and a neon-on-black dev dashboard.

OWN-WORLD: House-blue rail and header ink (#1d2347) over script-white pages (#f5f6f9 ground, #fff sheets, #dde0e8 rules, #15171e ink). Script revision colours in their industry order (blue, pink, yellow, green, goldenrod) as tinted tabs with dark same-hue ink, used only for versions and compilations. Cue lights for state: go green, standby red, amber warn, always with icon and words. Public Sans for UI, Courier Prime only for cues, tool names, IDs and numbers that are data.

STORY: The reviewer sees which cue produced which part of the answer and which revision a fact belongs to; the engineer sees which revision regressed and how long each cue took. Both act without asking each other.

FIRST VIEWPORT: Chat: 264px house-blue rail (brand, agent card with live light, threads with cue counts); white header with thread title and the revision tabs in play; conversation on a 76px cue margin (Q1, Q2 with status lights and brackets) beside an 860px column; tool cards with name, summary, status and a duration bar; memory timeline as revision-coloured lanes with superseded facts struck and stamped; composer showing the next cue number. Dashboard: same rail (views, prompt-version filters as revision tabs, "saved results, nothing runs here"); summary strip of four figures with deltas; pass-rate chart with revision bands and red regression rings; runs table numbered R24…

FORM: Theatre prompt book, candidate 3 of 7 on my ordered list (assigned by the roll). Raises: duration bars on one time scale (Labanotation); nothing disappears, it cancels (ticket wallet); data density with tabular numerals (datamatics); fixed name plate per test case (character catalog). seed:652ade2d7d93a420

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

## Spec and requirements

- Approved design canvas (the build spec for both surfaces): https://claude.ai/artifact/Mk2WJnqcRtTGBqxYhKS9i3 . The new DESIGN.md MUST link to it (user requirement).
- Critique 2026-11-13 (29/40) P1 fixes are already in the canvas: revision colours only for prompt versions and compilations (nth version = nth colour; source documents use lettered neutral chips); AA colours (status text #17703c, secondary text at least #6b7080, control borders #8a90a3); dashboard opens with a verdict sentence and a numbered regressions list plus a Δ vs previous column; plain-word copy (Running, not Standby), a "How to read this" key in the chat header, config collapsed in Explorer, unexpected tool calls styled as warnings.
- Open P2 for the build: keyboard model (roving tabindex in Explorer lists, / to search), 44px targets, narrow-width layouts, tool-card headers wrap or ellipsize.

## Adaptations

- 2026-11-14, header revision tabs (Main.html "1 Jul 2020 compilation" / "1 Jul 2026 compilation"): not built in the chat. Scope says "the chat stays generic for any agent", and no tool result in the sample carries a version field it could read. `family_assistance_act_search` (a `hierarchical_document` tool) returns `SearchResult.format()` plain text: "Score | Source: <path>", "Location", "Section" and the chunk text (src/holodeck/lib/hybrid_search.py). The compilation number and `effective_date` exist only in each document's front matter (sample/temporal-memory/fetch_corpus.py, `front_matter`), not as result fields. The memory results (`memory_timeline`, `memory_search`, `memory_provenance`; schemas/memory) have no `compilation`, `compilation_date` or `version` field either. A rule keyed on those names would never fire on real data, and reading compilation numbers out of file paths would tie the generic chat to one agent's corpus. Source documents keep the lettered neutral chips. The revision colours stay reserved for a result that carries a version field, and for the dashboard's prompt versions.
