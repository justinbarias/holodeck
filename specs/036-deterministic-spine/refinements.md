# Refinement Memo — Deterministic Spine (036)

> **ARCHIVED (2026-08-29).** Part of the archived 036 spec — see the
> banner in `spec.md` and the successor `specs/040-holodeck-temporal/spec.md`.


> Date: 2026-06-10 · Produced by an idea-refine interview run against
> `spec.md` (Draft for review). The spec is **not** edited by this memo;
> fold these resolutions in manually (suggested deltas below are precise
> enough to apply verbatim).

## Why these six questions

The spec was already sharp, so the refine run skipped re-divergence and
interviewed only where (a) the codebase contradicts a spec assumption, or
(b) the spec text is genuinely ambiguous. Codebase grounding:

- `ExecutionResult.structured_output` exists (`src/holodeck/lib/backends/base.py`),
  but **only the Claude backend produces and validates it**; the SK backend
  has no structured-output path at all.
- `TestExecutor` (`src/holodeck/lib/test_runner/executor.py`) is hard-wired
  to agent invocation — there is no seam for asserting a pure function's
  verdict.
- **No confidence convention exists anywhere** in the codebase.
- `workflow` is a free CLI verb (resolves spec Open Question 4). No prior
  FEEL/DMN/SpiffWorkflow dependency — clean slate.
- Artifact conventions favor `.holodeck/` (`deployments.json`, optimizer
  `output_dir/<run-id>/trials.jsonl`) — supports `.holodeck/runs/<id>.json`
  for the run record (spec Open Question 2, suggestion only, decide in plan).

## Decisions

### 1. Edge backends — Claude-only for the POC

SK structured output is a hidden dependency the spec didn't price in.
**Decision:** edge nodes still dispatch via `BackendSelector` (the protocol
contract stands), but the POC validates the **Claude backend only**. SK
structured-output support becomes a named follow-up, not an implicit
prerequisite.

- **Delta — FR-006:** append: "The POC validates the Claude backend only;
  SK-backend structured output is a tracked follow-up, not a POC requirement."
- **Delta — Open Question 3:** resolved — binding is confirmed for Claude
  (`Agent.response_format` → `structured_output`, already schema-validated
  in `claude_backend.py`); SK is out of POC scope.

### 2. Human node — the table computes a *recommendation*

The spec left the human node's DMN table semantically dangling (a table
with a hit policy that's never evaluated would be dead config).
**Decision:** at a human node the table **is evaluated** over the composed
inputs and produces a **recommended verdict**. The CLI presents:
composed inputs → table recommendation (with matched rule) → AI-drafted
reasons (advisory) → prompt. The human **accepts or overrides**; an
override is recorded as an override (recommended vs. decided, both in the
run record). The human's choice is the verdict, always.

- **Delta — FR-014:** the CLI presents the composed inputs, **the table's
  recommended verdict and matched rule**, and (if declared) AI-drafted
  reasons, then prompts for the human selection.
- **Delta — FR-015:** the run record stores both the table's recommendation
  and the human's decision; when they differ, the verdict is flagged as an
  override.
- **Delta — US3 acceptance scenarios:** add: "**Given** the table recommends
  `refer` and the human selects `decline`, **Then** the verdict is `decline`
  and the record marks it as an override of the recommendation."
- **Replay note:** replay re-evaluates the table (recomputing the
  recommendation from snapshots) and re-applies the *recorded* human
  decision — identical verdict guaranteed, override flag reproduced.

### 3. Policy tests — a new `WorkflowTestExecutor`, same shape

`TestExecutor` won't be refactored inside a POC. **Decision:** introduce a
**`WorkflowTestExecutor`** that mirrors the existing executor's shape
(TestCaseModel-like YAML cases, TestResult-like outputs, same AAA flow)
but evaluates a decision table over given inputs instead of invoking an
agent. No agent, no backend, no LLM.

- **Delta — FR-023:** "…via a `WorkflowTestExecutor` that mirrors the
  existing test/eval framework's case and result shapes" (replaces "via the
  existing HoloDeck test/eval framework", which implied reusing
  `TestExecutor` as-is).
- Convergence of the two executors is North Star, not POC.

### 4. Confidence gating — cut from the POC

No convention exists, and inventing one means trusting self-reported LLM
confidence. **Decision:** gates validate **schema only** (types, required
fields, enums, ranges). Confidence-threshold rejection/routing moves to
Out of Scope.

- **Delta — Edge Cases:** delete the "Low-confidence edge output" bullet.
- **Delta — Out of Scope:** add "Confidence-scored gate fields /
  low-confidence routing (no convention exists; self-reported confidence is
  unreliable). *North Star.*"
- **Delta — Open Question 3:** drop the "low-confidence-field convention"
  clause.

### 5. FEEL — the syntax is the contract; the subset may shrink

**Decision:** decision-table cells must be valid **FEEL syntax**. If the
spike finds no Python evaluator covering the full FR-010 subset, the subset
narrows further (e.g. drop date difference before dropping ranges) but
stays FEEL-compatible so a fuller evaluator can swap in later. **The sample
bends to the subset, never the reverse.** No pragmatic non-FEEL expression
language fallback.

- **Delta — FR-010:** keep the NEEDS CLARIFICATION (library choice is still
  the plan-phase spike) but add the fallback posture: "If no library covers
  the full subset, the subset narrows (remaining FEEL-syntax-compatible);
  non-FEEL expression syntaxes are not an acceptable fallback."
- **Delta — SC-005:** caveat "…or the documented narrowed subset, with the
  sample adjusted accordingly."

### 6. Human identity — declared in `workflow.yaml`, confirmed at the prompt

**Decision:** the human node declares **`decided_by`** (a role or name) in
`workflow.yaml`. At the prompt, the CLI displays the declared `decided_by`
and asks the operator to confirm or enter their name; the confirmed name is
recorded verbatim in the run record with a timestamp. This models
accountability ahead of time while staying honest that a CLI POC makes no
authentication claim.

- **Delta — FR-002:** human node gains optional `decided_by`.
- **Delta — FR-015:** "…attributed to the named human" becomes "…attributed
  to the `decided_by` identity declared in the workflow and confirmed (or
  entered) at the CLI prompt; no authentication is claimed in the POC."
- **Delta — Key Entities (RunRecord):** add `decided_by` (declared +
  confirmed) alongside the human decision.

## Still open (unchanged, resolve in `/speckit.plan`)

1. Which FEEL evaluator to embed — the spike remains the gating risk
   (posture now fixed by Decision 5).
2. Run record JSON shape; location suggestion: `.holodeck/runs/<id>.json`.

## Not doing (added by this refinement)

- SK-backend structured output (Decision 1) — follow-up feature.
- Confidence gating in any form (Decision 4) — North Star.
- Refactoring `TestExecutor` (Decision 3) — North Star convergence.
- Any non-FEEL expression fallback (Decision 5) — the standards claim is
  the point.
