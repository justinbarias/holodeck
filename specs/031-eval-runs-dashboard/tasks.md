---
description: "Task index — 031-eval-runs-dashboard"
---

# Tasks: Eval Runs, Prompt Versioning, and Test View Dashboard

**Feature**: 031-eval-runs-dashboard
**Prerequisites**: plan.md (required), spec.md (required — 5 user stories), research.md, data-model.md, contracts/

**Tests**: TDD is in scope. Every user-story task file writes failing tests first (tasks marked "(TDD)") and implementation only after.

**Organization**: One task file per user story. Each file is independently executable and carries its own Setup / Foundational / Tests / Implementation phases. Shared setup is duplicated deliberately so any single story can be picked up without cross-referencing siblings.

## ⭐ Design handoff is authoritative for US4 and US5

The `design_handoff_holodeck_eval_dashboard/` bundle is the **primary source of truth** for the dashboard (US4 + US5). When `spec.md` and the handoff differ on visual/interaction details, **the handoff wins** — the FR-032 4KB collapse threshold becomes 500B per `explorer.js:156`; the single "Summary + Explorer" spec expands to a three-view app (Summary · Explorer · Compare) per the handoff README. The handoff's HTML prototype (`Evaluation Dashboard.html`) is the ground-truth reference for every pixel and interaction.

**Every US4/US5 task cites specific handoff line ranges** (e.g. `summary.js:173–175`, `compare.js::CompareTray:512–549`). Those citations are mandatory reading before implementation. Phase 8 of each file uses the Chrome MCP tools (`mcp__claude-in-chrome__*`) to drive a real browser and diff the live Streamlit app against the HTML prototype — visual delta at merge time is a blocker.

## Per-Story Task Files

| Priority | Story | Goal | File |
|---|---|---|---|
| P1 🎯 MVP | US1 | Persist strongly-typed `EvalRun` per `holodeck test` invocation (atomic write, redaction, slugified path) **+ runtime-shape migrations** (`MetricResult.kind`, `TestResult.tool_events`, `TestResult.conversation`) that unblock the dashboard's real-run rendering | [tasks-us1.md](./tasks-us1.md) |
| P1 | US2 | Prompt versioning via YAML frontmatter (`python-frontmatter`) with `auto-<sha256[:8]>` fallback | [tasks-us2.md](./tasks-us2.md) |
| P1 | US3 | `EvalRun.metadata.agent_config` is a complete, frozen, round-trippable snapshot of the validated `Agent` | [tasks-us3.md](./tasks-us3.md) |
| P2 | US4 | Scaffold Streamlit app per handoff (three-view shell, terminal-green theme, seed data ported from `data.js`, Plotly charts) + **Summary** view with KPI strip, pass-rate chart with regression dots and version boundaries, metric trend with kind toggle, three breakdown panels, filterable runs table. Chrome MCP visual-parity sweep. | [tasks-us4.md](./tasks-us4.md) |
| P2 | US5 | **Explorer** (3-column drilldown per `explorer.js`: runs/cases/detail with agent-config, chat-style conversation, amber-tinted tool-call panels, expected-tools coverage, evaluations-by-kind) + **Compare** (3-variant layout per `compare.js`: side-by-side, baseline+deltas, matrix-first; compare tray; heatmap case matrix; delta pills with inverted polarity). Chrome MCP visual-parity sweep. | [tasks-us5.md](./tasks-us5.md) |

## Cross-Story Dependencies

```
Phase 1 — Setup & Foundational (each story owns its own setup/foundational tasks)
  US1 Phase 2:  SecretStr migration (BLOCKS US1 AC3)
  US1 Phase 2b: Runtime-shape migrations — MetricResult.kind + TestResult.tool_events +
                TestResult.conversation (BLOCKS US4/US5 real-run rendering)
    ↓
  US1 Core (writer + models + CLI wiring with fail-loud stub guard)
    ↓                 ↓
  US2 (replaces      US3 (snapshot fidelity — hardens US1's Agent serialization)
       US1 stub
       call)
    ↓
  US4 (scaffold + Summary — works against seed immediately; real runs require US1 Phase 2b)
    ↓
  US5 (Explorer + Compare — extends US4; real-run Conversation panel requires US1 Phase 2b)
```

## MVP

**MVP scope = US1 only.** After US1 ships:
- Developers can diff run JSON files in their editor to spot regressions.
- Every subsequent `holodeck test` invocation builds experiment history automatically.
- US2/US3/US4/US5 each deliver independently valuable increments on top.

## Parallel Team Strategy

Two developers can parallelize after US1's Foundational phase (T001–T010):
- Developer A finishes US1 core (T011–T032) then picks up US3.
- Developer B starts US2 (T101–T123) in parallel — US2's only hard dep on US1 is the `PromptVersion` stub location (US1 T024).

A third developer can start US4 as soon as US1 ships any persisted `EvalRun` files (even fixtures). US5 depends on US4 infrastructure and should be sequential to US4.

## Polish & Cross-Cutting (after all stories land)

- [ ] TPOL0 [P] Commit `specs/031-eval-runs-dashboard/visual-baselines/` — screenshots + `dashboard_tour.gif` captured via Chrome MCP during US4/US5 Phase 8. These are the durable visual-parity evidence against the handoff prototype
- [ ] TPOL1 [P] Update `docs/` with: persistence layout, dashboard usage, frontmatter schema, redaction policy, dashboard visual-fidelity approach (reference handoff as the authoritative source)
- [ ] TPOL2 [P] Update `CLAUDE.md` "Recent Changes" and "Active Technologies" sections with the new deps (`python-frontmatter`, `streamlit[dashboard]`)
- [ ] TPOL3 Run quickstart.md end-to-end; tick every checkbox in §9 Success checklist
- [ ] TPOL4 Performance check: measure persistence overhead per `holodeck test` invocation and assert <200 ms (SC-008)
- [ ] TPOL5 Performance check: populate 1000 synthetic run files, launch dashboard, assert Summary first-render <5 s P95 (SC-010)
- [ ] TPOL6 [P] `make format && make lint && make type-check && make security` — all clean
- [ ] TPOL7 Coverage check: persistence writer, redactor, PromptVersion resolver, slugifier, data loader, filters, view CLI — each ≥80% (plan.md §Constitution Check)

## Notes

- Every task file carries its own Acceptance Scenario Traceability table so story-level test coverage can be audited without reading this index.
- `tasks.md` is intentionally thin — the per-story files are the authoritative execution lists.
- Per `.specify/templates/tasks-template.md`, every task uses the strict checklist format: `- [ ] TID [P?] [US#] description with file path`.
