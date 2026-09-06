# Spec and execution-plan drift reconciliation

**Date:** 2026-09-06. **Owner:** Feature maintainers. **Debt closed:** H-001.

## Objective

Make the spec inventory and the execution-plan index say what the code and merge history say.
Shipped features stop reading as pending. Work that the two pivots made obsolete stops reading as open.
No task checkbox is ticked or unticked by this pass; historical task lists remain as recorded.

## Evidence used

- Code: CLI surface (`holodeck --help`), `lib/backends/selector.py` routing, `lib/evaluators/deepeval/`, `lib/hybrid_search.py`, `lib/definition_extractor.py`, `serve/server.py` health and session endpoints, `deploy/deployers/`, `lib/observability/exporters/`.
- Merges: PR #308 (032), #309 (029), #335 (033 MVP), #338 (035 MVP), #345 (038), stack #369 (040), stack #376–#380 (035 T0–T3).
- Pivots: Semantic Kernel agent execution replaced by native SDKs (021, 035); the 036 workflow engine replaced by Temporal (040).

## Dispositions

| Spec | Disposition | Reason |
| --- | --- | --- |
| 001, 004, 005, 006, 007, 008, 011, 012, 013, 021, 029, 032, 038 | `shipped`; plans moved to `completed/` | Feature in daily use; open boxes are historical verification or file checklists |
| 014, 017, 018, 019, 020 | `shipped`; plans moved to `completed/` | Core shipped; unbuilt user stories named in the inventory notes and tracked in GitHub issues |
| 026, 027, 028, 030 | `shipped` | Delivered inside 021 and 035; no execution plan of their own |
| 040 | `shipped` | Was `complete`; vocabulary normalised |
| 009, 010 | `archived`; plans moved to `completed/` | Open tasks targeted Semantic Kernel `AgentFactory` and SK MCP plugins, which no longer exist |
| 015, 016 | `archived`; plans moved to `completed/` | Reranking folded into 020 (#252); GraphRAG has research only |
| 023 | `archived`; plan moved to `completed/` | Google ADK and Microsoft Agent Framework backends contradict the two-native-SDK stance (035, 042) |
| 039 | `archived` | Depended on 036; the workflow-engine surfaces it targeted were removed by 040 |
| 024, 031, 033, 034, 035 | `pending`; remain in `active/` | Real open scope with current plans |
| 037, 041, 042 | `draft` | Specs without execution plans |

## Related corrections

- Spec `**Status**` lines updated for the 28 affected specs so a spec no longer says Draft when it shipped or was superseded.
- Duplicate debt identifier: the deferred Envoy profile is now H-022 (it shared H-010 with the Qdrant record-key gap).
- Migration ledger destinations and buckets updated for the 68 moved files.
- Feature indexes under `product-specs/` and `design-docs/` relink to the new plan locations.

## Not changed

- README roadmap and changelog release headers (release decision, not spec drift).
- Semantic Kernel retrieval-layer removal remains H-008 and spec 042.
- Open GitHub issues for the unbuilt user stories remain open; the inventory names them.

## Verification

`make harness-check` and `uv run pytest tests/unit/test_harness.py tests/unit/test_mkdocs_harness.py -n auto -q` recorded below.
