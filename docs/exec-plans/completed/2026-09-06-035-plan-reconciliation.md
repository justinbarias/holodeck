# Feature 035 plan reconciliation

## Objective and acceptance criteria

Reconcile all six feature 035 plans and TODO records against the current repository.
Each record must distinguish implemented work, pending work, superseded decisions, and unverified acceptance criteria.
Update the feature inventory and indexes with the reconciled evidence.

## Scope and constraints

Documentation only. Do not implement pending features or run live provider or deployment checks.
Preserve historical acceptance evidence without presenting it as a new test result.
The [feature records](../active/035-openai-agents-backend/index.md) define the audit scope.

## Steps

1. Inspect each record, relevant source, tests, and Git history.
2. Update task statuses and record differences from the original design.
3. Reconcile the feature inventory, indexes, and deferred technical debt.
4. Run focused backend tests and the documentation harness check.
5. Review the complete diff and record verification results.

## Progress and decisions

- Baseline: `b585e80`, clean working tree on `main`.
- PR #338 (`3805e80`) contains the native backend, SK cleanup, and LiteLLM migration.
- Independent audits cover full parity, documentation, and SK/LiteLLM cleanup.
- Existing checkbox marks require context: implementation does not prove every live acceptance criterion.

## Validation and unresolved work

Completed all six plan/TODO audits and updated the specification status, indexes, inventory, and technical debt.
The parity TODO now records 9/43 checked items. Reopened acceptance gates explain the reduced count.
The [report](../active/035-openai-agents-backend/reconciliation.md) contains task dispositions and exact verification commands.

- Focused backend and selector tests: 128 passed.
- Focused RAG and parity helper tests: 196 passed.
- `make harness-check`: passed.
- `uv run mkdocs build --strict`: passed after the new report resolved two temporary missing-link warnings.
- `git diff --check`: passed.

No runtime code changed. Live acceptance, visual checks, and full-suite validation remain outside this audit.
Pending product work stays in the feature plans. Final SK removal and LiteLLM acceptance gaps are recorded as H-008 and H-009.
