# Migrate specifications into the harness

Date: 2026-09-06. Owner: repository maintainers. Status: complete.

## Objective

Move all 316 existing files from the repository-root specification directory into the harness knowledge base.
Preserve requirements, decisions, task completion marks, and binary assets.
Move all 30 `plan.md` files and other execution plans into the execution-plan hierarchy.

## Placement

- Product specs retain feature IDs, requirements, contracts, checklists, quickstarts, and corpus inputs.
- Design documents contain research, data models, analysis, design handoffs, and visual baselines.
- Execution-plan directories contain plans and associated task lists, including nested task directories.
- Unfinished work remains active, even when the feature inventory records an earlier release as shipped.
- Finished plans and frozen historical plans use `completed/`, with archived work explicitly identified.
- The old inventory moves into the product-spec directory without changing recorded status or task counts.

## Acceptance criteria

1. Every original file has exactly one recorded destination. Binary hashes remain unchanged.
2. Every plan and task list is reachable from the execution-plan index.
3. Local links, code/test references, ownership rules, and legacy path tooling use the new destinations.
4. The repository-root specification directory is absent, and new-feature tooling does not recreate it.
5. Harness checks, affected tests, and the documentation build pass, with historical gaps recorded explicitly.

## Progress

- Inventoried all files and recorded status evidence from the existing feature inventory.
- Delegated legacy path-tooling updates while migrating content and references locally.
- Moved all 316 files: 136 product artifacts, 71 design artifacts, 108 execution artifacts, and the original feature inventory.
- Added feature indexes that connect requirements, design records, plans, task lists, and assets.
- Updated repository references, the OpenAPI contract-test fixture path, ownership rules, and legacy shell helpers.
- Preserved all 30 `plan.md` files, including the nested plans for features 036 and 040.
- Kept unresolved work active. Marked finished work completed and explicitly identified frozen historical work as archived.
- Recorded each original path, destination, and source hash in the [migration ledger](../../generated/spec-migration.json).
- Removed all 20 repository custom commands from `.claude/commands/` and `.opencode/command/` at the user's request.
- Updated retained templates and shell diagnostics to avoid recommending the removed commands.
- Adapted publishing to display historical task labels without treating them as API cross-references.

## Decisions

This is a location migration. It does not certify that historical specs match current implementation.
Preserve original task checkboxes and status statements rather than inferring completion from file location.
Keep related contracts and binary assets together so examples and design handoffs remain usable.

## Validation

These results describe the migration before the later [workflow cleanup](2026-09-06-workflow-cleanup.md) removed the shell helpers and their tests.

- Compared the ledger with the original Git inventory: every source has one unique existing destination, and the old root directory is absent.
- Verified all seven binary hashes and all task checkbox sequences against the migration baseline.
- Verified that every original `plan.md` is under `docs/exec-plans/`.
- `make harness-check` passed: document discovery, local links, required structure, entry points, and generated schema inventory.
- The migration ran 52 focused harness, publishing, path-resolution, and OpenAPI contract tests with `-n auto`; all passed. Tests specific to the deleted generators were later removed.
- Black and Ruff passed for all 36 changed or new Python files. Runtime source changes are documentation references only.
- `uv run --no-sync mypy --explicit-package-bases scripts/check_harness.py scripts/mkdocs_hooks.py tests/unit/test_harness.py tests/unit/test_mkdocs_harness.py` passed.
- The additional MyPy check of `tests/contract/serve/test_openapi_contract.py` reports its existing `no-any-return` at line 31, where `yaml.safe_load` returns untyped data. Only its fixture path changed; all contract tests pass.
- Strict MkDocs build, shell syntax checks, and `git diff --check` passed.

Historical status drift remains tracked in [technical debt](../tech-debt-tracker.md).
The migration preserves the original inventory's status claims and counts; it does not certify implementation completeness.
