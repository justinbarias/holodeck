# Remove retired command workflow

Date: 2026-09-06. Owner: repository maintainers. Status: complete.

## Objective

Remove the retired command workflow, its scaffolding, and all references to it.
Preserve feature requirements, task completion states, and product principles.

## Steps and acceptance criteria

1. Remove unused command templates, generators, and tests of the deleted tooling.
2. Move the constitution into the design documents and update all references.
3. Replace obsolete command instructions with plain descriptions of the intended work.
4. Verify no retired workflow references remain in repository files.
5. Validate document links, publishing, and the remaining harness tests.

## Progress and decisions

- Removed the command templates and shell generators.
- Moved the constitution to `docs/design-docs/constitution.md`; all five core principles are unchanged.
- Updated governance references to use the harness documents.
- Removed tests that only exercised deleted tooling.
- Historical feature documents retain their requirements and task states while obsolete workflow references are removed.

## Validation

- Scanned all 1,151 existing tracked and untracked repository files, including hidden files and binary content: no retired workflow names, directory references, script names, or template references remain.
- Verified that the constitution's five core principles are unchanged.
- Verified all 316 migrated files still exist, all seven binary hashes match, and every original task checkbox state is preserved.
- `uv run --no-sync pytest tests/unit/test_harness.py tests/unit/test_mkdocs_harness.py tests/contract/serve/test_openapi_contract.py -n auto -q`: 41 passed.
- Black, Ruff, and focused MyPy checks passed for the harness and publishing scripts and their tests.
- `make harness-check`, the strict MkDocs build, and `git diff --check` passed.

The removed tests covered the five deleted shell scripts. Runtime implementation is unchanged.
