# Harness engineering adoption

Date: 2026-09-06. Owner: repository maintainers. Status: complete.

## Objective and acceptance criteria

Audit the root instruction files and adopt the exact knowledge-base structure in the requested harness-engineering article.
Keep `AGENTS.md` within 100 lines and make `CLAUDE.md` a single import.
Provide a code-verified architecture map, indexed guidance, plans, quality evidence, and mechanical drift checks.
Apply the requested Astra prompting guidance to engineering instructions without changing runtime model configuration.

## Audit findings

| Previous guidance | Evidence | Resolution |
| --- | --- | --- |
| `AGENTS.md` contained 2,260 lines, with duplicated examples, setup, and rules. | Root file before this change | Replace it with a map and concise working agreement. |
| `CLAUDE.md` contained a separate 230-line policy. | Root file before this change | Use only `@AGENTS.md`. |
| AGENTS described Semantic Kernel runtime routing and deployment as planned. | [Selector](../../../src/holodeck/lib/backends/selector.py), [deployment code](../../../src/holodeck/deploy) | Document current backends and deployment implementation. |
| Mandatory command-driven workflow conflicted with the Temporal spec. | [Spec 040](../../product-specs/040-holodeck-temporal/spec.md) replaces that workflow. | Preserve historical specs and use indexed execution plans. |
| Automatic broad format, test, and security commands applied after every task. | Previous root instructions | Use proportional task verification while retaining commit and CI requirements. |
| Plain Make test aliases were described as parallel. | [Makefile](../../../Makefile), pytest configuration | Document explicit parallel commands and track helper cleanup. |
| CLAUDE called all models pure, but decision-table models load files and import library validation. | [Decision-table model](../../../src/holodeck/models/decision_table.py) | Record the actual boundary and enforcement gap. |
| Deploy instructions assumed a personal sample, registry, and endpoint. | Previous root instructions | Preserve deployment authorization and base-image caveats in a portable runbook. |
| Contributing guide had a placeholder clone URL, missing architecture path, and inaccurate hook/security descriptions. | [Contributing](../../contributing.md) before this change | Replace duplicate material with verified commands and links. |

## Structure and decisions

The article's directories and named files are retained:

```text
AGENTS.md
ARCHITECTURE.md
docs/
├── design-docs/
│   ├── index.md
│   └── core-beliefs.md
├── exec-plans/
│   ├── active/
│   ├── completed/
│   └── tech-debt-tracker.md
├── generated/
│   └── db-schema.md
├── product-specs/
│   ├── index.md
│   └── new-user-onboarding.md
├── references/
│   ├── design-system-reference-llms.txt
│   ├── nixpacks-llms.txt
│   └── uv-llms.txt
├── DESIGN.md
├── FRONTEND.md
├── PLANS.md
├── PRODUCT_SENSE.md
├── QUALITY_SCORE.md
├── RELIABILITY.md
└── SECURITY.md
```

Existing product guides remain canonical and linked. The initial adoption retained the root specification tree.
A [subsequent migration](2026-09-06-spec-migration.md) moved its contents into product specs, design documents, and execution plans.
The schema inventory is generated from committed JSON contracts because HoloDeck has no central relational database.
The Nixpacks reference explicitly records non-applicability. Environment management remains UV-based.
The checker validates local file targets and discovery. It does not claim semantic freshness or external-link availability.
Existing schema and Temporal tests remain the executable architectural constraints.
Global import-layer enforcement and an isolated observability stack remain explicit technical debt.

The Astra guidance is reflected in scope-aware autonomy, conflict handling, concise communication, bounded delegation, and proportional checks.
The change does not alter API parameters, runtime model defaults, publishing permissions, or merge requirements.

## Progress and validation

- Audited both root files, contributor guidance, build commands, hooks, and relevant source boundaries.
- Created the mapped documentation structure and generated-inventory checker.
- Added checker regression tests and pre-commit integration, which also runs in CI.
- Protected both root entry points from the legacy context updater.
- Added a typed MkDocs hook so repository source links remain usable on the published site.

These results describe the initial adoption. Later [workflow cleanup](2026-09-06-workflow-cleanup.md) removed the context generator and its tests.
Validation used the existing UV environment with `--no-sync` for Python tooling:

| Command | Result |
| --- | --- |
| `make harness-check` | Passed: exact required structure, local targets, discovery, entry-point limits, and generated inventory. |
| `uv run --no-sync pytest tests/unit/test_harness.py tests/unit/test_mkdocs_harness.py -n auto -q` | 15 passed. Includes invalid layout, stale schema, orphan plan, legacy overwrite prevention, and publishing behavior. |
| `uv run --no-sync black --target-version py310 --check scripts/check_harness.py scripts/mkdocs_hooks.py tests/unit/test_harness.py tests/unit/test_mkdocs_harness.py` | Passed. Python 3.10 target is explicit for the local formatter runtime. |
| `uv run --no-sync ruff check scripts/check_harness.py scripts/mkdocs_hooks.py tests/unit/test_harness.py tests/unit/test_mkdocs_harness.py` | Passed. |
| `uv run --no-sync mypy --explicit-package-bases scripts/check_harness.py scripts/mkdocs_hooks.py tests/unit/test_harness.py tests/unit/test_mkdocs_harness.py` | Passed for all four files. Explicit package bases avoid duplicate module discovery for namespace imports. |
| `uv run --no-sync mkdocs build --strict --site-dir /tmp/holodeck-harness-docs` | Passed after resolving repository-source links. |
| `git diff --check` | Passed. |
| `uv run --no-sync pre-commit run harness-check --all-files` | Passed. |

Runtime implementation, schemas, and dependencies did not change. The full runtime suite and live deployments were not run.

## Sources

- [Harness engineering](https://openai.com/index/harness-engineering/), reviewed 2026-09-06.
- [Architecture template](https://architecture.md/), reviewed 2026-09-06.
- [GPT-6 Astra guidance](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra), reviewed 2026-09-06.

See [technical debt](../tech-debt-tracker.md) for deferred work and [quality score](../../QUALITY_SCORE.md) for verification coverage.
