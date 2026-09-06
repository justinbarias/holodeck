# Quality score

Review date: 2026-09-06. Owner: maintainers of each listed area.
This score describes available enforcement and evidence, not a release certification.

Grades: **A** has named checks run for this audit. **B** has named checks located but not rerun here.
**C** has a documented gap without sufficient mechanical enforcement.
Update a row after relevant changes and record the command and result.

| Area / layer | Grade | Evidence and verification route | Gap |
| --- | --- | --- | --- |
| Repository harness | A | `make harness-check`, 15 [checker](../tests/unit/test_harness.py) and [publishing](../tests/unit/test_mkdocs_harness.py) tests passed. Strict MkDocs build, Black, Ruff, and focused MyPy passed. | Semantic freshness still requires code review |
| Configuration and schemas | B | [Schema sync](../tests/unit/test_agent_schema_sync.py), [config tests](../tests/unit/config), `make schema-check` | Full suite not run for this documentation audit |
| Runtime selection and contracts | B | [Backend tests](../tests/unit/lib/backends), especially `test_selector.py` and `test_base.py` | Provider parity still has open spec tasks |
| Tools and retrieval | B | [Tool tests](../tests/unit/tools), [integration tests](../tests/integration) | Live stores require service-specific verification |
| Evaluations and optimizer | B | [Evaluator tests](../tests/unit/lib/evaluators), [optimizer tests](../tests/unit/optimizer) | Model quality needs representative evaluation runs |
| Serving and deployment | B | [Serve tests](../tests/unit/serve), [deploy tests](../tests/unit/deploy) | Cloud behavior requires authorized end-to-end validation |
| Temporal deterministic boundary | B | [Import purity](../tests/unit/workflow/test_import_purity.py), [sandbox safety](../tests/unit/temporal/test_sandbox_safety.py), [optional import guard](../tests/unit/temporal/test_import_guard.py) | Replay/service integration needs the corresponding environment |
| Dashboard | B | [Dashboard tests](../tests/unit/dashboard), [frontend guidance](FRONTEND.md) | Visual changes require rendered inspection |
| Global dependency layering | C | [Architecture exceptions](../ARCHITECTURE.md) | No repository-wide import-layer enforcement |
| Spec and guide freshness | C | [Spec inventory](product-specs/inventory.md), harness links | Historical task/status drift and docsite warnings require manual reconciliation |

## Verification policy

For a behavior change, run its regression tests and affected contracts with `-n auto`.
For shared schema or protocol changes, broaden to their consumers.
For Python changes, run focused formatting, lint, and type checks.
Keep required commit and CI checks. See [Contributing](contributing.md).

The constitution sets an 80% coverage target. Measure it with explicit coverage options.
Do not infer coverage from test counts or from the presence of tests.
Document environmental blockers and skipped live checks separately from failures.

## Maintenance loop

When a check fails repeatedly, identify the missing contract, fixture, diagnostic, or instruction.
Fix that cause with a focused change and a remediation message.
Review this map when a module or its test strategy changes.
Track unresolved gaps in [technical debt](exec-plans/tech-debt-tracker.md).
