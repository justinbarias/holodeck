# Technical debt tracker

Review date: 2026-09-06. Owner identifies the responsible module maintainers, not an assigned individual.
Update an entry when its evidence or status changes. Close it only with a linked result.

| ID | Gap / evidence | Owner | Exit criterion | Status |
| --- | --- | --- | --- | --- |
| H-001 | [Spec inventory](../product-specs/inventory.md) has historical task drift and inconsistent status vocabulary, including `complete`. | Feature maintainers | Reconcile each affected row against merge and acceptance evidence. | Open |
| H-002 | [Architecture](../../ARCHITECTURE.md) has no global dependency-layer enforcement. Some models import library logic. | Architecture maintainers | Agree actual allowed edges, record exceptions, and add focused structural checks without inventing a new runtime architecture. | Open |
| H-003 | [Makefile](../../Makefile) includes serial test aliases, placeholder package helpers, and Sphinx-style docs targets despite MkDocs. | Developer tooling | Align affected helpers with current package and documented commands, with command-level verification. | Open |
| H-004 | No automatically isolated per-worktree logs/metrics/traces environment. | Observability maintainers | Provide a reproducible local setup and a demonstrated query against a representative request. | Open |
| H-005 | Legacy context generation could repopulate root instruction files. | Repository harness | Removed command scaffolding and generators; the harness checker validates both entry points. | Resolved in workflow cleanup |
| H-006 | Repository-source links broke when published as relative docsite links. | Documentation maintainers | [Publishing hook](../../scripts/mkdocs_hooks.py) resolves existing source links. [Regression tests](../../tests/unit/test_mkdocs_harness.py) and strict MkDocs build passed. | Resolved in harness adoption |
| H-007 | NLTK 3.10.3 has no patch for PYSEC-2026-3740. The [scoped audit exception](../SECURITY.md#nltk-model-artifact-exception) records inspected evaluator paths and does not cover arbitrary custom tools. | Evaluation and dependency maintainers | Lock a patched release and remove the exception; re-review by 2026-10-06 or before evaluator/model persistence changes. | Open; affected APIs absent from inspected paths |

The harness checker detects layout, link-target, discovery, and generated-inventory drift.
Semantic freshness still requires code inspection and named test evidence.
No scheduled cleanup service is configured by this change.
