# Quality score

Review date: 2026-09-06; rows for retrieval, evaluation and install profiles updated 2026-11-10 for [spec 042](product-specs/042-dependency-stack-revamp/index.md). Owner: maintainers of each listed area.
This score describes available enforcement and evidence, not a release certification.

Grades: **A** has named checks run for this audit. **B** has named checks located but not rerun here.
**C** has a documented gap without sufficient mechanical enforcement.
Update a row after relevant changes and record the command and result.

| Area / layer | Grade | Evidence and verification route | Gap |
| --- | --- | --- | --- |
| Repository harness | A | `make harness-check`, 15 [checker](../tests/unit/test_harness.py) and [publishing](../tests/unit/test_mkdocs_harness.py) tests passed. Strict MkDocs build, Black, Ruff, and focused MyPy passed. | Semantic freshness still requires code review |
| Configuration and schemas | B | [Schema sync](../tests/unit/test_agent_schema_sync.py), [config tests](../tests/unit/config), `make schema-check` | Full suite not run for this documentation audit |
| Runtime selection and contracts | B | [Backend tests](../tests/unit/lib/backends), especially `test_selector.py` and `test_base.py` | Provider parity still has open spec tasks |
| Tools and retrieval | B | [Tool tests](../tests/unit/tools), [adapter unit tests](../tests/unit/lib/retrieval), [integration tests](../tests/integration) | Live stores require service-specific verification |
| Retrieval adapter contract | B | [Contract suite](../tests/contract/retrieval/test_adapter_contract.py): one parametrized suite for all eight native adapters, with capability errors checked before ingestion | Pinecone ran on Pinecone Local (end of life) with a fake control plane, Azure AI Search on a fake service with the real SDK over a stub transport, Cosmos DB hybrid search on a fake only. No hosted-service run for Pinecone, Azure AI Search or Cosmos DB |
| Stored-data compatibility | B | [Legacy-layout fixtures](../tests/fixtures/legacy_layout/README.md) (Qdrant, Chroma, Postgres layouts written by the removed connectors), [Qdrant](../tests/integration/retrieval/test_qdrant_compat.py), [Chroma](../tests/integration/retrieval/test_chroma_compat.py) and [Postgres](../tests/integration/retrieval/test_postgres_compat.py) compatibility tests, [migration tests](../tests/integration/retrieval/test_migration.py) | Manifest and fingerprint not run on Pinecone, Azure AI Search, Cosmos DB or Weaviate. Concurrent writers are narrowed, not eliminated, on stores without compare-and-set |
| Golden fixtures | B | [Goldens](../tests/fixtures/golden) and [their tests](../tests/integration/golden): retrieval ranking, parser output, plain chunks and NLP scores. A change to a golden needs a stated reason | The goldens freeze behavior, not relevance quality |
| Provider verify matrix | B | `make verify-providers` and the provider cases of `make verify-live`: real Qdrant, Chroma, Postgres, Weaviate and Cosmos emulator, with a check that the store holds records. See [Reliability](RELIABILITY.md#verify-suite) | Pinecone and Azure AI Search cases skip without credentials. One flaky case: the OpenAI model sometimes declines to repeat a code word in an AG-UI turn |
| Install profiles | B | [Profile tests](../tests/unit/test_verify_profiles.py), [missing-extra tests](../tests/unit/test_install_profiles.py), [no-Semantic-Kernel guard](../tests/unit/test_no_semantic_kernel.py); `make verify-profiles` and the `profiles` CI workflow install the built wheel for 16 profiles | The hosted workflow's first run was on PR #442. The Node.js install in the image template has no provenance review (FR-029) |
| Evaluations and optimizer | B | [Evaluator tests](../tests/unit/lib/evaluators), [retired-metric tests](../tests/unit/models/test_retired_metrics.py), [optimizer tests](../tests/unit/optimizer) | Model quality needs representative evaluation runs. METEOR tests skip when the NLTK data is absent |
| Serving and deployment | B | [Serve tests](../tests/unit/serve), [deploy tests](../tests/unit/deploy) | Cloud behavior requires authorized end-to-end validation |
| Temporal deterministic boundary | B | [Import purity](../tests/unit/workflow/test_import_purity.py), [sandbox safety](../tests/unit/temporal/test_sandbox_safety.py), [optional import guard](../tests/unit/temporal/test_import_guard.py) | Replay/service integration needs the corresponding environment |
| Dashboard | B | [Dashboard tests](../tests/unit/dashboard), [frontend guidance](FRONTEND.md) | Visual changes require rendered inspection |
| Global dependency layering | C | [Architecture exceptions](../ARCHITECTURE.md) | No repository-wide import-layer enforcement |
| Spec and guide freshness | C | [Spec inventory](product-specs/inventory.md), harness links | Historical task/status drift and docsite warnings require manual reconciliation |
| Published template schemas | C | `make schema-check` covers `schemas/agent.schema.json` | The copies under `src/holodeck/templates/*/schemas/` are hand-kept and stale; `make schema` does not regenerate them |

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
