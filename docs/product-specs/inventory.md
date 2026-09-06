# Specs Index

One row per feature directory. **Status** values: `shipped` (built and merged, git evidence), `pending` (spec exists; build partial or not started), `draft` (spec-only by declaration), `archived` (superseded, frozen, or will not be built). **Tasks** counts markdown checkboxes in the directory's task files (`checklists/requirements.md` is a spec-quality checklist and is never counted). † marks a discrepancy: git says shipped but the task files were never checked off.

Maintenance: update the row when a spec's status or task list changes. Generated 2026-08-29 from a full survey; treat unfamiliar rows as of that date.

Migrated on 2026-09-06 with the recorded statuses and task counts unchanged.
Reconciled on 2026-09-06 against code and merge evidence ([record](../exec-plans/completed/2026-09-06-spec-drift-reconciliation.md)): shipped features moved to `shipped`, Semantic Kernel-era and workflow-engine work moved to `archived`, and their execution records moved to `docs/exec-plans/completed/`. Task counts stay as recorded; unticked boxes in a shipped feature are historical, not open work. Remaining scope is named in the notes column.
Each feature index links its requirements, design documents, and relocated [execution records](../PLANS.md).
Task lists now live under `docs/exec-plans/`, grouped by the same feature ID.
The original CLI feature README is now `001-cli-core-engine/overview.md`.

| Spec | Title | Status | Tasks | Notes |
| --- | --- | --- | --- | --- |
| [001](001-cli-core-engine/index.md) | CLI & Core Agent Engine | shipped | 75/127 | The v0.1 foundation; only dir with its own README. Unticked boxes are historical file checklists |
| [004](004-init-agent-project/index.md) | Init Agent Project | shipped | 137/152 | Residual edge-case tasks unscheduled |
| [005](005-global-settings-response-format/index.md) | Global Settings & Response Format | shipped | 25/32 |  |
| [006](006-agent-test-execution/index.md) | Agent Test Execution | shipped | 144/183 | Side plans: logging, spinner. Historical SC checklist never ticked |
| [007](007-interactive-chat/index.md) | Interactive Chat | shipped | 28/37 |  |
| [008](008-unstructured-vector-ingestion-search/index.md) | Unstructured Vector Ingestion & Search | shipped | 68/94 | Qdrant fixes landed 2026-09-06 (035 T3) |
| [009](009-ollama-endpoint-support/index.md) | Ollama Endpoint Support | archived | 26/47 | Superseded: Ollama routes to the Claude backend (021/035); open tasks targeted the Semantic Kernel `_create_kernel` path |
| [010](010-mcp-tool-operations/index.md) | MCP Tool Operations | archived | 7/33 | Superseded: MCP tools run through the Claude MCP bridge (021) and OpenAI Agents MCP servers (035); open tasks targeted SK plugins |
| [011](011-interactive-init-wizard/index.md) | Interactive Init Wizard | shipped | 47/78 |  |
| [012](012-deepeval-metrics/index.md) | DeepEval Metrics | shipped | 30/69 | GEval, RAG, and answer-relevancy evaluators exist under `lib/evaluators/deepeval/` |
| [013](013-mcp-cli/index.md) | MCP CLI Command Group | shipped | 63/73 |  |
| [014](014-structured-data-ingestion/index.md) | Structured Data Ingestion | shipped | 38/74 | Core structured ingestion shipped; US2 multi-field embeddings and US3 database sources not built (#178, #180) |
| [015](015-vectorstore-reranking/index.md) | Vectorstore Reranking | archived | no task list | Folded into 020 optional reranking (#252) |
| [016](016-graphrag-integration/index.md) | GraphRAG Integration | archived | no task list | Research only; not scheduled. Supersedes legacy `graph-rag-integration/` |
| [017](017-agent-local-server/index.md) | Agent Local Server | shipped | 54/98 | AG-UI + REST serve, health/ready, session delete, CORS exist; phase issues #190–#195 predate them |
| [018](018-otel-observability/index.md) | OTel Observability | shipped | 65/134 | OTLP + console exporters and redaction shipped; Prometheus, Azure Monitor, multiple exporters not built (#208, #209, #211) |
| [019](019-deploy-command/index.md) | Deploy Command | shipped | 24/52 | `deploy build/run/status/destroy` on Azure Container Apps; registry push, AWS, GCP not built (#237–#240) |
| [020](020-structured-document-tool/index.md) | HierarchicalDocumentTool | shipped | 84/130 | Hybrid search, definitions, cross-references shipped; optional reranking not built (#252); polish #253, #254 |
| [021](021-claude-agent-sdk/index.md) | Native Claude Agent SDK | shipped | 154/163 | Phase 5 tool adapters exist (`lib/backends/tool_adapters.py`) |
| [022](022-otel-genai-semconv/index.md) | OTel GenAI Semconv in Claude Backend | shipped | 28/28 | |
| [023](023-choose-your-backend/index.md) | Choose Your Backend (ADK + MAF) | archived | 0/212 | Superseded by the two-native-SDK stance (035, 042); ADK and MAF backends will not be built |
| [024](024-claude-serve-deploy/index.md) | Claude Serve & Deploy Parity | pending | 46/139 | US1–US2 done; US3–US5 open and overlap 034 hardening |
| [025](025-tool-init-endpoints/index.md) | Async Tool Init Endpoints | shipped | 53/53 | |
| [026](026-sdk-config-additions/index.md) | Simple SDK Config Additions | shipped | no task list | Landed as feat(026) |
| [027](027-mcp-http-sse-transport/index.md) | MCP HTTP/SSE Transport | shipped | no task list | Delivered via 035 MCP transports and the Claude MCP bridge |
| [028](028-yaml-hooks-system/index.md) | YAML Hooks System | shipped | no task list | Delivered on Claude via 021; OpenAI parity is 035 T6 |
| [029](029-subagent-orchestration/index.md) | Subagent Orchestration | shipped | 0/63 | Merged as PR #309; checkboxes never ticked |
| [030](030-skills-support/index.md) | Skills Support | shipped | no task list | Delivered via 021 (Claude) and 035 T3 (`type: skill`) |
| [031](031-eval-runs-dashboard/index.md) | Eval Runs & Test View Dashboard | pending | 103/252 | US4 done; dashboard moved Streamlit → Dash |
| [032](032-multi-turn-test-cases/index.md) | Multi-Turn Test Cases & Evaluators | shipped | 224/226 | PR #308 |
| [033](033-holodeck-test-optimizer/index.md) | Test Optimizer | pending | 9/12 T + 0/30 | MVP shipped in PR #335; post-MVP + text proposer open; mixed task conventions |
| [034](034-production-hardening/index.md) | Production Hardening | pending | 4/226 | Checkboxes live inside phase plan docs |
| [035](035-openai-agents-backend/index.md) | OpenAI Agents SDK Backend | pending | 1/12 | T0 contract complete. [Completion plan](../exec-plans/active/035-openai-agents-backend/2026-09-06-complete-035.md) is the active checklist. MVP shipped in #338; audited legacy TODO remains 9/43. |
| [036](036-deterministic-spine/index.md) | Deterministic Spine | archived | 13/30 | Frozen after Phase 1 (2026-08-29); superseded by 040 |
| [037](037-gepa-optimizer/index.md) | GEPA Optimizer Backend | draft | no task list | Builds on 033 |
| [038](038-optimizer-progress-stream/index.md) | Optimizer Progress Stream | shipped | 4/35 | Merged as PR #345; checkboxes stale |
| [039](039-policy-generator/index.md) | Policy Generator | archived | no task list | Depended on 036, replaced by Temporal-first 040; its workflow-engine surfaces no longer exist |
| [040](040-holodeck-temporal/index.md) | HoloDeck Agents on Temporal | shipped | 16/16 | Merged to main 2026-08-30 (stack #369: PRs #367, #368, #371, #372); AC-1..AC-6 demonstrated by named tests; T14 moved to 041; budget-retry follow-up in #373 |
| [041](041-temporal-file-inputs/index.md) | File and Bytestream Inputs for Temporal Agents | draft | 0/0 | Depends on 040; parse_document activity + pass-through attachments + gate-schema codegen (moved from 040 T14); decisions settled 2026-08-30 |
| [—](graph-rag-integration/index.md) | GraphRAG Integration Plan (legacy) | archived | no task list | Early unnumbered feature directory; superseded by 016 |

Numbers 002 and 003 have no spec directory (stale branches only).
