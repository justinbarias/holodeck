# Specs Index

One row per feature directory. **Status** values: `shipped` (built and merged, git evidence), `pending` (spec exists; build partial or not started), `draft` (spec-only by declaration), `archived` (superseded or frozen). **Tasks** counts markdown checkboxes in the directory's task files (`checklists/requirements.md` is a spec-quality checklist and is never counted). † marks a discrepancy: git says shipped but the task files were never checked off.

Maintenance: update the row when a spec's status or task list changes. Generated 2026-08-29 from a full survey; treat unfamiliar rows as of that date.

| Spec | Title | Status | Tasks | Notes |
| --- | --- | --- | --- | --- |
| [001](001-cli-core-engine/) | CLI & Core Agent Engine | pending | 75/127 | The v0.1 foundation; only dir with its own README |
| [004](004-init-agent-project/) | Init Agent Project | pending | 137/152 | |
| [005](005-global-settings-response-format/) | Global Settings & Response Format | pending | 25/32 | |
| [006](006-agent-test-execution/) | Agent Test Execution | pending | 144/183 | Side plans: logging, spinner |
| [007](007-interactive-chat/) | Interactive Chat | pending | 28/37 | |
| [008](008-unstructured-vector-ingestion-search/) | Unstructured Vector Ingestion & Search | pending | 68/94 | |
| [009](009-ollama-endpoint-support/) | Ollama Endpoint Support | pending | 26/47 | |
| [010](010-mcp-tool-operations/) | MCP Tool Operations | pending | 7/33 | Barely started |
| [011](011-interactive-init-wizard/) | Interactive Init Wizard | pending | 47/78 | |
| [012](012-deepeval-metrics/) | DeepEval Metrics | pending | 30/69 | |
| [013](013-mcp-cli/) | MCP CLI Command Group | pending | 63/73 | |
| [014](014-structured-data-ingestion/) | Structured Data Ingestion | pending | 38/74 | |
| [015](015-vectorstore-reranking/) | Vectorstore Reranking | pending | no task list | Plan/research artifacts only |
| [016](016-graphrag-integration/) | GraphRAG Integration | pending | no task list | Supersedes legacy `graph-rag-integration/` |
| [017](017-agent-local-server/) | Agent Local Server | pending | 54/98 | AG-UI + REST serve |
| [018](018-otel-observability/) | OTel Observability | pending | 65/134 | |
| [019](019-deploy-command/) | Deploy Command | pending | 24/52 | |
| [020](020-structured-document-tool/) | HierarchicalDocumentTool | pending | 84/130 | |
| [021](021-claude-agent-sdk/) | Native Claude Agent SDK | shipped | 154/163 | Phase 5 (0/8) is the only unstarted phase |
| [022](022-otel-genai-semconv/) | OTel GenAI Semconv in Claude Backend | shipped | 28/28 | |
| [023](023-choose-your-backend/) | Choose Your Backend (ADK + MAF) | pending | 0/212 | Not started; US7 blocked on US1/US2 |
| [024](024-claude-serve-deploy/) | Claude Serve & Deploy Parity | pending | 46/139 | US1–US2 done; US3–US5 not started |
| [025](025-tool-init-endpoints/) | Async Tool Init Endpoints | shipped | 53/53 | |
| [026](026-sdk-config-additions/) | Simple SDK Config Additions | shipped † | no task list | Landed as feat(026); doc still says Draft |
| [027](027-mcp-http-sse-transport/) | MCP HTTP/SSE Transport | pending | no task list | Related transports landed via 035 |
| [028](028-yaml-hooks-system/) | YAML Hooks System | pending | no task list | |
| [029](029-subagent-orchestration/) | Subagent Orchestration | shipped † | 0/63 | Merged as PR #309; checkboxes never ticked |
| [030](030-skills-support/) | Skills Support | pending | no task list | |
| [031](031-eval-runs-dashboard/) | Eval Runs & Test View Dashboard | pending | 103/252 | US4 done; dashboard moved Streamlit → Dash |
| [032](032-multi-turn-test-cases/) | Multi-Turn Test Cases & Evaluators | shipped | 224/226 | PR #308 |
| [033](033-holodeck-test-optimizer/) | Test Optimizer | pending | 9/12 T + 0/30 | MVP shipped in PR #335; post-MVP + text proposer open; mixed task conventions |
| [034](034-production-hardening/) | Production Hardening | pending | 4/226 | Checkboxes live inside phase plan docs |
| [035](035-openai-agents-backend/) | OpenAI Agents SDK Backend | pending | 14/43 | MVP shipped in PR #338; full-parity plan open |
| [036](036-deterministic-spine/) | Deterministic Spine | archived | 13/30 | Frozen after Phase 1 (2026-08-29); superseded by 040 |
| [037](037-gepa-optimizer/) | GEPA Optimizer Backend | draft | no task list | Builds on 033 |
| [038](038-optimizer-progress-stream/) | Optimizer Progress Stream | shipped † | 4/35 | Merged as PR #345; checkboxes stale |
| [039](039-policy-generator/) | Policy Generator | draft | no task list | Blocked on the pivot; depended on archived 036 |
| [040](040-holodeck-temporal/) | HoloDeck Agents on Temporal | pending | 0/13 | Active successor to 036; first post-speckit spec; plan in `tasks/plan.md`; deferred items in issues #364/#365 |
| [041](041-temporal-file-inputs/) | File and Bytestream Inputs for Temporal Agents | draft | 0/0 | Depends on 040; parse_document activity + pass-through attachments; decisions settled 2026-08-30 |
| [—](graph-rag-integration/) | GraphRAG Integration Plan (legacy) | archived | no task list | Pre-speckit dir, no `NNN-` prefix; superseded by 016 |

Numbers 002 and 003 have no spec directory (stale branches only).
