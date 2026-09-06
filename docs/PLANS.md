# Execution plans

Small, reversible changes need a brief task-local plan and verification result.
For work across boundaries or multiple sessions, create a committed plan in `docs/exec-plans/active/`.
Use `YYYY-MM-DD-short-name.md` as its filename.

## Plan contents

Include these sections:

1. Objective and observable acceptance criteria.
2. Scope, constraints, and links to the relevant code and product spec.
3. Ordered steps with verification for each meaningful change.
4. Progress and decisions, including why an approach changed.
5. Validation results, blockers, and remaining work.

Keep the plan current as work proceeds. Preserve enough context for another agent to continue.
When all acceptance criteria pass, move the plan to `completed/` and update this index.
Move deferred gaps into [technical debt](exec-plans/tech-debt-tracker.md) with an owner and exit criterion.

## Active plans


Imported feature plans keep their original filenames and task lists. Feature IDs group related execution records.
An earlier release does not close unfinished work. The inventory remains the source of status evidence.

| Feature | Execution records |
| --- | --- |
| Claude Serve & Deploy Parity | [024-claude-serve-deploy](exec-plans/active/024-claude-serve-deploy/index.md) |
| Eval Runs & Test View Dashboard | [031-eval-runs-dashboard](exec-plans/active/031-eval-runs-dashboard/index.md) |
| Test Optimizer | [033-holodeck-test-optimizer](exec-plans/active/033-holodeck-test-optimizer/index.md) |
| Production Hardening | [034-production-hardening](exec-plans/active/034-production-hardening/index.md) |
| OpenAI Agents SDK Backend | [Completion execution plan](exec-plans/active/035-openai-agents-backend/2026-09-06-complete-035.md); [historical records](exec-plans/active/035-openai-agents-backend/index.md) |

Drafts without an execution plan yet: [037](product-specs/037-gepa-optimizer/index.md), [041](product-specs/041-temporal-file-inputs/index.md), [042](product-specs/042-dependency-stack-revamp/index.md).

## Completed and archived plans

- [2026-09-06: Spec and plan drift reconciliation](exec-plans/completed/2026-09-06-spec-drift-reconciliation.md)
- [2026-09-06: Feature 035 plan reconciliation](exec-plans/completed/2026-09-06-035-plan-reconciliation.md)

- [2026-09-06: Remove retired command workflow](exec-plans/completed/2026-09-06-workflow-cleanup.md)
- [2026-09-06: Specification migration and custom command removal](exec-plans/completed/2026-09-06-spec-migration.md)
- [2026-09-06: Harness engineering adoption](exec-plans/completed/2026-09-06-harness-engineering.md)

Frozen historical plans share `completed/` but are explicitly labeled archived. They do not represent completed implementation.

| Feature | Disposition | Execution records |
| --- | --- | --- |
| CLI & Core Agent Engine | Completed | [001-cli-core-engine](exec-plans/completed/001-cli-core-engine/index.md) |
| Init Agent Project | Completed | [004-init-agent-project](exec-plans/completed/004-init-agent-project/index.md) |
| Global Settings & Response Format | Completed | [005-global-settings-response-format](exec-plans/completed/005-global-settings-response-format/index.md) |
| Agent Test Execution | Completed | [006-agent-test-execution](exec-plans/completed/006-agent-test-execution/index.md) |
| Interactive Chat | Completed | [007-interactive-chat](exec-plans/completed/007-interactive-chat/index.md) |
| Unstructured Vector Ingestion & Search | Completed | [008-unstructured-vector-ingestion-search](exec-plans/completed/008-unstructured-vector-ingestion-search/index.md) |
| Interactive Init Wizard | Completed | [011-interactive-init-wizard](exec-plans/completed/011-interactive-init-wizard/index.md) |
| DeepEval Metrics | Completed | [012-deepeval-metrics](exec-plans/completed/012-deepeval-metrics/index.md) |
| MCP CLI Command Group | Completed | [013-mcp-cli](exec-plans/completed/013-mcp-cli/index.md) |
| Structured Data Ingestion | Completed | [014-structured-data-ingestion](exec-plans/completed/014-structured-data-ingestion/index.md) |
| Agent Local Server | Completed | [017-agent-local-server](exec-plans/completed/017-agent-local-server/index.md) |
| OTel Observability | Completed | [018-otel-observability](exec-plans/completed/018-otel-observability/index.md) |
| Deploy Command | Completed | [019-deploy-command](exec-plans/completed/019-deploy-command/index.md) |
| HierarchicalDocumentTool | Completed | [020-structured-document-tool](exec-plans/completed/020-structured-document-tool/index.md) |
| Native Claude Agent SDK | Completed | [021-claude-agent-sdk](exec-plans/completed/021-claude-agent-sdk/index.md) |
| OTel GenAI Semconv in Claude Backend | Completed | [022-otel-genai-semconv](exec-plans/completed/022-otel-genai-semconv/index.md) |
| Async Tool Init Endpoints | Completed | [025-tool-init-endpoints](exec-plans/completed/025-tool-init-endpoints/index.md) |
| Subagent Orchestration | Completed | [029-subagent-orchestration](exec-plans/completed/029-subagent-orchestration/index.md) |
| Multi-Turn Test Cases & Evaluators | Completed | [032-multi-turn-test-cases](exec-plans/completed/032-multi-turn-test-cases/index.md) |
| Optimizer Progress Stream | Completed | [038-optimizer-progress-stream](exec-plans/completed/038-optimizer-progress-stream/index.md) |
| HoloDeck Agents on Temporal | Completed | [040-holodeck-temporal](exec-plans/completed/040-holodeck-temporal/index.md) |
| Ollama Endpoint Support | Archived / superseded by Claude-backend routing (021) | [009-ollama-endpoint-support](exec-plans/completed/009-ollama-endpoint-support/index.md) |
| MCP Tool Operations | Archived / superseded by 021 and 035 MCP paths | [010-mcp-tool-operations](exec-plans/completed/010-mcp-tool-operations/index.md) |
| Vectorstore Reranking | Archived / folded into 020 | [015-vectorstore-reranking](exec-plans/completed/015-vectorstore-reranking/index.md) |
| GraphRAG Integration | Archived / not scheduled | [016-graphrag-integration](exec-plans/completed/016-graphrag-integration/index.md) |
| Choose Your Backend (ADK + MAF) | Archived / superseded by 035 and 042 | [023-choose-your-backend](exec-plans/completed/023-choose-your-backend/index.md) |
| Deterministic Spine | Archived / superseded by 040 | [036-deterministic-spine](exec-plans/completed/036-deterministic-spine/index.md) |
| GraphRAG Integration Plan (legacy) | Archived / superseded by 016, itself archived | [graph-rag-integration](exec-plans/completed/graph-rag-integration/index.md) |

Completed rows whose remaining scope lives in GitHub issues (014, 017, 018, 019, 020) name those issues in the [inventory](product-specs/inventory.md).
Specs 039 (policy generator) and 023 are archived in the inventory; 039 never had an execution plan.

## Relationship to feature specs

The [product-spec index](product-specs/index.md) links canonical feature requirements in `docs/product-specs/`.
Research and design records live in `docs/design-docs/<feature>/`. Plans and task lists live in their execution-plan directory.
New specs use the format established by [spec 040](product-specs/040-holodeck-temporal/spec.md).
An execution plan links the relevant spec instead of copying its requirements.
When feature status changes, update `docs/product-specs/inventory.md` as well.

Earlier plans remain historical artifacts. Repository custom commands and their scaffolding have been removed.
Maintain specifications, design documents, and execution plans directly in the directories above.
Keep `AGENTS.md` as the knowledge map and `CLAUDE.md` as its single import.

## Migration ledger

The [migration ledger](generated/spec-migration.json) records one destination for each of the 316 original files.
When a migrated file moves again, update its destination in the ledger and its index links.
Source hashes record the migration baseline. They do not freeze future content edits.
