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
| CLI & Core Agent Engine | [001-cli-core-engine](exec-plans/active/001-cli-core-engine/index.md) |
| Init Agent Project | [004-init-agent-project](exec-plans/active/004-init-agent-project/index.md) |
| Global Settings & Response Format | [005-global-settings-response-format](exec-plans/active/005-global-settings-response-format/index.md) |
| Agent Test Execution | [006-agent-test-execution](exec-plans/active/006-agent-test-execution/index.md) |
| Interactive Chat | [007-interactive-chat](exec-plans/active/007-interactive-chat/index.md) |
| Unstructured Vector Ingestion & Search | [008-unstructured-vector-ingestion-search](exec-plans/active/008-unstructured-vector-ingestion-search/index.md) |
| Ollama Endpoint Support | [009-ollama-endpoint-support](exec-plans/active/009-ollama-endpoint-support/index.md) |
| MCP Tool Operations | [010-mcp-tool-operations](exec-plans/active/010-mcp-tool-operations/index.md) |
| Interactive Init Wizard | [011-interactive-init-wizard](exec-plans/active/011-interactive-init-wizard/index.md) |
| DeepEval Metrics | [012-deepeval-metrics](exec-plans/active/012-deepeval-metrics/index.md) |
| MCP CLI Command Group | [013-mcp-cli](exec-plans/active/013-mcp-cli/index.md) |
| Structured Data Ingestion | [014-structured-data-ingestion](exec-plans/active/014-structured-data-ingestion/index.md) |
| Vectorstore Reranking | [015-vectorstore-reranking](exec-plans/active/015-vectorstore-reranking/index.md) |
| GraphRAG Integration | [016-graphrag-integration](exec-plans/active/016-graphrag-integration/index.md) |
| Agent Local Server | [017-agent-local-server](exec-plans/active/017-agent-local-server/index.md) |
| OTel Observability | [018-otel-observability](exec-plans/active/018-otel-observability/index.md) |
| Deploy Command | [019-deploy-command](exec-plans/active/019-deploy-command/index.md) |
| HierarchicalDocumentTool | [020-structured-document-tool](exec-plans/active/020-structured-document-tool/index.md) |
| Native Claude Agent SDK | [021-claude-agent-sdk](exec-plans/active/021-claude-agent-sdk/index.md) |
| Choose Your Backend (ADK + MAF) | [023-choose-your-backend](exec-plans/active/023-choose-your-backend/index.md) |
| Claude Serve & Deploy Parity | [024-claude-serve-deploy](exec-plans/active/024-claude-serve-deploy/index.md) |
| Subagent Orchestration | [029-subagent-orchestration](exec-plans/active/029-subagent-orchestration/index.md) |
| Eval Runs & Test View Dashboard | [031-eval-runs-dashboard](exec-plans/active/031-eval-runs-dashboard/index.md) |
| Multi-Turn Test Cases & Evaluators | [032-multi-turn-test-cases](exec-plans/active/032-multi-turn-test-cases/index.md) |
| Test Optimizer | [033-holodeck-test-optimizer](exec-plans/active/033-holodeck-test-optimizer/index.md) |
| Production Hardening | [034-production-hardening](exec-plans/active/034-production-hardening/index.md) |
| OpenAI Agents SDK Backend | [035-openai-agents-backend](exec-plans/active/035-openai-agents-backend/index.md) |
| Optimizer Progress Stream | [038-optimizer-progress-stream](exec-plans/active/038-optimizer-progress-stream/index.md) |

## Completed and archived plans

- [2026-09-06: Remove retired command workflow](exec-plans/completed/2026-09-06-workflow-cleanup.md)
- [2026-09-06: Specification migration and custom command removal](exec-plans/completed/2026-09-06-spec-migration.md)
- [2026-09-06: Harness engineering adoption](exec-plans/completed/2026-09-06-harness-engineering.md)

Frozen historical plans share `completed/` but are explicitly labeled archived. They do not represent completed implementation.

| Feature | Disposition | Execution records |
| --- | --- | --- |
| OTel GenAI Semconv in Claude Backend | Completed | [022-otel-genai-semconv](exec-plans/completed/022-otel-genai-semconv/index.md) |
| Async Tool Init Endpoints | Completed | [025-tool-init-endpoints](exec-plans/completed/025-tool-init-endpoints/index.md) |
| Deterministic Spine | Archived / superseded | [036-deterministic-spine](exec-plans/completed/036-deterministic-spine/index.md) |
| HoloDeck Agents on Temporal | Completed | [040-holodeck-temporal](exec-plans/completed/040-holodeck-temporal/index.md) |
| GraphRAG Integration Plan (legacy) | Archived / superseded | [graph-rag-integration](exec-plans/completed/graph-rag-integration/index.md) |

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
