# Design document index

Owner means the maintainer of the affected module. Review these entries when that module changes.
“Code reviewed” means inspected on 2026-09-06. It does not claim current tests passed.

| Document | Scope / owner | Verification status |
| --- | --- | --- |
| [Constitution](constitution.md) | Repository maintainers | Governing product principles |
| [Core beliefs](core-beliefs.md) | Repository maintainers | Adopted engineering guidance |
| [Architecture](../../ARCHITECTURE.md) | Cross-module maintainers | Code reviewed against selector, server, deployment, and Temporal surfaces |
| [Harness adoption](../exec-plans/completed/2026-09-06-harness-engineering.md) | Repository harness | Audit, decisions, and validation record |
| [Backend contracts](../../src/holodeck/lib/backends/base.py) | Backend maintainers | Source contract. Tests listed in quality score |
| [OpenAI backend spec](../product-specs/035-openai-agents-backend/index.md) | Backend maintainers | Partial implementation. Consult spec inventory |
| [Temporal integration](../product-specs/040-holodeck-temporal/spec.md) | Temporal maintainers | Code reviewed. Spec records acceptance evidence |
| [Workflow comparison](../ideas/agent-workflows-comparative-analysis-2026.md) | Workflow design | Historical decision input |
| [Deterministic spine](../product-specs/036-deterministic-spine/index.md) | Workflow design | Archived. Only retained primitives describe current code |

See [Quality score](../QUALITY_SCORE.md) for test evidence and [Plans](../PLANS.md) for open work.

## Feature design records

These records retain their historical claims. Migration does not establish current implementation status.

- [CLI & Core Agent Engine](001-cli-core-engine/index.md)
- [Init Agent Project](004-init-agent-project/index.md)
- [Global Settings & Response Format](005-global-settings-response-format/index.md)
- [Agent Test Execution](006-agent-test-execution/index.md)
- [Interactive Chat](007-interactive-chat/index.md)
- [Unstructured Vector Ingestion & Search](008-unstructured-vector-ingestion-search/index.md)
- [Ollama Endpoint Support](009-ollama-endpoint-support/index.md)
- [MCP Tool Operations](010-mcp-tool-operations/index.md)
- [Interactive Init Wizard](011-interactive-init-wizard/index.md)
- [DeepEval Metrics](012-deepeval-metrics/index.md)
- [MCP CLI Command Group](013-mcp-cli/index.md)
- [Structured Data Ingestion](014-structured-data-ingestion/index.md)
- [Vectorstore Reranking](015-vectorstore-reranking/index.md)
- [GraphRAG Integration](016-graphrag-integration/index.md)
- [Agent Local Server](017-agent-local-server/index.md)
- [OTel Observability](018-otel-observability/index.md)
- [Deploy Command](019-deploy-command/index.md)
- [HierarchicalDocumentTool](020-structured-document-tool/index.md)
- [Native Claude Agent SDK](021-claude-agent-sdk/index.md)
- [OTel GenAI Semconv in Claude Backend](022-otel-genai-semconv/index.md)
- [Choose Your Backend (ADK + MAF)](023-choose-your-backend/index.md)
- [Claude Serve & Deploy Parity](024-claude-serve-deploy/index.md)
- [Async Tool Init Endpoints](025-tool-init-endpoints/index.md)
- [Subagent Orchestration](029-subagent-orchestration/index.md)
- [Eval Runs & Test View Dashboard](031-eval-runs-dashboard/index.md)
- [Multi-Turn Test Cases & Evaluators](032-multi-turn-test-cases/index.md)
- [Test Optimizer](033-holodeck-test-optimizer/index.md)
- [Deterministic Spine](036-deterministic-spine/index.md)
- [Dependency Stack Revamp selection evidence](042-dependency-stack-revamp/selection-evidence.md)
