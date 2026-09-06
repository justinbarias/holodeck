# HoloDeck architecture

HoloDeck is a Python package and CLI for agents defined in YAML.
Its runtime, evaluation framework, and deployment engine share typed contracts.
Python-authored Temporal workflows can call HoloDeck agents as activities.

This map describes implemented code. [Product specs](docs/product-specs/index.md)
separate implementation evidence from proposals and historical status.

## System flow

```text
agent.yaml + environment + referenced files
                    |
             config/ -> models/
                    |
       CLI / chat / HTTP server / Temporal activity
                    |
             BackendSelector
               /         \
      OpenAI Agents      Claude Agent SDK
      OpenAI / Azure     Anthropic / Ollama
               \         /
       execution results, streams, tools, telemetry
                    |
       evaluation -> result files -> dashboard / optimizer

agent configuration -> deploy/ -> Docker image -> Azure Container Apps
```

The diagram shows execution flow, not an enforced global import hierarchy.
Semantic Kernel remains in retrieval and supporting integrations. It is no longer
the runtime backend selected for OpenAI, Azure OpenAI, or Ollama.

## Code map

Paths are relative to the repository root. Each directory owns the responsibility shown.

| Area | Responsibility and entry points |
| --- | --- |
| [cli](src/holodeck/cli) | Click commands, including `init`, `test`, `chat`, `serve`, `deploy`, and `worker`. Start at `main.py`. |
| [config](src/holodeck/config) | YAML loading, environment substitution, merging, and validation. Start at `loader.py`. |
| [models](src/holodeck/models) | Pydantic configuration and result contracts. `agent.py` defines `Agent`. |
| [backends](src/holodeck/lib/backends) | Provider selection, lifecycle protocols, SDK adapters, MCP bridges, and tracing. Start at `base.py` and `selector.py`. |
| [tools](src/holodeck/tools) | Vector search, hierarchical document retrieval, and MCP integration. |
| [services](src/holodeck/services) | Shared services, including embedding support. |
| [test runner](src/holodeck/lib/test_runner) | Agent execution and evaluation orchestration. Start at `executor.py`. |
| [evaluators](src/holodeck/lib/evaluators) | NLP, Azure AI, and DeepEval metrics. |
| [eval run](src/holodeck/lib/eval_run) | Result serialization, metadata, redaction, and atomic file writes. |
| [optimizer](src/holodeck/optimizer) | Candidate proposals, scoring, and optimization artifacts. |
| [chat](src/holodeck/chat) | Interactive sessions, execution, and streaming. |
| [serve](src/holodeck/serve) | FastAPI server, AG-UI and REST protocols, session storage, and tool initialization. |
| [deploy](src/holodeck/deploy) | Docker packaging, image builds, deployment state, and Azure Container Apps deployment. |
| [temporal](src/holodeck/temporal) | Agent activities, worker configuration, plugins, and deterministic helpers. |
| [workflow primitives](src/holodeck/lib/workflow) | Schema gates, decision tables, and FEEL expressions reused by Temporal. |
| [observability](src/holodeck/lib/observability) | OpenTelemetry configuration, GenAI conventions, metrics, and exporters. |
| [dashboard](src/holodeck/dashboard) | Dash application for saved evaluation results. See [Frontend](docs/FRONTEND.md). |
| [templates](src/holodeck/templates) | Agent project templates used by `holodeck init`. |
| [tests](tests) | Unit, integration, and contract tests plus committed fixtures. |
| [schemas](schemas) | Published JSON contracts. See the [generated inventory](docs/generated/db-schema.md). |
| [product specs](docs/product-specs/index.md) | Feature requirements, contracts, and acceptance criteria. |
| [design documents](docs/design-docs/index.md) | Research, data models, and design decisions. |
| [execution plans](docs/PLANS.md) | Active work, completed plans, and historical task lists. |
| [docs](docs) | Product guides and the engineering knowledge base mapped by [AGENTS.md](AGENTS.md). |

## Boundaries and exceptions

Runtime consumers use `AgentBackend`, `AgentSession`, and `ExecutionResult` from
[base.py](src/holodeck/lib/backends/base.py). Backend construction belongs in
[selector.py](src/holodeck/lib/backends/selector.py), except backend implementation tests.
SDK-specific values must become typed HoloDeck values at adapters.

Configuration belongs in Pydantic models, YAML, and environment variables.
`models/` is not currently a pure dependency layer. For example,
[decision_table.py](src/holodeck/models/decision_table.py) imports FEEL validation
and exposes a file loader. Do not claim that a global layer check enforces purity.

Temporal workflow execution must be deterministic. LLM calls belong in
[activity.py](src/holodeck/temporal/activity.py). The
[deterministic surface](src/holodeck/temporal/deterministic.py) exposes reusable helpers.
Decision-table loading occurs at import time in a sibling module, outside the workflow sandbox.
Table changes require the same replay-versioning discipline as workflow code changes.

External API tools use MCP. Local retrieval tools use shared initialization and
provider-specific adapters. These boundaries let both runtimes reuse tool behavior.

## Data, external systems, and operations

HoloDeck stores evaluation results and deployment state in local JSON files.
Retrieval stores embeddings in the configured vector store. Temporal owns workflow history.
The [schema inventory](docs/generated/db-schema.md) links the storage implementations.

External dependencies include model providers, MCP servers, vector stores,
OpenTelemetry collectors, Docker registries, Azure Container Apps, and optional Temporal services.
Exact dependencies and extras live in [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock).

[Security](docs/SECURITY.md) describes credentials, redaction, and input boundaries.
[Reliability](docs/RELIABILITY.md) describes lifecycle, local diagnostics, and deployment validation.
[Contributing](docs/contributing.md) provides local setup and verification commands.

## Direction and known gaps

[Spec 040](docs/product-specs/040-holodeck-temporal/spec.md) replaced the proposed YAML workflow engine with Temporal integration.
The surviving gate and table primitives remain in `lib/workflow/`.
[Spec 041](docs/product-specs/041-temporal-file-inputs/spec.md) describes proposed Temporal file inputs.

[Quality score](docs/QUALITY_SCORE.md) records available evidence without assuming a passing test suite.
[Technical debt](docs/exec-plans/tech-debt-tracker.md) tracks known gaps and their exit criteria.
