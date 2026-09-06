# Product specification index

Canonical feature requirements and contracts live in this directory.
The [inventory](inventory.md) links all 40 migrated feature collections, including historical entries.
Each feature index links its product, design, and execution records.
The [spec inventory](inventory.md) records status, task counts, and known discrepancies.
This index supplies navigation without duplicating requirements.

| Product area | Requirements and current user guidance |
| --- | --- |
| New user onboarding | [Onboarding journey](new-user-onboarding.md), [quickstart](../getting-started/quickstart.md) |
| Agent definition | [Configuration guide](../guides/agent-configuration.md), [published schema](../../schemas/agent.schema.json) |
| Runtime backends | [Claude](../guides/claude-backend.md), [OpenAI](../guides/openai-backend.md), [backend spec](035-openai-agents-backend/index.md) |
| Tools and retrieval | [Tools](../guides/tools.md), [vector stores](../guides/vector-stores.md) |
| Evaluation and comparison | [Evaluations](../guides/evaluations.md), [dashboard](../guides/dashboard.md), [eval-run spec](031-eval-runs-dashboard/index.md) |
| Optimization | [Optimizer guide](../guides/optimizer.md), [optimizer spec](033-holodeck-test-optimizer/index.md) |
| Serving and deployment | [Server guide](../guides/serve.md), [deployment guide](../guides/deployment.md) |
| Temporal activities | [Spec 040](040-holodeck-temporal/spec.md), [Temporal guide](../guides/temporal.md), [worker guide](../guides/temporal-worker.md) |
| Proposed Temporal file inputs | [Spec 041](041-temporal-file-inputs/spec.md), currently draft |
| Proposed dependency stack revamp | [Spec 042](042-dependency-stack-revamp/index.md): native SDKs, LlamaIndex retrieval, retained DeepEval, Microsoft/legacy evaluation removal |

Read the target feature's requirements, decisions, and relevant acceptance tests before implementation.
Older status labels and unchecked tasks can lag merged code. Resolve uncertainty through code and git evidence.
Track discrepancies in [technical debt](../exec-plans/tech-debt-tracker.md).

The [migration ledger](../generated/spec-migration.json) preserves original-to-current file locations.
Research is indexed under [design documents](../design-docs/index.md). Plans and tasks are indexed under [execution plans](../PLANS.md).
