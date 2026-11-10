# Dependency Stack Revamp — specification

Feature: `042-dependency-stack-revamp`. **Status:** Implemented 2026-11-10; live evidence open.

## Documents

- [Product specification](spec.md)
- [Dependency inventory](dependency-inventory.md)
- [Selection evidence and provider assessment](../../design-docs/042-dependency-stack-revamp/selection-evidence.md)

## Related records

- [Execution plan](../../exec-plans/active/042-dependency-stack-revamp/index.md) and its [acceptance evidence](../../exec-plans/active/042-dependency-stack-revamp/2026-11-10-remove-semantic-kernel.md#acceptance-evidence)
- [Baseline and result measurements](../../exec-plans/active/042-dependency-stack-revamp/baseline-measurements.md)
- [Upgrade guide for 0.8](../../guides/upgrading-to-0.8.md)
- [Feature inventory](../inventory.md)
- [Native OpenAI backend migration](../035-openai-agents-backend/index.md)
- [SK decommission debt](../../exec-plans/tech-debt-tracker.md#h-008)
- [Architecture](../../../ARCHITECTURE.md)

This spec owned the dependency consolidation and the final Semantic Kernel removal. The code and the plan show that work is implemented. It does not claim hosted-service evidence for Pinecone, Azure AI Search or Cosmos DB; the plan lists what is open. It does not close 035 acceptance.
