# Dependency Stack Revamp — execution plans

Feature: `042-dependency-stack-revamp`. Product contract: [spec 042](../../../product-specs/042-dependency-stack-revamp/index.md).

**Status:** Implementation complete 2026-11-10; live evidence open. The [execution plan](2026-11-10-remove-semantic-kernel.md) records the acceptance evidence and the items that keep it in `active/`: no hosted-service run for Pinecone, Azure AI Search and Cosmos DB, the CopilotKit browser step, and the FR-029 provenance review. Each is in the [technical debt tracker](../../tech-debt-tracker.md) (H-023 to H-031).

## Documents

- [2026-11-10-remove-semantic-kernel.md](2026-11-10-remove-semantic-kernel.md) — execution checklist and record: 25 tasks (T0 to T24, all done), three checkpoints, acceptance table with evidence, operation matrix, open items
- [baseline-measurements.md](baseline-measurements.md) — install size, package count and startup time for base and all-extras installs: the T0 baseline and the result after spec 042
