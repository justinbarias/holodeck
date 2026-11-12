# Temporal Knowledge Graph Memory — specification

Feature: `043-temporal-memory`. **Status:** approved for build 2026-11-11; plan in progress.

## Documents

- [Product specification](spec.md)

## Related records

- [Design](../../design-docs/043-temporal-memory/design.md)
- [Graphiti evidence](../../design-docs/043-temporal-memory/graphiti-evidence.md)
- [Execution plan](../../exec-plans/active/043-temporal-memory/index.md)
- [Feature inventory](../inventory.md)
- [GraphRAG research (archived)](../016-graphrag-integration/index.md)
- [HoloDeck Agents on Temporal](../040-holodeck-temporal/index.md)
- [Dependency stack revamp](../042-dependency-stack-revamp/index.md)
- [Architecture](../../../ARCHITECTURE.md)

This spec binds a host-neutral temporal knowledge graph design to HoloDeck. The MVP is a `temporal_memory` tool on a Graphiti backend behind a `MemoryBackend` port. The native PostgreSQL backend is specified as Phase 2 and is not built.
