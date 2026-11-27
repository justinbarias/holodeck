# Backend-Agnostic Chat Thread Store — specification

Feature: `045-thread-store`. **Status:** draft 2026-11-15; not built.

## Documents

- [Product specification](spec.md)

## Related records

- [Design](../../design-docs/045-thread-store/design.md)
- [Feature inventory](../inventory.md)
- [Agent local server](../017-agent-local-server/index.md)
- [Production hardening](../034-production-hardening/index.md)
- [OpenAI Agents backend](../035-openai-agents-backend/index.md)
- [HoloDeck Agents on Temporal](../040-holodeck-temporal/index.md)
- [Architecture](../../../ARCHITECTURE.md)

This spec adds durable chat threads to `holodeck serve`. One `ThreadStore` port holds the thread record, a canonical message log and each SDK's native state, with in-memory, SQLite and PostgreSQL adapters.
