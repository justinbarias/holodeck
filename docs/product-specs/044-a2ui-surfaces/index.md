# Native A2UI Surfaces — specification

Feature: `044-a2ui-surfaces`. **Status:** draft 2026-11-27. Nothing is built.

## Documents

- [Product specification](spec.md)

## Related records

- [Execution plan](../../exec-plans/active/044-a2ui-surfaces/index.md)
- [Agent local server](../017-agent-local-server/index.md)
- [OpenAI Agents backend](../035-openai-agents-backend/index.md)
- [Feature inventory](../inventory.md)
- [Architecture](../../../ARCHITECTURE.md)

This spec adds an `a2ui` tool type. HoloDeck validates A2UI 1.0 component trees in Python and emits A2UI surfaces on the `holodeck serve` AG-UI stream, in the same wire shape as `@ag-ui/a2ui-middleware`. The OpenAI Agents AG-UI path gains parity with the Claude path first.
