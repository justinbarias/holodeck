# OpenAI Agents full-capability sample

One agent that exercises every shipped surface of the OpenAI Agents backend
(`model.provider: openai` / `azure_openai`). Use it to smoke-test a HoloDeck
install, to see handoffs and skills in the chat tools panel, or as a template.

| Capability | Where in `agent.yaml` |
| --- | --- |
| Function tools (sync Python) | `get_inventory`, `reserve_stock` in `tools/warehouse.py` |
| Disallowed tool filtering | `purge_inventory` declared, removed by `openai.disallowed_tools` |
| Vectorstore RAG with `embedding_dimensions` | `knowledge_base` over `data/kb/*.md` |
| Hierarchical-document hybrid RAG | `handbook` over `docs/handbook.md` |
| MCP server (stdio) with a static tool filter | `filesystem` (needs Node.js / `npx`) |
| Subagents / handoffs | `openai.agents`: `researcher` (restricted), `analyst` (inherits all), `writer` (no tools, prompt file, prefix opt-out) |
| Skills | `summarise` (inline) and `research-assistant` (`skills/research-assistant/SKILL.md`) |
| Reasoning effort + `thinking` | `openai.effort: low` on `gpt-5-mini` |
| Budget cap | `openai.max_budget_usd` |
| Fallback model | `openai.fallback_model` |
| Tracing (OTel mirror) | `observability` block (disabled by default) |
| Structured output | `schemas/answer.schema.json` (commented `response_format`) |
| Test cases: tool assertions, ground truth, LLM-graded metric | `test_cases`, `evaluations` |

## Run

```bash
cd sample/openai-agents-full
cp .env.example .env            # OPENAI_API_KEY
holodeck test run agent.yaml -n 1   # smoke: one function-tool case
holodeck test run agent.yaml        # all seven cases (RAG ingest runs first)
holodeck chat agent.yaml            # watch handoffs nest in the tools panel
```

Requirements: the `openai-agents` extra, and Node.js for the MCP server
(remove the `filesystem` tool if `npx` is unavailable). The first run ingests
the two RAG sources with `text-embedding-3-small`; later runs reuse the cache
(`-f` forces re-ingest).

## Azure OpenAI

Set `model.provider: azure_openai`, make `model.name` (and `fallback_model`,
and every subagent `model`) deployment names, add
`endpoint: ${AZURE_OPENAI_ENDPOINT}`, and use `AZURE_OPENAI_API_KEY`. Do the
same for `embedding_provider`. Trace upload to platform.openai.com is
suppressed automatically for Azure.

## What each test case proves

1. `function tool lookup` — a plain function tool call.
2. `vectorstore RAG` — `knowledge_base_search` is called (substring match on `knowledge_base`).
3. `hierarchical document RAG` — `handbook_search` over structured sections.
4. `handoff to researcher` — `transfer_to_researcher` fires; the researcher only sees the two RAG tools.
5. `handoff to analyst with inherited tools` — the analyst inherited `reserve_stock` from the parent.
6. `skill route` — the inline skill is a handoff target (`transfer_to_summarise`).
7. `disallowed tool never offered` — `purge_inventory` was filtered before build, so the model cannot call it.

The `grounded-answer` G-Eval metric runs on every case using the agent's model.
