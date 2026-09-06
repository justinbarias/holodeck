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
| Test cases: tool assertions, ground truth, G-Eval + RAG faithfulness metrics | `test_cases`, `evaluations` |

## Run

```bash
cd sample/openai-agents-full
cp .env.example .env            # OPENAI_API_KEY (or the Azure variables below)
holodeck test run agent.yaml -n 1   # smoke: one function-tool case
holodeck test run agent.yaml        # all seven cases (RAG ingest runs first)
holodeck chat agent.yaml            # watch handoffs nest in the tools panel
```

Each run writes an EvalRun JSON under `results/` (gitignored).

Requirements: the `openai-agents` extra, and Node.js for the MCP server
(remove the `filesystem` tool if `npx` is unavailable). The first run ingests
the two RAG sources with `text-embedding-3-small`; later runs reuse the cache
(`-f` forces re-ingest).

## Azure OpenAI

`agent.azure.yaml` is the same agent pointed at Azure OpenAI deployments. It
reads `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`,
`AZURE_OPENAI_DEPLOYMENT_NAME` (a `gpt-5-mini` deployment) and
`AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` from `.env`; `fallback_model` is
commented out because it needs a second deployment. Trace upload to
platform.openai.com is suppressed automatically for Azure.

```bash
holodeck test run agent.azure.yaml
```

## Qdrant instead of the in-memory vector store

`agent.azure.yaml` persists the `knowledge_base` vectors in Qdrant; `agent.yaml`
has the same block commented out. Start a server and set `QDRANT_URL`:

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
echo 'QDRANT_URL=http://localhost:6333' >> .env
```

Re-runs skip unchanged files (record IDs are deterministic UUIDs on Qdrant).
The loader only expands plain `${VAR}` references, so there are no
`${VAR:-default}` fallbacks in either file.

## What each test case proves

1. `function tool lookup` — a plain function tool call.
2. `vectorstore RAG` — `knowledge_base_search` is called (substring match on `knowledge_base`).
3. `hierarchical document RAG` — `handbook_search` over structured sections.
4. `handoff to researcher` — `transfer_to_researcher` fires; the researcher only sees the two RAG tools.
5. `handoff to analyst with inherited tools` — the analyst inherited `reserve_stock` from the parent.
6. `skill route` — the inline skill is a handoff target (`transfer_to_summarise`).
7. `disallowed tool never offered` — `purge_inventory` was filtered before build, so the model cannot call it.

The `direct-answer` G-Eval metric runs on every case and compares the reply with `ground_truth`; tool usage is asserted by `expected_tools`. The `faithfulness` RAG metric scores every case that called `knowledge_base_search` or `handbook_search` (their output is forwarded as `retrieval_context`, also from inside handoffs) and is skipped on the others.
