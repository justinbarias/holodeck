# Docs Overhaul Plan — Ship the OpenAI Native Backend

**Goal:** Make the docsite ship-ready for spec 035: (1) remove Semantic Kernel as a
user-facing concept, (2) document the new OpenAI Agents native backend, (3) restructure
every guide to lead with a light "quick start" section, (4) update the CHANGELOG.
This plan absorbs and extends task **K3** from `plan-full.md`.

## Ground truth (verified 2026-06-13)

- **Routing** (`lib/backends/selector.py`): `openai`/`azure_openai` → `OpenAIAgentsBackend`;
  `anthropic`/`ollama` → `ClaudeBackend`. **`SKBackend` no longer exists.**
- **`semantic-kernel` remains a pinned internal dependency** — used by
  `lib/vector_store.py` (store connectors incl. qdrant native hybrid),
  `lib/text_chunker.py`, observability glue. It is an implementation detail,
  not a backend.
- Implemented 035 surface to document: `openai:` block (`max_turns`, `effort`
  incl. `max → "xhigh"`, `max_budget_usd`, `fallback_model`, `disallowed_tools`,
  `permissions`), function/vectorstore/hier-doc tools, MCP stdio/sse/http
  (websocket skipped), structured output (`response_format` → strict-eligible
  JSON schema; **`oneOf` rejected by OpenAI — use `anyOf`**), `thinking` via
  reasoning summaries, OTel trace mirror + `observability.disable_provider_tracing`,
  `trace_include_sensitive_data` ← `capture_content`. NOT yet shipped (don't
  document as available): subagents/handoffs (D), YAML hooks (E), hosted tools (G),
  serve cap / deploy sizing (I), redaction guardrails (J).

## Decisions

1. **SK mentions:** removed everywhere user-facing. Two carve-outs:
   - `docs/CHANGELOG.md` **historical entries stay untouched** (a changelog is a
     record; rewriting it is revisionism). The grep gate excludes it.
   - Internal API docs (`api/utilities.md` chunker, `api/observability.md`,
     vector-store internals) **reword** to "vector store connectors" /
     "chunking pipeline" — SK branding only where factually unavoidable, and
     never as "backend".
2. **`guides/semantic-kernel-backend.md` is deleted** (not redirected — the docsite
   has no redirect infra; nav entry removed). Its still-true content (Ollama via
   Claude backend, provider env vars) migrates to `llm-providers.md`.
3. **New guide `guides/openai-backend.md`** mirrors `claude-backend.md`'s role.
4. **Guide template** (applies to every guide):

   ```markdown
   # <Feature>
   ## Quick start            ← ≤30 lines: minimal agent.yaml + one command + expected output
   ## How it works           ← 3–6 sentences of concepts, link out, no exhaustive tables
   ## <Detailed sections>    ← existing reference content, condensed
   ## Troubleshooting        ← keep last
   ```

   Verbosity rule: quick start must be copy-paste runnable; reference tables move
   below the fold; duplicated explanations across guides become links.

## Phase 1 — Backend realignment (the correctness phase)

| File | Action |
|---|---|
| `guides/openai-backend.md` | **NEW.** Quick start (azure_openai warehouse-style agent + `holodeck test`); `openai:` block reference; effort (`max → xhigh` deviation); budget/fallback semantics (retry-exhaust-then-one-fallback); structured output + **`anyOf`-not-`oneOf` portability callout**; thinking; MCP transports (websocket skip); RAG tools; tracing behaviour table (openai = mirror + platform upload, azure = mirror only, `disable_provider_tracing`, `capture_content` coupling); per-backend semantics matrix (K3) for the shipped surface; "coming soon" list for D/E/G/I/J. |
| `guides/semantic-kernel-backend.md` | **DELETE**; salvage Ollama/env-var content into `llm-providers.md`. |
| `guides/llm-providers.md` (174 ln) | Rewrite routing table: provider → backend (openai/azure → OpenAI Agents; anthropic/ollama → Claude). Quick-start-first. |
| `mkdocs.yml` | Nav: remove SK entry; add "OpenAI Backend" after "Claude Backend". |
| `api/backends.md` (253 ln) | Protocol docs: replace SKBackend with OpenAIAgentsBackend; selector routing; ExecutionResult fields incl. `structured_output`/`thinking`. |
| `index.md`, `getting-started/installation.md`, `getting-started/quickstart.md` | Sweep SK mentions; quickstart gets a provider-tabbed (openai/azure/anthropic) example. |
| `CLAUDE.md` + `AGENTS.md` | Fix the stale routing table (currently says OpenAI/Azure/Ollama → SKBackend). |

## Phase 2 — Guide restructure (quick-start-first, 13 guides)

Apply the template. Current sizes → targets are guidance, not hard caps:

| Guide | Lines | Restructure notes |
|---|---|---|
| `tools.md` | 1583 | Biggest offender. Quick start = one function tool. Split detail by tool type; MCP detail links to `mcp-cli.md`. Target ≤800. |
| `evaluations.md` | 1135 | Quick start = one numeric metric on one test case. Metric catalog → reference tables below. Target ≤700. |
| `agent-configuration.md` | 801 | Quick start = minimal valid agent.yaml. Field-by-field detail stays, grouped. |
| `vector-stores.md` | 753 | Quick start = local qdrant + one vectorstore tool. Reword SK-connector mentions per Decision 1. |
| `global-config.md` | 650 | Quick start = `~/.holodeck/.env` + precedence one-liner. |
| `deployment.md` | 618 | Quick start = `deploy build` + `deploy run` happy path. SK mention sweep. |
| `observability.md` | 612 | Quick start = Aspire docker one-liner + otlp block (validated live this week). Add openai-backend trace-mirror subsection + link to new guide. |
| `serve.md` | 577 | Quick start = `holodeck serve` + curl. SK mention sweep. |
| `claude-backend.md` | 556 | Already backend guide; apply template; cross-link comparison matrix. |
| `file-references.md` | 550 | Template only. |
| `mcp-cli.md` | 415 | Template only. |
| `optimizer.md` / `dashboard.md` | 188/151 | Light touch — verify template shape, likely fine. |

## Phase 3 — Reference + periphery sweep

- `api/*.md` (models, evaluators, utilities, observability, test-runner): SK rewording
  per Decision 1; add `OpenAIConfig` to `api/models.md`.
- `examples/README.md`: add the financial-assistant openai sample walk-through
  (note `sample/` is gitignored; show the YAML inline).
- `docs/security/aca-limitations.md`: deferred-surfaces note (P3/sandbox/computer-use)
  per K3.

## Phase 4 — CHANGELOG + verification gates

1. **CHANGELOG** (`docs/CHANGELOG.md`, `[Unreleased]`): Added — OpenAI Agents native
   backend (tools, MCP, structured output/thinking, effort/budget/fallback/disallowed,
   OTel trace mirror, `openai:` block); Changed — `openai`/`azure_openai` now route to
   the native backend (was SK), `ollama` routes to the Claude backend; Removed —
   Semantic Kernel backend; docs overhaul note.
2. **Gates (run in order):**
   - `grep -riE "semantic.kernel|skbackend" docs/ --exclude=CHANGELOG.md` → **zero hits**
     outside the Decision-1 carve-outs (target: zero, carve-outs justified inline).
   - `mkdocs build --strict` clean (catches broken nav/links).
   - Every guide's first H2 is "Quick start" — `grep -L` check.
   - Quick-start snippets in the two backend guides validated against
     `schemas/agent.schema.json` (the financial-assistant + warehouse YAMLs already
     pass live — reuse them).
   - Manual: render `openai-backend.md` and `llm-providers.md`, eyeball nav.

## Execution shape

Phase 1 first (correctness before style), then Phase 2 fanned out per-guide
(independent files — parallel subagents), Phases 3–4 sequential close-out.
Commits: one per phase, conventional (`docs(035): …`), no AI attribution.
