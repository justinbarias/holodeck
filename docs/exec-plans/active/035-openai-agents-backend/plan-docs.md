# Docs Overhaul Plan — Ship the OpenAI Native Backend

**Execution handoff:** Use the [completion execution plan](2026-09-06-complete-035.md) for current task order, acceptance gates, and progress.
This record retains the reconciled design and historical task evidence.

**Goal:** Make the docsite ship-ready for spec 035: (1) remove Semantic Kernel as a
user-facing concept, (2) document the new OpenAI Agents native backend, (3) restructure
every guide to lead with a light "quick start" section, (4) update the CHANGELOG.
This plan absorbs and extends task **K3** from `plan-full.md`.

## Reconciliation — 2026-09-06

**Status: substantially implemented; documentation and validation follow-ups remain.**
The phase tables below preserve the original scope. This checklist records current
completion, based on tracked files at `b585e80` and implementation commit `3805e80`
(PR #338). Presence of documentation does not establish runtime support or prove
that its examples have passed validation.

### Completed or superseded

- [x] Phase 1: `docs/guides/openai-backend.md` covers the configuration reference,
  effort, budget/fallback, structured output, thinking, MCP, RAG, tracing,
  backend comparison, and deferred surfaces.
- [x] Phase 1: the SK guide is absent; `docs/guides/llm-providers.md` documents
  current routing, Ollama setup, and provider environment variables. The MkDocs
  navigation places OpenAI Backend after Claude Backend.
- [x] Phase 1: `docs/api/backends.md` documents the current selector and
  `ExecutionResult.structured_output` / `thinking`. The landing, installation,
  and quickstart pages contain no SK references.
- [x] Phase 1: `AGENTS.md` states current routing; `CLAUDE.md` imports it.
- [x] Phase 2: all 13 named guides have `Quick start` as their first H2.
  `tools.md` is 507 lines and `evaluations.md` is 464 lines, below the suggested
  targets. The observability guide includes the OpenAI trace-mirror subsection.
  Full template compliance and snippet runnability remain separate checks below.
- [x] Phase 3: `docs/api/models.md` includes `OpenAIConfig`. The surveyed API pages
  no longer refer to an SK backend. The remaining `enable_semantic_kernel_telemetry`
  references in `docs/api/observability.md` document an existing internal symbol
  and fit Decision 1's API carve-out.
- [x] Phase 3: `docs/security/aca-limitations.md` explicitly identifies P3,
  sandbox mode, and computer-use as deferred.
- [x] Phase 3 examples approach superseded: `docs/examples/README.md` links to the
  external `holodeck-samples` repository, lists Financial Assistant, and supplies
  clone/run instructions for its OpenAI variant. Inline financial-assistant YAML
  was not added; the tracked implementation uses the sample repository instead.
- [x] Phase 4: the Unreleased changelog records the backend, routing change,
  removed SK agent path, and documentation overhaul. Accuracy issues remain below.

### Remaining work

- [ ] Add the planned provider-tabbed OpenAI/Azure/Anthropic example to
  `docs/getting-started/quickstart.md`; its current agent example is Azure-only.
- [ ] Make the OpenAI backend quick start self-contained. It references
  `tools/warehouse.py` without supplying the implementation, describes three
  function tools but declares one, and exceeds the proposed 30-line section limit.
  Validate both backend examples after fixing prerequisites and dependencies.
- [ ] Reconcile the remaining Phase 2 template deviations: `dashboard.md` lacks
  `How it works`; 12 of the 13 guides place next steps/resources after
  troubleshooting; the Claude guide links to the OpenAI guide but does not link
  directly to its comparison matrix. Either complete the original template or
  explicitly relax those requirements in a follow-up decision.
- [ ] Remove or explicitly approve the SK connector branding in
  `docs/guides/vector-stores.md` under Decision 1. It accurately describes an
  internal dependency, but the original carve-out only covers internal API docs.
- [ ] Correct the Unreleased changelog's residual SK scope: its Removed entry
  still assigns embeddings and context generation to SK, although those paths
  now use LiteLLM. Its blanket claim that every quick start is runnable is also
  unsupported by the warehouse example above.
- [ ] Reconcile the guide/changelog deferred feature lists with
  [the full implementation plan](plan-full.md), especially broad statements that
  OpenAI `serve` / `deploy` are entirely unavailable versus unfinished capacity
  enforcement and sizing. Do not infer availability from accepted config fields.

### Verification evidence and outstanding gates

- [x] Read the changed documentation and inspected relevant Git history.
- [x] Mechanically checked the first H2 and line counts of the 13 Phase 2 guides.
- [x] Searched current guides, API docs, getting-started pages, and `docs/index.md`
  for SK references; the hits are the vector-store paragraph and telemetry API
  symbol described above.
- [ ] Run schema validation on both backend quick-start YAML snippets.
- [x] `uv run mkdocs build --strict` passed during the parent audit; see [verification](reconciliation.md#verification-in-this-audit).
- [ ] Render the OpenAI backend and provider guides and inspect navigation;
  not rerun in this audit.

The original whole-`docs/` grep gate predates the migration of historical specs
and execution plans into `docs/`. Use the current user-facing pages for this gate;
historical specs, execution plans, ideas, and changelog entries are evidence, not
current backend instructions. This scope change does not waive the outstanding
vector-store wording issue. Repository harness/build results for this
reconciliation are recorded in the parent reconciliation plan.

## Historical ground truth (verified 2026-06-13)

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
