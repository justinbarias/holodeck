# Execution plan: Temporal Knowledge Graph Memory on Graphiti

Feature: `043-temporal-memory`. Product contract: [spec 043](../../../product-specs/043-temporal-memory/spec.md).
**Status:** Plan approved 2026-11-11. T0 and T1 done; Checkpoint A passed; Phase 0 committed on `feat/043-spec`; T2 done on `feat/043-contracts` (uncommitted); T3 next. Nothing pushed.
**Code baseline:** `6203c82` (release 0.8.0).

## 1. Objective and acceptance

Ship the `temporal_memory` tool on a Graphiti backend behind the `MemoryBackend` port, as the spec's MVP defines. The work is done when the twelve success criteria in spec section 11 pass. Section 6 maps each criterion to the task that proves it.

## 2. Scope and constraints

- In scope: spec sections 4 to 12. Out of scope: the native backend and everything else the spec marks Phase 2.
- Ask-first items are held at checkpoints and never guessed: the `graphiti-core` dependency (T0), the chunk accessor on `HierarchicalDocumentTool` (T5), any front-matter change in the chunker, a new port operation, and live LLM spend (T1, T22).
- Delivery is five stacked PRs made with `gh stack`, one per phase, bottom first. The bottom branch, `feat/043-spec`, carries the spec, this plan, the evidence and the dependency. Each layer passes its own gates. Remote writes need maintainer authorization.
- Each PR layer gets one adversarial review before commit and push. The maintainer picks Codex (gpt-6-astra, high) or an Opus 5.5 subagent (high) at that point. The stack gets one integrated pass before merge.
- Spawning any subagent needs the maintainer's approval at spawn time. The model and effort below are the proposal reviewed with this plan.

### Code this plan builds on

| Concern | Location |
| --- | --- |
| Tool config models and the tool union | `src/holodeck/models/tool.py`, `src/holodeck/models/agent.py` |
| Tool initialization and the shared embedding service | `src/holodeck/lib/tool_initializer.py` |
| Chunks held by the hierarchical tool (`_chunks`, `preview_chunks`, `search`) | `src/holodeck/tools/hierarchical_document_tool.py` |
| Claude tool adapters | `src/holodeck/lib/backends/tool_adapters.py` |
| OpenAI Agents tool adapters | `src/holodeck/lib/backends/openai_agents_tool_adapters.py` |
| Lazy CLI group and early telemetry environment variables | `src/holodeck/cli/main.py`, `src/holodeck/cli/commands/` |
| Extras and the `all` aggregate test | `pyproject.toml`, `tests/unit/test_install_profiles.py` |
| Schema generation | `scripts/generate_agent_schema.py`, `make schema`, `make schema-check` |

### Dependency graph

```text
T0 dependency ─▶ T1 Graphiti evidence ─▶ Checkpoint A
                                             │
T2 port, types, fake ─▶ T3 config model ─▶ T4 tool schemas
        │                    │
        └──▶ T5 chunk accessor, effective date ─▶ Checkpoint B
                                             │
        T6 service ingest ──┐                │
        T7 Graphiti ingest ─┼▶ T23 Agent SDK client, T8 spans ─▶ T9 CLI ─▶ T10 integration ─▶ Checkpoint C
                            │
        T11 Graphiti read ──┐
        T12 service search ─┼▶ T13 tool, Claude ─▶ T14 OpenAI Agents ─▶ T15 Temporal ─▶ Checkpoint D
                            │
        T24 sample location, timeline ─▶ T25 as-of results, provenance ─▶ Checkpoint D2 (after T13)
                            │
        T16 sample ─▶ T17 fetch ─▶ T18 questions ─▶ T19 eval ─▶ T22 live runs ─▶ Checkpoint E
        T20 ID coverage check, T21 docs (after T15)
```

Parallel pairs once their inputs exist: T6 with T7, T11 with T12, T17 with T16, T20 with T21. Each pair has separate file ownership.

## 3. Tasks

Sizes: S is 1 to 2 files, M is 3 to 5. Every task also runs focused Black, Ruff and MyPy on the files it touches.

### Phase 0: De-risk (PR 0, `feat/043-spec`)

- [x] **T0: Resolve the `graphiti-core` dependency.** Add a `graphiti` extra with `graphiti-core[anthropic]>=0.31,<0.32`, list it in `all`, and lock.
  - Acceptance: `uv lock` resolves with `openai>=3,<4`; `tests/unit/test_install_profiles.py` passes; new transitive packages are listed in the evidence file.
  - Verify: `uv lock && uv sync --extra graphiti && uv run pytest tests/unit/test_install_profiles.py -n auto -q`
  - Depends on: none. Files: `pyproject.toml`, `uv.lock`, `src/holodeck/deploy/constraints.txt`, `scripts/verify/profiles.py`, `scripts/verify/profile_probe.py`. Size: S. Owner: orchestrator, inline. **Ask first.**
  - If the lock fails, stop and report; do not add an override.

- [x] **T1: Graphiti evidence run.** A throwaway script against Neo4j in Docker answers the spec's open questions 4 and 5 and assumptions A3 and A4.
  - Acceptance: `docs/design-docs/043-temporal-memory/graphiti-evidence.md` records, with the command and output for each: (a) what `add_episode` does when the `uuid` already exists; (b) that `SearchFilters` on `valid_at`, `invalid_at`, `created_at` and `expired_at` return the expected edges on a three-episode fixture; (c) how breadth-first search depth is limited to 1 and 2; (d) the result of ingesting an earlier-dated episode after a later one; (e) a search configuration that makes no cross-encoder call; (f) no PostHog request with telemetry off.
  - Verify: the evidence file exists and each of (a) to (f) has an observed result, not an expectation.
  - Depends on: T0. Files: evidence file, a script under the scratchpad (not committed). Size: M. Model: Opus 5.5, medium. Uses live LLM calls on a tiny fixture: **ask first**.

**Checkpoint A (maintainer).**
- [x] Dependency approved or rejected. Evidence (a) to (f) reviewed. The `ChunkSource` accessor shape in T5 approved. Spec updated first if any finding changes it.

### Phase 1: Contracts (PR 1, `feat/043-contracts`)

- [x] **T2: Port, types, errors and the fake backend.**
  - Acceptance: `MemoryBackend` and `MemoryCapabilities` in `ports.py`; frozen dataclasses in `types.py` with UTC-aware datetimes rejected when naive; three typed errors; `FakeBackend` implements every port method deterministically and counts calls.
  - Verify: `uv run pytest tests/unit/lib/memory/test_ports.py tests/unit/lib/memory/test_fake_backend.py -n auto -q` (covers MB-01 to MB-04).
  - Depends on: Checkpoint A. Files: `src/holodeck/lib/memory/{__init__,ports,types,errors}.py`, `backends/fake.py`, tests. Size: M. Owner: orchestrator, inline (boundary contract).

- [x] **T3: `TemporalMemoryToolConfig` and agent-level validation.**
  - Acceptance: the model has `extra="forbid"` and the fields in spec section 4; loading fails with the exact tool name for an empty, repeated or wrong-type `source_tools` entry (CFG-02), a second memory tool (CFG-03), `single_valued_predicates` on Graphiti (CFG-04) and mismatched embeddings (CFG-06); `schemas/agent.schema.json` regenerated.
  - Verify: `uv run pytest tests/unit/models/test_temporal_memory_tool.py -n auto -q && make schema && make schema-check`
  - Depends on: T2. Files: `src/holodeck/models/tool.py`, `src/holodeck/models/agent.py`, `schemas/agent.schema.json`, template schema copies, tests. Size: M. Owner: orchestrator, inline (schema).

- [x] **T4: Host-neutral tool schemas.**
  - Acceptance: `src/holodeck/lib/memory/tool_api.py` holds the Pydantic argument and result models of the three tools, the error result, and converters from the port dataclasses (design seam 6); `schemas/memory/` holds JSON Schemas generated from those models; a check fails if they drift. The `call()` dispatcher is T13.
  - Verify: `uv run pytest tests/unit/lib/memory/test_tool_api.py -n auto -q && uv run python scripts/generate_memory_schemas.py --check`
  - Depends on: T2. Files: `src/holodeck/lib/memory/tool_api.py`, `scripts/generate_memory_schemas.py`, `schemas/memory/*.json`, `Makefile`, tests. Size: S. Model: Sonnet 5.5, medium.

- [x] **T5: Chunk accessor and effective-date reader.**
  - Acceptance: the `ChunkSource` protocol of design seam 2 (`name`, `documents()`, `chunks()`, `find()`), an adapter in `src/holodeck/tools/hierarchical_chunk_source.py`, and one read-only `document_chunks()` on `HierarchicalDocumentTool`, with no change to its search behaviour; `effective_date.py` returns the date and its source (`front_matter`, `mtime`, `ingestion`) for a configurable key; chunk text is unchanged (A6).
  - Verify: `uv run pytest tests/unit/lib/memory/test_effective_date.py tests/unit/tools/test_hierarchical_chunk_source.py -n auto -q`; the existing hierarchical tool tests still pass.
  - Depends on: T2, Checkpoint A. Files: `src/holodeck/lib/memory/{chunk_source,effective_date}.py`, `src/holodeck/tools/{hierarchical_document_tool,hierarchical_chunk_source}.py`, tests. Size: M. Model: Opus 5.5, medium (interface extension).

**Checkpoint B.**
- [x] `make schema-check` and `make harness-check` pass; unit tests green; adversarial review of PR 1 has no open finding; PR 1 committed.

### Phase 2: Ingestion (PR 2, `feat/043-ingest`)

- [x] **T6: Service ingest path.**
  - Acceptance: `episode_id_for` matches the spec formula; sources from all source tools are submitted in one ascending effective-date order (ING-04); existing episodes are skipped with zero backend writes (ING-01, ING-02); `dry_run` reports the new-episode count with zero LLM calls (ING-08); a backend failure stops the run and the report lists what was written (ING-05).
  - Verify: `uv run pytest tests/unit/lib/memory/test_service_ingest.py -n auto -q`, including Hypothesis properties: re-ingest is a no-op; one changed chunk yields exactly one new episode.
  - Depends on: T5. Files: `src/holodeck/lib/memory/service.py`, tests. Size: M. Model: Sonnet 5.5, high.

- [x] **T7: Graphiti backend, write side.**
  - Acceptance: builds the driver for `neo4j`, `falkordb`, `neptune` and `kuzu` (a warning for `kuzu`) (GPH-03); maps `extraction_model` to the four LLM clients; sets telemetry off before the client exists (GPH-02); passes no cross-encoder (GPH-01); a missing extra raises `MemoryConfigError` with the install command (CFG-05); implements `has_episode`, `ingest`, `purge` and `capabilities`; `has_episode` is true only for a complete episode, and `ingest` pre-saves the node, calls `add_episode`, then marks it complete (D21, ING-09); passes the extraction instructions (ING-10); always passes a cross-encoder object and copies search recipes before use; a missing driver package fails with its install command (GPH-05); no Graphiti type leaves the module.
  - Verify: `uv run pytest tests/unit/lib/memory/test_graphiti_backend_write.py -n auto -q` with Graphiti mocked.
  - Depends on: T2, T1 findings. Files: `src/holodeck/lib/memory/backends/graphiti.py`, `registry.py`, `src/holodeck/cli/main.py` (environment variable), tests. Size: M. Model: Opus 5.5, medium.

- [x] **T23: Agent SDK client for Graphiti.**
  - Acceptance: a Graphiti `LLMClient` backed by the Claude Agent SDK returns schema-valid output or raises after three attempts; the backend selects it for `anthropic` when `auth_provider` is `oauth_token`, and refuses `bedrock`, `vertex`, `foundry` and `custom` at load (GPH-04); it runs with no tools and without `bypassPermissions`, or the task records why a stricter mode is not possible; `--dry-run` shows an estimated duration (ING-08).
  - Verify: `uv run pytest tests/unit/lib/memory/test_agent_sdk_llm_client.py -n auto -q` with the SDK mocked, plus one live episode at Checkpoint C.
  - Depends on: T7. Files: `src/holodeck/lib/memory/backends/graphiti_agent_sdk.py`, `backends/graphiti.py`, tests. Size: S. Model: Opus 5.5, medium.

- [x] **T8: Ingestion telemetry.**
  - Acceptance: one trace per run, a `holodeck.memory.ingest_episode` span per episode, token counts as attributes; the existing redaction rules apply (ING-06).
  - Verify: `uv run pytest tests/unit/lib/memory/test_ingest_spans.py -n auto -q` with an in-memory span exporter.
  - Depends on: T6, T7. Files: `service.py`, `backends/graphiti.py`, test. Size: S. Model: Sonnet 5.5, medium.

- [x] **T9: `holodeck memory` commands.**
  - Acceptance: `ingest` (with `--tool` and `--dry-run`) prints the report (CLI-01); `status` prints backend, namespace, episode count and capabilities (CLI-02); `purge` without `--yes` asks and names the namespace (CLI-03); the group is lazy, so `holodeck init` imports no memory module.
  - Verify: `uv run pytest tests/unit/cli/test_memory_commands.py -n auto -q`
  - Depends on: T6, T7. Files: `src/holodeck/cli/commands/memory.py`, `src/holodeck/cli/main.py`, tests. Size: M. Model: Sonnet 5.5, medium.

- [x] **T10: Graphiti integration suite.**
  - Acceptance: a compose file starts Neo4j; tests marked `graphiti` use a stub LLM client and prove ingest, skip on re-ingest with zero LLM calls, and purge of one namespace only; a nightly CI job runs the marker.
  - Verify: `docker compose -f tests/integration/memory/compose.yaml up -d neo4j && uv run pytest tests/integration/memory -n auto -q -m graphiti`
  - Depends on: T7, T9. Files: `tests/integration/memory/{compose.yaml,conftest.py,test_graphiti_ingest.py}`, the CI workflow, `pyproject.toml` marker. Size: M. Model: Sonnet 5.5, high.

**Checkpoint C.**
- [x] `holodeck memory ingest --dry-run` then a live ingest of a three-file fixture succeeds and a second run reports zero new episodes (ask first: small LLM spend). Adversarial review of PR 2 has no open finding; PR 2 committed.

### Phase 3: Retrieval and tools (PR 3, `feat/043-retrieval`)

- [x] **T11: Graphiti backend, read side.**
  - Acceptance: `search` applies `valid_at` and `known_at` with the known-at rule in spec section 4 (a closed fact is still returned, as open, when `expired_at > known_at`; where `SearchFilters` cannot express that, the backend filters the returned edges), each dated filter with an `IS NULL` branch and at most one dated group per field, and the hop limit (RET-03, RET-04); expansion is seeded from entity uuids (RET-09); `timeline` returns open and closed facts ordered by `valid_from` (RET-08); `provenance` resolves a fact to episodes, source tools and chunk ids (RET-06); each not-supported row in the capability table raises `MemoryNotSupportedError` with the operation name (MB-03).
  - Verify: unit tests with Graphiti mocked, plus `uv run pytest tests/integration/memory/test_graphiti_read.py -n auto -q -m graphiti`, which includes the "between two compilations" case.
  - Depends on: T7. Files: `backends/graphiti.py`, tests. Size: M. Model: Opus 5.5, medium.

- [x] **T12: Service search and fusion.**
  - Acceptance: defaults `valid_at` and `known_at` to the query time (RET-01); calls every source tool's search and maps chunk hits to episode ids (RET-02); fuses by reciprocal rank fusion with the configured weights (RET-05); every hit carries `source_tool`, `chunk_id`, `source_uri` and the valid period (RET-06); an unavailable backend gives chunk-only hits with `degraded: true` (RET-07).
  - Verify: `uv run pytest tests/unit/lib/memory/test_service_search.py -n auto -q`, including a Hypothesis property that fusion is stable under equal ranks.
  - Depends on: T5, T6. Files: `service.py`, tests. Size: M. Model: Sonnet 5.5, high.

- [x] **T13: The tool and the Claude backend binding.**
  - Acceptance: `TemporalMemoryTool` is created by the tool initializer after its source tools; `ingest_on_start` runs ingestion at initialization (ING-07); the Claude adapter exposes `memory_search`, `memory_timeline` and `memory_provenance` whose results validate against `schemas/memory/` with ISO 8601 UTC timestamps (HST-01, HST-02).
  - Verify: `uv run pytest tests/unit/tools/test_temporal_memory_tool.py tests/unit/lib/backends/test_tool_adapters.py -n auto -q`
  - Depends on: T11, T12. Files: `src/holodeck/tools/temporal_memory_tool.py`, `tool_initializer.py`, `backends/tool_adapters.py`, tests. Size: M. Model: Opus 5.5, medium.

- [x] **T14: OpenAI Agents binding and the two-backend contract test.**
  - Acceptance: the same three tools with the same schemas on the OpenAI Agents backend; one parametrized contract test runs against both backends.
  - Verify: `uv run pytest tests/unit/lib/backends/test_memory_tool_contract.py -n auto -q`
  - Depends on: T13. Files: `backends/openai_agents_tool_adapters.py`, test. Size: S. Model: Sonnet 5.5, high.

- [x] **T15: Temporal activities.**
  - Acceptance: ingestion and each memory tool call run in activities; the workflow sandbox import check still passes (HST-03).
  - Verify: the existing Temporal determinism tests plus `uv run pytest tests/unit/temporal -n auto -q -k memory`
  - Depends on: T13. Files: under `src/holodeck/temporal/`, tests. Size: S. Model: Opus 5.5, medium.

**Checkpoint D.**
- [x] Live validation loop steps 1 and 2 on the sample: `holodeck test run`, then `holodeck serve` with raw AG-UI requests on two threads, two turns each, with `TOOL_CALL_*` events for the memory tools inspected. Adversarial review of PR 3 has no open finding; PR 3 committed.

### Phase 3b: Memory tool renderers in the CopilotKit sample (PR 3b, `feat/043-ui`)

Added 2026-11-11 at the maintainer's request. The sample's wildcard tool renderer shows every backend tool call as raw JSON. This phase adds a named renderer for each memory tool. A renderer draws the tool's own typed result (`schemas/memory/`), so the model writes none of the displayed data. No agent, backend or schema change. A frontend tool that the agent calls to show a view (supported on the Claude backend through AG-UI) is not in this phase.

- [ ] **T24: Sample location and the fact timeline.**
  - Acceptance: `sample/copilotkit` is tracked in the repository through an exception to `/sample/*` in `.gitignore` (decided by the maintainer), with no `node_modules`, build output or `.env` file committed; a `memory_timeline` call renders `FactTimeline`: one lane per attribute, one bar per fact from `valid_from` to `valid_to`, closed facts dimmed, the open fact highlighted, and a draggable "as of" marker that filters in the browser with no model call; a fact with no start date is listed below the chart, not drawn; an error result (`invalid_argument`, `not_supported`, `backend_unavailable`) renders its message, not an empty chart; any other tool still uses the wildcard renderer.
  - Verify: component tests on fixture results taken from the live `cpc-live` namespace (rates 90, 100, 120, 150 and the administration change); `npm run build` and `npm run lint` pass; light and dark themes, and a 375 px wide viewport, checked in the browser.
  - Depends on: T13. Files: `sample/copilotkit/src/components/`, `.gitignore`. Size: M. Model: Opus 5.5, medium.

- [ ] **T25: As-of results and provenance.**
  - Acceptance: a `memory_search` call renders `AsOfResults`: a header with the resolved `valid_at` and `known_at`, fact hits with their valid period and a link to the superseding fact, chunk hits after them, and a banner when `degraded` is true; a `memory_provenance` call renders `ProvenanceCard`: the fact, each source document with its effective date, and the chunk text; `found: false` renders a plain "not found" line; all timestamps are shown in UTC as the result gives them.
  - Verify: component tests on fixture results, including a degraded result and a not-found result; `npm run build` and `npm run lint` pass.
  - Depends on: T24. Files: `sample/copilotkit/src/components/`. Size: S. Model: Sonnet 5.5, high.

**Checkpoint D2.**
- [ ] Live validation loop step 3: the maintainer drives the CopilotKit browser flow against `holodeck serve` on the fixture and sees the three views. `make verify-deps` passes with the sample present (npm audit). Adversarial review of PR 3b has no open finding; PR 3b committed.

### Phase 4: Evaluation and documentation (PR 4, `feat/043-eval-docs`)

- [ ] **T16: Sample agent.**
  - Acceptance: `sample/temporal-memory/` has an agent with two source tools, a small dated fixture corpus and test cases that assert memory tool calls.
  - Verify: `cd sample/temporal-memory && uv run holodeck test run agent.yaml`
  - Depends on: T14. Files: the sample directory. Size: S. Model: Sonnet 5.5, medium.

- [ ] **T17: Golden corpus fetch script.**
  - Acceptance: `manifest.yaml` pins four to six compilations of each Act with URL, start date and SHA-256; the script downloads, verifies, converts to markdown and writes `effective_date` front matter; a hash mismatch fails with the compilation id; the downloaded text is git-ignored.
  - Verify: `uv run python scripts/fetch_golden_corpus.py --out tests/golden/family-assistance/corpus` twice; the second run downloads nothing.
  - Depends on: spec open questions 1 and 2 answered. Files: `scripts/fetch_golden_corpus.py`, `tests/golden/family-assistance/manifest.yaml`, `.gitignore`. Size: M. Model: Sonnet 5.5, high.

- [ ] **T18: Labelled questions.**
  - Acceptance: `questions.yaml` has at least 30 questions with expected answers, `valid_at` dates and expected source sections, covering the five cases in spec section 9; the maintainer has reviewed every label.
  - Verify: a schema test for the file; maintainer sign-off recorded in section 5.
  - Depends on: T17. Files: `tests/golden/family-assistance/questions.yaml`, test. Size: M. Owner: orchestrator drafts, maintainer reviews.

- [ ] **T19: Eval runner.**
  - Acceptance: reports the five measures in EVAL-01 for memory and for a chunk-only baseline, as JSON and a short markdown summary; runs offline against `FakeBackend` in a unit test.
  - Verify: `uv run pytest tests/unit/lib/memory/test_eval.py -n auto -q`
  - Depends on: T12, T18. Files: `src/holodeck/lib/memory/eval.py`, tests. Size: M. Model: Opus 5.5, medium.

- [ ] **T20: Requirement-ID coverage check.**
  - Acceptance: a script reads the IDs in spec section 5 and fails, naming each ID, when no test name or case id contains it; wired into CI.
  - Verify: `uv run python scripts/check_requirement_coverage.py docs/product-specs/043-temporal-memory/spec.md` exits 0; removing one ID from a test name makes it exit 1.
  - Depends on: T15. Files: the script, the CI workflow. Size: S. Model: Sonnet 5.5, medium.

- [ ] **T21: Documentation.**
  - Acceptance: `docs/guides/temporal-memory.md` covers setup, configuration, the three tools and the capability table; `ARCHITECTURE.md`, the tools guide, the quality map, the schema inventory and the spec inventory row are updated.
  - Verify: `make harness-check && uv run mkdocs build`
  - Depends on: T15. Files: the guide and the listed documents. Size: M. Model: Sonnet 5.5, medium.

- [ ] **T22: Live evidence.**
  - Acceptance: the dry-run episode count for the full corpus is recorded and its spend approved; the live golden-corpus eval report is committed under this directory; the CopilotKit browser step of the live validation loop is done with the maintainer.
  - Verify: the report file exists with all five measures; section 5 records the CopilotKit observations.
  - Depends on: T19, T16. Files: evidence files in this directory. Size: S. Owner: orchestrator with maintainer. **Ask first** (LLM spend).

**Checkpoint E.**
- [ ] Adversarial review of PR 4, then one integrated adversarial pass over `git diff main...feat/043-eval-docs`, have no open finding. Every row in section 6 has evidence. Spec inventory row updated. The stack is ready for the maintainer to authorize merge.

## 4. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| `graphiti-core` does not resolve against `openai>=3,<4` | High: blocks the MVP | T0 runs first; on failure stop and return to the maintainer |
| Graphiti duplicates an episode when the uuid exists | High: breaks idempotent re-ingest | T1 (a) measures it; the backend checks `has_episode` before any write regardless |
| Hop limit or retroactive invalidation does not behave as the spec assumes | Medium | T1 (c) and (d); update the spec before Phase 1 |
| Ingest cost of the full corpus | Medium | Dry-run count before any live run; chunk-hash skipping; approval at T22 |
| Graphiti minor releases change signatures | Medium | Pin to one minor; all Graphiti calls in one module; contract tests |
| Legislation licence or labelling effort | Medium: blocks T17 and T18 only | Answer open questions 1 and 2 during Phase 1; Phases 1 to 3 do not depend on them |
| The chunk accessor changes hierarchical tool behaviour | Medium | Read-only protocol; existing tool tests are part of T5's gate |

## 5. Progress and decisions

- 2026-11-11: plan drafted from spec 043.
- 2026-11-11: maintainer approved the plan and the `graphiti-core` dependency.
- 2026-11-11: T0 done. `uv lock` resolved 331 packages with no override; two new packages (`graphiti-core` 0.31.2, `neo4j` 6.4.1). 83 install-profile, constraint and verifier tests pass. T0 grew by three files: the deploy constraints are generated from the lock, and a repository guard requires each new extra to have a verifier profile. `make verify-profiles` (wheel build per profile) was not run. Evidence: [graphiti-evidence.md](../../../design-docs/043-temporal-memory/graphiti-evidence.md).
- 2026-11-11: T1 done. Items (b), (d), (e), (f), (g) and (h) support the spec. Three findings change it and are held for Checkpoint A: `add_episode(uuid=...)` cannot create an episode, so the episode-identity design needs another mechanism; breadth-first expansion follows outgoing edges only; default extraction produces no facts for amounts. The LLM ran through the Claude Agent SDK with the OAuth token at the maintainer's request: it worked with no schema failures at 6.5 to 15.5 seconds per episode. No call was made through the `openai` 3 client.
- 2026-11-11: Checkpoint A passed. The maintainer decided D21 (pre-saved episode node with a completion marker), D22 (extraction instructions with a built-in default), D23 (accept outgoing-only expansion and measure) and D24 (Anthropic client chosen by credential). The spec was updated first. T23 was added for the Agent SDK client. The `ChunkSource` shape is the one in T5, approved with this plan.
- 2026-11-11: Checkpoint C live run done with the maintainer's approval; evidence is in [graphiti-evidence.md](../../../design-docs/043-temporal-memory/graphiti-evidence.md). Dry run 4 new episodes; ingest wrote 4 with 14 LLM calls in 55.7 seconds; second run 0 new and 0 LLM calls. The Agent SDK probe settled the T23 items: structured output arrives with no tools, the default permission mode does not block it, no user settings load, and a successful call takes 2 turns, so `max_turns = 2` is required. One result to carry into phase 4: the 100-dollar fact was not closed by the 120-dollar fact on this run. Invalidation is a model decision inside Graphiti; T17 measures it.
- 2026-11-11: Phase 2 adversarial review by Codex (gpt-6-astra, high) on the diff from `5c237d4`: 6 high, 10 medium, 3 low. Codex could not run probes. 18 findings are fixed, each with a regression test or a strengthened test; finding 2 is open for the maintainer. Fixed: the completion marker is a conditional update that cannot recreate a purged episode (1); the OpenAI and Anthropic clients get an explicit endpoint, so `OPENAI_BASE_URL` and `ANTHROPIC_BASE_URL` cannot redirect extraction (3); the OAuth path refuses to start when `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL` or a `CLAUDE_CODE_USE_*` variable is set (4); Graphiti's own log records are redacted (5); redaction covers keys from the environment and passwords inside a URI (6); context episodes from another namespace are left out (7); span status carries a fixed text or an exception type, never error text (8); a node under our id with another name or body is refused (9); a date tie is broken by position in `source_tools`, so the first tool owns a shared chunk (10); `DatabaseUnavailable` and every `DriverError` raise unavailable (11); the OAuth path warns that `temperature` and `max_tokens` are not applied (12); tokens of a failed SDK result are counted (13); complete is decided by the name prefix in both `has_episode` and `episode_count` (14); Graphiti is built in a thread, because the Kuzu driver does blocking I/O in its constructor (15); a YAML that does not load prints one line (16); the purge integration test counts entities and relationships (17); the CLI secret test goes through the real backend (18); integration cleanup purges every namespace (19). Open, finding 2: `--dry-run` still initializes the source tools, which can embed and store new documents and, with contextual embeddings, call a model. The tool's preview path returns text with no section ids, so episode ids cannot be computed from it without a second change to `HierarchicalDocumentTool`. The help text now says what the dry run covers. After the fixes: unit suite 7351 passed; the six integration tests pass on the local Neo4j.
- 2026-11-11: Codex phase 2 finding 2 fixed at the maintainer's direction (an interface extension: `HierarchicalDocumentTool.preview_document_chunks()`). `--dry-run` now builds the source tools read-only and initializes none, so it makes no embedding call and no vector store write (ING-08 reworded to say so). Evidence: 10 new unit tests, including preview listing equal to stored listing for local and remote sources; unit suite 7,361 passed, 1 skipped; live dry run on the Checkpoint C fixture with one added document reported 4 documents, 1 new, 4 skipped, 0 LLM calls, and no source-ingestion log line. Not re-reviewed by Codex.
- 2026-11-11: Supersession probe on the live fixture, at the maintainer's direction. The default extraction instructions gained a sentence that makes a new value say it replaces the earlier one; rate changes closed went from 1 of 3 to 2 of 3. Graphiti's contradiction prompt does not see the instructions and refuses to infer replacement from dates. Evidence and limits are in graphiti-evidence.md. Unclosed superseded facts remain the main risk for T16 and T17.
- 2026-11-11: Second supersession probe, at the maintainer's direction. `build_graphiti` now passes a prompt library that replaces Graphiti's `dedupe_edges.resolve_edge` with a prompt for dated provisions (`backends/graphiti_prompts.py`); the replacement sentence in the default instructions is reverted. Live result: 3 of 3 rate changes closed, the explicit change closed. A unit test pins the hash of Graphiti's own prompt so an upgrade forces a review. Over-closing (facts that should stay open) is not tested yet and is added to T16's scope.
- 2026-11-11: Second Codex pass (gpt-6-astra, high) on the dry-run fix and the prompt override: 1 high, 5 medium, 1 low; Codex could not run tests. Finding 5 (the prompt called every role single-valued) is fixed in the prompt with a test and a live re-run. Findings 1 to 4 are limits now stated in the spec (ING-08), the CLI help and the design doc: schema creation on connect by the Kuzu and FalkorDB drivers, stale stored chunks, chunking-setting changes on unchanged files, and no closure without two start dates. Findings 6 and 7 are older defects outside this feature and are listed under open items. Not refuted: `preview_chunks` callers unchanged, the override changes one prompt only, the hash guard, string construction.
- 2026-11-11: Phase 3 started on `feat/043-retrieval`. T12 (Sonnet 5.5, high) done by a subagent and validated: 41 tests, the worked fusion values hold, and the composition now passes `top_k`, `hops` and the fusion weights. Phase 3b (T24, T25, Checkpoint D2) added at the maintainer's request: renderers for the three memory tools in the CopilotKit sample.
- 2026-11-11: T11 (Opus 5.5, medium) done by a subagent and validated: 26 unit tests and 9 Neo4j tests, including the between-two-compilations case; unit suite 7,434 passed; integration suite 15 passed on the local Neo4j. Read-only probe of the live `cpc-live` namespace through the service: the base rate is 150 dollars now, 100 at 2021-01-01, 90 at 2019-12-01 and absent at 2018-01-01; the timeline lists 8 facts in order; provenance returns the chunk text. Differences from the fake that the backend cannot remove: one edge per fact (no second version on closing), `superseded_by` always empty, timeline matches an entity by name at either end of a fact. A fact with no start date is returned for every earlier date (the known-at rule's null branch). Not run: FalkorDB, Kuzu, Neptune.
- 2026-11-11: order changed by the maintainer. After Checkpoint D: phase 3b (T24, T25), then T16 (sample agent) brought forward from phase 4 onto the `feat/043-ui` branch, so the CopilotKit app has a sample to run against. Then the stack is submitted (PR cut) and work pauses while the maintainer tries the CopilotKit app. Checkpoint D2's browser step is that session. T17 to T22 stay in phase 4.
- 2026-11-11: T13 (Opus 5.5, medium), T14 (Sonnet 5.5, high) and T15 (Opus 5.5, medium) done by subagents and validated. Unit suite 7,581 passed, 1 skipped; Neo4j integration suite 15 passed; Ruff, Black, the repository's MyPy command, `make schema-check` and `make harness-check` clean. T15's tests are under `tests/unit/temporal` (the plan named a directory that does not exist). Known differences and gaps: the Claude SDK validates arguments against the schema before the handler runs, so schema-invalid arguments return the SDK's error there and `invalid_argument` on OpenAI Agents; a subagent or skill on OpenAI Agents cannot be granted the memory tool by its config name; the test runner does not collect memory results as retrieval context; `ActivityParameters` has no heartbeat timeout field, so a workflow passes it to `execute_activity` directly.
- 2026-11-11: Checkpoint D live validation, steps 1 and 2, on the `cpc-live` fixture (Claude Haiku through OAuth, Ollama embeddings, local Neo4j). Step 1, `holodeck test run`: 3 of 3 cases passed with the expected memory tool called; answers 150 dollars today, 100 on 1 January 2021, and the full 90/100/120/150 history. Step 2, `holodeck serve` with raw AG-UI requests: two threads, two turns each, four HTTP 200 runs, each with `RUN_STARTED`, `TOOL_CALL_START/ARGS/END/RESULT` for the memory functions, `TEXT_MESSAGE_*` and `RUN_FINISHED`, and no `RUN_ERROR`; the second turn on each thread used the first turn's context. Found there and fixed: `memory_timeline` returned nothing for "Family Benefit base rate" because it matched an exact entity name only. The timeline now also uses an entity whose whole name appears inside the reference when nothing matches exactly (backend, fake, port text, tests; 8 entries on the live graph).
- 2026-11-11: Phase 3 adversarial review by Codex (gpt-6-astra, high): 2 high, 10 medium, 2 low; Codex could not run tests. Fixed by three subagents split by file ownership (the maintainer's instruction): 1, the present-time shortcut is removed and search always runs both queries, with a 25-row table test of the rule; 2, the ingestion activity forwards no exception text, cause or detail, and only an unreachable store and a failed episode are retryable; 3 and 13, the timeline name fallback matches a whole name at word boundaries through one shared function used by Graphiti and the fake; 4, only an unreachable store degrades a search; 5 and 6, `ChunkSource.document()` added with the maintainer's approval, so a search reads each hit document's date from the file with no listing and no cache; 7, `invalid_argument` messages are fixed text; 9, granting the memory tool's config name on OpenAI Agents grants all three functions (maintainer decision); 11, a backend built after its caller was cancelled is closed; 12, unparsable JSON arguments return `invalid_argument`; 14, weak tests strengthened and checked by breaking the code. Accepted with the maintainer's decision: 8, the Claude SDK validates the schema before the handler, now stated in the design and pinned by a test through the real dispatch. Not fixed: 10, downloaded source directories are never removed on the success path either, an older tool-lifecycle defect, listed under open items. Gates after the fixes: unit suite 7,704 passed, 1 skipped. The fixes were not re-reviewed.
- 2026-11-11: T23 (Opus 5.5, medium), T8 and T9 (Sonnet 5.5, medium) and T10 (Sonnet 5.5, high) done by subagents and validated: unit suite 7332 passed; the six `graphiti` integration tests pass on the local Neo4j with stub clients. Orchestrator changes: the reserved function names moved to the model layer so that loading an agent imports no memory module (T9's lazy-import rule); the nightly job fails, and does not skip, when Neo4j is unreachable. T23 deviations kept: `max_turns` is 2, so the CLI's own schema check can correct the model once; `strict_mcp_config=True` is set; no `bypassPermissions`. T23 made no live call. The live run at Checkpoint C must confirm: structured output arrives with `tools=[]`; the default permission mode does not block it; where token counts are; whether 1 turn is always enough; that no user settings or MCP servers load; and the time per episode at 4 concurrent calls.
- 2026-11-11: Phase 2 started on `feat/043-ingest`. T6 (Sonnet 5.5, high) and T7 (Opus 5.5, medium) done by subagents and validated: unit suite 7275 passed. T6 uses seeded random corpora for its three properties, because Hypothesis is not installed and a new dependency is an ask-first item. T7 settled the five "verify" items of design seam 3 on the local Neo4j with a stub LLM client; the results are in the design doc. Still open from design seam 4: no real model call has gone through any Graphiti client, and FalkorDB, Neptune and Kuzu have not run.
- 2026-11-11: T4 (Sonnet 5.5, medium) and T5 (Opus 5.5, medium) done by subagents and validated by the orchestrator. Orchestrator changes: `instants.py` written first as the shared date parser; provenance text returned once; `find()` no longer loads the corpus (a hit's ordinal is `UNKNOWN_ORDINAL`); one vacuous T5 test fixed. T5 deviations kept and recorded in design seam 2: `document_chunks()` always loads from the store; a front-matter-only chunk is left out.
- 2026-11-11: Phase 1 adversarial review by Codex (gpt-6-astra, high) on the diff from `35616a3`: 0 high, 11 medium, 2 low. Codex could not run probes, so each finding got a regression test first; all 13 reproduced and all are fixed. Fixed: a fact that returns after a change now starts a new period in the fake backend; an extraction error is reported, not raised; a past-known fact shows no later expiry; seeds require the `EXPAND` capability; another tool cannot take a memory function name, including a search tool named `memory`; a missing file falls back to ingestion time; a byte order mark, an impossible YAML date and a four-dash rule are handled in front matter; an out-of-range instant is a `MemoryValueError`; an empty degraded search stays degraded; fusion weights must be finite; a boolean is not a hop count. One part of finding 10 was not taken: the fake does not require `VALID_AT` or `KNOWN_AT`, because a query always carries both instants and the port cannot tell a default from a chosen date. T12 checks those capabilities in the service when the caller passes a date. After the fixes: unit suite 7075 passed, `make schema-check` and `make harness-check` pass.
- 2026-11-11: T3 done. `TemporalMemoryToolConfig`, `GraphBackendConfig`, `FusionWeights` and `MemorySchemaHints` are in `models/tool.py`; the agent-level rules are in `models/agent.py`. 33 tests in `tests/unit/models/test_temporal_memory_tool.py` pass, the unit suite passes (6995), and `make schema-check` and `make harness-check` pass. Two points differ from the spec text and were updated there: the backend gained `aoss_host`, which the Neptune driver cannot run without; and CFG-06 compares embedding models only, because a hierarchical tool has no dimension setting. The tool loads from YAML but gives the agent no function until T13.
- 2026-11-11: Design seam 6 accepted. The graph database being down at start gives degraded mode unless `ingest_on_start` is set; a Temporal worker refuses `ingest_on_start`; the three function names stay fixed. `EpisodeRef` gained an optional `text` field for provenance. The spec tools table and ING-07 were updated.
- 2026-11-11: Design seam 5 accepted: the known-at rule runs as two filtered searches, which replaces the "filter the returned edges" option in T11. The T15 file path was corrected to `src/holodeck/temporal/`.
- 2026-11-11: Design seams 2 to 4 accepted in the [design doc](../../../design-docs/043-temporal-memory/design.md). Seam 3 added `context_episode_ids` to `EpisodeIn`. Seam 4 changed D24 and GPH-04: `auth_provider` selects the Anthropic client, and `bedrock`, `vertex`, `foundry` and `custom` are refused at load. The T23 acceptance line was updated.
- 2026-11-11: the maintainer chose one stacked PR per phase. Phase 0 is the bottom branch `feat/043-spec`.
- 2026-11-11: T2 done. 27 unit tests pass; Black, Ruff and MyPy clean. The port gained four things the spec's first sketch lacked, and the spec was updated: `name`, `episode_count` (for `holodeck memory status`), `close`, and an `instructions` argument on `ingest` (D22). `provenance` returns `None` for an unknown fact. Writing the fake exposed a rule the T1 filter test did not: with Graphiti's single-edge model, filtering on `expired_at` alone hides every closed fact from current queries. The correct known-at rule is now in spec section 4 and in T11's acceptance.

## 6. Acceptance evidence

| Spec criterion | Proving task | Evidence |
| --- | --- | --- |
| 1. `make schema-check` passes | T3 | — |
| 2. Contract tests on both backends | T14 | — |
| 3. Unchanged re-ingest: zero LLM calls, zero writes | T6, T10 | — |
| 4. Second compilation: only new hashes | T6 | — |
| 5. Every fact resolves through `memory_provenance` | T11, T19 | — |
| 6. `valid_at` between compilations returns the earlier fact | T11 | — |
| 7. Not-supported operations raise by name | T11 | — |
| 8. Graph stopped: chunk hits with `degraded: true` | T12 | — |
| 9. Eval report with five measures | T19, T22 | — |
| 10. Every requirement ID has a test | T20 | — |
| 11. Live validation loop recorded | Checkpoint D, T22 | — |
| 12. Two source tools, one graph | T6, T16 | — |
| 13. Default instruction yields dated rate facts | T10 | — |

## 7. Open items

- Spec open questions 1 (licence), 2 (compilations and labelling) and 3 (spend) need maintainer answers before T17, T18 and T22.
- Spec open questions 4 and 5 are answered by T1.
- Spec open question 6 (front matter in chunk text) stays open; A6 holds for this plan.
- Temporary directories that `SourceResolver` downloads for a remote source are not owned by the tool's lifetime: nothing removes them on success or failure except the orphan sweep at `holodeck serve` start. Temporal workers never sweep. Older defect; the memory ingestion activity adds one more caller.
- On OpenAI Agents, the vectorstore, hierarchical-document and function-tool argument parsers still let a `ValueError` or `RecursionError` from `json.loads` escape (a 4,301-digit integer, very deep nesting). Fixed for the memory functions only.
- The ingestion activity logs the full traceback at DEBUG, which can contain document text. Nothing at INFO or above does.
- The Claude dispatch test for schema-invalid arguments skips on `mcp` 1.x, which the dependency range still allows.
- The test runner does not collect memory results as retrieval context for evaluation metrics. Decide in T16.
- Phase 3b: the maintainer decided on 2026-11-11 to track `sample/copilotkit` in this repository with an exception to `/sample/*` in `.gitignore`. T24 copies it from the main checkout without `node_modules`, build output or any `.env` file. A node-and-edge graph view would need subject and object entities in the tool results, which is a schema change and is not planned.
- `SourceResolver.resolve` leaks its temporary directory when cancelled during a download (it catches `Exception` only). Older defect; the dry-run preview is a new caller.
- `HierarchicalDocumentTool._convert_to_markdown` runs blocking file conversion on the event loop, in ingestion and in the preview. Older defect.
- Extraction produced no fact for "X may appoint inspectors" statements on the live fixture. T16 needs power and role cases, for extraction and for over-closing.
