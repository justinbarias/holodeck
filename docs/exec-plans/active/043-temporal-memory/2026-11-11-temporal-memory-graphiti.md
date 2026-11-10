# Execution plan: Temporal Knowledge Graph Memory on Graphiti

Feature: `043-temporal-memory`. Product contract: [spec 043](../../../product-specs/043-temporal-memory/spec.md).
**Status:** Plan approved 2026-11-11. T0 and T1 done; Checkpoint A passed; Phase 0 committed on `feat/043-spec`; T2 next. Nothing pushed.
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

- [ ] **T2: Port, types, errors and the fake backend.**
  - Acceptance: `MemoryBackend` and `MemoryCapabilities` in `ports.py`; frozen dataclasses in `types.py` with UTC-aware datetimes rejected when naive; three typed errors; `FakeBackend` implements every port method deterministically and counts calls.
  - Verify: `uv run pytest tests/unit/lib/memory/test_ports.py tests/unit/lib/memory/test_fake_backend.py -n auto -q` (covers MB-01 to MB-04).
  - Depends on: Checkpoint A. Files: `src/holodeck/lib/memory/{__init__,ports,types,errors}.py`, `backends/fake.py`, tests. Size: M. Owner: orchestrator, inline (boundary contract).

- [ ] **T3: `TemporalMemoryToolConfig` and agent-level validation.**
  - Acceptance: the model has `extra="forbid"` and the fields in spec section 4; loading fails with the exact tool name for an empty, repeated or wrong-type `source_tools` entry (CFG-02), a second memory tool (CFG-03), `single_valued_predicates` on Graphiti (CFG-04) and mismatched embeddings (CFG-06); `schemas/agent.schema.json` regenerated.
  - Verify: `uv run pytest tests/unit/models/test_temporal_memory_tool.py -n auto -q && make schema && make schema-check`
  - Depends on: T2. Files: `src/holodeck/models/tool.py`, `src/holodeck/models/agent.py`, `schemas/agent.schema.json`, template schema copies, tests. Size: M. Owner: orchestrator, inline (schema).

- [ ] **T4: Host-neutral tool schemas.**
  - Acceptance: `schemas/memory/` holds JSON Schemas for the arguments and results of the three tools; a test fails if they drift from the Pydantic or dataclass types.
  - Verify: `uv run pytest tests/unit/lib/memory/test_tool_schemas.py -n auto -q`
  - Depends on: T2. Files: `schemas/memory/*.json`, one generator or test. Size: S. Model: Sonnet 5.5, medium.

- [ ] **T5: Chunk accessor and effective-date reader.**
  - Acceptance: a typed `ChunkSource` protocol (`tool_name`, `iter_chunks()`, `search()`) that `HierarchicalDocumentTool` satisfies without changing its search behaviour; `effective_date.py` returns the date and its source (`front_matter`, `mtime`, `ingestion`) for a configurable key; chunk text is unchanged (A6).
  - Verify: `uv run pytest tests/unit/lib/memory/test_effective_date.py tests/unit/tools/test_hierarchical_chunk_source.py -n auto -q`; the existing hierarchical tool tests still pass.
  - Depends on: T2, Checkpoint A. Files: `src/holodeck/lib/memory/{chunk_source,effective_date}.py`, `src/holodeck/tools/hierarchical_document_tool.py`, tests. Size: M. Model: Opus 5.5, medium (interface extension).

**Checkpoint B.**
- [ ] `make schema-check` and `make harness-check` pass; unit tests green; adversarial review of PR 1 has no open finding; PR 1 committed.

### Phase 2: Ingestion (PR 2, `feat/043-ingest`)

- [ ] **T6: Service ingest path.**
  - Acceptance: `episode_id_for` matches the spec formula; sources from all source tools are submitted in one ascending effective-date order (ING-04); existing episodes are skipped with zero backend writes (ING-01, ING-02); `dry_run` reports the new-episode count with zero LLM calls (ING-08); a backend failure stops the run and the report lists what was written (ING-05).
  - Verify: `uv run pytest tests/unit/lib/memory/test_service_ingest.py -n auto -q`, including Hypothesis properties: re-ingest is a no-op; one changed chunk yields exactly one new episode.
  - Depends on: T5. Files: `src/holodeck/lib/memory/service.py`, tests. Size: M. Model: Sonnet 5.5, high.

- [ ] **T7: Graphiti backend, write side.**
  - Acceptance: builds the driver for `neo4j`, `falkordb`, `neptune` and `kuzu` (a warning for `kuzu`) (GPH-03); maps `extraction_model` to the four LLM clients; sets telemetry off before the client exists (GPH-02); passes no cross-encoder (GPH-01); a missing extra raises `MemoryConfigError` with the install command (CFG-05); implements `has_episode`, `ingest`, `purge` and `capabilities`; `has_episode` is true only for a complete episode, and `ingest` pre-saves the node, calls `add_episode`, then marks it complete (D21, ING-09); passes the extraction instructions (ING-10); always passes a cross-encoder object and copies search recipes before use; a missing driver package fails with its install command (GPH-05); no Graphiti type leaves the module.
  - Verify: `uv run pytest tests/unit/lib/memory/test_graphiti_backend_write.py -n auto -q` with Graphiti mocked.
  - Depends on: T2, T1 findings. Files: `src/holodeck/lib/memory/backends/graphiti.py`, `registry.py`, `src/holodeck/cli/main.py` (environment variable), tests. Size: M. Model: Opus 5.5, medium.

- [ ] **T23: Agent SDK client for Graphiti.**
  - Acceptance: a Graphiti `LLMClient` backed by the Claude Agent SDK returns schema-valid output or raises after three attempts; the backend selects it for `anthropic` when `ANTHROPIC_API_KEY` is unset and the OAuth token is set (GPH-04); it runs with no tools and without `bypassPermissions`, or the task records why a stricter mode is not possible; `--dry-run` shows an estimated duration (ING-08).
  - Verify: `uv run pytest tests/unit/lib/memory/test_agent_sdk_llm_client.py -n auto -q` with the SDK mocked, plus one live episode at Checkpoint C.
  - Depends on: T7. Files: `src/holodeck/lib/memory/backends/graphiti_agent_sdk.py`, `backends/graphiti.py`, tests. Size: S. Model: Opus 5.5, medium.

- [ ] **T8: Ingestion telemetry.**
  - Acceptance: one trace per run, a `holodeck.memory.ingest_episode` span per episode, token counts as attributes; the existing redaction rules apply (ING-06).
  - Verify: `uv run pytest tests/unit/lib/memory/test_ingest_spans.py -n auto -q` with an in-memory span exporter.
  - Depends on: T6, T7. Files: `service.py`, `backends/graphiti.py`, test. Size: S. Model: Sonnet 5.5, medium.

- [ ] **T9: `holodeck memory` commands.**
  - Acceptance: `ingest` (with `--tool` and `--dry-run`) prints the report (CLI-01); `status` prints backend, namespace, episode count and capabilities (CLI-02); `purge` without `--yes` asks and names the namespace (CLI-03); the group is lazy, so `holodeck init` imports no memory module.
  - Verify: `uv run pytest tests/unit/cli/test_memory_commands.py -n auto -q`
  - Depends on: T6, T7. Files: `src/holodeck/cli/commands/memory.py`, `src/holodeck/cli/main.py`, tests. Size: M. Model: Sonnet 5.5, medium.

- [ ] **T10: Graphiti integration suite.**
  - Acceptance: a compose file starts Neo4j; tests marked `graphiti` use a stub LLM client and prove ingest, skip on re-ingest with zero LLM calls, and purge of one namespace only; a nightly CI job runs the marker.
  - Verify: `docker compose -f tests/integration/memory/compose.yaml up -d neo4j && uv run pytest tests/integration/memory -n auto -q -m graphiti`
  - Depends on: T7, T9. Files: `tests/integration/memory/{compose.yaml,conftest.py,test_graphiti_ingest.py}`, the CI workflow, `pyproject.toml` marker. Size: M. Model: Sonnet 5.5, high.

**Checkpoint C.**
- [ ] `holodeck memory ingest --dry-run` then a live ingest of a three-file fixture succeeds and a second run reports zero new episodes (ask first: small LLM spend). Adversarial review of PR 2 has no open finding; PR 2 committed.

### Phase 3: Retrieval and tools (PR 3, `feat/043-retrieval`)

- [ ] **T11: Graphiti backend, read side.**
  - Acceptance: `search` applies `valid_at` and `known_at` filters, each with an `IS NULL` branch and at most one dated group per field, and the hop limit (RET-03, RET-04); expansion is seeded from entity uuids (RET-09); `timeline` returns open and closed facts ordered by `valid_from` (RET-08); `provenance` resolves a fact to episodes, source tools and chunk ids (RET-06); each not-supported row in the capability table raises `MemoryNotSupportedError` with the operation name (MB-03).
  - Verify: unit tests with Graphiti mocked, plus `uv run pytest tests/integration/memory/test_graphiti_read.py -n auto -q -m graphiti`, which includes the "between two compilations" case.
  - Depends on: T7. Files: `backends/graphiti.py`, tests. Size: M. Model: Opus 5.5, medium.

- [ ] **T12: Service search and fusion.**
  - Acceptance: defaults `valid_at` and `known_at` to the query time (RET-01); calls every source tool's search and maps chunk hits to episode ids (RET-02); fuses by reciprocal rank fusion with the configured weights (RET-05); every hit carries `source_tool`, `chunk_id`, `source_uri` and the valid period (RET-06); an unavailable backend gives chunk-only hits with `degraded: true` (RET-07).
  - Verify: `uv run pytest tests/unit/lib/memory/test_service_search.py -n auto -q`, including a Hypothesis property that fusion is stable under equal ranks.
  - Depends on: T5, T6. Files: `service.py`, tests. Size: M. Model: Sonnet 5.5, high.

- [ ] **T13: The tool and the Claude backend binding.**
  - Acceptance: `TemporalMemoryTool` is created by the tool initializer after its source tools; `ingest_on_start` runs ingestion at initialization (ING-07); the Claude adapter exposes `memory_search`, `memory_timeline` and `memory_provenance` whose results validate against `schemas/memory/` with ISO 8601 UTC timestamps (HST-01, HST-02).
  - Verify: `uv run pytest tests/unit/tools/test_temporal_memory_tool.py tests/unit/lib/backends/test_tool_adapters.py -n auto -q`
  - Depends on: T11, T12. Files: `src/holodeck/tools/temporal_memory_tool.py`, `tool_initializer.py`, `backends/tool_adapters.py`, tests. Size: M. Model: Opus 5.5, medium.

- [ ] **T14: OpenAI Agents binding and the two-backend contract test.**
  - Acceptance: the same three tools with the same schemas on the OpenAI Agents backend; one parametrized contract test runs against both backends.
  - Verify: `uv run pytest tests/unit/lib/backends/test_memory_tool_contract.py -n auto -q`
  - Depends on: T13. Files: `backends/openai_agents_tool_adapters.py`, test. Size: S. Model: Sonnet 5.5, high.

- [ ] **T15: Temporal activities.**
  - Acceptance: ingestion and each memory tool call run in activities; the workflow sandbox import check still passes (HST-03).
  - Verify: the existing Temporal determinism tests plus `uv run pytest tests/unit/lib/workflow -n auto -q -k memory`
  - Depends on: T13. Files: under `src/holodeck/lib/workflow/`, tests. Size: S. Model: Opus 5.5, medium.

**Checkpoint D.**
- [ ] Live validation loop steps 1 and 2 on the sample: `holodeck test run`, then `holodeck serve` with raw AG-UI requests on two threads, two turns each, with `TOOL_CALL_*` events for the memory tools inspected. Adversarial review of PR 3 has no open finding; PR 3 committed.

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
- 2026-11-11: the maintainer chose one stacked PR per phase. Phase 0 is the bottom branch `feat/043-spec`.

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
