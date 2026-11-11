# SPEC: Temporal Knowledge Graph Memory

**Status:** Approved for build 2026-11-11. Decisions in section 3 were settled with the maintainer; D21 to D24 follow the [Graphiti evidence run](../../design-docs/043-temporal-memory/graphiti-evidence.md). Only the dependency is in the tree.
**Date:** 2026-11-11.
**Code baseline:** `6203c82` (release 0.8.0).
**Source:** the host-neutral spec "Temporal Knowledge Graph Memory" (generic spec). This document binds it to HoloDeck and records where the MVP differs.
**Related:** [016 GraphRAG](../016-graphrag-integration/index.md) (archived research), [040 Temporal](../040-holodeck-temporal/spec.md), [042 dependency stack](../042-dependency-stack-revamp/spec.md).

## 1. Objective

Give a HoloDeck agent a memory that answers "what was true, when, and according to which document" across a corpus whose documents amend and contradict each other. The `hierarchical_document` tool finds relevant passages. It cannot say which of two conflicting passages is current, or link one entity across many documents. This feature adds that layer beside the retriever and does not replace it.

The MVP runs on Graphiti. Graphiti does extraction, entity resolution and edge invalidation with its own pipeline. HoloDeck owns the configuration, the port, ingestion control, fusion with chunk retrieval, the agent tools and the evaluation.

Users:

- Agent authors, who add one `temporal_memory` tool to `agent.yaml`.
- Agents, which call three memory tools at run time.
- Operators, who run ingestion and pay for its LLM calls.

User stories:

1. As an agent, I ask for facts about an entity as of a past date and get only the facts valid then, each with its source chunk.
2. As an agent, I start from chunks that hybrid search returned and expand to related entities in other documents within two hops.
3. As an agent author, I enable memory with one tool block and no Python.
4. As an operator, re-ingesting an unchanged corpus makes no LLM calls and no writes.
5. As an operator, ingesting a new compilation of a document extracts only the chunks whose text changed.
6. As an agent author, I build one graph across several document collections by listing more than one source tool.

## 2. Scope

| Generic module | 043 MVP | Notes |
| --- | --- | --- |
| `tkg-model` | Port only | `MemoryBackend` port and types. The SQL schema, `GraphStore` port and PostgreSQL adapter are Phase 2. |
| `tkg-ingest` | Partly | HoloDeck: episode intake, hashing, effective dates, ordering. Graphiti: extraction and entity resolution. |
| `tkg-temporal` | Graphiti's | Graphiti invalidates edges. No HoloDeck judge, no supersession audit. |
| `tkg-retrieve` | Yes | Point-in-time filters, seed-and-expand, fusion with the chunk retriever. |
| `tkg-host` | Python only | Three tools on the Claude and OpenAI Agents backends. No context provider. |
| `tkg-graphiti` | Yes | The MVP backend. |

Phase 2, specified in the generic spec and not built here: the native backend (PostgreSQL and in-memory `GraphStore`, HoloDeck extractor, resolver and judge), SQL migrations by plain versioned files and a `tkg_schema_version` table, the review-queue CLI, and the supersession audit trail.

Out of scope: the .NET host, conversation turns as episodes, the context provider (HST-04), dashboard UI, RAPTOR summaries.

## 3. Decisions

| # | Decision | Choice |
| --- | --- | --- |
| D1 | v1 modules | Core, Python host, Graphiti. |
| D2 | YAML surface | New tool type `temporal_memory` with `source_tools`, a list of one or more `hierarchical_document` tools. One graph spans all of them. |
| D3 | Chunk store and graph store | Decoupled. The graph links to chunks by id as text; chunks stay in any vector store. |
| D4 | Effective date | Front-matter key (default `effective_date`), then file mtime, then ingestion time. The source used is recorded. |
| D5 | Ingestion trigger | `holodeck memory ingest`, plus opt-in `ingest_on_start`. An activity under Temporal. |
| D6 | Model | One optional `extraction_model`, default the agent's model. |
| D7 | Predicates | Free text, with optional `schema_hints`. |
| D8 | Review queue | CLI list and resolve. Phase 2, because Graphiti has no queue. |
| D9 | Golden corpus | Real legislation: *A New Tax System (Family Assistance) Act 1999* and *A New Tax System (Family Assistance) (Administration) Act 1999*, volumes 1 and 2. |
| D10 | Memory scope | Documents only. |
| D11 | Context provider | Deferred. |
| D12 | Migrations | Plain SQL and a version table. Phase 2. |
| D13 | Graphiti database | Any Graphiti driver passes through from YAML. Only Neo4j is tested. |
| D14 | Graphiti depth | Graphiti's own pipeline (`add_episode`), not storage only. |
| D15 | Swappable seam | A `MemoryBackend` port. Graphiti is the MVP implementation; the native backend is deferred. |
| D16 | Graphiti gaps | Emulate what is cheap; raise a named not-supported error for the rest. |
| D17 | Graphiti tests | Contract tests per commit, live golden-corpus eval nightly, no score gate in v1. |
| D18 | MVP cut | Migrations, in-memory `GraphStore`, review CLI and supersession audit all move to Phase 2. |
| D19 | Episode size | One chunk per episode. |
| D20 | Spec home | This directory, plus an HTML artifact. |
| D21 | Episode identity on Graphiti | Pre-save the episode node under the content-hash uuid, run `add_episode` on it, then mark it complete. Skip only complete episodes. Decided after the T1 evidence run. |
| D22 | Extraction instructions | Optional `extraction_instructions` field, with a built-in default that asks for amounts, rates, dates and thresholds as entities. |
| D23 | Expansion direction | Accept Graphiti's outgoing-only traversal in the MVP. Seed from every entity the hit chunks mention, and measure recall in the eval. |
| D24 | Anthropic extraction | By `auth_provider` on `extraction_model`: `api_key` (the default) uses Graphiti's `AnthropicClient`; `oauth_token` uses a HoloDeck client on the Claude Agent SDK. `bedrock`, `vertex`, `foundry` and `custom` are refused at load in the MVP. No environment variable selects the client. |

### Working assumptions

These are filled in, not confirmed. Correct them before the Plan phase.

- A1. One `temporal_memory` tool per agent in the MVP, so the function names `memory_search`, `memory_timeline` and `memory_provenance` are fixed, as HST-01 requires.
- A2. The package pin is `graphiti-core>=0.31,<0.32` in a new `graphiti` extra. Version 0.31.2 requires `openai>=1.91.0`, `neo4j>=5.26.0` and `pydantic>=2.11.5`; compatibility with HoloDeck's `openai>=3,<4` pin is checked with `uv lock` at the start of Plan.
- A3. Graphiti's default cross-encoder calls OpenAI. HoloDeck uses a search configuration with no cross-encoder, so no content goes to an endpoint the YAML does not name.
- A4. Graphiti sends anonymous usage telemetry through PostHog. HoloDeck sets `GRAPHITI_TELEMETRY_ENABLED=false` before it creates a client.
- A5. Kuzu passes through like any driver but logs a warning, because the Graphiti README marks it deprecated.
- A6. Front matter is read by the memory library only. The chunker's handling of front matter does not change.
- A7. Each historical compilation of an Act is a separate source file with its own effective date. Replacing a file in place is not the model for versions.
- A8. The episode key does not include the source tool. The same chunk text in two collections is extracted once, and its facts cite the collection that comes first by effective date, then by position in `source_tools`.
- A9. All source tools must use the same embedding model and dimensions, because the graph has one embedder.

## 4. Architecture

Two pipelines share one backend. Ingestion turns chunks into dated episodes. Retrieval seeds from the chunk retriever, expands through the graph at a point in time, and fuses both result sets.

```text
agent.yaml
  hierarchical_document tools (1..n) ──chunks──┐
                                       ▼
  holodeck memory ingest ──▶ TemporalMemory service ──▶ MemoryBackend port
                              · effective date           ├─ GraphitiBackend ──▶ Neo4j / FalkorDB / Neptune
                              · chunk hash, skip         ├─ FakeBackend (unit tests)
                              · order by date            └─ NativeBackend (Phase 2)
  agent turn
    memory_search ──▶ TemporalMemory.search
                        1. each source tool's hybrid search ──▶ chunk hits
                        2. chunk hits ──▶ episode ids ──▶ seed entities
                        3. backend.search(valid_at, known_at, hops)
                        4. reciprocal rank fusion ──▶ MemoryHit[]
```

### Bi-temporal mapping

| Generic concept | Graphiti field | Set by |
| --- | --- | --- |
| `valid_from` | edge `valid_at` | Graphiti, from the text or the episode `reference_time` |
| `valid_to` | edge `invalid_at` | Graphiti, on invalidation |
| `recorded_at` | edge `created_at` | Graphiti, on write |
| `expired_at` | edge `expired_at` | Graphiti, on invalidation |
| namespace | `group_id` | HoloDeck |
| episode | `EpisodicNode`, one per chunk | HoloDeck |

`memory_search` turns `valid_at` and `known_at` into Graphiti `SearchFilters` on those four fields.

### Episode identity and skipping

Chunk ids are `{source}_chunk_{index}`, so they shift when a section is inserted. The episode key is therefore content-based:

```text
chunk_hash   = sha256(section_id + "\n" + chunk_text)
episode_uuid = uuid5(namespace, chunk_hash)
```

Graphiti's `add_episode(uuid=...)` re-processes an existing episode and raises when the uuid is absent, so the backend creates the episode in three steps:

1. Look the episode up by uuid. This makes no LLM call. If it exists and is marked complete, skip the chunk.
2. Otherwise save an episode node under that uuid, marked incomplete, and call `add_episode` on it.
3. Mark the episode complete.

An episode left incomplete by a crash is processed again on the next run. A complete episode is never passed to `add_episode` again, because a re-run costs LLM calls and adds duplicate mention edges. A chunk whose text is unchanged between two compilations is extracted once, and its facts cite the first compilation that contained it.

### Port

```python
class MemoryBackend(Protocol):
    @property
    def name(self) -> str: ...
    def capabilities(self) -> MemoryCapabilities: ...
    async def has_episode(self, namespace: str, episode_id: UUID) -> bool: ...
    async def episode_count(self, namespace: str) -> int: ...
    async def ingest(self, namespace: str, episodes: Sequence[EpisodeIn], instructions: str) -> IngestReport: ...
    async def search(self, namespace: str, query: MemoryQuery) -> list[FactHit]: ...
    async def timeline(self, namespace: str, entity_ref: str) -> list[TimelineEntry]: ...
    async def provenance(self, namespace: str, fact_id: str) -> Provenance | None: ...
    async def purge(self, namespace: str) -> None: ...
    async def close(self) -> None: ...
```

No Graphiti type appears in a port signature. Time is a timezone-aware UTC `datetime`. Errors are typed: `MemoryNotSupportedError`, `MemoryBackendUnavailableError`, `MemoryConfigError` and `MemoryValueError`, all under `TemporalMemoryError`.

`has_episode` is true only for a completed episode. `ingest` stops at the first failed episode and reports it in `IngestReport` instead of raising. A namespace uses letters, digits, `_` and `-` only, which is Graphiti's `group_id` rule applied to every backend.

**Known-at semantics.** A fact closed by a later one stays visible to a query whose `known_at` is before the closing was recorded, and it is seen as still open. Graphiti keeps one edge and stamps `expired_at` when it closes it, so a filter of `expired_at IS NULL OR expired_at > known_at` alone would hide closed facts from every current query. The Graphiti backend must apply the rule `created_at <= known_at`, and treat `invalid_at` as unset when `expired_at > known_at`.

### Configuration

```yaml
tools:
  - name: family_law
    type: hierarchical_document
    source: corpus/**/*.md
    database: { provider: postgres, connection_string: ${PG_DSN} }

  - name: family_guides
    type: hierarchical_document
    source: guides/**/*.md
    database: { provider: qdrant, url: ${QDRANT_URL} }

  - name: family_law_memory
    type: temporal_memory
    description: Facts about family assistance law and when they applied.
    source_tools: [family_law, family_guides]   # one graph across all of them
    namespace: family-assistance        # default: the tool name
    backend:
      provider: graphiti                # only value in the MVP
      driver: neo4j                     # neo4j | falkordb | neptune | kuzu
      uri: ${NEO4J_URI}
      user: ${NEO4J_USER}
      password: ${NEO4J_PASSWORD}
      # aoss_host: required for neptune only (its OpenSearch host)
    extraction_model: { provider: openai, name: gpt-5.4-mini }   # default: agent model
    extraction_instructions: null       # default: built-in instruction for amounts, rates and dates
    effective_date_key: effective_date
    ingest_on_start: false
    hops: 1                             # 1 or 2
    top_k: 10
    fusion_weights: { chunks: 1.0, graph: 1.0 }
    schema_hints:
      entity_types: [Act, Section, Payment, Person, Rate]
```

Provider mapping for `extraction_model`: `openai` to `OpenAIClient`, `azure_openai` to `AzureOpenAILLMClient`, `ollama` to `OpenAIGenericClient`. For `anthropic`, `auth_provider` decides: `api_key` or not set, Graphiti's `AnthropicClient`; `oauth_token`, a HoloDeck client that calls the Claude Agent SDK; any other value fails at load with `MemoryConfigError`. The Agent SDK path measured 6.5 to 15.5 seconds per chunk. The embedder comes from the source tools' shared embedding settings (CFG-06).

### Tools

| Tool | Arguments | Returns |
| --- | --- | --- |
| `memory_search` | `query`, `valid_at?`, `known_at?`, `hops?` | An object with the `valid_at` and `known_at` used, `degraded`, and `hits`: `MemoryHit[]` with fact or chunk text, score, `fact_id?`, `source_tool`, `chunk_id`, `source_uri`, `valid_from`, `valid_to`, `superseded_by?` |
| `memory_timeline` | `entity` | `TimelineEntry[]` ordered by `valid_from`, including closed facts |
| `memory_provenance` | `fact_id` | `Provenance`: episodes, source tools, chunk ids, source URIs, chunk text, effective date and its source |

Results are JSON with ISO 8601 UTC timestamps. A date argument with no time means the start of that day in UTC. A cause the model can correct is returned as an error result, not raised. The [design doc](../../design-docs/043-temporal-memory/design.md) section 6 has the detail. JSON schemas for arguments and results live in `schemas/memory/` and are host-neutral.

### Graphiti capability table

| Generic operation | MVP on Graphiti |
| --- | --- |
| Idempotent re-ingest (ING-01, ING-02) | Emulated: uuid lookup, pre-saved node, completion marker |
| Entity type hints | Native: `entity_types` |
| `valid_at` and `known_at` queries | Native: `SearchFilters` |
| Seed-and-expand, 1 to 2 hops | Native: breadth-first search from seed nodes, following outgoing edges only |
| Facts for amounts and rates | Needs extraction instructions; the default prompt extracts none |
| Provenance to chunk | Native: edge to episodes; episode name is the chunk id |
| `single_valued_predicates` (TMP-07) | Not supported: rejected at config load |
| Supersession rationale, model id, prompt version (TMP-06) | Not supported |
| Retraction when a source version is replaced (TMP-04) | Not supported; see A7 |
| Pending-merge and low-confidence review (ING-07, TMP-08) | Not supported |
| Per-step extraction retry policy (ING-04) | Graphiti's own |

## 5. Requirements (EARS)

"The memory" is the `TemporalMemory` service with its backend. The last column gives the generic requirement each one binds.

| ID | Pattern | Requirement | Generic |
| --- | --- | --- | --- |
| MB-01 | Ubiquitous | The memory shall reach every backend only through the `MemoryBackend` port. | MOD-06 |
| MB-02 | Ubiquitous | The memory shall scope every read and write to one namespace. | MOD-03 |
| MB-03 | Unwanted | If a backend cannot perform an operation, then it shall raise `MemoryNotSupportedError` naming the operation and shall not return partial results. | GPH-02 |
| MB-04 | Event-driven | When `purge` is called, the memory shall remove every episode, entity and fact in that namespace and no other. | MOD-05 |
| CFG-01 | Ubiquitous | The agent schema shall accept a `temporal_memory` tool validated by a Pydantic model with `extra="forbid"`. | — |
| CFG-02 | Unwanted | If `source_tools` is empty, repeats a name, or names anything other than a `hierarchical_document` tool in the same agent, then loading shall fail with the tool name. | — |
| CFG-06 | Unwanted | If the source tools set different embedding models, then loading shall fail and name the tools that differ. A hierarchical tool has no dimension setting, so the model alone is compared. | — |
| CFG-03 | Unwanted | If more than one `temporal_memory` tool is configured, then loading shall fail. | HST-01 |
| CFG-04 | Unwanted | If `schema_hints.single_valued_predicates` is set with the Graphiti backend, then loading shall fail and name the field. | TMP-07 |
| CFG-05 | Unwanted | If the `graphiti` extra is not installed, then loading shall fail with the install command. | — |
| ING-01 | Event-driven | When ingestion runs, the memory shall compute each chunk's hash and submit only chunks that have no complete episode. | ING-01 |
| ING-02 | Unwanted | If every chunk already has an episode, then the memory shall make no LLM calls and no writes. | ING-02 |
| ING-03 | Ubiquitous | The memory shall set each episode's reference time from the front-matter key, else file mtime, else ingestion time, and record which was used. | ING-05 |
| ING-04 | Ubiquitous | The memory shall submit sources from all source tools in one ascending effective-date order, and chunks in document order within a source. | TMP-05 |
| ING-05 | Unwanted | If the backend fails on an episode, then the memory shall stop, report the episodes written, and leave the rest for the next run. | ING-04 |
| ING-06 | Ubiquitous | The memory shall emit one trace per ingestion run with a span per episode and token counts. | ING-10 |
| ING-07 | Optional | Where `ingest_on_start` is true, the tool shall run ingestion during initialization, and shall fail to start if the backend is unavailable. Under a Temporal worker, `ingest_on_start: true` shall fail at worker start and name the ingestion activity. | HST-03 |
| ING-08 | Event-driven | When `--dry-run` is given, ingestion shall report the count of new episodes and an estimated duration, and make no LLM calls. | — |
| ING-09 | Unwanted | If an episode exists but is not marked complete, then the memory shall process it again. | ING-08 |
| ING-10 | Ubiquitous | The memory shall pass `extraction_instructions` to the backend, or the built-in default when the field is unset. | ING-03 |
| RET-01 | Ubiquitous | The memory shall default `valid_at` and `known_at` to the time of the query. | RET-01 |
| RET-02 | Event-driven | When `memory_search` is called, the memory shall call every source tool's search, map returned chunks to episodes, and expand from those episodes' entities. | RET-02 |
| RET-03 | Ubiquitous | The memory shall return only facts valid at `valid_at` and known at `known_at`. | RET-03 |
| RET-04 | Ubiquitous | The memory shall limit expansion to the configured hop count, 1 or 2, counted in entity-to-entity edges, and to `top_k` facts. | RET-04 |
| RET-09 | Ubiquitous | The memory shall seed expansion from every entity that the hit chunks' episodes mention. | RET-02 |
| RET-05 | Event-driven | When chunk hits from the source tools and fact hits are both available, the memory shall fuse them by reciprocal rank fusion with the configured weights. | RET-05 |
| RET-06 | Ubiquitous | The memory shall attach to every hit its source tool, chunk id, source URI and valid period. | RET-06 |
| RET-07 | Unwanted | If the backend is unavailable, then `memory_search` shall return chunk-only hits with `degraded: true`. | RET-07 |
| RET-08 | Event-driven | When `memory_timeline` is called, the memory shall return all facts about the entity ordered by `valid_from`, including closed ones. | RET-08 |
| HST-01 | Ubiquitous | The binding shall expose `memory_search`, `memory_timeline` and `memory_provenance` on the Claude and OpenAI Agents backends with the schemas in `schemas/memory/`. | HST-01 |
| HST-02 | Ubiquitous | Tool results shall be JSON with ISO 8601 UTC timestamps. | HST-05 |
| HST-03 | Ubiquitous | Under Temporal, ingestion and every memory tool call shall run in activities. | — |
| GPH-01 | Ubiquitous | The Graphiti backend shall pass no document content to a model or endpoint that the YAML does not name. | Boundary |
| GPH-02 | Ubiquitous | The Graphiti backend shall disable Graphiti telemetry. | Boundary |
| GPH-03 | Ubiquitous | The Graphiti backend shall accept the drivers `neo4j`, `falkordb`, `neptune` and `kuzu`. | D13 |
| GPH-04 | Ubiquitous | For an `anthropic` extraction model, the Graphiti backend shall use the API-key client when `auth_provider` is `api_key` or not set, the Agent SDK client when it is `oauth_token`, and shall fail at load for any other value. | D24 |
| GPH-05 | Unwanted | If the driver's package is not installed, then loading shall fail with the install command for that driver. | — |
| CLI-01 | Event-driven | When `holodeck memory ingest <agent.yaml>` runs, it shall ingest and print an `IngestReport`: sources, episodes new and skipped, tokens, duration. | — |
| CLI-02 | Event-driven | When `holodeck memory status <agent.yaml>` runs, it shall print the backend, namespace, episode count and declared capabilities. | — |
| CLI-03 | Unwanted | If `holodeck memory purge` runs without `--yes`, then it shall ask for confirmation and name the namespace. | MOD-05 |
| EVAL-01 | Ubiquitous | The golden-corpus eval shall report point-in-time accuracy, the cross-document delta against chunk-only retrieval, supersession precision and recall, provenance correctness, and search latency. | — |

## 6. Commands

```bash
uv sync --extra graphiti --extra postgres
docker compose -f tests/integration/memory/compose.yaml up -d neo4j
uv run holodeck memory ingest sample/temporal-memory/agent.yaml --dry-run
uv run holodeck memory ingest sample/temporal-memory/agent.yaml
uv run holodeck memory status sample/temporal-memory/agent.yaml
uv run pytest tests/unit/lib/memory -n auto -q
uv run pytest tests/integration/memory -n auto -q -m graphiti
uv run python scripts/fetch_golden_corpus.py --out tests/golden/family-assistance/corpus
uv run python -m holodeck.lib.memory.eval --corpus tests/golden/family-assistance --report out/memory-eval.json
make schema && make schema-check
make harness-check
```

## 7. Project structure

```text
src/holodeck/lib/memory/
  ports.py            MemoryBackend protocol, MemoryCapabilities
  types.py            EpisodeIn, MemoryQuery, FactHit, MemoryHit, TimelineEntry, Provenance, IngestReport
  errors.py           typed errors
  service.py          TemporalMemory: hashing, effective date, ordering, fusion, degradation
  registry.py         backend selection by provider name
  eval.py             golden-corpus eval
  backends/fake.py    deterministic backend for unit tests
  backends/graphiti.py
src/holodeck/tools/temporal_memory_tool.py
src/holodeck/models/tool.py                 TemporalMemoryToolConfig
src/holodeck/cli/commands/memory.py
schemas/memory/                             host-neutral tool argument and result schemas
scripts/fetch_golden_corpus.py
tests/unit/lib/memory/  tests/integration/memory/
tests/golden/family-assistance/             manifest.yaml (pinned compilations, hashes), questions.yaml
sample/temporal-memory/
docs/guides/temporal-memory.md
```

## 8. Code style

House rules apply: concrete domain types, no `Any` or `getattr` around known models, SDK data converted at the boundary, no blocking I/O in async paths.

```python
@dataclass(frozen=True, slots=True)
class EpisodeIn:
    episode_id: UUID
    chunk_id: str
    source_uri: str
    text: str
    reference_time: datetime          # timezone-aware UTC
    reference_time_source: Literal["front_matter", "mtime", "ingestion"]


def episode_id_for(namespace: str, section_id: str, chunk_text: str) -> UUID:
    digest = hashlib.sha256(f"{section_id}\n{chunk_text}".encode()).hexdigest()
    return uuid5(uuid5(NAMESPACE_URL, f"holodeck/memory/{namespace}"), digest)
```

Every public operation emits a span named `holodeck.memory.<operation>`.

## 9. Testing strategy

| Level | Covers | Data | Runs |
| --- | --- | --- | --- |
| Unit | Episode ids, effective-date selection, ordering, fusion, config validation, degradation | `FakeBackend` | Every commit |
| Property-based | Re-ingest is a no-op; a changed chunk yields exactly one new episode; fusion is stable under equal ranks | Generated (Hypothesis) | Every commit |
| Contract | Tool names, argument and result schemas on both backends; `MemoryNotSupportedError` for each row marked not supported; no telemetry, no cross-encoder | Graphiti mocked | Every commit |
| Integration | Graphiti backend against Neo4j with a stub LLM client: ingest, skip, filters, timeline, provenance, purge | Docker, small fixture | Nightly, marker `graphiti` |
| Golden-corpus eval | Live models on the legislation corpus | `tests/golden/family-assistance/` | Nightly and before release |
| Live validation loop | `holodeck test run`, `holodeck serve` with raw AG-UI requests, CopilotKit flow | `sample/temporal-memory` | Before closing |

Every requirement ID appears in at least one test name. A CI check fails if an ID in section 5 has no test.

### Golden corpus

A fetch script downloads four to six pinned historical compilations of each Act from the Federal Register of Legislation, checks SHA-256 hashes, converts them to markdown, and writes `effective_date` front matter from each compilation's start date. The repository holds only the manifest, the labelled questions and the expected answers. The questions cover: a provision amended at least twice, a rate or threshold that changed between compilations, one entity named differently across the two Acts, a cross-Act reference, and a full re-run with no change.

## 10. Boundaries

**Always:** cite a requirement ID in every PR; keep `schemas/memory/` host-neutral; use UTC everywhere; run `--dry-run` and record its episode count before a live ingest of the golden corpus.

**Ask first:** adding the `graphiti-core` dependency and its pin; the accessor that lets the memory read chunks from `HierarchicalDocumentTool` (an interface extension); any change to how the chunker treats front matter; adding a port operation; changing the default `hops`, `top_k` or fusion weights; a live golden-corpus run, because it costs LLM spend; starting the native backend.

**Never:** put Graphiti types in a port signature or tool result; return partial results without an error; send document content to an endpoint not set in YAML; run LLM calls or I/O inside Temporal workflow code; commit the downloaded legislation text.

## 11. Success criteria

1. `make schema-check` passes with the `temporal_memory` tool type.
2. The contract tests pass on the Claude and OpenAI Agents backends.
3. Re-ingesting an unchanged corpus makes zero LLM calls and zero writes, counted by a stub LLM client.
4. Ingesting a second compilation creates episodes only for chunks whose hash is new.
5. Every fact returned by `memory_search` resolves to a source chunk through `memory_provenance` (100%).
6. A query with `valid_at` between two compilations returns the earlier fact and not the later one, on the integration fixture.
7. Each not-supported row in the capability table raises `MemoryNotSupportedError` with the operation name.
8. With the graph database stopped, `memory_search` returns chunk hits with `degraded: true`.
9. The golden-corpus eval produces a report with the five measures in EVAL-01. Proposed targets, reported and not gated in v1: point-in-time accuracy at least 90%, cross-document gain at least 15 points over chunk-only, supersession precision at least 0.9 and recall at least 0.8.
10. Every requirement ID is covered by at least one test (CI-enforced).
11. The live validation loop in `AGENTS.md` is recorded in the execution plan.
12. With two source tools on different vector stores, one ingest builds one graph, and a `memory_search` hit from each tool carries its `source_tool`.
13. On the integration fixture with the default extraction instruction, a rate that changes between two dated documents yields two facts, and the earlier one is closed at the later document's effective date.

## 12. Delivery

A stacked PR set, one per phase, bottom first. Each layer passes its own gates. The bottom branch, `feat/043-spec`, carries this spec, the plan, the Graphiti evidence and the dependency.

| Layer | Content | Requirements |
| --- | --- | --- |
| 1. Contracts | Tool config model, schema, port, types, errors, `FakeBackend`, `schemas/memory/` | MB, CFG |
| 2. Ingestion | Service ingest path, Graphiti backend ingest, `holodeck memory` CLI | ING, GPH, CLI |
| 3. Retrieval and tools | Search, timeline, provenance, fusion, tool binding on both backends | RET, HST |
| 4. Eval and docs | Fetch script, golden questions, eval, sample, guide | EVAL |

## 13. Trade-offs and what to revisit

| Decision | Gain | Cost | Revisit when |
| --- | --- | --- | --- |
| Graphiti first, native later | A working temporal graph without building extraction, resolution and judging | A graph database to run; no audit trail or review queue; results are Graphiti's | Auditors need rationale, or users will not run Neo4j |
| Graphiti's own pipeline | Least code; Graphiti's invalidation logic | Several LLM calls per chunk; behaviour cannot be fixture-tested | Ingest cost per compilation is too high |
| One chunk per episode | Fact-to-chunk provenance; unchanged chunks skip | More episodes and less context per extraction | Cross-chunk facts are missed in the eval |
| Content-hash episode ids | Stable across chunk renumbering; duplicates across collections extract once | Facts cite the first compilation or collection with that text | Users need "last confirmed in" as well |
| No score gate for Graphiti in v1 | The feature ships before targets are proven | No quality floor | The first eval report exists |
| Outgoing-only expansion (D23) | No dependency on Graphiti's storage schema | A seed that is only an edge target finds nothing upstream | The eval shows cross-document recall below target |
| Agent SDK client for OAuth-only users (D24) | Works with the credential HoloDeck's Claude users already have | About 3.4 seconds per LLM call | Ingest time for a real corpus is unacceptable |
| Real legislation as the corpus | Realistic amendments and cross-references | Hand labelling; a fetch step; LLM spend | Labelling takes longer than the build |

## 14. Open questions

1. Licence: confirm that Federal Register of Legislation text may be fetched and used for evaluation, and whether the labelled questions can quote it.
2. Which compilations of each Act, and who labels the questions?
3. What is the dry-run episode count for the full corpus, and is the resulting LLM spend acceptable?
4. Answered by T1: the hop limit can be set to 1 or 2; traversal is outgoing only (D23).
5. Partly answered by T1: one retroactive episode was closed correctly. Ingestion still submits in date order (ING-04); out-of-order safety is not proven.
6. Should the hierarchical tool strip front matter from chunk text? Today it does not.
