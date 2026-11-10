# Graphiti evidence for spec 043

Measured facts that the [spec](../../product-specs/043-temporal-memory/spec.md) and the [execution plan](../../exec-plans/active/043-temporal-memory/2026-11-11-temporal-memory-graphiti.md) depend on. Each entry gives the command and what it returned.

## T0: dependency resolution (2026-11-11)

**Result:** `graphiti-core` 0.31.2 resolves and imports with HoloDeck's `openai>=3,<4` pin. Spec assumption A2 holds.

The extra added to `pyproject.toml`:

```toml
graphiti = [
    "graphiti-core[anthropic]>=0.31,<0.32",
]
```

| Check | Command | Result |
| --- | --- | --- |
| Lock | `uv lock` | Resolved 331 packages. No override or constraint was added. |
| New packages | diff of package names in `uv.lock` before and after | Two: `graphiti-core` 0.31.2 and `neo4j` 6.4.1. Every other requirement of `graphiti-core` was already locked. |
| Versions in use | `importlib.metadata.version` after `uv sync --all-extras` | `openai` 3.48.0, `anthropic` 2.6.0, `pydantic` 2.13.5, `posthog` 7.86.4 |
| Import | `from graphiti_core import Graphiti`, `SearchFilters`, `AnthropicClient`, `OpenAIGenericClient`, `Neo4jDriver` | All import. |
| Install profiles | `uv run pytest tests/unit/test_install_profiles.py tests/unit/test_constraints_sync.py tests/unit/test_dockerfile_pins.py tests/unit/test_verify_profiles.py -n auto -q` | 83 passed |

What this does not show: that any Graphiti call works against `openai` 3 at run time. The import succeeds; the first real LLM call is made in T1.

Notes for later tasks:

- `posthog` was already a locked package before this change. Graphiti's own telemetry still has to be turned off (spec A4, task T7).
- Only the Neo4j driver comes with the base package. `falkordb`, `neptune` and `kuzu` need their own `graphiti-core` extras, which the `graphiti` extra does not install. T7 must report a missing driver with its install command.
- The repository has a guard that a new extra needs a row in the install-profile verifier. The `graphiti` profile and its `graphiti-imports` probe were added to `scripts/verify/`. The full `make verify-profiles` run, which builds a wheel and installs each profile, was not run.

## T1: Graphiti behaviour (2026-11-11)

Run by a subagent against `neo4j:5.26` in Docker with `graphiti-core` 0.31.2. The orchestrator checked the raw outputs against the findings below. Scripts and outputs were kept in the session scratchpad and are not committed.

**Setup**

- LLM: `claude-haiku-5-5` through the Claude Agent SDK with `CLAUDE_CODE_OAUTH_TOKEN`, by a custom subclass of Graphiti's `LLMClient`. Graphiti's own `AnthropicClient` needs an API key and was not used.
- Embeddings: local Ollama `nomic-embed-text` (768 dimensions) through `OpenAIEmbedder`.
- Cross-encoder: a stub that raises when called.
- Fixture: four one-sentence episodes about one benefit rate (90, 100 and 120 dollars, dated 2019, 2020 and 2022) and its administering body. The 2019 episode was ingested last.
- Size: 10 `add_episode` runs, 31 LLM calls, 0 schema-validation failures.

**Findings**

| Item | Observed | Effect on the spec |
| --- | --- | --- |
| (a) Episode uuid | `add_episode(uuid=U)` raises `NodeNotFoundError` when U does not exist, with 0 LLM calls. The argument re-processes an existing episode; it cannot create one. A re-run costs 2 to 3 LLM calls, ignores a changed body, and adds a duplicate `MENTIONS` edge each time. `EpisodicNode.get_by_uuid` raises `NodeNotFoundError` for an absent uuid in about 3 ms with no LLM call; `get_by_uuids` returns `[]`. | **Contradicts** the "Episode identity and skipping" design. The existence check works; creating an episode under a chosen uuid needs another mechanism. |
| (b) Time filters | Graphiti closed the 100-dollar fact when the 120-dollar fact arrived: `invalid_at` = 2022-07-01 (the new episode's reference time), `expired_at` = the wall-clock time of that ingest. "Valid at 2021-01-01" returned the 100 fact only; "valid at 2023-01-01" returned the 120 fact only; a "known at" filter between the two ingests returned the 100 fact only. Each needs an explicit `IS NULL` branch: a plain `invalid_at > t` returned nothing. | Confirms the bi-temporal mapping, with an implementation rule. |
| (b) Filter defect | Two dated OR-groups on one field share a parameter name. `valid_at = 2019-07-01 OR valid_at = 2022-07-01` generated `(e.valid_at = $valid_at_0) OR (e.valid_at = $valid_at_0)` and returned only the 2022 facts. The `IS NULL` pattern binds no parameter and is not affected. | Implementation rule: at most one dated group per field. |
| (c) Hop limit | `bfs_max_depth` (default 3) can be set to 1 or 2 through `search_` with `bfs_origin_node_uuids`. On a chain, depth 1 returned 5 edges and depth 2 returned 6. The traversal follows outgoing edges only: a node that is only an edge target returned 0 edges at every depth. From an episode node, `MENTIONS` uses the first hop. Breadth-first search returns closed facts unless a time filter is passed. | **Changes** RET-04: the limit exists, but expansion is one-directional. |
| (d) Retroactive ingest | The 2019 fact, ingested last, was closed at 2020-07-01, and the later facts were unchanged. One run only. Its `expired_at` is its own ingest time, so "known at" history is not meaningful for back-filled facts. | Supports ING-04 as written (ingest in date order); out-of-order ingest is not proven safe. |
| (e) Cross-encoder | `search()` and `search_(COMBINED_HYBRID_SEARCH_RRF)` made 0 cross-encoder calls. The default `search_` configuration called it. Constructing `Graphiti` with no `cross_encoder` and no `OPENAI_API_KEY` raised `OpenAIError` at construction. `search()` sets `limit` on a shared module-level recipe object. | Confirms A3. The backend must always pass a cross-encoder object and must copy recipe configurations. |
| (f) Telemetry | One PostHog event, `graphiti_initialized`, is sent from `Graphiti.__init__`. The variable is read at call time. With `GRAPHITI_TELEMETRY_ENABLED=false` set before construction, the patched capture counter stayed at 0 through construction and one `add_episode`; unset, it recorded 1. Enabled telemetry also writes `~/.cache/graphiti/telemetry_anon_id`. | Confirms A4. |
| (g) Agent SDK as the LLM client | Worked: 31 calls, 0 schema failures, 2 to 5 calls and 6.5 to 15.5 seconds per episode, about 3.4 seconds per call because each call starts the CLI. | The OAuth-token path is viable for correctness; it is slow. |
| (h) Seeds and provenance | `get_nodes_and_edges_by_episode([uuid]).nodes` returned the episode's entity uuids. `EntityEdge.episodes` returned the episode that produced each fact. `EpisodicNode.entity_edges` is overwritten on each run and is not a cumulative record. | Confirms RET-02 and provenance. |
| Extraction of amounts | With the default prompts, "the base rate is 100 dollars per fortnight" produced one entity and **0 facts**: Graphiti's extraction prompt tells the model not to extract numbers as entities. Dated rate facts appeared only after passing `custom_extraction_instructions`. | **Changes** the spec: the tool needs extraction instructions, or amounts never become facts. |

**Other observations**

- `group_id` must match `^[a-zA-Z0-9_-]+$`. On Neo4j all groups share one database and are separated by the `group_id` property.
- `Graphiti(tracer=..., trace_span_prefix=...)` is the only tracing hook; there is no per-call tracer. Span attributes use Graphiti's names, not the OpenTelemetry GenAI conventions.
- The experiment client set `permission_mode="bypassPermissions"` with `tools=[]`. A product client must not copy that setting; it needs a review of the least-privilege options first.

**Not measured**

- Any provider other than Claude through the Agent SDK. No Graphiti call was made through the `openai` 3 client.
- Concurrency, and ingest time at corpus scale.
- Retroactive ingest more than once, or with several competing facts.
