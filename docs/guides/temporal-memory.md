# Temporal Memory

The `temporal_memory` tool gives an agent a memory of dated facts. It builds a graph from the documents of your [hierarchical document tools](tools.md#hierarchical-document-tools), and it answers "what was true on this date, and according to which document?".

Use it when your documents amend each other: legislation with several compilations, policies with versions, price lists with effective dates. A hierarchical document tool finds passages. It cannot tell you which of two conflicting passages is current. The memory can.

The tool runs on [Graphiti](https://github.com/getzep/graphiti). Graphiti extracts the facts with a language model and stores them in a graph database. The design and the measured behaviour are in the [spec](../product-specs/043-temporal-memory/spec.md) and the [design](../design-docs/043-temporal-memory/design.md). A working example is in [`sample/temporal-memory`](https://github.com/justinbarias/holodeck/tree/main/sample/temporal-memory).

## How it works

Each fact has two periods:

- **Valid period.** When the fact was true in the documents. It starts at the effective date of the document that states it. It ends at the effective date of the document that replaces it.
- **Known period.** When the memory recorded the fact, and when it recorded that the fact had ended.

The agent asks "as of" a date. `memory_search` returns the facts that were valid on that date, with the source passage of each. A query with `known_at` before a later document was ingested returns what the memory believed at that time.

Two steps build the graph:

1. **Ingestion.** `holodeck memory ingest` reads the chunks of every source tool. It dates each document, orders the documents by date, and sends one chunk at a time to Graphiti as an episode. Graphiti extracts entities and facts, and closes older facts that a new one replaces.
2. **Retrieval.** `memory_search` runs the source tools' hybrid search, finds the entities that the hit chunks mention, expands through the graph, and merges chunk hits and fact hits by weighted reciprocal rank fusion.

## Setup

### Install

```bash
pip install 'holodeck-ai[graphiti]'
```

The `graphiti` extra installs `graphiti-core` and the Neo4j driver. If the extra is missing, the agent fails to load and the error names this command.

### Graph database

The tool needs a graph database. `backend.driver` accepts four values. Any other value fails validation.

| Driver | Status | Extra install |
| --- | --- | --- |
| `neo4j` | Tested. Use this one. | None. It comes with the `graphiti` extra. |
| `falkordb` | Configured the same way, not tested. Closure check and text closing are skipped. | `pip install 'graphiti-core[falkordb]>=0.31,<0.32'` |
| `neptune` | Not tested. Needs `aoss_host`. Closure check and text closing are skipped. | `pip install 'graphiti-core[neptune]>=0.31,<0.32'` |
| `kuzu` | Not tested. Deprecated upstream; HoloDeck logs a warning. Closure check and text closing are skipped. | `pip install 'graphiti-core[kuzu]>=0.31,<0.32'` |

If a driver package is missing, the error names the package and the install command.

Start a local Neo4j with the sample's [`compose.yaml`](https://github.com/justinbarias/holodeck/blob/main/sample/temporal-memory/compose.yaml), or use the integration suite's file, `tests/integration/memory/compose.yaml`.

### Models

Extraction needs a language model and an embedding model.

- **Extraction model.** `extraction_model` takes the same fields as the agent `model`. When it is unset, the agent's model is used. The supported providers are `openai`, `azure_openai`, `ollama` and `anthropic`. For `anthropic`, `auth_provider` chooses the client: `api_key` (the default) or `oauth_token`. Any other value fails at load. The `oauth_token` path runs each call through the Claude Agent SDK and is slow: 6.5 to 15.5 seconds per chunk in the first measurement.
- **Embedding model.** The memory uses the agent's `embedding_provider`, the same one as the source tools. All source tools must use the same embedding model. If they differ, loading fails and names the tools that differ. A change of embedding model needs a fresh graph.
- **Cross-encoder.** None. No document content goes to an endpoint that the YAML does not name.
- **Telemetry.** HoloDeck sets `GRAPHITI_TELEMETRY_ENABLED=false` before it creates a client.

### Environment variables

The sample reads these names. Set them in `.env` or in the process environment. The values are not shown here.

| Variable | Used for |
| --- | --- |
| `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` | The `backend` block, through `${...}` substitution. |
| `OPENAI_API_KEY` | The agent model, extraction and embeddings, when they use OpenAI. |
| `AZURE_OPENAI_API_KEY` | Extraction through `azure_openai`, when `extraction_model.api_key` is unset. |

For an OpenAI extraction model, the key comes from `extraction_model.api_key` or `OPENAI_API_KEY`. With `auth_provider: oauth_token`, the Claude CLI reads its token from the environment. Ingestion then refuses to run while `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`, `CLAUDE_CODE_USE_BEDROCK`, `CLAUDE_CODE_USE_FOUNDRY` or `CLAUDE_CODE_USE_VERTEX` is set, because they make the CLI use another provider.

### Minimal configuration

```yaml
tools:
  - name: policy_docs
    type: hierarchical_document
    description: Policy documents, one file per version.
    source: docs/policy/
    database: { provider: qdrant, connection_string: ${QDRANT_URL} }

  - name: policy_memory
    type: temporal_memory
    description: Dated facts from the policy documents and when they applied.
    source_tools: [policy_docs]
    backend:
      provider: graphiti
      driver: neo4j
      uri: ${NEO4J_URI}
      user: ${NEO4J_USER}
      password: ${NEO4J_PASSWORD}
```

An agent can have one `temporal_memory` tool. The agent also needs an `embedding_provider`.

## Configuration

Every field of a `temporal_memory` tool:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `name` | string | required | Tool name. Letters, digits and underscores. |
| `description` | string | required | Added in front of the built-in description that the agent reads. |
| `type` | `temporal_memory` | required | Tool type. |
| `source_tools` | list of strings | required | Names of `hierarchical_document` tools in the same agent. One graph covers all of them. No repeats. |
| `namespace` | string | tool name | Graph namespace (Graphiti `group_id`). Letters, digits, `_` and `-`. |
| `backend` | object | required | Graph connection. See below. |
| `extraction_model` | model block | agent model | The model that extracts facts. |
| `extraction_reasoning_effort` | `none`, `minimal`, `low`, `medium`, `high` or `xhigh` | unset | Reasoning effort for an OpenAI or Azure OpenAI extraction model. Unset keeps Graphiti's `auto`, which is `none` for GPT-5.1 and later. Other providers ignore it and log a warning. |
| `close_on_text_change` | boolean | `false` | Close an open fact when a newer version of its section no longer states it. See [How facts close](#how-facts-close). Neo4j only. |
| `extraction_instructions` | string | built-in text | Guidance for extraction. The default asks for amounts, rates, percentages, thresholds and dates as entities, stated as facts with their unit and period. |
| `effective_date_key` | string | `effective_date` | Front-matter key that holds a document's effective date. No spaces. |
| `ingest_on_start` | boolean | `false` | Run ingestion when the tool starts. |
| `hops` | integer, 1 or 2 | `1` | How many entity-to-entity edges `memory_search` follows from its seeds. |
| `top_k` | integer, 1 to 100 | `10` | Hits that `memory_search` returns. |
| `fusion_weights` | object | `chunks: 1.0`, `graph: 1.0` | Weights for merging chunk hits and fact hits. Each is 0 or more. They cannot both be 0. |
| `schema_hints` | object | unset | Guidance for the extraction model. See [Schema hints](#schema-hints). |

`backend` fields:

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `provider` | `graphiti` | `graphiti` | The only backend. |
| `driver` | `neo4j`, `falkordb`, `neptune` or `kuzu` | required | Graph database driver. |
| `uri` | string | none | Connection URI. Required for every driver except `kuzu`. For `kuzu`, a database path; unset means an in-memory database. |
| `user` | string | none | Database user. |
| `password` | string | none | Database password. Keep it in an environment variable. |
| `aoss_host` | string | none | OpenSearch host. Required for `neptune`. |

Loading fails, and the message names the tool, when:

- `source_tools` is empty, repeats a name, or names anything other than a `hierarchical_document` tool in the same agent.
- The source tools use different embedding models.
- The agent has more than one `temporal_memory` tool.
- A function tool in the same agent uses the name `memory_search`, `memory_timeline` or `memory_provenance`.
- `schema_hints.single_valued_predicates` is set. The Graphiti backend does not support it.
- `extraction_model` uses `anthropic` with an `auth_provider` other than `api_key` or `oauth_token`.

### Schema hints

`schema_hints` tells the extraction model which kinds of things to look for. All parts are optional.

```yaml
schema_hints:
  entity_types:
    - Person
    - name: Payment
      description: A benefit or payment, such as Family Tax Benefit.
  edge_types:
    - name: HAS_RATE
      description: A payment has a rate or amount.
      attributes:
        - { name: amount, type: number }
        - { name: unit, type: string }
        - { name: category, type: string, description: Who the rate applies to }
      single_valued: true
      key_attributes: [category]
  edge_type_map:
    - { source: Payment, target: Entity, edges: [HAS_RATE] }
```

| Field | Meaning |
| --- | --- |
| `entity_types` | Entity types to prefer. Each item is a name, or an object with `name` and `description`. |
| `edge_types` | Relation types to use for facts. Each has a `name`, a required `description`, optional typed `attributes` (`string`, `number`, `date` or `boolean`), `single_valued` and `key_attributes`. |
| `edge_type_map` | Which edge types may join which entity types. Each entry has `source`, `target` and `edges`. Use `Entity` for any type. |

Rules checked at load:

- Names start with a letter and use letters, digits and `_`. Names are unique.
- `edge_type_map` names only declared types, and each pair appears once.
- An edge attribute cannot reuse a Graphiti edge field (for example `name`, `fact`, `valid_at`).
- `key_attributes` names attributes and needs `single_valued: true`.

`single_valued` and `key_attributes` are written into the type's description for the extraction model. The closure check does not read them.

#### Designing schema hints

Section numbers and Acts should stay provenance. Do not make them entity types.

A full ingest of the Family Assistance Acts used a typed ontology with `Section` and `Act` entity types and a `REFERS_TO` edge. In the resulting graph, 72% of the entities were Sections and only 13% of the facts joined two domain entities. The graph recorded which section refers to which, and little about the payments, people and amounts.

Instead:

- Choose types for the domain: people, payments, amounts, conditions, powers and duties.
- Write a clear description for each type and edge. Graphiti uses the descriptions to choose a type.
- Map only the pairs you care about. A pair that is not in `edge_type_map` keeps the model's own edge names.
- The chunk's source already gives the section. `memory_provenance` returns it.

## Dating documents

The effective date of a document becomes the start of the facts it states. HoloDeck takes it from the first of these that exists:

1. The front-matter key named by `effective_date_key` (default `effective_date`).
2. The file's modified time, when the file is a local source.
3. The time of ingestion.

```markdown
---
effective_date: 2026-07-01
series_id: family-assistance-act
---

# Part 1 ...
```

A date with no time means the start of that day in UTC. A date with an offset is converted to UTC. A date with no offset is taken as UTC. A value that is not an ISO 8601 date fails the ingest, and the error names the key. It does not fall back to the modified time.

Each episode records which source was used (`front_matter`, `mtime` or `ingestion`), and `memory_provenance` returns it. Use front matter. A modified time changes when you copy a file, and a downloaded source has only its download time.

Front matter must be a YAML mapping between two `---` lines at the start of the file. The memory reads it. The chunker is not changed by it.

Put each version of a document in its own file with its own effective date. Replacing a file in place does not create a new version.

### `series_id`

`series_id` is an optional front-matter key. Documents with the same `series_id` are versions of one document. Documents with different ids are never versions of each other.

Set it when one source tool holds more than one document, for example two Acts in one folder. Without it, the memory treats documents from the same source tool as versions of each other, and a heading that two documents share counts as one section. Text closing and the closure check can then compare the wrong documents. The value is a string or an integer.

## Ingestion

```bash
holodeck memory ingest agent.yaml --dry-run   # count only
holodeck memory ingest agent.yaml             # extract and write
holodeck memory status agent.yaml
holodeck memory purge agent.yaml              # asks first; --yes skips the question
```

| Command | What it does |
| --- | --- |
| `ingest` | Ingests the source tools' own documents, then extracts facts into the graph. Prints documents, episodes new and skipped, episodes written, LLM calls, tokens and duration. `--tool NAME` limits it to one source tool; repeat it for several. |
| `ingest --dry-run` | Counts the new episodes from the files on disk. It makes no model call, no embedding call and no vector store write, and writes nothing to the graph. It prints an estimated duration only for an `anthropic` extraction model with `auth_provider: oauth_token` (11 seconds per episode). For other models it prints "no estimate for this model". |
| `status` | Prints the backend, the namespace, the episode count and the operations the backend supports. |
| `purge` | Removes every episode, entity and fact in the namespace, and no other namespace. It asks for confirmation and names the namespace. |

How ingestion behaves:

- **One chunk is one episode.** The episode id is a hash of the chunk's section id and text. An unchanged chunk has the same id in every run.
- **An unchanged re-ingest makes no model calls.** A chunk whose episode is complete is skipped. A new compilation of a document extracts only the chunks whose text changed. A chunk that is identical in two compilations is extracted once, and its facts cite the first compilation that had it.
- **Order.** Documents from all source tools go in one ascending effective-date order. Chunks go in document order inside a document.
- **An interrupted run resumes.** Ingestion stops at the first episode that fails and reports it. Run the command again. Complete episodes are skipped, and an episode that was left incomplete is processed again.
- **One writer per namespace.** Run one ingestion at a time for a namespace. HoloDeck does not enforce this. The CLI, `ingest_on_start` and the Temporal activity use no lock, and two writers can corrupt the closure checks.
- **`ingest_on_start: true`** ingests when the tool starts, and the agent fails to start if the graph is down. Leave it `false` for a large corpus and run the CLI.

Ingestion emits one trace with a span per episode when observability is on. See [Observability](observability.md).

## How facts close

A fact closes when a newer document replaces it. Three mechanisms do this. The first is Graphiti's. The other two are HoloDeck's.

**Graphiti's contradiction step.** When a new fact arrives, Graphiti asks the model which older facts it contradicts. Graphiti then ends those facts on the new fact's start date. HoloDeck replaces Graphiti's prompt for this step with one written for dated documents: two facts that state different values of the same attribute of the same subject (a rate, an amount, a threshold, a period) contradict each other.

**The closure check.** The model closes facts that did not change. In one live run, about 41 of 63 closures were wrong. After each episode, the backend checks every fact that this episode closed, and it reopens the closure when:

- an episode dated after this one states the fact, so this episode is an older version ingested late;
- the fact's earliest other source is in another document (by `series_id`, else by source tool);
- that source is in another section;
- no open fact written by this episode could replace it (same two entities, or the same endpoint and relation name);
- every line of the older section is still in the new section; or
- the line that matches the fact best is still in the new text.

Otherwise the fact stays closed. Facts with no start date are given the earliest start of the episodes that state them, so that a later document can close them.

**Text closing** (`close_on_text_change: true`, default off). The closure check only reopens. It cannot close a fact that Graphiti never proposed to close. With this setting, the backend also closes an open fact when the newer version of its section no longer states it: the fact's source line changed, and the new text matches the fact clearly worse than the old line did. It does not close a fact that was only reworded, for example "subsection (4) does not apply" changed to "subsections (4) and (4A) do not apply".

Limits:

- Closing is still partly a model decision. A changed rate can stay open next to the new one. Check `memory_timeline` against the source text.
- The closure check and text closing run on Neo4j only. Other drivers skip them and log one line.
- The matching is lexical. A fact paraphrased far from its source line is never closed by text. A changed line that keeps all of the fact's key words and adds no word such as "not" or "only" is kept open, for example "applies to children" changed to "applies to children over 16".
- A section that is split in two numbers ("3" into "3" and "3A"), or a heading renamed with its number, counts as another section, so a real change there is reopened.
- A replacement fact that the model gives other entities or another relation name than the closed fact is not counted as a replacement, so that closure is undone.
- The thresholds of text closing are starting values. They are not tuned on live data, and the 178 text closures in the full Family Assistance run are not audited.
- `memory_search` returns chunk hits from superseded versions too. A chunk has a start date but no end date. Only fact hits carry an end date.

## Tools

The agent gets three functions. The names are fixed and have no tool-name prefix. Argument and result schemas are in [`schemas/memory/`](https://github.com/justinbarias/holodeck/tree/main/schemas/memory).

| Function | Arguments | Result |
| --- | --- | --- |
| `memory_search` | `query` (required); `valid_at`, `known_at` (ISO 8601 date or datetime, default now); `hops` (1 or 2, default from the config) | `valid_at` and `known_at` as used, `degraded`, and `hits`. |
| `memory_timeline` | `entity` (required): the entity's name as the facts state it | `entity` and `entries`: every fact about the entity, open and closed, ordered by `valid_from`. |
| `memory_provenance` | `fact_id` (required): an id from another memory tool | `found`, the `fact`, and `sources`: each episode with its source tool, chunk id, source URI, passage text, effective date and the source of that date. |

`top_k` is not an argument. It comes from the configuration.

A hit has `kind` (`fact` or `chunk`), `text`, `score`, `source_tool`, `chunk_id`, `source_uri`, `fact_id`, `valid_from`, `valid_to` and `superseded_by`. A `chunk` hit has no `fact_id`. On Graphiti `superseded_by` is always null. Every timestamp is ISO 8601 UTC with a `Z` suffix.

```json
{"query": "base rate of Family Benefit", "valid_at": "2021-01-01"}
```

```json
{
  "valid_at": "2021-01-01T00:00:00Z",
  "known_at": "2026-11-12T03:10:00Z",
  "degraded": false,
  "hits": [
    {
      "kind": "fact",
      "text": "The base rate of Family Benefit is 100 dollars per fortnight.",
      "score": 0.031,
      "source_tool": "policy_docs",
      "chunk_id": "policy_2020_chunk_4",
      "source_uri": "docs/policy/policy-2020.md",
      "fact_id": "3f1c0b52-6b1e-4a64-9c53-1b9d1f0f6a7e",
      "valid_from": "2020-07-01T00:00:00Z",
      "valid_to": "2022-07-01T00:00:00Z",
      "superseded_by": null
    }
  ]
}
```

`memory_timeline` takes a name, not a question: `{"entity": "Family Benefit"}`. It matches an entity whose name equals the text, ignoring case. If none does, it matches an entity whose whole name appears in the text. A paraphrase finds nothing.

`memory_provenance` takes the id from a hit: `{"fact_id": "3f1c0b52-6b1e-4a64-9c53-1b9d1f0f6a7e"}`. An id that is not in the graph returns `found: false` and is not an error.

### Errors and degraded mode

A cause the model can correct comes back as a result, not an exception:

```json
{"error": {"code": "invalid_argument", "message": "..."}}
```

| `code` | Cause |
| --- | --- |
| `invalid_argument` | An argument fails validation, for example a date that is not ISO 8601. |
| `not_supported` | The backend cannot do the operation. The message names it. |
| `backend_unavailable` | The graph database is down, for `memory_timeline` or `memory_provenance`. |

When the graph database is down, `memory_search` does not fail. It returns the chunk hits of the source tools with `degraded: true`. If the database is down at start and `ingest_on_start` is `false`, the tool starts and logs a warning.

On the Claude backend, the SDK checks the arguments against the schema before the handler runs. A call that breaks the schema (for example `hops: 3`) returns the SDK's own validation text. The OpenAI Agents backend returns the `invalid_argument` result.

## Capabilities

What the Graphiti backend supports. `holodeck memory status` lists the supported operations.

| Operation | Graphiti backend |
| --- | --- |
| Idempotent re-ingest | Supported. A lookup by episode id, a pre-saved episode node and a completion marker. |
| Entity type hints | Supported (`schema_hints`). |
| `valid_at` and `known_at` queries | Supported, with Graphiti search filters. |
| Seed-and-expand, 1 or 2 hops | Supported. Breadth-first search that follows outgoing edges only, so a seed that is only the target of an edge finds nothing upstream. |
| Facts for amounts and rates | Needs extraction instructions. The built-in default asks for them. |
| Provenance to the chunk | Supported. |
| `single_valued_predicates` | Not supported. Loading fails and names the field. |
| Supersession rationale, model id, prompt version | Not supported. Raises `MemoryNotSupportedError`. |
| Retraction when a source version is replaced | Not supported. Add a new version as a new file. |
| Pending-merge and low-confidence review | Not supported. |
| Per-step extraction retry policy | Graphiti's own. |

`MemoryNotSupportedError` names the operation, and a backend never returns a partial result.

## Backends and Temporal

The three functions work the same on the [Claude backend](claude-backend.md) and the [OpenAI backend](openai-backend.md). OpenAI and Azure models run on the OpenAI Agents backend. Anthropic and Ollama models run on the Claude backend. The extraction model is separate from the agent model, so either backend can use any supported extraction provider.

Under a [Temporal worker](temporal-worker.md), an agent node is one activity, so every memory call runs inside that activity. Ingestion has its own activity, `holodeck.memory_ingest`, which sends a heartbeat after each episode and is safe to retry. A worker refuses an agent whose memory tool has `ingest_on_start: true` and names `holodeck memory ingest` as the alternative. No memory code runs in workflow code.

## Cost and time

Each episode costs several model calls. Two measured runs of the sample with OpenAI extraction:

| Run | Episodes | Time per episode | OpenAI calls per episode |
| --- | --- | --- | --- |
| 2026-11-12: both Acts, `extraction_reasoning_effort: medium`, typed `schema_hints` | 851 | about 33 to 47 s | about 14 |
| 2026-11-11: the Family Assistance Act only, effort unset, no `schema_hints` | 418 | about 10.6 s | about 7 |

The runs differ in corpus as well as settings, so read the gap as a rough guide. Typed schema hints add calls per episode, and a higher reasoning effort makes each call slower.

The Agent SDK path (`anthropic` with `oauth_token`) takes about 3.4 seconds per model call, because each call starts the Claude CLI.

Run `--dry-run` first. It tells you how many episodes are new. Ingest one source tool at a time with `--tool` to spread the cost, and run the command again after an interruption. The second compilation of a document costs only the chunks that changed.

A repeat run over an unchanged corpus skips every episode and makes no model calls.

## Sample and browser views

- [`sample/temporal-memory`](https://github.com/justinbarias/holodeck/tree/main/sample/temporal-memory) ingests two Acts in two compilations each, with five test cases. Its README lists every command.
- [`sample/copilotkit`](https://github.com/justinbarias/holodeck/tree/main/sample/copilotkit) renders the three memory tools: `memory_timeline` as a fact timeline with an as-of date marker, `memory_search` as dated fact and passage hits with a degraded banner, and `memory_provenance` as the fact with its source documents and passage. Open `/memory-demo` to see the views from fixtures, with no backend.

## Next steps

- [Tools](tools.md): every tool type.
- [Vector Stores](vector-stores.md): the stores behind the source tools.
- [Temporal Worker](temporal-worker.md): run memory under Temporal.
- [Observability](observability.md): traces for ingestion.
