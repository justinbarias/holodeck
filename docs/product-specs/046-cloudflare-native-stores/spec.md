# SPEC: Cloudflare-native stores

**Status:** Draft for review
**Date:** 2026-11-15
**Depends on:** [045 Deploy v2](../045-deploy-v2/spec.md) (Cloudflare target),
[042 Dependency Stack Revamp](../042-dependency-stack-revamp/index.md)
(native retrieval adapters and registry)
**Related:** [043 Temporal Knowledge Graph Memory](../043-temporal-memory/index.md)
**Decision history:** The decisions in §3 were settled in a design interview on
2026-11-15. An adversarial review on the same day (gpt-6-astra, 24 findings)
changed D1, D2 and D6 and added D7.

## 1. Objective

A HoloDeck agent hosted on Cloudflare can keep its retrieval data on
Cloudflare. This spec adds three providers and one piece of guidance:

| Need | Today | This spec |
| --- | --- | --- |
| Vector store | postgres, qdrant, weaviate, chromadb, pinecone, azure-ai-search, azure-cosmos-nosql, in-memory | adds `cloudflare-vectorize` |
| Keyword index (hybrid search) | in-memory, opensearch | adds `cloudflare-d1` (SQLite FTS5) |
| Embeddings | openai, azure_openai, ollama | adds `cloudflare_workers_ai` |
| Graph store (`temporal_memory`) | Graphiti on neo4j, falkordb, neptune, kuzu | no new provider; an external managed graph database |

These providers are for agents that run on the Cloudflare target. A developer
can also run the same agent on a local machine to test it (D1). They are not
for agents hosted on Azure or any other target.

Cloudflare has no graph database product. §7 covers that case.

The chat thread store is not in this spec. It is specified separately.

## 2. Platform facts

Read from the Cloudflare documentation on 2026-11-15.

**Vectorize**

- At most 1536 dimensions for each vector, and 20,000,000 vectors for each index.
- At most 10 KiB of metadata for each vector.
- At most 10 metadata indexes for each index. A string metadata index covers
  only the first 64 bytes of the value.
- A vector ID is at most 64 bytes.
- `topK` is at most 50 when the query returns values or metadata, and 100 when
  it returns neither.
- Writes are asynchronous. A completed write is not searchable at once. An
  upsert replaces the vector and all its metadata. The index information
  reports the last mutation that was processed.
- The dimension count and distance metric of an index cannot change.
- No keyword or hybrid search. Cloudflare AI Search has hybrid retrieval, but
  it is a different product.

**D1**

- A database is at most 10 GB on the paid plan and 500 MB on the free plan.
  The 10 GB limit cannot be raised.
- A row is at most 2 MB. A statement is at most 100 KB with at most 100 bound
  parameters. A query runs for at most 30 seconds.
- FTS5 is supported.

**Workers AI**

- Embedding models and their dimensions: `@cf/baai/bge-small-en-v1.5` 384,
  `@cf/baai/bge-base-en-v1.5` 768, `@cf/baai/bge-large-en-v1.5` 1024.

**Access paths**

- A Worker reaches these services through bindings. A container reaches a
  Worker's bindings through outbound interception: the container sends a plain
  HTTP request to a virtual host name, and the Durable Object maps that host
  to a Worker entrypoint that holds the bindings. No credential is needed in
  the container.
- The REST API also reaches them with an API token. Cloudflare describes D1
  over REST as best suited to administrative use. The REST API allows 1,200
  requests in five minutes for each user, counted across all tokens.

## 3. Decisions

**D1 — Two transports: bindings when deployed, REST for local testing.**
*(Changed in review.)* On the Cloudflare target, the providers use Worker
bindings. On a developer machine, they use the REST API with a token from the
local environment.
*Why:* these stores serve Cloudflare-hosted agents, and bindings are the
application path: no rate limit and no account token in the container. A
developer still needs to run `holodeck test` and `holodeck serve` locally
against the real stores. The REST rate limit is acceptable for that.
*Cost:* two transports to build and validate.

**D2 — HoloDeck creates the resources where they are first needed.**
*(Changed in review.)* `holodeck deploy run` creates and binds any resource
that is missing. A local run creates a missing resource at first ingestion.
Both record what they created.
*Why:* a binding must name a resource that exists at deploy time, and a local
test can run before any deployment. The record gives `deploy destroy
--include-data` its ownership rule (045 D9).

**D3 — Hybrid search is in scope.** `cloudflare-d1` is a keyword index provider
with SQLite FTS5.
*Why:* `hierarchical_document` needs a keyword side, and Vectorize has none.
Without it, a Cloudflare deployment needs OpenSearch or rebuilds an in-memory
index on every start.

**D4 — Workers AI is an embedding provider only.** `cloudflare_workers_ai` is
valid as an embedding provider. It is not valid as a chat model provider.
*Why:* embeddings complete the Cloudflare-only path. Chat models need a backend
routing decision that this spec does not make.

**D5 — Graph store: external managed service.** On Cloudflare,
`temporal_memory` uses a managed graph database reached over the network. A
native `MemoryBackend` on D1 and Vectorize is deferred to 043 Phase 2.
*Why:* a native backend replaces Graphiti's extraction and bi-temporal logic.
That is a large and separate piece of work. A graph database in a Cloudflare
container loses its data on restart.

**D6 — Record text lives in Vectorize metadata, with a size limit.**
*(Changed in review.)* A record that does not fit in the 10 KiB metadata limit
fails ingestion.
*Why:* one store and one write. Two stores cannot be kept consistent by write
order alone, because Vectorize writes are asynchronous. Cloudflare documents
metadata as the place for small source content.
*Cost:* a chunk with its context must fit in 10 KiB, less the other fields.
`topK` stays at 50, because every query returns metadata.

**D7 — One Vectorize index for each logical collection.** *(Added in review.)*
The adapter derives the index name from `index_name` and the collection name.
*Why:* HoloDeck keeps a manifest collection beside the data collection, with a
different dimension count. One index cannot hold both.

## 4. Configuration

```yaml
embedding_provider:
  provider: cloudflare_workers_ai

tools:
  - type: hierarchical_document
    name: policies
    source: ./docs
    embedding_model: "@cf/baai/bge-base-en-v1.5"
    embedding_dimensions: 768
    database:
      provider: cloudflare-vectorize
      index_name: policies
    keyword_index:
      provider: cloudflare-d1
      d1_database: holodeck-policies
```

The account comes from `deployment.target.cloudflare.account_id` when
deployed, and from `CLOUDFLARE_ACCOUNT_ID` on a developer machine. Field names
are a proposal. The plan sets the final names against the existing models.

## 5. Requirements

**Transport and credentials**

- **FR-T1:** On the Cloudflare target, the generated Worker project (045
  FR-C10) MUST declare the bindings for every store the agent config names,
  and MUST make them reachable from the container through outbound interception.
- **FR-T2:** In a container on the Cloudflare target, the providers MUST use
  the binding transport. The container MUST NOT hold a Cloudflare API token
  for these stores.
- **FR-T3:** Outside a Cloudflare deployment, the providers MUST use the REST
  transport with `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` from the
  environment. They MUST log once that this transport is for testing and name
  the rate limit.
- **FR-T4:** The deployment sets the transport. A provider MUST choose the
  binding transport only from a signal that `deploy run` sets, and MUST fail
  when that signal is present and the binding host cannot be reached. It MUST
  NOT fall back to REST.
- **FR-T5:** Both transports MUST give the same results for the same
  operations. The provider contract tests MUST run against both.
- **FR-T6:** `holodeck deploy run` to a target other than `cloudflare` MUST
  fail when the agent uses one of these providers. The message MUST say that
  the provider needs the Cloudflare target.
- **FR-T7:** A REST rate-limit response MUST be retried with backoff and then
  reported as a rate-limit error that names the limit. Other transient errors
  MUST be retried with backoff. A store that cannot be reached MUST raise the
  adapter's existing unavailable error.
- **FR-T8:** API tokens MUST be `SecretStr` and MUST NOT appear in logs,
  errors, traces or the tool manifest.
- **FR-T9:** Documentation MUST list the token permissions for a local test
  run (Vectorize, D1, Workers AI) and for a deployment, and MUST say that the
  deployment token never enters the container (045 FR-C20).

**Provisioning and ownership**

- **FR-P1:** `deploy run` MUST create each Vectorize index and D1 database that
  the agent config needs and that does not exist, before it deploys the Worker.
- **FR-P2:** A local run MUST create a missing resource at first ingestion.
- **FR-P3:** The physical resource names MUST be derived from the agent config
  alone, so that `deploy run` and a local run find the same resources.
- **FR-P4:** HoloDeck MUST record every index, database and table it creates,
  with the account ID and the resource ID, in the deploy state record or in a
  local record for a local run. It MUST record a resource as soon as it exists.
- **FR-P5:** HoloDeck MUST NOT record a resource that already existed as its own.
- **FR-P6:** When an index with the derived name exists with a different
  dimension count or metric, the provider MUST fail and MUST NOT recreate it.

**Vector store — `cloudflare-vectorize`**

- **FR-V1:** `DatabaseConfig.provider` MUST accept `cloudflare-vectorize`. The
  provider MUST be registered in the retrieval registry and implement
  `VectorStoreAdapter`.
- **FR-V2:** The adapter MUST map each logical collection to its own index
  (D7), including the manifest collection and migration collections. The
  manifest adapter for this provider MUST work. Failure to build it MUST be an
  error and MUST NOT continue without fingerprint protection.
- **FR-V3:** An embedding dimension above 1536 MUST fail before any store is
  created or changed. The error MUST name the limit and the configured model.
- **FR-V4:** A `top_k` above 50 MUST fail, both in the config and when a
  caller overrides it at search time.
- **FR-V5:** The adapter MUST create the metadata indexes for the filterable
  fields before the first upsert. A schema that needs more than 10 metadata
  indexes MUST fail before any store is created or changed.
- **FR-V6:** A record whose metadata, with its text, is larger than 10 KiB
  MUST fail ingestion. The error MUST name the record and its size. The
  adapter MUST NOT shorten the text.
- **FR-V7:** Physical vector IDs MUST be at most 64 bytes and MUST NOT collide
  for different logical IDs.
- **FR-V8:** An exact-match filter on a string longer than 64 bytes MUST give
  exact results. The adapter MUST NOT rely on the 64-byte index prefix alone.
- **FR-V9:** After a write, ingestion MUST wait until the index reports that
  the mutation is processed, with a fixed upper wait. It MUST report the tool
  as ready only after that. A timeout is an error, not a silent pass.
- **FR-V10:** Delete and re-ingest MUST remove the record from the index.
  Search MUST NOT return a deleted record after the delete is processed.
- **FR-V11:** Two instances that start at the same time MUST NOT corrupt or
  duplicate the collection.

**Keyword index — `cloudflare-d1`**

- **FR-K1:** `KeywordIndexConfig.provider` MUST accept `cloudflare-d1`, with
  `d1_database` required.
- **FR-K2:** The provider MUST keep the index in an FTS5 table and rank with BM25.
- **FR-K3:** Table names MUST include the tool name, so that several tools can
  share one database.
- **FR-K4:** At start, the provider MUST open an index that already exists and
  MUST NOT rebuild or clear it. It MUST apply only the changes that the source
  fingerprints show.
- **FR-K5:** The provider MUST support adding, replacing and deleting single documents.
- **FR-K6:** A second instance that starts while the first one serves MUST NOT
  rewrite or empty the index.
- **FR-K7:** Calls to D1 MUST NOT block the event loop.
- **FR-K8:** Writes MUST be split into batches that respect the D1 statement,
  parameter and row limits in §2.
- **FR-K9:** User query text MUST be passed as a bound parameter and escaped
  for FTS5 syntax. It MUST NOT be put into SQL text.
- **FR-K10:** When the keyword index cannot be reached, the tool MUST fail its
  readiness check or report a degraded state that the caller can see. It MUST
  NOT return semantic-only results as if they were hybrid results.
- **FR-K11:** When a write fails because the database is full, the error MUST
  name the database and the 10 GB limit.
- **FR-K12:** Hybrid search results MUST match the result contract of the
  existing keyword providers.

**Embeddings — `cloudflare_workers_ai`**

- **FR-E1:** `ProviderEnum` MUST gain `cloudflare_workers_ai`.
- **FR-E2:** The value MUST be accepted where an embedding provider is
  configured. Where a chat model is configured (agent `model`, evaluation
  models), it MUST fail when the config loads, with a message that says it is
  embedding-only. The check MUST be at those positions, not in the shared
  `LLMProvider` model.
- **FR-E3:** When `embedding_provider` is set, it MUST be used for every chat
  backend, not only for Anthropic agents. The spec for this rule is: tool
  `embedding_model`, then the `embedding_provider` default, then today's
  default for the chat provider.
- **FR-E4:** The provider and model MUST be part of the ingestion fingerprint,
  so that a change of either causes re-ingestion.
- **FR-E5:** `BackendSelector` routing MUST NOT change.
- **FR-E6:** The vector dimension MUST come from `embedding_dimensions` when
  it is set. When it is not set, the provider MUST find it with one embedding
  request before any store is created or changed. The provider MUST check the
  dimension of every response against it.
- **FR-E7:** The provider MUST batch requests within the limits of the model.
- **FR-E8:** The provider MUST support the bge-small, bge-base and bge-large
  English models in §2. Other models are supported only after a live test.

**Shared**

- **FR-S1:** The providers MUST use the existing HTTP client. They MUST NOT add
  an install extra or the Cloudflare SDK.
- **FR-S2:** The published schema MUST be regenerated (`make schema`).
- **FR-S3:** The [vector stores guide](../../guides/vector-stores.md) MUST
  document the three providers, the limits in §2, the two transports, and the
  local-testing setup.

## 6. Local testing

A developer runs `holodeck test` or `holodeck serve` on a local machine with
`CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` set. The providers use REST
against the same indexes and database that the deployment uses (FR-P3).

- The local run and the deployment share data. A local ingestion changes what
  the deployed agent finds. To keep them apart, use a different `index_name`
  and `d1_database` in a local override.
- The REST limit of 1,200 requests in five minutes is shared with every other
  use of the account by that user. A large ingestion can reach it.

## 7. Graph store on Cloudflare

- **FR-G1:** The deployment guide MUST state that `temporal_memory` on
  Cloudflare uses an external managed graph database, with the connection URI
  and credentials passed as Worker secrets (045 D6).
- **FR-G2:** The guide MUST state that Neo4j is the only tested driver. It MUST
  list FalkorDB as not verified, and say that fact dating and closure run only
  on Neo4j.
- **FR-G3:** The image for an agent with a `temporal_memory` tool MUST include
  the packages for its graph driver. Today the extras rule skips this tool type.
- **FR-G4:** `holodeck deploy run` to the `cloudflare` target MUST fail when a
  `temporal_memory` tool uses `driver: kuzu`. The message MUST say that the
  container disk is ephemeral and name the other drivers.
- **FR-G5:** `holodeck deploy run` to the `cloudflare` target MUST warn when a
  vector store or keyword index is configured with a local path or `in-memory`.

A native graph backend on D1 and Vectorize is a candidate for 043 Phase 2. Its
port already exists (`holodeck.lib.memory.ports.MemoryBackend`).

## 8. Out of scope

- Use of these providers by an agent hosted outside Cloudflare.
- The chat thread store.
- A native graph or temporal memory backend.
- Workers AI chat models.
- Durable Object storage, KV and R2.
- Cloudflare AI Search (the managed retrieval pipeline).
- Records larger than the Vectorize metadata limit.
- Migration of data from another vector provider.

## 9. Success criteria

- **SC-1:** An agent on the Cloudflare target, with `cloudflare_workers_ai`
  embeddings, a `cloudflare-vectorize` store and a `cloudflare-d1` keyword
  index, ingests a document set and answers a retrieval question. Its
  container holds no Cloudflare API token.
- **SC-2:** The same agent config passes `holodeck test` on a developer machine
  over REST, and finds the data that the deployment ingested.
- **SC-3:** A restart of the serving container causes no re-ingestion and no
  rebuild of the keyword index, and search results are the same after it.
- **SC-4:** With `instances: 2`, both instances start at the same time and the
  collection and keyword index are correct afterwards.
- **SC-5:** `text-embedding-3-large` at its default 3072 dimensions fails
  before any store is created (FR-V3).
- **SC-6:** A record of more than 10 KiB fails ingestion with the FR-V6 message.
- **SC-7:** The provider contract tests that every retrieval adapter passes
  also pass for `cloudflare-vectorize` on both transports. They include a long
  Unicode source path, two paths with the same first 64 bytes, a long logical
  ID, and a schema with more than 10 filterable fields.
- **SC-8:** Manifest protection is active for this provider: a changed source
  is re-ingested and an unchanged source is not.

## 10. Verification

Unit tests use recorded responses for both transports. They cover validation,
request shape, retry, batch splitting, ID mapping and the readiness wait.

Acceptance needs a live run against a Cloudflare account, as spec 042 requires
for hosted providers: `holodeck test run` for a sample over REST, then a
deployment and the serve loop in [AGENTS.md](../../../AGENTS.md) over
bindings. The user must authorize each live run.

## 11. Open questions

1. Outbound interception is documented for KV, R2, D1 and Durable Objects
   "and others". Confirm with a spike that a Worker entrypoint reached this
   way can serve Vectorize and Workers AI calls.
2. The Workers AI request and response shape for embeddings. The bge-m3 guide
   and its schemas disagree, so bge-m3 is not in FR-E8. Cloudflare also has an
   OpenAI-compatible embeddings endpoint; the plan should evaluate it for the
   REST transport.
3. Latency of a hybrid search over each transport. No measurement exists.
4. How much text fits in practice once the other metadata fields are counted
   (D6), measured on the sample document sets.
5. Whether `embedding_dimensions` exists on the tool models today or is a new field.
