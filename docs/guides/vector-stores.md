# Vector Stores Guide

This guide explains how to set up and configure vector stores for semantic search (RAG) in HoloDeck agents.

## Quick start

Run a local Qdrant, point one `vectorstore` tool at a folder of documents, and chat with your agent. HoloDeck ingests the source on first run and answers from it.

```bash
# 1. Start a local Qdrant
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

```yaml
# agent.yaml
name: knowledge-agent
model:
  provider: openai          # also works on provider: anthropic
  name: gpt-4o
instructions:
  inline: Search the knowledge base before answering.
tools:
  - name: search-kb
    type: vectorstore
    description: Search the knowledge base
    source: knowledge_base/  # folder of docs to ingest
    database:
      provider: qdrant
      connection_string: http://localhost:6333
```

```bash
holodeck chat   # ingests knowledge_base/ on first run, then answers from it
```

Expected: the agent calls `search-kb`, retrieves matching chunks, and grounds its reply in your documents.

## How it works

A `vectorstore` tool embeds your `source` documents into chunks and stores them in a vector database, then retrieves the most semantically similar chunks at query time to ground the agent's answer (RAG). Embeddings work across providers, and `vectorstore` (and `hierarchical_document`) RAG tools run on **both** the Claude and OpenAI backends. The vector store connectors are internally built on the Semantic Kernel connector library — an implementation detail that gives you a uniform API across providers; on Qdrant the connector layer also supports native hybrid (dense + sparse) search. See the [OpenAI backend](openai-backend.md) and [Claude backend](claude-backend.md) guides for backend specifics, and the [Tools Guide](tools.md) for the full tool schema.

---

## Installing vector store providers

HoloDeck uses optional dependencies for vector database providers. Install only what you need:

```bash
uv add holodeck-ai[qdrant]        # Qdrant
uv add holodeck-ai[postgres]      # PostgreSQL with pgvector
uv add holodeck-ai[chromadb]      # ChromaDB
uv add holodeck-ai[pinecone]      # Pinecone
uv add holodeck-ai[weaviate]      # Weaviate
uv add holodeck-ai[azure-ai-search]  # Azure AI Search
uv add holodeck-ai[azure-cosmos]  # Azure Cosmos DB NoSQL
uv add holodeck-ai[vectorstores]  # all of the above
```

The same extras work with `pip install holodeck-ai[<extra>]`.

## Supported providers

| Provider    | Best for                               | Setup complexity |
| ----------- | -------------------------------------- | ---------------- |
| `qdrant`    | High-performance production, hybrid search | Medium       |
| `chromadb`  | Lightweight development, Python-native | Low              |
| `postgres`  | Existing PostgreSQL infrastructure     | Medium           |
| `pinecone`  | Serverless, managed cloud              | Low              |
| `weaviate`  | Self-hosted or cloud, native hybrid search | Medium       |
| `azure-ai-search` | Azure-hosted search, native hybrid | Medium         |
| `azure-cosmos-nosql` | Existing Azure Cosmos DB accounts | Medium        |
| `in-memory` | Testing and prototyping                | None             |

See the [Tools Guide](tools.md) for the complete list.

### Removed vector store providers

HoloDeck 0.8.0 removed three providers. An `agent.yaml` or `config.yaml` that
names one fails at config load with the migration advice below.

| Removed              | Use instead                                                  |
| -------------------- | ------------------------------------------------------------ |
| `faiss`              | `chromadb` (embedded, local files) or `in-memory` for local development |
| `sql-server`         | `postgres` (pgvector) for a SQL database                     |
| `azure-cosmos-mongo` | `azure-cosmos-nosql`                                         |

## Configuring vector stores

You can configure a store inline (as in the Quick start) or define it once in global config and reference it by name.

### Global configuration (config.yaml)

Define reusable vector store connections, then reference them from any agent:

```yaml
# config.yaml (project root or ~/.holodeck/config.yaml)
vectorstores:
  my-qdrant-store:
    provider: qdrant
    url: http://localhost:6333
  my-chroma-store:
    provider: chromadb
    connection_string: http://localhost:8000
  my-postgres-store:
    provider: postgres
    connection_string: ${DATABASE_URL}
```

```yaml
# agent.yaml
tools:
  - name: search-kb
    type: vectorstore
    description: Search the knowledge base
    source: knowledge_base/
    database: my-qdrant-store   # reference to config.yaml
```

### Tuning ingestion and embeddings

Set embedding model and chunking on the tool. The embedding model is provider-neutral — pick one your configured model provider exposes:

```yaml
tools:
  - name: search-kb
    type: vectorstore
    description: Search the knowledge base
    source: knowledge_base/
    database: my-qdrant-store
    embedding_model: text-embedding-3-small
    chunk_size: 512
    chunk_overlap: 50
```

### Environment variables

Store sensitive connection details in environment variables and reference them with `${VAR}`:

```bash
# .env file (DO NOT commit to version control)
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
QDRANT_API_KEY=your-api-key
```

> **Security**: Always use `${VAR_NAME}` for passwords, API keys, and sensitive connection details. Never commit plaintext credentials to version control.

---

## Running a vector database

All providers below run locally in a container. You need a container runtime — [Docker](https://docs.docker.com/get-docker/) (recommended) or Podman. Podman commands are Docker-compatible; replace `docker` with `podman` in the examples.

### Qdrant

[Qdrant](https://qdrant.tech/) is a high-performance vector database designed for production workloads. It supports HTTP and gRPC, and native hybrid (dense + sparse) search.

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant
```

Exposes port **6333** (HTTP API) and **6334** (gRPC). Verify:

```bash
curl http://localhost:6333/healthz
```

Docker Compose:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: holodeck-qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant-data:/qdrant/storage
    restart: unless-stopped

volumes:
  qdrant-data:
```

For production, enable API key auth with `QDRANT__SERVICE__API_KEY=${QDRANT_API_KEY}`, and pin a version (e.g. `qdrant/qdrant:v1.9.0`) instead of `latest`.

**Record IDs.** A UUID record ID is the point ID. Any other ID becomes a stable UUIDv5 and the original is kept in the `holodeck_id` payload field. Two different IDs can never share a point: a write that would give two IDs of one call the same point ID (`invoice-7` and the text of its own UUID), or overwrite a point that holds another ID, fails with a `SchemaError`. The stored layout does not change.

### ChromaDB

[ChromaDB](https://www.trychroma.com/) is a lightweight, Python-native embedding database that's ideal for development.

```bash
docker run -d \
  --name chromadb \
  -p 8000:8000 \
  -v ./chroma-data:/chroma/chroma \
  -e IS_PERSISTENT=TRUE \
  -e ANONYMIZED_TELEMETRY=FALSE \
  chromadb/chroma:latest
```

Exposes port **8000** (HTTP API). Verify:

```bash
curl http://localhost:8000/api/v2/heartbeat
```

| Variable | Description | Default |
|----------|-------------|---------|
| `IS_PERSISTENT` | Enable data persistence | `FALSE` |
| `PERSIST_DIRECTORY` | Path for persistent storage | `/chroma/chroma` |
| `ANONYMIZED_TELEMETRY` | Send anonymous usage data | `TRUE` |

> **Security Advisory**: `chromadb` is currently affected by
> [CVE-2026-45829 / GHSA-f4j7-r4q5-qw2c](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c),
> a pre-authentication code injection issue in ChromaDB server versions 1.0.0
> and later when the server accepts a malicious model repository with
> `trust_remote_code=true`. As of this documentation update, upstream has not
> published a fixed `chromadb` release. Until a fix is available, run ChromaDB
> only on trusted networks, do not expose it directly to the public internet,
> and treat model repository inputs as trusted administrative data.

### PostgreSQL with pgvector

[PostgreSQL with pgvector](https://github.com/pgvector/pgvector) provides production-grade vector storage with full SQL capabilities — ideal if you already run PostgreSQL.

```bash
docker run -d \
  --name postgres \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=${POSTGRES_PASSWORD} \
  -v pgdata:/var/lib/postgresql/data \
  pgvector/pgvector:pg17
```

Exposes port **5432**. Verify:

```bash
docker exec -it postgres psql -U postgres -c "SELECT 1;"
```

| Variable | Description | Default |
|----------|-------------|---------|
| `POSTGRES_PASSWORD` | Database password (required) | - |
| `POSTGRES_USER` | Database user | `postgres` |
| `POSTGRES_DB` | Default database name | `postgres` |

### Pinecone

[Pinecone](https://www.pinecone.io/) is a fully managed, serverless vector database. HoloDeck uses the `pinecone` client 10.x (`holodeck-ai[pinecone]`) over asyncio and creates one serverless, cosine index per collection:

```yaml
database:
  provider: pinecone
  api_key: ${PINECONE_API_KEY}
  namespace: my-namespace  # optional
  cloud: aws               # optional, for a new index
  region: us-east-1        # optional, for a new index
```

| Setting | Meaning |
|---------|---------|
| `api_key`, `connection_string` | API key, or `pinecone://key[@namespace]`. `PINECONE_API_KEY` is the fallback. |
| `namespace` | Namespace of the records. Default: the default namespace. |
| `cloud`, `region` | Where a new serverless index runs. Default `aws`, `us-east-1`. |
| `index_name` | Index name instead of the collection name. |
| `host`, `index_host` | Control plane and data plane URLs, for a gateway or private endpoint. |

Pinecone index names are lowercase letters, digits and hyphens, at most 45 characters. A name that is already valid is used unchanged. Otherwise HoloDeck lower-cases it, turns other characters into hyphens and adds a short hash of the original name, so `documents_dim1536` becomes `documents-dim1536-<hash>` and `a_b` and `a-b` never share an index. Set `index_name` or `collection_name` to a valid name to choose the index name yourself.

Record IDs must be ASCII (at most 512 characters). A source file with a non-ASCII path is rejected at ingestion with a clear error. Pinecone reads are eventually consistent, so a search right after ingestion can briefly miss new chunks.

A HoloDeck collection on Pinecone is the pair (index, namespace):

- With the default namespace the collection is the whole index. Deleting the collection (`holodeck vectorstore prune`) deletes the index.
- With a named `namespace` the collection is that namespace only. Deleting it deletes every record in the namespace and keeps the index, which other namespaces can share. Record counts and "exists" checks read the namespace. Pinecone has no empty namespaces, so a named namespace with no records reads as absent.

The pipeline manifest (see [Pipeline fingerprint and migration](#pipeline-fingerprint-and-migration)) is one more small serverless index, `holodeck-manifest-<hash>`, per project and API key. Set `manifest: false` in the database settings if you cannot afford the extra index; fingerprint protection and `holodeck vectorstore migrate` are then off.

> **Note**: Pinecone Local reached end of life on 2026-10-07. The last emulator image (`ghcr.io/pinecone-io/pinecone-local:latest`, 1.0.0-rc0) speaks the pre-10 control plane, so the 10.x client cannot create indexes on it. HoloDeck's own tests use it with a small test-only control plane. For local development use the cloud service, or the in-memory, Qdrant or ChromaDB providers.

### Weaviate

[Weaviate](https://weaviate.io/) is an open-source vector database with a managed cloud. HoloDeck uses the `weaviate-client` 4.x asynchronous client (`holodeck-ai[weaviate]`) and supports every retrieval operation, including Weaviate's own hybrid search (vector plus BM25), so a `hybrid` tool does not build an in-process keyword index.

```yaml
# A server, for example Docker on this machine
database:
  provider: weaviate
  local_host: localhost
  local_port: 8080        # optional, default 8080
  local_grpc_port: 50051  # optional, default 50051

# Weaviate Cloud
database:
  provider: weaviate
  url: https://my-cluster.weaviate.cloud
  api_key: ${WEAVIATE_API_KEY}
### Azure AI Search

[Azure AI Search](https://learn.microsoft.com/azure/search/) is Azure's managed search service. HoloDeck uses the `azure-search-documents` 12.0 asyncio clients (`holodeck-ai[azure-ai-search]`, which also installs `azure-identity`) and keeps one index per collection:

```yaml
database:
  provider: azure-ai-search
  connection_string: https://my-service.search.windows.net
  api_key: ${AZURE_AI_SEARCH_API_KEY}   # omit to use Microsoft Entra ID
### Azure Cosmos DB NoSQL

HoloDeck uses the `azure-cosmos` async client (`holodeck-ai[azure-cosmos]`) and stores each collection in one container of a Cosmos DB for NoSQL account. The provider name is `azure-cosmos-nosql`:

```yaml
database:
  provider: azure-cosmos-nosql
  url: https://my-account.documents.azure.com:443/
  key: ${COSMOS_KEY}
  database_name: holodeck
  create_database: true   # optional
```

| Setting | Meaning |
|---------|---------|
| `url`, `api_key` | Weaviate Cloud cluster (`https://` URL or host name) and its API key. The key is required. |
| `local_host`, `local_port`, `local_grpc_port` | A server over plain HTTP and gRPC. Set `api_key` too when the server uses API key authentication. |
| `connection_string` | `https://<cluster>` for Weaviate Cloud, or `http://host[:port]` for a server. Explicit settings win. |
| `collection_name` | Collection name instead of the default. |

`WEAVIATE_URL`, `WEAVIATE_LOCAL_HOST`, `WEAVIATE_LOCAL_PORT`, `WEAVIATE_LOCAL_GRPC_PORT` and `WEAVIATE_API_KEY` are used when the settings name no server. Set exactly one of a cloud `url` and `local_host`; unknown settings are an error. The embedded Weaviate mode that Semantic Kernel offered (`use_embed`) is not supported; run a server.

To start a local server:

```bash
docker run -d -p 8080:8080 -p 50051:50051 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  -e DEFAULT_VECTORIZER_MODULE=none \
  cr.weaviate.io/semitechnologies/weaviate:1.40.4
```

**Stored layout.** One Weaviate collection per HoloDeck collection. Weaviate names start with an upper-case letter and use only `A-Z a-z 0-9 _`. A name that already fits is used unchanged. Any other name is cleaned (the first letter is upper-cased, every other character becomes `_`) and gets `_` plus a short hash of the original name, so `documents_dim1536` becomes `Documents_dim1536_<hash>` and `docs`, `Docs` and `a-b`/`a_b` never share a collection. Set `collection_name` to a name that fits to choose the collection name yourself. It holds one self-provided named vector `embedding` (cosine, flat index), one property per data field, and a `holodeck_id` text property. The record ID is the object UUID: a UUID is stored as it is, any other ID (chunk IDs are free text) becomes a stable UUIDv5 of the collection and the ID, and the original goes into `holodeck_id`. Two different IDs can never share an object: a write that would give two IDs of one call the same UUID (`invoice-7` and the text of its own UUID), or overwrite an object that holds another ID, fails with a `SchemaError`. Fields marked `indexed` get a filter index, and string fields use field tokenization, so a filter such as `source_path` matches the whole value exactly and case-sensitively; other fields have no index, and a filter on one is an error. This is the layout Semantic Kernel's Weaviate connector writes, apart from the property indexes, `holodeck_id` and the field tokenization.

Limits to know:

- Semantic Kernel's Weaviate path never worked against a server (its cosine setting was rejected and it could not store free-text IDs), so there are no earlier HoloDeck Weaviate collections to migrate.
- Weaviate returns at most `QUERY_MAXIMUM_RESULTS` (10,000 by default) results for one query. HoloDeck pages with the filter on the server while it stays under that limit. For a source with more chunks it reads the whole collection by cursor and applies the filter itself, so `find`, `delete_where` and replacing a large source still work, more slowly.
- An existing collection must have the `embedding` named vector with cosine distance, matching property types and field tokenization on the string properties that tools filter on, or HoloDeck raises an `IncompatibleCollectionError` and leaves it alone. A collection that Semantic Kernel's connector creates uses word tokenization, which cannot match exactly, so it is refused.
| `connection_string`, `endpoint`, `search_endpoint` | Service URL, `https://<service>.search.windows.net`. `AZURE_AI_SEARCH_ENDPOINT` is the fallback. |
| `api_key` | Admin key. `AZURE_AI_SEARCH_API_KEY` is the fallback. With no key, `DefaultAzureCredential` (Microsoft Entra ID) signs in. |
| `index_name` | Index name instead of the collection name. |

Unknown settings are an error. The index is created on first use with a key field `id`, one field per data field, and a vector field `embedding` with an HNSW cosine profile. Only fields that the tool filters on are `filterable`, and only the text field the keyword leg searches is `searchable`.

**Search.** All of vector, keyword and hybrid search run in the service. Hybrid mode is one request that carries the text query and the vector query, and the service fuses the two lists with Reciprocal Rank Fusion. Keyword text is reduced to words first, so characters such as `-` or `"` in a question are not read as query operators. Vector scores are converted from Azure's `1 / (1 + distance)` value to cosine similarity.

**Record IDs and names.** Azure keys allow only letters, digits, `_`, `-` and `=`, so an ID such as `docs/policy.md_chunk_0` is stored as `b64-` plus its URL-safe Base64 text. An ID that is already a valid key is stored unchanged. An ID that starts with `b64-` is always encoded too, so a key that starts with `b64-` is always an encoding and the mapping is one to one. Index names are lowercase letters, digits, `-` and `_`. A name that already fits is used unchanged (`documents_dim1536`). Any other name is lower-cased, cleaned and gets `-` plus a short hash of the original name (`Product Catalog.v2` becomes `product-catalog-v2-<hash>`), so `a_b` and `a-b` never share an index.

**Existing indexes.** An index that an earlier Semantic Kernel setup created has a key that cannot be sorted. Azure cannot change that on an existing index, so HoloDeck stops with an `IncompatibleCollectionError`. Delete the index or use another `index_name`. An index whose vector field compares vectors with another metric than cosine is refused in the same way.

Azure indexes documents asynchronously. After a delete, HoloDeck checks for up to 10 seconds that a search shows none of the deleted documents. After an upload it checks that a search shows every written document and that the last one has the written values and embedding. Then it continues, with a warning if the wait ran out. The document count is approximate while indexing runs.

> **Note**: Azure AI Search has no local emulator, and this adapter has been verified only against an in-memory fake of the client and against the SDK's request shapes with a stubbed transport. It has not been run against the Azure service. Ordering of string keys in range filters, indexing delay, and score values are therefore unobserved. Treat the first run on a real service as a check, and report problems.
| `url` (or `endpoint`), `key` | Account URL and key. The environment variables `AZURE_COSMOS_DB_NO_SQL_URL` and `AZURE_COSMOS_DB_NO_SQL_KEY` are the fallbacks. Token credentials cannot be set from YAML. |
| `connection_string` | The portal form `AccountEndpoint=...;AccountKey=...;` instead of `url` and `key`. |
| `database_name` | Database that holds the container. Fallback: `AZURE_COSMOS_DB_NO_SQL_DATABASE_NAME`. |
| `create_database` | Create the database when it is missing. Default `false`: a missing database is an error. |
| `vector_index` | `flat`, `quantizedFlat` or `diskANN`, for a container HoloDeck creates. Default: `flat` up to 505 dimensions, `quantizedFlat` above (4096 at most). |
| `full_text_search` | `true` creates a full-text policy and index and uses the service's hybrid search (`ORDER BY RANK RRF`). Default `false`. See below. |
| `partition_key` | Only `id` or `/id`. |

The account needs the vector search capability (`EnableNoSQLVectorSearch`). `full_text_search` also needs the full-text search capability. Unknown settings are an error.

**Layout.** The container has the collection's name and the partition key `/id`. A document is `{id, <fields>, embedding}`. The vector embedding policy is `float32`, `cosine`, with the collection's dimensions. Fields that are neither `indexed` nor full-text are excluded from the index, and so is the vector. This is the layout Semantic Kernel created, so containers created before the move keep working. HoloDeck never changes an existing container. It rejects one with other dimensions or another partition key (`IncompatibleCollectionError`).

**Vector index.** Semantic Kernel always used `flat`, which Cosmos DB limits to 505 dimensions, so it could not create a container for the default 1536-dimension embeddings. HoloDeck chooses `quantizedFlat` there. Cosmos DB cannot change a vector index in place, so set `vector_index` before the first run.

**Record IDs.** Cosmos DB rejects item IDs with `/`, `\`, `?` or `#`, and HoloDeck chunk IDs often contain a file path. An ID that Cosmos DB accepts is stored unchanged. Any other ID is stored as `~hd~` plus the URL-safe base64 of the ID, and the original is kept in the `holodeck_id` field. HoloDeck decodes the ID on every read.

**Hybrid search.** With `full_text_search: false` (the default) HoloDeck fuses its own BM25 results with the vector results, as it does for Postgres. With `true`, the keyword leg runs in Cosmos DB. This needs a full-text index. Cosmos DB cannot add one to an existing container, and Semantic Kernel never created one. For a container from the old path, keep the default or use a new container (a new `collection_name`). HoloDeck reports a container without the index with `IncompatibleCollectionError`. The service does not return the fused score, so hybrid results carry a rank-based score: only their order means something.

**What was verified where.** The adapter passes the shared contract suite on the Linux Cosmos DB emulator (`mcr.microsoft.com/cosmosdb/linux/azure-cosmos-emulator:vnext-preview`, `--protocol http`) for every operation except hybrid search. It passes the whole suite, hybrid search included, on a test fake of the container API. The emulator accepts only the `diskANN` index, does not check the account key, does not report the vector embedding policy back, and does not support full-text search (`FullTextScore`, `RRF`): it drops the full-text policy and index. So `full_text_search: true`, the `ORDER BY RANK RRF` query, the `flat` and `quantizedFlat` indexes, key authentication and throttling have **not run against a Cosmos DB service**. The hybrid query text follows the service documentation and is checked by unit tests only.

---

## Connection string formats

### PostgreSQL

| Format | Example |
|--------|---------|
| Basic | `postgresql://localhost:5432/mydb` |
| With credentials | `postgresql://user:password@localhost:5432/mydb` |
| With SSL | `postgresql://user:password@host:5432/mydb?sslmode=require` |

### ChromaDB

| Format | Example |
|--------|---------|
| HTTP | `http://localhost:8000` |
| HTTPS | `https://chroma.example.com` |
| Local persistent | use `persist_directory: ./data/chromadb` instead |

### Qdrant

| Format | Example |
|--------|---------|
| HTTP (local) | `http://localhost:6333` |
| HTTPS (remote) | `https://qdrant.example.com:6333` |
| gRPC (high-performance) | `qdrant+grpc://localhost:6334` |
| In-memory | `:memory:` |
| Local path | `/path/to/qdrant/data` |

For remote Qdrant, add `api_key: ${QDRANT_API_KEY}`.

### Azure AI Search

| Format | Example |
|--------|---------|
| Endpoint | `https://my-service.search.windows.net` |

The URL must use `https`. Add `api_key`, or leave it out to use Microsoft Entra ID.

### Pinecone

Pinecone uses API key authentication, or the `pinecone://pc-abc123@my-namespace` connection-string form (the key keeps its case; a namespace after `@` is lower-cased, as in earlier releases, so existing data stays visible; a `/namespace` path and the `namespace` setting keep their case).

### Weaviate

| Format | Example |
|--------|---------|
| Server (plain HTTP) | `http://localhost:8080` |
| Weaviate Cloud | `https://my-cluster.weaviate.cloud` |

For Weaviate Cloud, add `api_key: ${WEAVIATE_API_KEY}`.
### Azure Cosmos DB NoSQL

| Format | Example |
|--------|---------|
| URL and key | `url: https://my-account.documents.azure.com:443/` and `key: ...` |
| Portal form | `AccountEndpoint=https://my-account.documents.azure.com:443/;AccountKey=...;` |
| Local emulator | `url: http://127.0.0.1:8081` (the emulator's published key) |

---

## Complete example

```
my-agent-project/
├── config.yaml
├── agent.yaml
├── .env
├── knowledge_base/
│   ├── faq.json
│   └── docs.md
└── docker-compose.yml
```

```yaml
# config.yaml
providers:
  openai:
    provider: openai
    name: gpt-4o
    api_key: ${OPENAI_API_KEY}

vectorstores:
  knowledge-store:
    provider: qdrant
    url: http://localhost:6333
```

```yaml
# agent.yaml
name: support-agent
description: Customer support agent with knowledge base search

model:
  provider: openai
  name: gpt-4o
  temperature: 0.7

instructions:
  inline: |
    You are a customer support specialist.
    Always search the knowledge base before answering questions.

tools:
  - name: search-kb
    description: Search knowledge base for answers to customer questions
    type: vectorstore
    source: knowledge_base/
    database: knowledge-store
    embedding_model: text-embedding-3-small
    chunk_size: 512
    chunk_overlap: 50

test_cases:
  - name: "FAQ lookup"
    input: "How do I reset my password?"
    expected_tools: [search-kb]
```

```yaml
# docker-compose.yml
services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: holodeck-qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant-data:/qdrant/storage

volumes:
  qdrant-data:
```

Run it:

```bash
docker compose up -d           # 1. Start Qdrant
curl http://localhost:6333/healthz   # 2. Verify it's up
holodeck test                  # 3. Run the agent (uses agent.yaml by default)
```

---

## Pipeline fingerprint and migration

Records in a collection depend on more than its schema. HoloDeck records the
**pipeline** that wrote a collection and refuses to mix records from two
different pipelines in one collection.

### What is fingerprinted

The fingerprint is a SHA-256 hash of these inputs:

| Input | `vectorstore` tool | `hierarchical_document` tool |
| --- | --- | --- |
| Embedding | provider, model, dimensions | provider, model, dimensions |
| Parser | `markitdown/1` (`structured-rows/1` for CSV/JSON rows) | `markitdown/1` |
| Chunker | `plain-tiktoken-cl100k/1`, `chunk_size`, `chunk_overlap` (rows: vector fields, field separator) | `structured-chunker/1`, `max_chunk_tokens`, `document_domain`, `max_subsection_depth` |
| Record layout | field names, types and index flags, key and vector field, dimensions, distance | same |
| Retrieval | none | contextual embeddings on or off, `searchable-text/1` recipe; when on, the context generator (`litellm-context/1` or `claude-sdk-context/1` recipe, its model, `context_max_tokens`, document truncation), `context_model` provider and name |
| Structured rows | vector fields, `field_separator`, `id_field`, `delimiter`, `meta_fields` | - |

Search settings such as `top_k`, `min_similarity_score` and the search mode are
not part of the fingerprint: they do not change stored records.

### The manifest collection

HoloDeck does not add fields to your collections. It keeps its state in one
extra collection per store connection, `holodeck_manifest`, which it creates
through the same provider. The manifest holds:

- one entry per collection: the fingerprint, the physical collection that
  currently serves the tool, and the collection kept for rollback;
- one entry per ingested source file: the expected chunk count and a
  completion flag. A source is marked incomplete before its chunks are written
  and complete after the last write. A source whose ingestion stopped midway
  (a crash, a failed write) is ingested again on the next start.

A collection's manifest entry is keyed by its **logical name**: the name the
provider is asked for, which is `index_name` when set, else `collection_name`,
else the default (`documents_dim<N>`, the tool name for structured data,
`hierarchical_docs_<tool>_dim<N>`). Two tools with the same logical name share
one collection and one entry; two tools with different names have their own.

A provider that cannot hold the manifest logs one warning and works as
before, without fingerprint protection. Settings that address one collection
(`collection_name`, `index_name`, Pinecone `index_host`) do not apply to the
manifest: it is stored in its own `holodeck_manifest` collection or index.

If the manifest collection does not exist and cannot be created (a quota, a
missing permission, a provider error), HoloDeck logs one warning that names
the cause and says fingerprint protection is off, and the tool works as
before. If the manifest exists but cannot be read or written, an established
pointer or fingerprint may be in it: searches still work and every write is
refused until the store answers again.

Set `manifest: false` in a tool's `database` settings to turn the manifest off
on purpose (no warning). Fingerprint protection, completion markers and the
`holodeck vectorstore` migration commands are then off for that store.

```yaml
database:
  provider: pinecone
  api_key: ${PINECONE_API_KEY}
  manifest: false
```

#### Concurrent starts

Two processes that start at the same time can both find no manifest entry.
Vector stores have no portable compare-and-set, so HoloDeck narrows the race
this way:

- every write of a collection entry carries a new random `revision`;
- after writing the entry, a tool reads it back and writes data only if the
  stored revision is its own. Otherwise the other process's entry decides:
  the same fingerprint allows writes, a different one refuses them;
- immediately before the first write of each source file, the tool reads the
  entry again (one read per file) and refuses to write if the fingerprint or
  the physical collection changed. If only the revision changed, the tool
  continues with the new entry once.

This narrows the race but cannot remove it on a store without transactions:
two writers can still interleave between the last check and a write. Do not
start agents with different pipelines against one collection at the same
time. The manifest update on PostgreSQL uses the same read-back rule; it does
not take a transaction or advisory lock.

### What happens at start-up

| Situation | Result |
| --- | --- |
| New or empty collection | The current fingerprint is recorded. |
| Structured data (CSV, JSON, JSONL) | The source file gets a completion marker with its row count. An unchanged file with a complete marker is not embedded again; a changed file is embedded again. |
| Collection created before this release (no manifest entry) | It stays readable. Unchanged files are not embedded again. A changed file is replaced as a whole. HoloDeck records the current pipeline marked `legacy` and logs that `holodeck vectorstore migrate` rebuilds the collection. |
| Same fingerprint | Normal incremental ingestion. |
| Different fingerprint | Searches read the existing records. Any write (a changed file, `--force-ingest`) fails with an error that lists the inputs that differ and names the commands below. |

HoloDeck never deletes or recreates a collection on its own.

### Commands

```bash
# Read-only dry run: collection, record count, compatibility, the inputs that
# differ, and the files, chunks and estimated tokens a migration embeds.
holodeck vectorstore inspect agent.yaml [--tool NAME] [--json]

# Build a new collection <name>__<fingerprint prefix> with the current
# pipeline, verify it, then switch to it. --tool selects the collection; every
# tool of the agent that uses it is rebuilt. The old collection is kept.
holodeck vectorstore migrate agent.yaml --tool NAME [--yes] [--restart] [--drop-unlisted-sources]

# Switch back to the kept collection.
holodeck vectorstore rollback agent.yaml --tool NAME [--yes]

# Delete the kept collection. The only command that deletes a collection.
holodeck vectorstore prune agent.yaml --tool NAME --yes
```

- `inspect` is strictly read-only: it downloads nothing, creates no file,
  directory, collection or manifest entry, and calls no model. It counts
  chunks by parsing and splitting local sources. A remote source (`s3://`,
  `https://`, ...) is listed as "remote source: not estimated". The token
  figure is an estimate: tiktoken `cl100k_base` tokens of the chunk text,
  without the text that contextual embeddings add.
- `migrate` works on a collection. Several tools can share one (plain
  `vectorstore` tools share `documents_dim<N>` unless they set
  `collection_name`), so every tool of the agent bound to the selected
  tool's collection is rebuilt together, and the plan lists them. They must
  have the same pipeline; if not, `migrate` refuses and shows the inputs that
  differ.
- `migrate` lists every source file before it embeds anything. A source that
  fails to parse, produces no chunks or does not finish fails the migration;
  nothing is skipped silently.
- Before it switches, `migrate` checks that the old collection holds no
  source that the migration did not rebuild (for example, files written by a
  tool in another agent file). If it does, `migrate` lists them and does not
  switch. Pass `--drop-unlisted-sources` to switch without them.
- `migrate` calls the embedding model. If it stops (for example, the
  embedding service fails), run it again: files that were completed are not
  embedded again. The tools keep using the old collection until every source
  of the new one is complete with its expected chunk count, the total record
  count matches, and it answers a vector search. A migration for a different
  pipeline that is still in progress blocks a new one; add `--restart` to
  abandon it. Run `prune` before a second migration if a collection is still
  kept from the first.
- `migrate`, `rollback` and `prune` take a lock on the collection in the
  manifest while they run (a `lock` record with an owner, the command, and
  the time it was taken). A second command on the same collection reports
  who holds the lock and exits with an error. A lock that is not refreshed
  for two hours is stale and the next command takes it over; a running
  migration refreshes it every ten minutes.
- `rollback` checks that the kept collection still exists, under the lock,
  immediately before it switches.
- `prune` first removes the kept collection from the manifest (it is marked
  reserved for deletion), then deletes it. If the delete fails, run `prune`
  again to finish; `rollback` never points at a collection being deleted.
- `migrate`, `rollback` and `prune` print what they will do and ask for
  confirmation unless you pass `--yes`.

---

## Troubleshooting

### Cannot connect to the vector database

**Error:** `Connection refused` or `Cannot connect to http://localhost:<port>`

1. Verify the container is running: `docker ps | grep <name>`
2. Check logs: `docker logs <name>`
3. Test connectivity (Qdrant: `curl http://localhost:6333/healthz`; ChromaDB: `curl http://localhost:8000/api/v2/heartbeat`)
4. Ensure the port isn't blocked by a firewall

Error messages never contain the API key, password or connection string you configured: HoloDeck replaces them with `***`. An API key or header value that contains a control character (a trailing newline from a file or environment variable, for example) is rejected when the adapter is created, with a message that names the setting but not its value.

### Data not persisting

Mount a named volume for persistence — e.g. `-v qdrant-data:/qdrant/storage` (Qdrant) or `-v chroma-data:/chroma/chroma -e IS_PERSISTENT=TRUE` (ChromaDB).

### Container already exists

**Error:** `container name "<name>" is already in use`

Remove the existing container: `docker rm -f <name>`

---

## Additional resources

- [Tools Reference Guide](tools.md) - Complete vectorstore tool configuration (chunk size, embedding models, hybrid search)
- [OpenAI Backend Guide](openai-backend.md) / [Claude Backend Guide](claude-backend.md) - Backend specifics for RAG tools
- [Agent Configuration](agent-configuration.md) - Complete agent setup
- [LLM Providers Guide](llm-providers.md) - Configuring the LLM that powers your agent
- [Evaluations Guide](evaluations.md) - Testing your agent's search quality
- [Global Configuration Guide](global-config.md) - Sharing vectorstore configs across agents
- [Qdrant Documentation](https://qdrant.tech/documentation/) · [ChromaDB Documentation](https://docs.trychroma.com/) · [PostgreSQL pgvector](https://github.com/pgvector/pgvector)
