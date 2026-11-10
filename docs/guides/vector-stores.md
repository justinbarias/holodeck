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

**Stored layout.** One Weaviate collection per HoloDeck collection, with the first letter upper-cased and every character outside `A-Z a-z 0-9 _` replaced by `_` (`documents_dim1536` becomes `Documents_dim1536`). It holds one self-provided named vector `embedding` (cosine, flat index), one property per data field, and a `holodeck_id` text property. The record ID is the object UUID: a UUID is stored as it is, any other ID (chunk IDs are free text) becomes a stable UUIDv5 of the collection and the ID, and the original goes into `holodeck_id`. Fields marked `indexed` get a filter index, and string fields use field tokenization, so a filter such as `source_path` matches the whole value exactly and case-sensitively; other fields have no index, and a filter on one is an error. This is the layout Semantic Kernel's Weaviate connector writes, apart from the property indexes and `holodeck_id`.

Limits to know:

- Semantic Kernel's Weaviate path never worked against a server (its cosine setting was rejected and it could not store free-text IDs), so there are no earlier HoloDeck Weaviate collections to migrate.
- Weaviate returns at most `QUERY_MAXIMUM_RESULTS` (10,000) results for one query. Replacing the chunks of a single source file that has more chunks than that fails with an error instead of leaving stale chunks.
- An existing collection must have the `embedding` named vector with cosine distance and matching property types, or HoloDeck raises an error and leaves it alone.

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

### Pinecone

Pinecone uses API key authentication, or the `pinecone://pc-abc123@my-namespace` connection-string form (the key keeps its case; a namespace after `@` is lower-cased, as in earlier releases, so existing data stays visible; a `/namespace` path and the `namespace` setting keep their case).

### Weaviate

| Format | Example |
|--------|---------|
| Server (plain HTTP) | `http://localhost:8080` |
| Weaviate Cloud | `https://my-cluster.weaviate.cloud` |

For Weaviate Cloud, add `api_key: ${WEAVIATE_API_KEY}`.

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
