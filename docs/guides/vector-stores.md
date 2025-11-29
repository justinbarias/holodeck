# Vector Stores Guide

This guide explains how to set up and configure vector stores for semantic search in HoloDeck agents.

## Overview

Vector stores enable semantic search capabilities for your agents, allowing them to search through documents, knowledge bases, and structured data using natural language queries. HoloDeck uses vector embeddings to find semantically similar content.

### Why Use Vector Stores?

- **Semantic Search**: Find relevant information based on meaning, not just keywords
- **RAG (Retrieval-Augmented Generation)**: Ground agent responses in your data
- **Knowledge Bases**: Build searchable document repositories
- **FAQ Systems**: Match user questions to relevant answers

## Prerequisites

Before setting up a vector store, you need a container runtime:

### Docker (Recommended)

Docker is the most common container runtime. Install it from [docker.com](https://docs.docker.com/get-docker/).

**Verify installation:**

```bash
docker --version
# Docker version 24.0.0, build ...
```

### Podman (Alternative)

Podman is a daemonless container engine, useful in environments where Docker isn't available.

**Install on Linux:**

```bash
# Ubuntu/Debian
sudo apt-get install podman

# Fedora/RHEL
sudo dnf install podman
```

**Install on macOS:**

```bash
brew install podman
podman machine init
podman machine start
```

**Verify installation:**

```bash
podman --version
# podman version 4.0.0
```

> **Note**: Podman commands are compatible with Docker. Replace `docker` with `podman` in the examples below.

---

## Setting Up ChromaDB

[ChromaDB](https://www.trychroma.com/) is an open-source embedding database that's simple to set up and ideal for development. It provides a lightweight vector database with native Python support.

### Quick Start with Docker

**Run ChromaDB:**

```bash
docker run -d \
  --name chromadb \
  -p 8000:8000 \
  -v ./chroma-data:/chroma/chroma \
  -e IS_PERSISTENT=TRUE \
  -e ANONYMIZED_TELEMETRY=FALSE \
  chromadb/chroma:latest
```

This exposes:

- **Port 8000**: ChromaDB HTTP API (for HoloDeck connections)

**Verify ChromaDB is running:**

```bash
curl http://localhost:8000/api/v2/heartbeat
# {"nanosecond heartbeat":1234567890}
```

### Docker Compose (Recommended for Projects)

Create a `docker-compose.yml` file in your project root:

```yaml
version: "3.9"

services:
  chromadb:
    image: chromadb/chroma:latest
    container_name: holodeck-chromadb
    ports:
      - "8000:8000"
    volumes:
      - chroma-data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE
      - PERSIST_DIRECTORY=/chroma/chroma
      - ANONYMIZED_TELEMETRY=FALSE
    restart: unless-stopped

volumes:
  chroma-data:
```

**Start the service:**

```bash
docker compose up -d
```

**Stop the service:**

```bash
docker compose down
```

### Podman Equivalent

```bash
podman run -d \
  --name chromadb \
  -p 8000:8000 \
  -v ./chroma-data:/chroma/chroma \
  -e IS_PERSISTENT=TRUE \
  -e ANONYMIZED_TELEMETRY=FALSE \
  docker.io/chromadb/chroma:latest
```

### ChromaDB Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `IS_PERSISTENT` | Enable data persistence | `FALSE` |
| `PERSIST_DIRECTORY` | Path for persistent storage | `/chroma/chroma` |
| `ANONYMIZED_TELEMETRY` | Send anonymous usage data | `TRUE` |

> **Version Pinning**: For stability, pin to a specific version (e.g., `chromadb/chroma:0.6.3`) instead of `latest` to avoid unexpected changes during upgrades.

---

## Configuring Vector Stores in HoloDeck

### Global Configuration (config.yaml)

Define reusable vector store connections in your global configuration:

```yaml
# config.yaml (project root or ~/.holodeck/config.yaml)

vectorstores:
  # ChromaDB (lightweight, Python-native)
  my-chroma-store:
    provider: chromadb
    connection_string: http://localhost:8000

  # PostgreSQL with pgvector
  my-postgres-store:
    provider: postgres
    connection_string: ${DATABASE_URL}

  # Qdrant (high-performance)
  my-qdrant-store:
    provider: qdrant
    url: http://localhost:6333
```

### Environment Variables

Store sensitive connection details in environment variables:

```bash
# .env file (DO NOT commit to version control)
DATABASE_URL=postgresql://user:password@localhost:5432/mydb
QDRANT_API_KEY=your-api-key
```

### Agent Configuration (agent.yaml)

Reference the global vector store in your agent's tools:

```yaml
# agent.yaml
name: knowledge-agent
description: Agent with semantic search capabilities

model:
  provider: openai
  name: gpt-4o

instructions:
  inline: |
    You are a helpful assistant with access to a knowledge base.
    Use the search tool to find relevant information before answering.

tools:
  - name: search-kb
    description: Search the knowledge base for relevant information
    type: vectorstore
    source: knowledge_base/
    database: my-chroma-store  # Reference to config.yaml
```

### Inline Database Configuration

Alternatively, configure the database directly in `agent.yaml`:

```yaml
tools:
  - name: search-docs
    description: Search technical documentation
    type: vectorstore
    source: docs/
    database:
      provider: chromadb
      connection_string: http://localhost:8000
```

---

## Connection String Formats

### PostgreSQL Connection Strings

| Format | Example |
|--------|---------|
| Basic | `postgresql://localhost:5432/mydb` |
| With credentials | `postgresql://user:password@localhost:5432/mydb` |
| With SSL | `postgresql://user:password@host:5432/mydb?sslmode=require` |

### ChromaDB Connection Strings

| Format | Example |
|--------|---------|
| HTTP | `http://localhost:8000` |
| HTTPS | `https://chroma.example.com` |

### Qdrant Connection Strings

| Format | Example |
|--------|---------|
| HTTP | `http://localhost:6333` |
| HTTPS | `https://qdrant.example.com` |

> **Security**: Always use environment variables (`${VAR_NAME}`) for passwords and sensitive connection details. Never commit plaintext passwords to version control.

---

## Complete Example

### Project Structure

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

### config.yaml

```yaml
providers:
  openai:
    provider: openai
    name: gpt-4o
    api_key: ${OPENAI_API_KEY}

vectorstores:
  knowledge-store:
    provider: chromadb
    connection_string: http://localhost:8000

execution:
  cache_enabled: true
  verbose: false
```

### agent.yaml

```yaml
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
    Provide accurate, helpful responses based on the documentation.

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

### .env

```bash
OPENAI_API_KEY=sk-...
```

### docker-compose.yml

```yaml
version: "3.9"

services:
  chromadb:
    image: chromadb/chroma:latest
    container_name: holodeck-chromadb
    ports:
      - "8000:8000"
    volumes:
      - chroma-data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE
      - ANONYMIZED_TELEMETRY=FALSE

volumes:
  chroma-data:
```

### Running the Agent

```bash
# 1. Start ChromaDB
docker compose up -d

# 2. Verify ChromaDB is running
curl http://localhost:8000/api/v2/heartbeat

# 3. Run the agent
holodeck test agent.yaml
```

---

## Supported Vector Store Providers

HoloDeck supports multiple vector database backends. See the [Tools Guide](tools.md#supported-vector-database-providers) for the complete list.

| Provider | Best For | Setup Complexity |
|----------|----------|------------------|
| `chromadb` | Lightweight development, Python-native | Low |
| `postgres` | Existing PostgreSQL infrastructure | Medium |
| `qdrant` | High-performance production | Medium |
| `in-memory` | Testing and prototyping | None |

---

## Troubleshooting

### Cannot connect to ChromaDB

**Error:** `Connection refused` or `Cannot connect to http://localhost:8000`

**Solutions:**

1. Verify ChromaDB is running:
   ```bash
   docker ps | grep chromadb
   ```

2. Check the container logs:
   ```bash
   docker logs holodeck-chromadb
   ```

3. Test connectivity:
   ```bash
   curl http://localhost:8000/api/v2/heartbeat
   ```

4. Ensure port 8000 is not blocked by firewall

### Data not persisting

**Solution:** Mount a volume for data persistence:

```bash
docker run -d \
  --name chromadb \
  -p 8000:8000 \
  -v chroma-data:/chroma/chroma \
  -e IS_PERSISTENT=TRUE \
  chromadb/chroma:latest
```

### Container already exists

**Error:** `container name "chromadb" is already in use`

**Solution:** Remove the existing container:

```bash
docker rm -f chromadb
```

---

## Additional Resources

- [ChromaDB on Docker Hub](https://hub.docker.com/r/chromadb/chroma)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [PostgreSQL pgvector](https://github.com/pgvector/pgvector)
- [Tools Reference Guide](tools.md) - Complete vectorstore tool configuration
- [Global Configuration Guide](global-config.md) - Shared settings across agents

## Next Steps

- See [Tools Reference](tools.md) for vectorstore tool options (chunk size, embedding models, etc.)
- See [Agent Configuration](agent-configuration.md) for complete agent setup
- See [LLM Providers Guide](llm-providers.md) for configuring the LLM that powers your agent
- See [Evaluations Guide](evaluations.md) for testing your agent's search quality
- See [Global Configuration](global-config.md) for sharing vectorstore configs across agents
