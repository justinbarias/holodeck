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

## Setting Up Redis Stack

[Redis Stack](https://hub.docker.com/r/redis/redis-stack) is the recommended vector store for development. It includes Redis with the RediSearch and RedisJSON modules required for vector similarity search.

### Quick Start with Docker

**Run Redis Stack:**

```bash
docker run -d \
  --name redis-stack \
  -p 6379:6379 \
  -p 8001:8001 \
  redis/redis-stack:latest
```

This exposes:

- **Port 6379**: Redis server (for HoloDeck connections)
- **Port 8001**: RedisInsight web UI (optional, for debugging)

**Verify Redis is running:**

```bash
docker exec -it redis-stack redis-cli ping
# PONG
```

### Docker Compose (Recommended for Projects)

Create a `docker-compose.yml` file in your project root:

```yaml
version: "3.8"

services:
  redis:
    image: redis/redis-stack:latest
    container_name: holodeck-redis
    ports:
      - "6379:6379"
      - "8001:8001"
    volumes:
      - redis-data:/data
    environment:
      - REDIS_ARGS=--save 60 1
    restart: unless-stopped

volumes:
  redis-data:
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
  --name redis-stack \
  -p 6379:6379 \
  -p 8001:8001 \
  docker.io/redis/redis-stack:latest
```

### Redis Stack Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_ARGS` | Additional Redis server arguments | None |
| `REDISEARCH_ARGS` | RediSearch module arguments | None |
| `REDISJSON_ARGS` | RedisJSON module arguments | None |

**Example with persistence:**

```bash
docker run -d \
  --name redis-stack \
  -p 6379:6379 \
  -v redis-data:/data \
  -e REDIS_ARGS="--save 60 1 --appendonly yes" \
  redis/redis-stack:latest
```

**Persistence flags explained:**

| Flag | Description | Use Case |
|------|-------------|----------|
| `--save 60 1` | Save to disk if at least 1 key changed in 60 seconds | Basic durability |
| `--appendonly yes` | Enable append-only file (AOF) for write logging | Maximum durability |
| `--save ""` | Disable RDB snapshots (use with AOF) | AOF-only persistence |

> **Production Tip**: For production deployments, use both `--save` and `--appendonly yes` for maximum data durability. The AOF file logs every write operation, while RDB provides point-in-time snapshots.

---

## Configuring Vector Stores in HoloDeck

### Global Configuration (config.yaml)

Define reusable vector store connections in your global configuration:

```yaml
# config.yaml (project root or ~/.holodeck/config.yaml)

vectorstores:
  # Redis with Hashset storage (recommended for most use cases)
  my-redis-store:
    provider: redis-hashset
    connection_string: redis://localhost:6379

  # Redis with JSON storage (for complex nested data)
  my-redis-json:
    provider: redis-json
    connection_string: ${REDIS_URL}

  # Production Redis with authentication
  production-redis:
    provider: redis-hashset
    connection_string: redis://:${REDIS_PASSWORD}@${REDIS_HOST}:6379
```

### Environment Variables

Store sensitive connection details in environment variables:

```bash
# .env file (DO NOT commit to version control)
REDIS_URL=redis://localhost:6379
REDIS_HOST=redis.example.com
REDIS_PASSWORD=your-secure-password
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
    database: my-redis-store  # Reference to config.yaml
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
      provider: redis-hashset
      connection_string: redis://localhost:6379
```

---

## Connection String Formats

### Redis Connection Strings

| Format | Example |
|--------|---------|
| Basic | `redis://localhost:6379` |
| With password | `redis://:${REDIS_PASSWORD}@localhost:6379` |
| With username | `redis://${REDIS_USER}:${REDIS_PASSWORD}@localhost:6379` |
| With database | `redis://localhost:6379/0` |
| TLS/SSL | `rediss://localhost:6379` |

### Full Connection String Reference

```
redis[s]://[[username:]password@]host[:port][/database]
```

**Examples (using environment variables for security):**

```yaml
# Local development
connection_string: redis://localhost:6379

# With authentication (use environment variables!)
connection_string: redis://:${REDIS_PASSWORD}@localhost:6379

# Remote server with TLS
connection_string: rediss://${REDIS_USER}:${REDIS_PASSWORD}@${REDIS_HOST}:6380

# Specific database (0-15)
connection_string: redis://localhost:6379/1
```

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
    provider: redis-hashset
    connection_string: ${REDIS_URL}

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
REDIS_URL=redis://localhost:6379
```

### docker-compose.yml

```yaml
version: "3.8"

services:
  redis:
    image: redis/redis-stack:latest
    ports:
      - "6379:6379"
      - "8001:8001"
    volumes:
      - redis-data:/data

volumes:
  redis-data:
```

### Running the Agent

```bash
# 1. Start Redis
docker compose up -d

# 2. Verify Redis is running
docker exec -it holodeck-redis redis-cli ping

# 3. Run the agent
holodeck test agent.yaml
```

---

## Supported Vector Store Providers

HoloDeck supports multiple vector database backends. See the [Tools Guide](tools.md#supported-vector-database-providers) for the complete list.

| Provider | Best For | Setup Complexity |
|----------|----------|------------------|
| `redis-hashset` | Development, small-medium datasets | Low |
| `redis-json` | Complex nested data structures | Low |
| `postgres` | Existing PostgreSQL infrastructure | Medium |
| `qdrant` | High-performance production | Medium |
| `in-memory` | Testing and prototyping | None |

---

## Troubleshooting

### Cannot connect to Redis

**Error:** `Connection refused` or `Cannot connect to redis://localhost:6379`

**Solutions:**

1. Verify Redis is running:
   ```bash
   docker ps | grep redis
   ```

2. Check the container logs:
   ```bash
   docker logs redis-stack
   ```

3. Test connectivity:
   ```bash
   docker exec -it redis-stack redis-cli ping
   ```

4. Ensure port 6379 is not blocked by firewall

### Authentication failed

**Error:** `NOAUTH Authentication required`

**Solution:** Include password in connection string:

```yaml
connection_string: redis://:your-password@localhost:6379
```

### Module not loaded

**Error:** `unknown command 'FT.CREATE'`

**Solution:** Use `redis/redis-stack` image instead of plain `redis`:

```bash
# Wrong (missing vector search modules)
docker run -d redis:latest

# Correct (includes RediSearch)
docker run -d redis/redis-stack:latest
```

### Data not persisting

**Solution:** Mount a volume for data persistence:

```bash
docker run -d \
  --name redis-stack \
  -p 6379:6379 \
  -v redis-data:/data \
  redis/redis-stack:latest
```

### Container already exists

**Error:** `container name "redis-stack" is already in use`

**Solution:** Remove the existing container:

```bash
docker rm -f redis-stack
```

---

## Additional Resources

- [Redis Stack on Docker Hub](https://hub.docker.com/r/redis/redis-stack)
- [Redis Vector Similarity Documentation](https://redis.io/docs/stack/search/reference/vectors/)
- [Tools Reference Guide](tools.md) - Complete vectorstore tool configuration
- [Global Configuration Guide](global-config.md) - Shared settings across agents

## Next Steps

- See [Tools Reference](tools.md) for vectorstore tool options (chunk size, embedding models, etc.)
- See [Agent Configuration](agent-configuration.md) for complete agent setup
- See [LLM Providers Guide](llm-providers.md) for configuring the LLM that powers your agent
- See [Evaluations Guide](evaluations.md) for testing your agent's search quality
- See [Global Configuration](global-config.md) for sharing vectorstore configs across agents
