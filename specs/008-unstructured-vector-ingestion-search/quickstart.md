# Quickstart: VectorStore Tool

**Feature**: Unstructured Vector Ingestion and Search
**Audience**: Agent developers using HoloDeck
**Time to complete**: 5 minutes

## What You'll Build

An agent with semantic search capabilities over your documentation files. The agent will be able to answer questions by searching through markdown, PDF, text, and other document files.

## Prerequisites

- HoloDeck CLI installed (`pip install holodeck-ai`)
- OpenAI API key (for embeddings and LLM)
- Optional: Redis server for persistent storage (falls back to in-memory if not configured)

## Step 1: Create Agent Configuration

Create `agent.yaml`:

```yaml
name: docs_assistant
description: AI assistant with access to product documentation

model:
  provider: openai
  name: gpt-4
  temperature: 0.7
  max_tokens: 1000

instructions:
  inline: |
    You are a helpful assistant with access to product documentation.
    When users ask questions, search the knowledge base and provide
    accurate answers based on the documentation.

tools:
  - type: vectorstore
    name: knowledge_base
    description: "Search product documentation and guides"
    source: data/docs/  # Point to your documentation directory
```

## Step 2: Add Documentation Files

Create a `data/docs/` directory and add your documentation:

```bash
mkdir -p data/docs
```

Add some sample files:

**data/docs/getting-started.md**:
```markdown
# Getting Started

Welcome to our product! This guide will help you get up and running quickly.

## Installation

```bash
pip install our-product
```

## Quick Example

```python
from our_product import Client

client = Client(api_key="your-key")
result = client.query("Hello world")
print(result)
```
```

**data/docs/api-reference.md**:
```markdown
# API Reference

## Authentication

All API requests require an API key in the Authorization header:

```
Authorization: Bearer YOUR_API_KEY
```

## Endpoints

### POST /query
Execute a query against the system.

**Parameters**:
- `query` (string, required): The query text
- `limit` (integer, optional): Maximum results (default: 10)

**Response**:
```json
{
  "results": [...],
  "count": 5
}
```
```

## Step 3: Run Your Agent

Start an interactive chat session:

```bash
export OPENAI_API_KEY=your-api-key-here
holodeck chat
```

The first run will:
1. Load your agent configuration
2. Discover files in `data/docs/`
3. Convert files to markdown (if needed)
4. Generate embeddings
5. Store in vector database (in-memory by default)

## Step 4: Ask Questions

Try these example queries:

```
You: How do I install the product?
Agent: Based on the documentation, you can install the product using pip:
       pip install our-product
       [Source: data/docs/getting-started.md]

You: What authentication does the API require?
Agent: The API requires authentication via an API key in the Authorization header.
       Use the format: Authorization: Bearer YOUR_API_KEY
       [Source: data/docs/api-reference.md]

You: Show me a quick example
Agent: Here's a quick example from the documentation:
       ```python
       from our_product import Client
       client = Client(api_key="your-key")
       result = client.query("Hello world")
       print(result)
       ```
       [Source: data/docs/getting-started.md]
```

## Next Steps

### Add More File Types

The vectorstore tool supports multiple formats:

```bash
data/docs/
├── guides/
│   ├── tutorial.pdf      # PDF documents
│   ├── overview.md       # Markdown
│   └── examples.txt      # Plain text
├── api/
│   ├── schemas.json      # JSON data
│   └── endpoints.csv     # CSV tables
└── README.md
```

All files are automatically discovered and processed!

### Configure Redis for Persistence

For production use, configure Redis to persist embeddings across sessions.

**Option 1: User-level config** (`~/.holodeck/config.yaml`):
```yaml
vectorstore:
  provider: redis
  connection_string: redis://localhost:6379
  index_name: holodeck_vectors
```

**Option 2: Project-level config** (`.holodeck/config.yaml`):
```yaml
vectorstore:
  provider: redis
  connection_string: redis://prod-cache:6379
  index_name: myproject_vectors
```

**Option 3: Tool-specific config** (`agent.yaml`):
```yaml
tools:
  - type: vectorstore
    name: knowledge_base
    source: data/docs/
    database:
      provider: redis
      connection_string: redis://localhost:6379
      index_name: holodeck_vectors
```

With Redis configured:
- Embeddings persist across agent restarts
- Faster startup (no re-ingestion needed)
- Support for larger document collections (10,000+ files)

### Customize Search Behavior

Control search results with optional parameters:

```yaml
tools:
  - type: vectorstore
    name: knowledge_base
    source: data/docs/
    top_k: 10  # Return top 10 results (default: 5)
    min_similarity_score: 0.75  # Only results above 75% similarity
```

### Use Custom Embedding Models

Override the default embedding model:

```yaml
tools:
  - type: vectorstore
    name: knowledge_base
    source: data/docs/
    embedding_model: text-embedding-3-large  # Higher quality, more expensive
```

**Available models**:
- `text-embedding-3-small` (default): Fast, cost-effective
- `text-embedding-3-large`: Higher accuracy, more expensive
- `text-embedding-ada-002`: Legacy model (still supported)

After changing embedding models, force re-ingestion:

```bash
holodeck chat --force-ingest
```

### Test Your Agent

Create test cases to validate search behavior:

```yaml
# agent.yaml
test_cases:
  - name: installation_query
    input: "How do I install the product?"
    expected_tools: [knowledge_base]
    ground_truth: "pip install our-product"

  - name: api_auth_query
    input: "What authentication does the API use?"
    expected_tools: [knowledge_base]
    ground_truth: "API key in Authorization header"
```

Run tests:

```bash
holodeck test
```

Force re-ingestion before testing:

```bash
holodeck test --force-ingest
```

## Advanced Usage

### Multiple VectorStore Tools

Use different vectorstores for different knowledge domains:

```yaml
tools:
  - type: vectorstore
    name: user_docs
    description: "User-facing documentation"
    source: data/user-docs/

  - type: vectorstore
    name: internal_docs
    description: "Internal technical documentation"
    source: data/internal-docs/

  - type: vectorstore
    name: api_specs
    description: "API specifications and schemas"
    source: data/api-specs/
    top_k: 3
    min_similarity_score: 0.8
```

The agent will automatically choose the right tool based on the query context.

### Single-File Knowledge Base

Point to a single file instead of a directory:

```yaml
tools:
  - type: vectorstore
    name: faq
    source: data/faq.md  # Single file
```

Useful for:
- FAQ documents
- Policy documents
- Single comprehensive guides

### Update Documentation

When you update documentation files:

1. **Automatic detection**: Modified files are automatically re-ingested on next run (via mtime tracking)
2. **Manual force**: Use `--force-ingest` flag to re-process all files

```bash
# After updating docs/api-reference.md
holodeck chat  # Automatically re-ingests only changed files

# Or force re-ingest everything
holodeck chat --force-ingest
```

### Directory Structure Best Practices

Organize documentation for optimal search:

```bash
data/
├── docs/
│   ├── user-guide/       # User-facing guides
│   │   ├── introduction.md
│   │   ├── tutorials/
│   │   └── faq.md
│   ├── api/              # API documentation
│   │   ├── reference.md
│   │   ├── examples/
│   │   └── schemas.json
│   └── internal/         # Internal docs
│       ├── architecture.md
│       └── deployment.md
```

Use separate vectorstore tools for each top-level category if you want more control.

## Troubleshooting

### "No relevant results found"

**Causes**:
- Query doesn't match document content semantically
- `min_similarity_score` threshold too high
- Source directory empty or no supported files

**Solutions**:
1. Lower `min_similarity_score` or remove it
2. Verify files exist in source directory
3. Check agent logs for skipped files

### "FileNotFoundError: data/docs not found"

**Cause**: Source path doesn't exist

**Solution**: Create directory and add documentation files:
```bash
mkdir -p data/docs
cp your-docs/* data/docs/
```

### Slow initialization

**Cause**: Large number of files being processed on first run

**Solutions**:
1. Configure Redis for persistence (faster subsequent runs)
2. Use `--force-ingest` only when needed
3. Split large directories into multiple vectorstore tools

### Redis connection failed

**Cause**: Redis server not running or connection string incorrect

**Behavior**: Tool automatically falls back to in-memory storage with warning log

**Solutions**:
1. Start Redis: `redis-server`
2. Verify connection string: `redis://localhost:6379`
3. Or remove database config to use in-memory explicitly

## Configuration Reference

### Minimal Config
```yaml
tools:
  - type: vectorstore
    name: my_tool
    source: data/docs/
```

### All Options
```yaml
tools:
  - type: vectorstore
    name: my_tool
    description: "Tool description"
    source: data/docs/  # File or directory
    embedding_model: text-embedding-3-small  # Optional
    database:  # Optional
      provider: redis
      connection_string: redis://localhost:6379
      index_name: holodeck_vectors
      vector_algorithm: HNSW
      distance_metric: COSINE
    top_k: 5  # Number of results
    min_similarity_score: 0.7  # Minimum relevance threshold
```

## What's Next?

- **Add more tools**: Combine vectorstore with function tools, MCP tools, or prompt tools
- **Deploy your agent**: Use `holodeck deploy` to create production API endpoint
- **Run evaluations**: Validate search quality with AI-powered metrics

## Resources

- [Full API documentation](../contracts/vectorstore-tool-interface.md)
- [Data model reference](../data-model.md)
- [Implementation research](../research.md)
- [Example agents](../../templates/)

---

**Questions?** File an issue or check the documentation.
