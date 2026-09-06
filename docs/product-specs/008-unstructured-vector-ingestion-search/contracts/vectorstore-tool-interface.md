# VectorStore Tool Interface Contract

**Version**: 1.0.0
**Feature**: 008-unstructured-vector-ingestion-search
**Date**: 2025-11-23

## Overview

This document defines the contract between the Agent Engine and the VectorStore Tool. The tool provides semantic search capabilities over unstructured documents through a standardized interface.

## Tool Configuration Contract (YAML)

### Minimal Configuration

```yaml
# agent.yaml
tools:
  - type: vectorstore
    name: knowledge_base
    source: data/docs/
```

### Full Configuration (Tool-Specific Database)

```yaml
# agent.yaml
tools:
  - type: vectorstore
    name: knowledge_base
    description: "Semantic search over product documentation"
    source: data/docs/  # File or directory path
    embedding_model: text-embedding-3-small  # Optional
    database:  # Optional - uses in-memory if omitted
      provider: redis
      connection_string: redis://localhost:6379
      index_name: holodeck_vectors
      vector_algorithm: HNSW
      distance_metric: COSINE
    top_k: 5  # Optional - default 5
    min_similarity_score: 0.7  # Optional - default None
```

### Configuration with Global Database (Shared)

**Project-level config** (`.holodeck/config.yaml` or `~/.holodeck/config.yaml`):
```yaml
# .holodeck/config.yaml - Global vector database configuration
vectorstore:
  provider: redis
  connection_string: redis://localhost:6379
  index_name: holodeck_vectors
  vector_algorithm: HNSW
  distance_metric: COSINE
```

**Agent configuration** (`agent.yaml`):
```yaml
# agent.yaml
tools:
  - type: vectorstore
    name: knowledge_base
    source: data/docs/
    # database: config omitted - inherits from global config.yaml

  - type: vectorstore
    name: faq_search
    source: data/faqs.md
    # Also inherits global config - both tools share same Redis instance
```

### Configuration Precedence

**Database Configuration Resolution Order** (highest to lowest priority):
1. **Tool-specific `database` config in agent.yaml** - if present, use this
2. **Project-level config** (`.holodeck/config.yaml`) - vectorstore section
3. **User-level config** (`~/.holodeck/config.yaml`) - vectorstore section
4. **In-memory fallback** - if no config found, use VolatileVectorStore

**Example Scenarios**:

**Scenario 1: User-level default Redis for all projects**
```yaml
# ~/.holodeck/config.yaml (user home directory)
vectorstore:
  provider: redis
  connection_string: redis://localhost:6379
```

```yaml
# agent.yaml (any project)
tools:
  - type: vectorstore
    name: docs
    source: data/docs/
    # Uses user-level Redis config
```

**Scenario 2: Project overrides user-level config**
```yaml
# ~/.holodeck/config.yaml
vectorstore:
  provider: redis
  connection_string: redis://localhost:6379
```

```yaml
# .holodeck/config.yaml (project directory)
vectorstore:
  provider: redis
  connection_string: redis://prod-cache:6379  # Different server
  index_name: myproject_vectors
```

```yaml
# agent.yaml
tools:
  - type: vectorstore
    name: docs
    source: data/docs/
    # Uses project-level config (prod-cache), not user-level
```

**Scenario 3: Tool-specific override of all configs**
```yaml
# ~/.holodeck/config.yaml
vectorstore:
  provider: redis
  connection_string: redis://localhost:6379
```

```yaml
# .holodeck/config.yaml
vectorstore:
  provider: redis
  connection_string: redis://prod-cache:6379
```

```yaml
# agent.yaml
tools:
  - type: vectorstore
    name: docs
    source: data/docs/
    # Uses project-level config (prod-cache)

  - type: vectorstore
    name: experimental
    source: data/experimental/
    database:
      provider: redis
      connection_string: redis://dev-cache:6379
    # Overrides all - uses dev-cache
```

**Scenario 4: No config anywhere - in-memory fallback**
```yaml
# No ~/.holodeck/config.yaml
# No .holodeck/config.yaml

# agent.yaml
tools:
  - type: vectorstore
    name: docs
    source: data/docs/
    # No database config - uses in-memory store
```

### Configuration Schema

```typescript
// Global configuration (config.yaml at project or user level)
interface GlobalConfig {
  vectorstore?: DatabaseConfig;  // OPTIONAL - Shared by all vectorstore tools
}

// Agent configuration (agent.yaml)
interface AgentConfig {
  tools: ToolConfig[];
}

interface VectorStoreToolConfig {
  type: "vectorstore";  // REQUIRED - Literal type discriminator
  name: string;  // REQUIRED - Tool identifier
  description?: string;  // OPTIONAL - Human-readable description
  source: string;  // REQUIRED - File path or directory path
  embedding_model?: string;  // OPTIONAL - Defaults to provider default
  database?: DatabaseConfig;  // OPTIONAL - Overrides global config, defaults to in-memory
  top_k?: number;  // OPTIONAL - Default 5
  min_similarity_score?: number;  // OPTIONAL - Default null (no filtering)
}

interface DatabaseConfig {
  provider: "redis";  // REQUIRED - Only Redis supported initially
  connection_string: string;  // REQUIRED - Redis connection URL
  index_name?: string;  // OPTIONAL - Default "holodeck_vectors"
  vector_algorithm?: "HNSW" | "FLAT";  // OPTIONAL - Default "HNSW"
  distance_metric?: "COSINE" | "L2" | "IP";  // OPTIONAL - Default "COSINE"
}
```

### Validation Rules

1. **type**: Must be exactly `"vectorstore"`
2. **name**: Must be non-empty string, alphanumeric with underscores
3. **source**: Must be valid file or directory path (validated at initialization)
4. **embedding_model**: If specified, must be supported by configured LLM provider
5. **top_k**: Must be positive integer (1-100 recommended)
6. **min_similarity_score**: Must be float between 0.0 and 1.0
7. **database.connection_string**: Must be valid Redis URL (`redis://` or `rediss://`)
8. **database precedence**: Tool-specific database → project config → user config → in-memory

## Tool Invocation Contract (Python API)

### Tool Interface

```python
class VectorStoreTool:
    """Vectorstore tool for semantic search."""

    async def initialize(self, force_ingest: bool = False) -> None:
        """Initialize tool and ingest source files.

        Args:
            force_ingest: If True, re-ingest all files regardless of mtime

        Raises:
            FileNotFoundError: If source path doesn't exist
            ConfigError: If configuration is invalid
            ConnectionError: If database connection fails (falls back to in-memory)
        """

    async def search(self, query: str) -> str:
        """Execute semantic search and return formatted results.

        Args:
            query: Natural language search query

        Returns:
            Formatted string with search results:

            ```
            Found 3 results:

            [1] Score: 0.89 | Source: data/docs/api.md
            Content: The API endpoint /users accepts GET and POST requests...

            [2] Score: 0.76 | Source: data/docs/auth.md
            Content: Authentication is handled via JWT tokens...

            [3] Score: 0.72 | Source: data/docs/deployment.md
            Content: Deploy the application using Docker...
            ```

        Raises:
            RuntimeError: If tool not initialized
            ValueError: If query is empty
        """
```

### Initialization Behavior

#### Success Case

**Preconditions**:
- Configuration is valid
- Source path exists and is readable
- At least one supported file exists in source

**Process**:
1. Validate configuration
2. Resolve database config (tool database → project config → user config → in-memory fallback)
3. Initialize vector store (Redis or in-memory)
4. Discover files in source (recursively if directory)
5. For each file:
   - Check if re-ingestion needed (mtime comparison or force_ingest)
   - Convert to markdown (via FileProcessor)
   - Chunk text (512 tokens with 50 token overlap)
   - Generate embeddings (batch processing)
   - Store DocumentRecords in vector store
6. Set `is_initialized = True`

**Postconditions**:
- Tool is ready for search queries
- All source files are indexed
- `document_count` reflects number of stored chunks

#### Error Cases

| Error Condition | Exception | Fallback Behavior |
|----------------|-----------|-------------------|
| Source path doesn't exist | `FileNotFoundError` | Fail fast - user must fix config |
| No supported files in source | `ValidationError` | Fail fast - user must add files |
| Database connection fails | `ConnectionError` | Log warning, fall back to in-memory |
| File processing error | N/A (logged) | Skip file, continue with remaining |
| Empty file | N/A (logged) | Skip file, continue with remaining |
| Encoding error | N/A (logged) | Skip file, continue with remaining |
| Embedding API error | `RuntimeError` | Fail fast - cannot proceed without embeddings |

### Search Behavior

#### Success Case

**Preconditions**:
- Tool is initialized (`is_initialized == True`)
- Query is non-empty string

**Process**:
1. Generate query embedding
2. Execute vector similarity search
3. Filter results by `min_similarity_score` (if configured)
4. Limit to `top_k` results
5. Format results as string
6. Return formatted string

**Postconditions**:
- Results ordered by descending relevance score
- All results include source attribution
- Result count ≤ top_k

#### Response Format

```python
@dataclass
class SearchResult:
    """Internal search result structure (formatted to string for agent)."""
    rank: int  # 1-indexed result rank
    score: float  # Relevance score (0.0-1.0)
    source_path: str  # Original source file path
    content: str  # Matched chunk content
```

**Formatted Output Template**:
```
Found {count} result(s):

[{rank}] Score: {score:.2f} | Source: {source_path}
{content}

[{rank}] Score: {score:.2f} | Source: {source_path}
{content}

...
```

**Edge Cases**:
- **No results**: `"No relevant results found for query: {query}"`
- **All filtered by min_score**: `"No results above similarity threshold {min_score}"`

#### Error Cases

| Error Condition | Exception | Error Message |
|----------------|-----------|---------------|
| Tool not initialized | `RuntimeError` | "VectorStoreTool must be initialized before search" |
| Empty query | `ValueError` | "Search query cannot be empty" |
| Embedding API error | `RuntimeError` | "Failed to generate query embedding: {error}" |

## File Ingestion Contract

### Supported File Types

| Extension | Format | Processing Method | Notes |
|-----------|--------|------------------|-------|
| `.txt` | Plain text | Direct read | UTF-8 encoding assumed, fallback to chardet |
| `.md` | Markdown | Direct read | Preserves markdown structure |
| `.pdf` | PDF document | markitdown conversion | Extracts text + OCR if needed |
| `.csv` | CSV data | markitdown conversion | Converts to markdown table |
| `.json` | JSON data | markitdown conversion | Pretty-printed as code block |

All conversions handled by existing `FileProcessor` (`src/holodeck/lib/file_processor.py`).

### Directory Processing

**Behavior**:
- Recursively traverse all subdirectories (FR-019)
- Process only files with supported extensions
- Skip unsupported files with WARNING log (FR-016)
- Skip empty files with WARNING log (FR-018)
- Continue processing on individual file errors (FR-016)

**Example Structure**:
```
data/docs/
├── api/
│   ├── endpoints.md      ✅ Processed
│   ├── schemas.json      ✅ Processed
│   └── diagram.png       ⚠️  Skipped (not supported)
├── guides/
│   ├── quickstart.md     ✅ Processed
│   └── tutorial.pdf      ✅ Processed
└── README.txt            ✅ Processed
```

### Modification Tracking

**Behavior**:
- Compare file `mtime` (modification timestamp) with stored `DocumentRecord.mtime`
- If `file.mtime > record.mtime` → Re-ingest file (delete old records, create new)
- If `file.mtime == record.mtime` → Skip ingestion (already up to date)
- If `--force-ingest` flag set → Ignore mtime, always re-ingest

**Re-ingestion Triggers**:
1. File content modified (mtime changed)
2. `--force-ingest` flag used
3. Embedding model changed in configuration (requires `--force-ingest`)
4. File previously failed to process (no stored record)

## CLI Integration Contract

### Chat Command

```bash
# Standard usage
holodeck chat

# Force re-ingestion of vectorstore sources
holodeck chat --force-ingest
```

**Behavior**:
- `--force-ingest` flag passed to all VectorStoreTool instances during initialization
- All source files re-ingested regardless of mtime
- Useful after changing embedding model or recovering from corrupted index

### Test Command

```bash
# Standard usage
holodeck test

# Force re-ingestion before running tests
holodeck test --force-ingest
```

**Behavior**: Same as chat command

## Agent LLM Tool Call Contract

### Tool Registration

**Function Definition Provided to LLM**:
```json
{
  "name": "knowledge_base",
  "description": "Semantic search over product documentation",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "Natural language search query"
      }
    },
    "required": ["query"]
  }
}
```

### Tool Call Example

**LLM Request**:
```json
{
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "knowledge_base",
        "arguments": "{\"query\": \"How do I authenticate API requests?\"}"
      }
    }
  ]
}
```

**Tool Response**:
```json
{
  "tool_call_id": "call_abc123",
  "content": "Found 2 result(s):\n\n[1] Score: 0.89 | Source: data/docs/auth.md\nAuthentication is handled via JWT tokens. Include the token in the Authorization header...\n\n[2] Score: 0.76 | Source: data/docs/api.md\nThe API requires authentication for all endpoints except /health..."
}
```

## Performance Guarantees

| Metric | Target | Measurement Context |
|--------|--------|-------------------|
| Search latency | < 2 seconds | Up to 1000 documents, p95 |
| Initialization time | < 5 seconds | Single file, cold start |
| Memory usage | < 100MB overhead | In-memory store, 1000 documents |
| Concurrent queries | 10+ queries/sec | Redis backend, warm cache |

## Backward Compatibility

**Version 1.0.0 (Initial)**:
- No backward compatibility concerns (initial implementation)

**Future Versions**:
- Configuration changes MUST be backward compatible (optional new fields only)
- Breaking changes require major version bump and migration guide
- Tool invocation interface MUST remain stable (return format can be enhanced)

## Security Considerations

1. **File Access**: Tool can only read files within configured source path (no path traversal)
2. **Redis Connection**: Connection string should use environment variables, not hardcoded
3. **API Keys**: Embedding service API keys loaded from environment, never from config files
4. **Input Validation**: All queries sanitized before embedding generation (no code injection)

## Testing Contracts

### Unit Test Coverage Requirements

- Configuration parsing and validation (including project/user config precedence)
- File discovery and filtering
- Text chunking logic
- Embedding batch processing
- Modification timestamp comparison
- Search result formatting
- Error handling for all error cases

### Integration Test Coverage Requirements

- End-to-end file ingestion with Redis backend
- End-to-end file ingestion with in-memory backend
- Project-level config inheritance
- User-level config inheritance
- Tool-specific config override of global configs
- Search accuracy validation (semantic relevance)
- Multi-file directory ingestion
- Re-ingestion on file modification
- `--force-ingest` flag behavior
- Edge cases: empty files, unsupported formats, large files

## Open Questions (Resolved)

All questions resolved during research phase. No open contract questions remain.

## Changelog

- **2025-11-23**: Initial contract definition (v1.0.0)
  - Database config resolution: tool-specific → project config → user config → in-memory
  - Global `vectorstore` config supported in `.holodeck/config.yaml` and `~/.holodeck/config.yaml`
