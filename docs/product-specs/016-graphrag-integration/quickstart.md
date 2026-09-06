# GraphRAG Engine Quickstart

**Feature**: 016-graphrag-integration
**Date**: 2025-12-27

---

## Prerequisites

1. **Install GraphRAG support**:
   ```bash
   pip install holodeck[graphrag]
   ```

2. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY=sk-...
   # OR for Azure OpenAI:
   export AZURE_OPENAI_API_KEY=...
   export AZURE_OPENAI_ENDPOINT=https://your-instance.openai.azure.com/
   ```

3. **Prepare source documents**:
   - Place text, markdown, or PDF files in a directory
   - Example: `data/documents/`

---

## Quick Start

### 1. Minimal Configuration

Create or update your `agent.yaml`:

```yaml
name: knowledge-agent
description: Agent with GraphRAG knowledge graph search

model:
  provider: openai
  name: gpt-4o

instructions:
  inline: |
    You are a helpful assistant with access to a knowledge graph.
    Use the knowledge_graph tool to find information about entities
    and their relationships.

tools:
  - name: knowledge_graph
    type: vectorstore
    engine: graphrag
    source: data/documents/
    description: Search for entities and relationships in documents
```

### 2. Run Your Agent

```bash
# First run will build the index (may take several minutes)
holodeck chat --agent agent.yaml

# Chat with your agent
You: Who is the CEO of Acme Corp?
```

---

## Search Modes

### Local Search (Default)

Best for entity-centric queries about specific people, organizations, or concepts.

```yaml
tools:
  - name: entity_search
    type: vectorstore
    engine: graphrag
    source: data/documents/
    description: Find specific entities and their relationships

    graphrag:
      search_mode: local
```

**Example queries**:
- "Who is John Smith and what does he do?"
- "What companies are related to Acme Corp?"
- "Explain the relationship between X and Y"

### Global Search

Best for analytical queries that require synthesizing themes across the entire corpus.

```yaml
tools:
  - name: theme_analyzer
    type: vectorstore
    engine: graphrag
    source: data/research/
    description: Analyze themes and patterns across documents

    graphrag:
      search_mode: global
      community_level: 1  # More detailed analysis
```

**Example queries**:
- "What are the main themes in this dataset?"
- "What patterns emerge across all documents?"
- "Summarize the key findings"

---

## Configuration Options

### Cost Optimization

Use cheaper models for indexing, better models for search:

```yaml
graphrag:
  search_mode: local

  indexing_model:
    provider: openai
    name: gpt-4o-mini      # Cheaper, ~$0.15/1M tokens
    temperature: 0.0

  search_model:
    provider: openai
    name: gpt-4o           # Better quality for user-facing results
    temperature: 0.0
```

### Custom Entity Types

Specify which types of entities to extract:

```yaml
graphrag:
  entity_types:
    - organization
    - person
    - product
    - technology
    - regulation
```

### Azure OpenAI

```yaml
graphrag:
  indexing_model:
    provider: azure_openai
    name: gpt-4o-mini
    api_base: ${AZURE_OPENAI_ENDPOINT}
    deployment_name: gpt-4o-mini
    api_version: "2024-02-15-preview"
```

---

## Understanding the Index

### What GraphRAG Extracts

1. **Entities**: People, organizations, locations, events, concepts
2. **Relationships**: Connections between entities with descriptions
3. **Communities**: Clusters of related entities (hierarchical)
4. **Community Reports**: LLM-generated summaries of each community

### Index Location

Artifacts are stored in:
```
.holodeck/graphrag/{tool_name}/
├── output/           # Parquet files (entities, relationships, etc.)
├── cache/            # LLM response cache (saves money on re-indexing)
├── logs/             # Pipeline logs
└── index.meta        # Metadata (entity counts, timestamps)
```

### Incremental Updates

- Index is cached and reused if source files haven't changed
- Modified files trigger automatic re-indexing
- Force rebuild: Delete `.holodeck/graphrag/{tool_name}/` directory

---

## Troubleshooting

### "GraphRAG package not installed"

```bash
pip install holodeck[graphrag]
```

### "API key missing"

Ensure `OPENAI_API_KEY` is set in your environment or `.env` file.

### Slow indexing

GraphRAG indexing is LLM-intensive. For large corpora:
- Use `gpt-4o-mini` for indexing (much cheaper)
- Enable `skip_claim_extraction: true` (default)
- Reduce `max_gleanings` to 0-1

### Empty search results

1. Check if index was built successfully:
   ```bash
   ls .holodeck/graphrag/{tool_name}/output/
   ```
2. Verify source documents contain relevant content
3. Try different queries (local search is entity-focused)

---

## Example: Full Configuration

```yaml
tools:
  - name: company_knowledge
    type: vectorstore
    engine: graphrag
    source: data/company_docs/
    description: >
      Search company knowledge base for information about
      people, organizations, projects, and their relationships

    graphrag:
      search_mode: local
      community_level: 2

      indexing_model:
        provider: openai
        name: gpt-4o-mini
        temperature: 0.0

      search_model:
        provider: openai
        name: gpt-4o
        temperature: 0.0

      entity_types:
        - person
        - organization
        - project
        - product
        - location

      chunk_size: 300
      chunk_overlap: 100
      max_gleanings: 1
      skip_claim_extraction: true
```

---

## Next Steps

1. **Explore your data**: Start with local search to find specific entities
2. **Analyze patterns**: Use global search for high-level insights
3. **Tune entity types**: Customize for your domain
4. **Monitor costs**: Check LLM usage in your provider dashboard
