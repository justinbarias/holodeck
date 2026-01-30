# Quickstart: HierarchicalDocumentTool

## Overview

The `HierarchicalDocumentTool` provides advanced document search capabilities combining:
- **Semantic search** (dense embeddings for conceptual queries)
- **Keyword search** (BM25 for precise terminology)
- **Exact match** (section numbers, identifiers)
- **Hybrid fusion** (Reciprocal Rank Fusion to combine all modalities)

It preserves document structure during parsing, so search results include the full hierarchical context (e.g., "Title I > Chapter 2 > Section 203").

## Quick Start (3 minutes)

### 1. Add to your agent.yaml

```yaml
name: my-agent
description: Agent with document search

model:
  provider: openai
  name: gpt-4o

instructions:
  inline: |
    You are a helpful assistant with access to a knowledge base.
    Use the docs_search tool to find relevant information.

tools:
  - name: docs_search
    type: hierarchical_document
    description: "Search company documentation"
    source: "./docs/"
```

### 2. Add your documents

Place your documents in the `./docs/` directory:
- Markdown (`.md`)
- PDF (`.pdf`)
- Word (`.docx`)
- Text (`.txt`)

### 3. Run your agent

```bash
holodeck chat agent.yaml
```

The tool automatically:
1. Converts documents to markdown (via markitdown)
2. Parses hierarchical structure from headings
3. Chunks by structure (max 800 tokens)
4. **Generates LLM context for each chunk** (Anthropic approach - calls Claude Haiku)
5. Generates embeddings from contextualized text
6. Builds BM25 (on contextualized text) and exact match indices

## Configuration Examples

### Semantic-Only Search
Best for conceptual questions without specific terminology.

```yaml
tools:
  - name: concept_search
    type: hierarchical_document
    description: "Search by concepts"
    source: "./docs/"
    search_mode: semantic
    contextual_embeddings: true
```

### Keyword-Heavy Search
Best for technical documents with specific terminology.

```yaml
tools:
  - name: tech_search
    type: hierarchical_document
    description: "Technical documentation search"
    source: "./technical/"
    search_mode: hybrid
    semantic_weight: 0.3
    keyword_weight: 0.5
    exact_weight: 0.2
```

### Legal/Regulatory Documents
Optimized for section references and definitions.

```yaml
tools:
  - name: legal_search
    type: hierarchical_document
    description: "Search legal documents"
    source: "./legal/"
    search_mode: hybrid
    extract_definitions: true
    extract_cross_references: true
    exact_weight: 0.3
```

### Custom Context Model
Specify a different LLM for context generation (defaults to Claude Haiku).

```yaml
tools:
  - name: docs_search
    type: hierarchical_document
    description: "Search with custom context model"
    source: "./docs/"

    # LLM context generation settings
    contextual_embeddings: true
    context_model:
      provider: anthropic
      name: claude-3-haiku-20240307
      temperature: 0.0
    context_max_tokens: 100
    context_concurrency: 10  # Parallel LLM calls
```

### Large Document Corpus with Persistence

```yaml
tools:
  - name: knowledge_base
    type: hierarchical_document
    description: "Persistent knowledge base"
    source: "./knowledge/"
    database:
      provider: postgres
      connection_string: "${POSTGRES_URL}"
    top_k: 20
```

### MongoDB Atlas with Native Hybrid Search

```yaml
tools:
  - name: atlas_search
    type: hierarchical_document
    description: "MongoDB Atlas with native hybrid search"
    source: "./docs/"
    database:
      provider: mongodb  # MongoDB Atlas - supports native hybrid_search()
      connection_string: "${MONGODB_ATLAS_URL}"
```

> **Important**: The `mongodb` provider refers to **MongoDB Atlas** (native MongoDB), which supports native hybrid search via Semantic Kernel's `MongoDBAtlasStore`. If you're using **Azure Cosmos DB's MongoDB API**, use `azure-cosmos-mongo` instead - this provider uses the BM25 fallback strategy since it doesn't support native hybrid search.

## Search Modes

| Mode | Use Case | Example Query |
|------|----------|---------------|
| `semantic` | Conceptual questions | "What are the compliance requirements?" |
| `keyword` | Technical terms | "authentication timeout error" |
| `exact` | Section references | "Section 203(a)(1)" |
| `hybrid` | General purpose | "reporting requirements in Section 5" |

## How It Works

### Preprocessing Pipeline (Anthropic Contextual Retrieval)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INGESTION (per document)                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  document.pdf  →  [markitdown]  →  markdown text                         │
│                                                                          │
│  markdown text →  [StructuredChunker]  →  chunks with parent_chain       │
│                                                                          │
│  For each chunk:                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ [LLM Context Generation] - Claude Haiku                           │  │
│  │                                                                    │  │
│  │ Input:  Whole document + chunk                                     │  │
│  │ Output: "This chunk describes the reporting requirements under     │  │
│  │          Section 203 of the Environmental Protection Act..."       │  │
│  │                                                                    │  │
│  │ Cost: ~$0.03 per 100-page document                                 │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  contextualized = "{LLM context}\n\n{original chunk}"                    │
│                                                                          │
│  contextualized  →  [Embedding]  →  dense vector                         │
│  contextualized  →  [BM25/Native] →  keyword index                       │
│  section_id      →  [Dict]        →  exact match index                   │
│                                                                          │
│  Result: 49% better retrieval vs standard RAG (Anthropic research)       │
└─────────────────────────────────────────────────────────────────────────┘
```

### Hybrid Search Flow

1. **Query Analysis**: Detects exact match patterns (section numbers, quoted phrases)
2. **Parallel Search**: Runs semantic, keyword, and exact match searches
3. **RRF Fusion**: Combines ranked results using Reciprocal Rank Fusion (k=60)
4. **Enrichment**: Adds relevant definitions if terms are defined in documents

```
Query: "reporting requirements in Section 403"
                    │
    ┌───────────────┼───────────────┐
    ▼               ▼               ▼
[Semantic]      [BM25]          [Exact]
contextualized  contextualized  section_id
embeddings      text search     lookup
    │               │               │
    └───────────────┼───────────────┘
                    ▼
              [RRF Fusion]
              k=60, weights
                    │
                    ▼
            [Top 10 Results]
```

## Result Format

Each search result includes:

```
[1] Score: 0.847 | Source: docs/compliance.md
Location: Chapter 5 > Section 5.3 > Reporting Requirements
Section: sec_5_3

Annual reporting must be submitted within 60 days of the fiscal year end.
All covered entities must include revenue breakdowns by category.

Relevant definitions:
  • Covered entity: Any organization with annual revenue exceeding...
```

## Performance Tips

1. **Chunk size**: Default 800 tokens works well. Increase for dense technical content, decrease for granular retrieval.

2. **Weights tuning**:
   - More semantic: Increase `semantic_weight` for conceptual queries
   - More precise: Increase `keyword_weight` for technical terminology
   - Section-heavy: Increase `exact_weight` for regulatory documents

3. **Persistence**: For large corpora (>1000 documents), use a persistent database to avoid re-indexing on restart.

4. **Contextual embeddings**: Keep enabled (default) for 49% better retrieval accuracy per Anthropic research.

5. **Context generation cost**: ~$0.03 per 100-page document using Claude Haiku. For cost-sensitive applications, increase `context_concurrency` to reduce ingestion time.

6. **Large documents**: For very large documents that exceed context limits, the tool automatically truncates the document context while preserving the chunk content.

## Testing Your Configuration

```bash
# Run with verbose output to see search details
holodeck test agent.yaml --verbose

# Or use chat mode to test interactively
holodeck chat agent.yaml
```

Test queries:
- Conceptual: "What are the main requirements?"
- Keyword: "authentication error handling"
- Exact: "Section 5.2.1"
- Mixed: "how to handle errors in Section 3"

## Common Issues

### "No results found"
- Check that documents exist in `source` path
- Verify document format is supported (.md, .pdf, .docx, .txt)
- Try lowering `min_score` threshold

### "Results missing context"
- Ensure `contextual_embeddings: true` (default)
- Check that documents have heading structure

### "Slow ingestion"
- Large PDFs take longer to process
- Consider splitting very large documents
- Use persistent storage to avoid re-indexing

## Next Steps

- [Full Configuration Reference](./contracts/hierarchical_document_tool_config.yaml)
- [Data Model Documentation](./data-model.md)
- [Research & Design Decisions](./research.md)
