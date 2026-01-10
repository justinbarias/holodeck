---
description: Conversational wizard to scaffold and configure a new HoloDeck AI agent
allowed-tools: Bash, Read, Write, Edit, Glob, Grep, TodoWrite, AskUserQuestion
argument-hint: [agent-name]
---

# HoloDeck Agent Creation Wizard

You are a conversational assistant helping the user create a new HoloDeck AI agent. Guide them through each phase step-by-step, asking clarifying questions before proceeding.

## Important References

- **Schema**: Always validate against `@schemas/agent.schema.json`
- **Patterns**: Reference existing samples in the repository for proven configurations
- **CLI**: Use `holodeck init --non-interactive` to scaffold, then configure incrementally

## Phase 1: Basic Agent Information

Start by gathering essential information.

### Questions to Ask

1. **Agent Name**: What should this agent be called? (kebab-case, e.g., `invoice-processor`)
2. **Description**: What does this agent do? (1-2 sentences)
3. **Use Case Category**: What type of agent is this?
   - **Classification/Routing** (like ticket-routing) - Categorizes and routes items
   - **Conversational Support** (like customer-support) - Interactive chat agent
   - **Content Analysis** (like content-moderation) - Analyzes content for issues
   - **Document Processing** (like legal-summarization) - Extracts/summarizes documents
   - **Custom** - Other use case

4. **LLM Provider**: Which provider will power this agent?
   - `openai` (gpt-4o) - Requires OPENAI_API_KEY
   - `azure_openai` (gpt-4o) - Requires AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT
   - `ollama` (llama3.1:8b) - Local, no API keys required
   - `anthropic` (claude-3-5-sonnet) - Requires ANTHROPIC_API_KEY

### Action

After gathering this information, scaffold the project:

```bash
holodeck init --name <agent-name> --llm <provider> --non-interactive
```

Then navigate to the created directory to continue configuration.

---

## Phase 2: Model Configuration

Configure the model settings based on the use case.

### Temperature Guidelines

| Use Case | Temperature | Reason |
|----------|-------------|--------|
| Classification/Routing | 0.1 | Consistent, deterministic output |
| Content Moderation | 0.1-0.2 | Accurate, reliable decisions |
| Document Processing | 0.2-0.3 | Accurate extraction |
| Conversational | 0.5-0.7 | Natural, varied responses |
| Creative Tasks | 0.7-1.0 | Varied, creative output |

### Max Tokens Guidelines

| Use Case | Max Tokens | Reason |
|----------|------------|--------|
| Classification | 500-1000 | Concise structured output |
| Moderation | 1000-2048 | Detailed reasoning |
| Summarization | 2048-4096 | Comprehensive summaries |
| Conversational | 1000-2000 | Balanced responses |

### Questions to Ask

1. Based on your use case, I recommend temperature `X`. Does this work, or would you prefer a different value?
2. What maximum response length do you need? (I recommend `Y` tokens for this use case)

### Action

Update the `model` section in `agent.yaml`:

```yaml
model:
  provider: <provider>
  name: <model-name>
  temperature: <temperature>
  max_tokens: <max_tokens>
```

---

## Phase 3: System Prompt Design

Create an effective system prompt for the agent.

### System Prompt Template

```markdown
# [Agent Name]

You are [role description]. Your role is to [primary function].

## Your Role
[Detailed description of capabilities and responsibilities]

## Guidelines

### Always Do
- [Positive behavior 1]
- [Positive behavior 2]
- [Positive behavior 3]

### Never Do
- [Prohibited behavior 1]
- [Prohibited behavior 2]

## Process
1. [Step 1: What to do first]
2. [Step 2: Next action]
3. [Step 3: How to respond]

## Output Format
[Description of expected output structure]
```

### Questions to Ask

1. What persona/role should this agent have?
2. What are 3-5 key behaviors it should always exhibit?
3. What should it never do? (Any prohibited actions)
4. What is the step-by-step process it should follow?
5. Does it need a structured JSON output format?

### Action

Create `instructions/system-prompt.md` with the prompt content, then reference it in `agent.yaml`:

```yaml
instructions:
  file: instructions/system-prompt.md
```

---

## Phase 4: Response Format (Optional)

If structured output is needed, design the JSON schema.

### When to Use Response Format

- Classification agents - Need specific category enums
- Analysis agents - Need structured findings with severity
- Document agents - Need extracted fields like dates, parties, amounts
- NOT needed for open-ended conversational agents

### Common Patterns

**Classification Pattern** (like ticket-routing):
```yaml
response_format:
  type: object
  properties:
    category:
      type: string
      enum: [option1, option2, option3]
    confidence_score:
      type: number
      minimum: 0
      maximum: 1
    reasoning:
      type: string
  required: [category, confidence_score, reasoning]
  additionalProperties: false
```

**Analysis Pattern** (like content-moderation):
```yaml
response_format:
  type: object
  properties:
    decision:
      type: string
      enum: [approve, warning, remove]
    violations:
      type: array
      items:
        type: object
        properties:
          category: { type: string }
          severity: { type: string, enum: [low, medium, high] }
    reasoning:
      type: string
  required: [decision, violations, reasoning]
  additionalProperties: false
```

**Document Pattern** (like legal-summarization):
```yaml
response_format:
  type: object
  properties:
    summary:
      type: string
    document_type:
      type: string
    key_dates:
      type: array
      items: { type: string }
    parties:
      type: array
      items: { type: string }
  required: [summary, document_type]
  additionalProperties: false
```

### Questions to Ask

1. Does your agent need structured JSON output? (Y/N)
2. If yes: What fields should be in the response?
3. Are there enum values (fixed options) for any fields?
4. Which fields are required vs optional?

### Action

If needed, add `response_format` section to `agent.yaml`.

---

## Phase 5: Tools Configuration

### 5a. Vectorstore Tools (RAG)

Help the user configure knowledge base search.

#### Questions to Ask

1. Does this agent need to search a knowledge base? (Y/N)
2. What type of data will it search?
   - JSON data (structured records)
   - Markdown documentation
   - Mixed content
3. Do you have the data files ready, or should we create templates?
4. Should we collect data for you?
   - **Internet search** - Search the web and compile information into structured data
   - **File system** - Read and process existing files from your local system
   - **Other sources** - APIs, databases, or other data sources
   - **No** - I'll provide the data myself

#### For Each Vectorstore

Collect:
- **name**: Tool identifier (snake_case, e.g., `search_knowledge_base`)
- **description**: What does this tool search for? (Clear, action-oriented)
- **source**: Path to data file (e.g., `data/knowledge_base.json`)
- **top_k**: Number of results to return (default: 5)
- **chunk_size**: Text chunk size (default: 512)
- **chunk_overlap**: Overlap between chunks (default: 64)
- **min_similarity_score**: Relevance threshold (default: 0.7)

**For JSON files, also collect structured vector fields:**
- **id_field**: Field name for unique identifier (e.g., `id`)
- **vector_field**: Field name for content to vectorize (e.g., `content`, `description`)
- **meta_fields**: List of additional fields to return in search results

#### Embedding Models by Provider

| Provider | Embedding Model |
|----------|-----------------|
| openai | text-embedding-3-small |
| azure_openai | text-embedding-ada-002 |
| ollama | nomic-embed-text:latest |

#### Vectorstore Configuration Template

**For Markdown files:**
```yaml
tools:
  - type: vectorstore
    name: <tool_name>
    description: <what it searches>
    source: data/<filename>.md
    embedding_model: <embedding-model>
    database: chromadb
    top_k: 5
    chunk_size: 512
    chunk_overlap: 64
    min_similarity_score: 0.7
```

**For JSON files (with structured vector fields):**
```yaml
tools:
  - type: vectorstore
    name: <tool_name>
    description: <what it searches>
    source: data/<filename>.json
    embedding_model: <embedding-model>
    database: chromadb
    top_k: 5
    chunk_size: 512
    chunk_overlap: 64
    min_similarity_score: 0.7
    id_field: id
    vector_field: content
    meta_fields:
      - category
      - topic
      - <other_fields>
```

#### Help Create Data Files

If user needs help creating data, provide templates.

**IMPORTANT: JSON data must be FLAT (no nested objects)**. Each record should be a self-contained, independently searchable item. Deeply nested JSON is NOT supported.

**JSON Data Template** (`data/knowledge_base.json`):
```json
[
  {
    "id": "item-001",
    "category": "category-name",
    "topic": "Topic Title",
    "content": "Detailed searchable content. This field will be vectorized for semantic search. Include all relevant information in a single paragraph or block of text.",
    "metadata_field": "additional info returned in search results"
  },
  {
    "id": "item-002",
    "category": "another-category",
    "topic": "Another Topic",
    "content": "More searchable content. Keep each record focused on a single topic or concept for better retrieval quality.",
    "metadata_field": "more metadata"
  }
]
```

**Data Structure Guidelines:**
1. **Flat structure**: No nested objects or arrays (except simple string arrays)
2. **One concept per record**: Each JSON object should cover one topic/concept
3. **Descriptive content field**: The `vector_field` should contain comprehensive, searchable text
4. **Unique IDs**: Each record needs a unique identifier
5. **Metadata for context**: Include category, topic, or other fields for filtering/display

**Bad (nested):**
```json
{
  "categories": {
    "billing": {
      "items": [...]
    }
  }
}
```

**Good (flat):**
```json
[
  {"id": "billing-001", "category": "billing", "content": "..."},
  {"id": "billing-002", "category": "billing", "content": "..."}
]
```

### 5b. MCP Tools

Suggest relevant MCP servers based on use case.

#### MCP Recommendations by Use Case

**For Conversation/Memory**:
```yaml
- type: mcp
  name: memory
  description: Store and retrieve conversation context
  command: npx
  args: ["-y", "@modelcontextprotocol/server-memory"]
```

**For Web Search**:
```yaml
- type: mcp
  name: brave_search
  description: Search the web for current information
  command: npx
  args: ["-y", "@anthropic/mcp-server-brave-search"]
  env:
    BRAVE_API_KEY: ${BRAVE_API_KEY}
```

**For File Operations**:
```yaml
- type: mcp
  name: filesystem
  description: Read and write files
  command: npx
  args: ["-y", "@modelcontextprotocol/server-filesystem", "./data"]
```

#### Questions to Ask

1. Does the agent need persistent memory across conversations?
2. Does it need web search capabilities?
3. Does it need file system access?
4. Any other external integrations needed?

---

## Phase 6: Evaluation Metrics

Configure how the agent will be evaluated.

### RAG Metrics (if using vectorstore)

| Metric | Purpose | Recommended Threshold |
|--------|---------|----------------------|
| faithfulness | Response grounded in retrieved context | 0.7-0.8 |
| answer_relevancy | Response answers the question | 0.7-0.8 |
| contextual_relevancy | Retrieved context is relevant | 0.65-0.7 |
| contextual_precision | Top results are most relevant | 0.7 |
| contextual_recall | All relevant context retrieved | 0.6-0.7 |

**RAG Metric Template**:
```yaml
evaluations:
  metrics:
    - type: rag
      metric_type: faithfulness
      threshold: 0.75
      include_reason: true
```

### GEval Custom Metrics

Create custom LLM-as-judge evaluations.

**GEval Template**:
```yaml
- type: geval
  name: MetricName
  threshold: 0.75
  criteria: |
    Evaluate [what to evaluate].
    Consider:
    - [Criterion 1]
    - [Criterion 2]
    - [Criterion 3]
  evaluation_steps:
    - Analyze [aspect 1]
    - Check [aspect 2]
    - Verify [aspect 3]
  evaluation_params:
    - actual_output
    - expected_output
```

### Standard Metrics (for text comparison)

| Metric | Use Case | Threshold |
|--------|----------|-----------|
| rouge | Summary quality | 0.5-0.6 |
| bleu | Translation/extraction | 0.4-0.5 |
| f1_score | Classification accuracy | 0.7 |

### Questions to Ask

1. Is this a RAG-based agent? (I'll suggest faithfulness, answer_relevancy)
2. What custom criteria should the agent be evaluated on?
3. Are there specific quality dimensions to measure? (e.g., accuracy, helpfulness, professionalism)

### Action

Add evaluation configuration:

```yaml
evaluations:
  model:
    provider: <provider>
    name: <model>
    temperature: 0.0  # Evaluators should be deterministic
  metrics:
    - <metrics list>
```

---

## Phase 7: Test Cases

Create comprehensive test cases to validate the agent.

### Test Case Structure

```yaml
test_cases:
  - name: Descriptive test name
    input: "User query or input text"
    ground_truth: "Expected key information or answer"
    expected_tools:
      - tool_name_1
```

### Guidelines

1. Create **at least 5** diverse test cases covering:
   - Happy path scenarios (typical, expected inputs)
   - Edge cases (unusual but valid inputs)
   - Different input types/categories
   - Boundary conditions

2. For each test case, capture:
   - Realistic user input
   - Expected output or key facts (ground_truth)
   - Which tools should be invoked (expected_tools)

### Questions to Ask

1. What are 3-5 typical queries this agent will handle?
2. What are edge cases or tricky scenarios?
3. What would incorrect behavior look like?

### Action

Create test cases interactively, confirming each one with the user before adding to the configuration.

---

## Phase 8: Observability Configuration

Configure OpenTelemetry observability for monitoring.

### Default Configuration

```yaml
observability:
  enabled: true
  service_name: holodeck-<agent-name>
  traces:
    enabled: true
    sample_rate: 1.0
    capture_content: false  # Set true only for debugging
  metrics:
    enabled: true
    export_interval_ms: 5000
  logs:
    enabled: true
    level: INFO
  exporters:
    otlp:
      enabled: true
      endpoint: http://localhost:4317
      protocol: grpc
      insecure: true
```

### Questions to Ask

1. Should trace content be captured? (Note: may include sensitive data)
2. What log level? (DEBUG, INFO, WARNING, ERROR)
3. Is OTLP export to localhost:4317 correct for your setup? (Works with `./start-infra.sh`)

---

## Phase 9: Finalization

Complete the agent setup and validate.

### Checklist

1. Review the complete `agent.yaml` configuration
2. Ensure all required directories exist:
   - `instructions/`
   - `data/`
3. Create `.env.example` with required environment variables
4. Create `config.yaml` for provider-specific settings if needed

### Validation Command

```bash
holodeck test agent.yaml --verbose
```

### Next Steps to Suggest

After creation, recommend:
1. Add more data to the vectorstore sources
2. Run `holodeck chat agent.yaml` for interactive testing
3. Iterate on system prompt based on test results
4. Consider running `/holodeck.tune` to optimize performance
5. Start the frontend with `cd copilotkit && npm install && npm run dev`

---

## Reference: Existing Sample Patterns

Use these as reference when configuring the agent:

| Sample | Pattern | Key Features |
|--------|---------|--------------|
| ticket-routing | Classification | response_format with enums, GEval accuracy metrics |
| customer-support | Conversational | MCP memory tool, RAG metrics |
| content-moderation | Analysis | Violations array, consistency metrics |
| legal-summarization | Document | ROUGE/BLEU metrics, complex response schema |

Read the relevant sample files for specific implementation patterns:
- `<sample>/openai/agent.yaml`
- `<sample>/openai/instructions/system-prompt.md`
- `<sample>/openai/data/*.json`
