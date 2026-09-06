# Data Model: CLI & Core Agent Engine - US1 (Agent Configuration)

**Focus**: User Story 1 - Define Agent Configuration

## Entities & Fields

### Agent

Root configuration entity representing a single AI agent instance.

```python
class Agent:
    # Metadata
    name: str                           # Required. Agent identifier
    description: str | None             # Optional. Human-readable description

    # Model Configuration
    model: ModelConfig                  # Required. LLM provider + settings

    # Instructions
    instructions: Instructions           # Required. System prompt (file or inline)

    # Tools
    tools: List[Tool]                   # Optional (empty list if not provided)

    # Evaluations
    evaluations: EvaluationConfig | None # Optional. Metrics configuration

    # Test Cases
    test_cases: List[TestCase] | None   # Optional. Test scenarios
```

**Validation Rules**:
- `name`: 1-100 characters, alphanumeric + hyphens, must start with letter
- `description`: Max 500 characters
- `instructions.file` and `instructions.inline`: Mutually exclusive (exactly one required)
- `tools`: Max 50 items
- `test_cases`: Max 100 items

**State Transitions**: None (stateless configuration entity)

---

### ModelConfig

LLM provider and model settings.

```python
class ModelConfig:
    provider: str                       # Required. Enum: "openai" | "azure_openai" | "anthropic"
    name: str                           # Required. Model identifier (e.g., "gpt-4o", "gpt-4o-mini")
    temperature: float | None           # Optional. 0.0-2.0 (default per provider)
    max_tokens: int | None              # Optional. Max generation tokens
    top_p: float | None                 # Optional. Nucleus sampling
```

**Validation Rules**:
- `provider`: Must be one of supported enum values
- `name`: Non-empty string, validated against provider's known models
- `temperature`: Must be 0.0-2.0 if provided
- `max_tokens`: Must be positive if provided

**Relationships**:
- Used by: Agent (1:1), EvaluationMetric (1:1 per metric override)

---

### Instructions

System prompt specification (file or inline).

```python
class Instructions:
    file: str | None                    # Optional. Path to instruction file (relative to agent.yaml)
    inline: str | None                  # Optional. Inline instruction text
```

**Validation Rules**:
- **Exactly one of `file` or `inline` must be provided** (mutual exclusivity)
- `file`: Must be non-empty path string; file must exist relative to agent.yaml directory
- `inline`: Must be non-empty string, max 5000 characters

**Relationships**:
- Used by: Agent (1:1)

---

### Tool

Represents an agent capability. Union type with specific configurations per tool type.

```python
class Tool:
    name: str                           # Required. Tool identifier (unique within agent)
    description: str                    # Required. Human-readable description
    type: str                           # Required. Enum: "vectorstore" | "function" | "mcp" | "prompt"

    # Discriminator: type determines which fields are required
```

**Base Validation Rules**:
- `name`: 1-100 characters, alphanumeric + underscores, unique per agent
- `description`: Non-empty, max 500 characters
- `type`: Must be one of supported enum values

**Subtype Configurations**:

#### Tool (type: vectorstore)
```python
class VectorstoreTool(Tool):
    source: str                         # Required. Path to data file/directory
    embedding_model: str | None         # Optional. Embedding model name
    vector_field: str | List[str] | None  # Optional. Which field(s) to vectorize
    meta_fields: List[str] | None       # Optional. Metadata field names
    chunk_size: int | None              # Optional. Chunk size for text splitting
    chunk_overlap: int | None           # Optional. Chunk overlap
    record_path: str | None             # Optional. Path to array in JSON (nested access)
    record_prefix: str | None           # Optional. Prefix for record fields
    meta_prefix: str | None             # Optional. Prefix for metadata fields
```

**Validation Rules**:
- `source`: Must be non-empty path; file/directory must exist relative to agent.yaml
- `vector_field`: If not provided, auto-detect text fields
- `chunk_size`: If provided, must be > 0
- `record_path`: Dot notation supported (e.g., "data.records")

#### Tool (type: function)
```python
class FunctionTool(Tool):
    file: str                           # Required. Path to Python file
    function: str                       # Required. Function name in file
    parameters: Dict[str, ParamSchema] | None  # Optional. Parameter schema
```

**Validation Rules**:
- `file`: Must be non-empty path; file must exist relative to agent.yaml
- `function`: Must be valid Python identifier
- `parameters`: Each parameter has `type` (str, int, float, bool, array, object) and `description`

#### Tool (type: mcp)
```python
class MCPTool(Tool):
    server: str                         # Required. MCP server identifier
    config: Dict[str, Any] | None       # Optional. MCP-specific configuration
```

**Validation Rules**:
- `server`: Non-empty string (e.g., "@modelcontextprotocol/server-filesystem" or file path)
- `config`: Free-form dict (MCP server validates at runtime)

#### Tool (type: prompt)
```python
class PromptTool(Tool):
    template: str | None                # Optional. Inline prompt template
    file: str | None                    # Optional. Path to prompt file
    parameters: Dict[str, ParamSchema]  # Required. Parameter definitions
    model: ModelConfig | None           # Optional. Specific model for this tool
```

**Validation Rules**:
- **Exactly one of `template` or `file` must be provided**
- `template`: Non-empty string, max 5000 characters
- `file`: Must be non-empty path; file must exist relative to agent.yaml
- `parameters`: Required, at least one parameter
- `model`: Optional; if provided, overrides agent's model for this tool

**Relationships**:
- Used by: Agent (1:N)
- References: ModelConfig (1:1 for prompt tools only)

---

### EvaluationConfig

Evaluation framework configuration.

```python
class EvaluationConfig:
    model: ModelConfig | None           # Optional. Default model for all metrics
    metrics: List[EvaluationMetric]     # Required. List of metrics to compute
```

**Validation Rules**:
- `metrics`: Required, at least one metric

**Relationships**:
- Used by: Agent (1:1)
- Contains: EvaluationMetric (1:N)

---

### EvaluationMetric

Represents a single evaluation metric with flexible model configuration.

```python
class EvaluationMetric:
    metric: str                         # Required. Enum: "groundedness" | "relevance" | "coherence" | "safety" | "f1_score" | "bleu" | "rouge" | "meteor" | etc.
    threshold: float | None             # Optional. Minimum passing score (1-5 scale)
    enabled: bool                       # Optional, default=true
    scale: int | None                   # Optional. Score scale (e.g., 5 for 1-5 scale)
    model: ModelConfig | None           # Optional. Model override for this metric only
    fail_on_error: bool                 # Optional, default=false (soft failure)
    retry_on_failure: int | None        # Optional. Retry count on failure
    timeout_ms: int | None              # Optional. LLM call timeout
    custom_prompt: str | None           # Optional. Custom evaluation prompt
```

**Validation Rules**:
- `metric`: Must be valid metric name from supported list
- `threshold`: If provided, must be valid for metric's scale (e.g., 1-5 for AI metrics)
- `enabled`: Boolean (default true)
- `model`: Optional override; if provided, used instead of EvaluationConfig.model
- `fail_on_error`: Boolean (default false, implements soft failure from clarifications)
- `retry_on_failure`: If provided, must be 1-3
- `timeout_ms`: If provided, must be > 0

**Relationships**:
- Used by: EvaluationConfig (1:N)
- References: ModelConfig (1:1 optional)

---

### TestCase

Test scenario for agent validation.

```python
class TestCase:
    name: str | None                    # Optional. Test case identifier
    input: str                          # Required. User query/prompt
    expected_tools: List[str] | None    # Optional. Tools expected to be called
    ground_truth: str | None            # Optional. Expected output for comparison
    files: List[FileInput] | None       # Optional. Multimodal file inputs
    evaluations: List[str] | None       # Optional. Specific metrics to run for this test
```

**Validation Rules**:
- `input`: Non-empty string, max 5000 characters
- `expected_tools`: Each tool name must exist in agent.tools
- `ground_truth`: Max 5000 characters
- `name`: 1-100 characters, unique per agent
- `files`: Max 10 files per test case

**Relationships**:
- Used by: Agent (1:N)

---

### FileInput

Represents a file input for multimodal test cases.

```python
class FileInput:
    path: str | None                    # Optional. Local file path
    url: str | None                     # Optional. Remote URL
    type: str                           # Required. File type: "image" | "pdf" | "text" | "excel" | "word" | "powerpoint" | "csv"
    description: str | None             # Optional. File description
    pages: List[int] | None             # Optional. Specific pages/slides to extract
    sheet: str | None                   # Optional. Excel sheet name
    range: str | None                   # Optional. Excel cell range (e.g., "A1:E100")
    cache: bool | None                  # Optional. Cache remote files (default true)
```

**Validation Rules**:
- **Exactly one of `path` or `url` must be provided**
- `type`: Must be one of supported file types
- `pages`: If provided, must be positive integers
- `cache`: Boolean (default true for URLs)

**Relationships**:
- Used by: TestCase (1:N)

---

## Relationships Summary

```
Agent
├── 1:1  ModelConfig
├── 1:1  Instructions (file or inline)
├── 1:N  Tool (max 50)
│   ├── 1:1 ModelConfig (prompt tools only)
│   └── *(depends on tool type)*
├── 1:1  EvaluationConfig (optional)
│   └── 1:N EvaluationMetric
│       └── 1:1 ModelConfig (optional override)
└── 1:N TestCase (max 100, optional)
    ├── 1:N FileInput (max 10 per test)
    └── Expected tool refs → Tool
```

---

## Validation Order (Pydantic)

1. **Type validation**: Each field matches its declared type
2. **Mutual exclusivity**:
   - Instructions: `file` XOR `inline`
   - FileInput: `path` XOR `url`
   - PromptTool: `template` XOR `file`
3. **Reference validation**:
   - Tools referenced in TestCase.expected_tools must exist in Agent.tools
   - File paths must exist (relative to agent.yaml directory)
   - Environment variables in values must be resolvable
4. **Constraints**:
   - Name uniqueness (tools)
   - Value ranges (temperature, threshold, etc.)
   - Enum validation (provider, metric types, file types)

---

## SearchResult Type (From Clarifications)

Though not part of agent.yaml schema, the SearchResult type is used throughout tool execution for consistent vectorstore results:

```python
class SearchResult:
    matched_content: str                # The semantic match content
    metadata_dict: Dict[str, Any]       # Associated metadata fields
    source_reference: str               # File/record reference
    relevance_score: float              # Similarity score (0-1)
```

This type MUST be used consistently across all vectorstore tool implementations.
