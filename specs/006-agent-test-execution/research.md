# Research: Test Execution Framework Technologies

**Date**: 2025-11-01
**Feature**: Execute Agent Against Test Cases
**Phase**: 0 - Research

## Overview

This document consolidates technology decisions for implementing the `holodeck test` command. All decisions align with the project constitution and leverage Microsoft's ecosystem for consistency.

## Technology Decisions

### 1. Agent Execution: Semantic Kernel

**Decision**: Use Semantic Kernel Agents module for LLM interactions

**Rationale**:

- Native Azure integration and Microsoft support
- Clean agent abstractions with built-in orchestration
- Support for tool call tracking and capture
- Timeout and retry configuration
- Aligns with existing Microsoft tooling (Azure AI Evaluation, markitdown)

**Installation**: `semantic-kernel[azure]>=1.37.0`

**Key Capabilities**:

- Agent invocation with ChatHistory context
- Tool call tracking and validation
- Configurable timeouts (default: 60s for LLM calls)
- Exponential backoff retry (up to 3 attempts)
- Multi-agent orchestration patterns

**Integration Pattern**:

```python
from semantic_kernel.agents import Agent
from semantic_kernel import Kernel

# Create kernel and agent
kernel = Kernel()
agent = Agent(kernel=kernel, name="test_agent")

# Execute with context
chat_history = ChatHistory()
chat_history.add_user_message(test_input)
response = await agent.invoke(chat_history)

# Capture tool calls
tool_calls = response.tool_calls  # For validation against expected_tools
```

**Alternatives Considered**:

- LangChain: More complex API, less Azure-native
- Direct OpenAI/Anthropic SDKs: Would require custom orchestration layer

---

### 2. File Processing: markitdown

**Decision**: Use Microsoft's markitdown for unified multimodal file processing

**Rationale**:

- Single library handles all file types (PDF, Office, images, etc.)
- Converts to markdown (optimal for LLM consumption)
- Official Microsoft tool with ongoing support
- Simple API with stream processing for large files
- Built-in LLM integration for image descriptions

**Installation**: `markitdown[all]>=0.1.0`

**Supported Formats**: DOCX, XLSX, PPTX, PDF, JPG, PNG, HTML, JSON, CSV, Audio (transcription)

**Integration Pattern**:

```python
from markitdown import MarkItDown

md = MarkItDown()
result = md.convert(file_path)
markdown_content = result.text_content

# For large files with streaming
with open(large_file, 'rb') as f:
    result = md.convert_stream(f)
```

**File-Specific Features**:

- **Excel**: Extract specific sheets/ranges via preprocessing
- **PowerPoint**: Extract specific slides via preprocessing
- **PDF**: Page extraction via preprocessing (pages parameter from TestCaseModel)
- **Images**: OCR + LLM-based image description
- **Large files**: Stream processing with configurable timeout (30s default)

**Alternatives Considered**:

- textract: Less maintained, complex dependencies
- Apache Tika: JVM requirement, heavier footprint
- Per-format libraries (PyPDF2, python-docx, etc.): Would require managing 5+ libraries

---

### 3. AI Evaluation: Azure AI Evaluation SDK

**Decision**: Use Azure AI Evaluation SDK for AI-powered metrics

**Rationale**:

- Standardized metrics backed by research
- Per-metric model configuration (Constitution Principle V)
- Official Microsoft implementation
- Built-in retry and error handling
- Aligns with Azure AI Foundry ecosystem

**Installation**: `azure-ai-evaluation>=1.0.0`

**Built-in Evaluators**:

- `GroundednessEvaluator`: Verify factual accuracy against ground truth
- `RelevanceEvaluator`: Measure response relevance to query
- `CoherenceEvaluator`: Assess logical flow and consistency
- `FluencyEvaluator`: Check language quality
- `SimilarityEvaluator`: Compare semantic similarity
- `ToolCallAccuracyEvaluator`: Validate tool selection (for expected_tools)

**Integration Pattern**:

```python
from azure.ai.evaluation import (
    GroundednessEvaluator,
    RelevanceEvaluator,
    evaluate
)

# Per-metric model configuration (tiered approach)
groundedness_eval = GroundednessEvaluator(model_config={
    "azure_deployment": "gpt-4o"  # Critical metric: use expensive model
})

relevance_eval = RelevanceEvaluator(model_config={
    "azure_deployment": "gpt-4o-mini"  # Less critical: use cheaper model
})

# Evaluate response
groundedness_score = groundedness_eval(
    response=agent_response,
    context=ground_truth
)
```

**Cost Optimization Strategy**:

- GPT-4o for critical metrics (groundedness, safety)
- GPT-4o-mini for general metrics (relevance, coherence)
- User-configurable per EvaluationMetric.model override

**Alternatives Considered**:

- Custom prompt-based evaluation: Less standardized, no research backing
- LangChain evaluators: Not Azure-native, less maintained

---

### 4. NLP Metrics: Hugging Face `evaluate`

**Decision**: Use Hugging Face `evaluate` library for NLP metrics

**Rationale**:

- Unified API for all NLP metrics (BLEU, ROUGE, F1, METEOR)
- Modern, actively maintained
- Includes SacreBLEU (standard BLEU implementation)
- No LLM API calls required (Constitution Principle V)
- Better than NLTK for production use

**Installation**: `evaluate>=0.4.0`, `sacrebleu>=2.3.0`

**Supported Metrics**:

- **BLEU/SacreBLEU**: Machine translation quality
- **ROUGE**: Summarization quality (ROUGE-1, ROUGE-2, ROUGE-L)
- **METEOR**: MT evaluation with synonyms and stemming
- **F1 Score**: Precision-recall harmonic mean
- **BERTScore**: Contextual embedding similarity

**Integration Pattern**:

```python
import evaluate

# Load metrics (cached after first use)
bleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

# Calculate scores
bleu_score = bleu.compute(
    predictions=[agent_response],
    references=[ground_truth]
)

rouge_scores = rouge.compute(
    predictions=[agent_response],
    references=[ground_truth]
)
```

**Alternatives Considered**:

- NLTK: Older implementation, less maintained
- Direct implementation: Would require research and validation

---

## Integration Architecture

### Component Stack

```
┌─────────────────────────────────────────┐
│     CLI Layer (holodeck test)           │
│     - Parse arguments                   │
│     - Load agent.yaml                   │
│     - Display progress                  │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Test Execution Engine (executor.py)    │
│  - Sequential test execution            │
│  - Timeout management (30s/60s/30s)     │
│  - Retry logic (3x exponential backoff) │
└──────────────┬──────────────────────────┘
               │
         ┌─────┴─────┐
         │           │
┌────────▼──────┐  ┌─▼────────────────────┐
│ File Processor│  │ Agent Executor       │
│ (markitdown)  │  │ (Semantic Kernel)    │
│               │  │                      │
│ - Convert to  │  │ - Invoke agent       │
│   markdown    │  │ - Capture response   │
│ - Cache files │  │ - Track tool calls   │
│ - Handle      │  │ - Handle timeouts    │
│   ranges/     │  └──────────────────────┘
│   pages       │
└───────────────┘
         │
┌────────▼───────────────────────────────┐
│  Evaluation Engine                     │
│  ┌─────────────────┬─────────────────┐ │
│  │ Azure AI Eval   │ NLP Metrics     │ │
│  │ - Groundedness  │ - BLEU/ROUGE    │ │
│  │ - Relevance     │ - METEOR        │ │
│  │ - Coherence     │ - F1 Score      │ │
│  │ - Tool accuracy │                 │ │
│  └─────────────────┴─────────────────┘ │
└────────────────────────────────────────┘
         │
┌────────▼──────────────────────────────┐
│  Reporter (reporter.py)               │
│  - JSON output                        │
│  - Markdown output                    │
│  - Summary statistics                 │
└───────────────────────────────────────┘
```

### Configuration Flow

1. **Load agent.yaml** → Parse with existing ConfigLoader
2. **Validate schema** → Use existing ConfigValidator
3. **Extract test cases** → TestCaseModel instances
4. **Process files** → markitdown conversion with caching (.holodeck/cache/)
5. **Execute agent** → Semantic Kernel invocation with timeout
6. **Evaluate metrics** → Azure AI + NLP metric calculation with retry
7. **Generate report** → JSON/Markdown with summary

---

## Error Handling Strategy

### Retry Configuration

**File Downloads** (remote URLs):

- Timeout: 30s (configurable)
- Retry: 3 attempts with exponential backoff (1s, 2s, 4s)
- Fallback: Skip file, warn user, continue test

**File Processing** (markitdown):

- Timeout: 30s per file (configurable)
- Error: Log warning, attempt to process remaining files
- Large files (>100MB): Warn user but attempt processing

**LLM API Calls** (Semantic Kernel):

- Timeout: 60s (configurable)
- Retry: 3 attempts with exponential backoff (2s, 4s, 8s)
- Fallback: Mark test as failed with error details

**Evaluation Metrics** (Azure AI Evaluation):

- Timeout: 60s per metric (inherited from LLM timeout)
- Retry: 3 attempts with exponential backoff
- Fallback: Mark metric as "ERROR" but continue with other metrics

### Error Message Format

Structured errors with context (per Clarifications):

```
ERROR: Test case "What are your business hours?" failed
  Cause: LLM API timeout after 60s
  Metric: groundedness
  Suggestion: Check API rate limits or increase timeout via --llm-timeout flag
  File: agent.yaml:15
```

---

## Performance Considerations

### Caching Strategy

**Remote Files**:

- Location: `.holodeck/cache/{hash(url)}`
- TTL: No expiration (manual cleanup)
- Cache key: URL + file size + last-modified header

**markitdown Results**:

- Location: In-memory only (no disk cache for conversions)
- Rationale: File content may change, conversions are fast (<1s for most files)

**Evaluation Metrics**:

- No caching (each test run is independent)

### Batch Processing

**Sequential Execution** (v1):

- One test at a time to simplify error handling and progress display
- Parallel execution deferred to future version

**Batch Metric Evaluation**:

- Azure AI Evaluation SDK supports batch evaluation
- Defer to Phase 3 for optimization

---

## Cost Optimization

### Tiered Model Approach

**Critical Metrics** (GPT-4o):

- Groundedness (factual accuracy)
- Safety (harmful content)
- Tool call accuracy (expected_tools validation)

**General Metrics** (GPT-4o-mini):

- Relevance
- Coherence
- Fluency
- Similarity

**No LLM Metrics** (free):

- BLEU, ROUGE, METEOR, F1 (via `evaluate` library)

**User Override**:

```yaml
evaluations:
  metrics:
    - metric: groundedness
      model:
        provider: azure_openai
        name: gpt-4o # Override: use expensive model for critical metric
    - metric: relevance
      # Uses global default (gpt-4o-mini) if not specified
```

---

## Dependencies Summary

Add to `pyproject.toml`:

```toml
[tool.poetry.dependencies]
# Existing dependencies
python = "^3.10"
click = "^8.0.0"
pydantic = "^2.0.0"
pyyaml = "^6.0.0"
requests = "^2.32.0"

# NEW: Test execution framework
semantic-kernel = {extras = ["azure"], version = "^1.37.0"}
markitdown = {extras = ["all"], version = "^0.1.0"}
azure-ai-evaluation = "^1.0.0"
evaluate = "^0.4.0"
sacrebleu = "^2.3.0"
aiofiles = "^23.0.0"  # For async file operations
```

---

## Risk Mitigation

### Risk: Azure AI Evaluation SDK Rate Limits

**Mitigation**:

- Implement exponential backoff (3 retries)
- Add configurable retry delay
- Graceful degradation (mark metric as ERROR, continue execution)
- Consider metric batching in Phase 3

### Risk: markitdown File Processing Failures

**Mitigation**:

- Timeout per file (30s default, configurable)
- Warn on large files (>100MB)
- Continue test execution if file fails to process
- Log detailed error messages with file path

### Risk: Semantic Kernel Version Compatibility

**Mitigation**:

- Pin to semantic-kernel>=1.37.0 (current stable)
- Monitor release notes for breaking changes
- Comprehensive integration tests for agent execution
- Abstract agent_bridge.py module for easy swapping

### Risk: Token Context Window Limits

**Mitigation**:

- Warn users about large files before processing
- Support page/sheet/range extraction to limit content
- markitdown streaming for large files
- Document recommended file size limits (<10MB per file)

---

## Implementation Phases

### Phase 1: Core Execution (Weeks 1-2)

- Semantic Kernel agent execution
- Basic markitdown file preprocessing
- Response and tool call capture
- Sequential test execution

### Phase 2: Evaluation Metrics (Weeks 3-4)

- Azure AI Evaluation SDK integration (groundedness, relevance)
- NLP metrics (BLEU, ROUGE, METEOR)
- Per-metric model configuration
- Tool call validation

### Phase 3: Advanced Features (Weeks 5-6)

- File caching (.holodeck/cache/)
- Page/sheet/range extraction
- Retry and timeout configuration
- Batch metric evaluation (optimization)

### Phase 4: Polish & Production (Weeks 7-8)

- Comprehensive error handling
- Progress indicators and CI/CD output
- Report generation (JSON/Markdown)
- Performance tuning and optimization

---

## References

- [Semantic Kernel Documentation](https://learn.microsoft.com/en-us/semantic-kernel/)
- [markitdown GitHub](https://github.com/microsoft/markitdown)
- [Azure AI Evaluation SDK](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/evaluations)
- [Hugging Face evaluate](https://huggingface.co/docs/evaluate)

**Detailed Research**: See `./research/test-execution-integration-research.md` for comprehensive technical research (7,200+ words, code examples, integration patterns).
