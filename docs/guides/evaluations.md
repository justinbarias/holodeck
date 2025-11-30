# Evaluations Guide

This guide explains HoloDeck's evaluation system for measuring agent quality.

## Overview

Evaluations measure how well your agent performs. You define metrics in `agent.yaml` to automatically grade agent responses against test cases.

HoloDeck supports three categories of metrics (in order of recommendation):

1. **DeepEval Metrics (Recommended)** - LLM-as-a-judge with custom criteria (GEval) and RAG-specific metrics
2. **NLP Metrics (Standard)** - Text comparison algorithms (F1, BLEU, ROUGE, METEOR)
3. **Legacy AI Metrics (Deprecated)** - Azure AI-based metrics (groundedness, relevance, coherence, safety)

## Basic Structure

```yaml
evaluations:
  model:      # Optional: Default LLM for evaluation
    provider: ollama
    name: llama3.2:latest
    temperature: 0.0

  metrics:    # Required: Metrics to compute
    # DeepEval GEval metric (recommended)
    - type: geval
      name: "Response Quality"
      criteria: "Evaluate if the response is helpful and accurate"
      threshold: 0.7

    # DeepEval RAG metric
    - type: rag
      metric_type: answer_relevancy
      threshold: 0.7
```

## Configuration Levels

Model configuration for evaluations works at three levels (priority order):

### Level 1: Per-Metric Override (Highest Priority)

Override model for a specific metric:

```yaml
evaluations:
  metrics:
    - type: geval
      name: "Critical Metric"
      criteria: "..."
      model:                    # Uses this model for this metric only
        provider: openai
        name: gpt-4
```

### Level 2: Evaluation-Wide Model

Default for all metrics without override:

```yaml
evaluations:
  model:                        # Uses for all metrics
    provider: ollama
    name: llama3.2:latest

  metrics:
    - type: geval
      name: "Coherence"
      criteria: "..."
      # Uses evaluation.model above
    - type: rag
      metric_type: faithfulness
      # Also uses evaluation.model above
```

### Level 3: Agent Model (Lowest Priority)

Used if neither Level 1 nor Level 2 specified:

```yaml
model:                          # Agent's main model
  provider: openai
  name: gpt-4o

evaluations:
  metrics:
    - type: geval
      name: "Quality"
      criteria: "..."
      # Falls back to agent.model above
```

---

## DeepEval Metrics (Recommended)

DeepEval provides powerful LLM-as-a-judge evaluation with two metric types:

- **GEval**: Custom criteria evaluation using chain-of-thought prompting
- **RAG Metrics**: Specialized metrics for retrieval-augmented generation pipelines

### Why DeepEval?

- **Flexible**: Define custom evaluation criteria in natural language
- **Local Models**: Works with Ollama for free, local evaluation
- **RAG-Focused**: Purpose-built metrics for RAG pipeline evaluation
- **Chain-of-Thought**: Uses G-Eval algorithm for more accurate scoring

### Supported Providers

```yaml
model:
  provider: ollama        # Free, local inference (recommended for development)
  # provider: openai      # OpenAI API
  # provider: anthropic   # Anthropic API
  # provider: azure_openai # Azure OpenAI
  name: llama3.2:latest
  temperature: 0.0        # Use 0 for deterministic evaluation
```

---

### GEval Metrics

GEval uses the G-Eval algorithm with chain-of-thought prompting to evaluate responses against custom criteria.

#### Basic Configuration

```yaml
- type: geval
  name: "Coherence"
  criteria: "Evaluate whether the response is clear, well-structured, and easy to understand."
  threshold: 0.7
```

#### Full Configuration

```yaml
- type: geval
  name: "Technical Accuracy"
  criteria: |
    Evaluate whether the response provides accurate technical information
    that correctly addresses the user's question.
  evaluation_steps:              # Optional: Auto-generated if omitted
    - "Check if the response directly addresses the user's question"
    - "Verify technical accuracy of any code or commands provided"
    - "Ensure explanations are correct and not misleading"
  evaluation_params:             # Which test case fields to use
    - actual_output              # Required: Agent's response
    - input                      # Optional: User's query
    - expected_output            # Optional: Ground truth
    - context                    # Optional: Additional context
    - retrieval_context          # Optional: Retrieved documents
  threshold: 0.8
  strict_mode: false             # Binary scoring (1.0 or 0.0) when true
  enabled: true
  fail_on_error: false
  model:                         # Optional: Per-metric model override
    provider: openai
    name: gpt-4
```

#### GEval Configuration Options

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Must be `"geval"` |
| `name` | string | Yes | Custom metric name (e.g., "Coherence", "Helpfulness") |
| `criteria` | string | Yes | Natural language evaluation criteria |
| `evaluation_steps` | list | No | Step-by-step evaluation instructions (auto-generated if omitted) |
| `evaluation_params` | list | No | Test case fields to use (default: `["actual_output"]`) |
| `threshold` | float | No | Minimum passing score (0-1) |
| `strict_mode` | bool | No | Binary scoring when true (default: false) |
| `enabled` | bool | No | Enable/disable metric (default: true) |
| `fail_on_error` | bool | No | Fail test on evaluation error (default: false) |
| `model` | object | No | Per-metric model override |

#### Evaluation Parameters

| Parameter | Description | When to Use |
|-----------|-------------|-------------|
| `actual_output` | Agent's response | Always (required for evaluation) |
| `input` | User's query/question | When relevance to query matters |
| `expected_output` | Ground truth answer | When comparing to expected response |
| `context` | Additional context provided | When evaluating context usage |
| `retrieval_context` | Retrieved documents | For RAG pipeline evaluation |

#### GEval Examples

**Coherence Check:**
```yaml
- type: geval
  name: "Coherence"
  criteria: "Evaluate whether the response is clear, well-structured, and easy to understand."
  evaluation_steps:
    - "Evaluate whether the response uses clear and direct language."
    - "Check if the explanation avoids jargon or explains it when used."
    - "Assess whether complex ideas are presented in a way that's easy to follow."
  evaluation_params:
    - actual_output
  threshold: 0.7
```

**Helpfulness Check:**
```yaml
- type: geval
  name: "Helpfulness"
  criteria: |
    Evaluate whether the response provides actionable, practical help
    that addresses the user's needs.
  evaluation_params:
    - actual_output
    - input
  threshold: 0.75
```

**Factual Accuracy:**
```yaml
- type: geval
  name: "Factual Accuracy"
  criteria: |
    Evaluate whether the response is factually accurate when compared
    to the expected answer and provided context.
  evaluation_params:
    - actual_output
    - expected_output
    - context
  threshold: 0.85
  strict_mode: true  # Binary pass/fail
```

---

### RAG Metrics

RAG (Retrieval-Augmented Generation) metrics evaluate the quality of responses generated using retrieved context.

#### Available RAG Metrics

| Metric Type | Purpose | Required Parameters |
|-------------|---------|-------------------|
| `faithfulness` | Detects hallucinations | input, actual_output, retrieval_context |
| `answer_relevancy` | Response relevance to query | input, actual_output |
| `contextual_relevancy` | Retrieved chunks relevance | input, actual_output, retrieval_context |
| `contextual_precision` | Chunk ranking quality | input, actual_output, expected_output, retrieval_context |
| `contextual_recall` | Retrieval completeness | input, actual_output, expected_output, retrieval_context |

#### Basic Configuration

```yaml
- type: rag
  metric_type: faithfulness
  threshold: 0.8

- type: rag
  metric_type: answer_relevancy
  threshold: 0.7
```

#### Full Configuration

```yaml
- type: rag
  metric_type: faithfulness
  threshold: 0.8
  include_reason: true           # Include reasoning in results
  enabled: true
  fail_on_error: false
  model:                         # Optional: Per-metric model override
    provider: openai
    name: gpt-4
```

#### RAG Metric Details

**Faithfulness** - Detects hallucinations by comparing response to retrieval context:
```yaml
- type: rag
  metric_type: faithfulness
  threshold: 0.8
  include_reason: true
```
- **What it measures**: Whether claims in the response are supported by retrieved documents
- **When to use**: Critical for factual accuracy in RAG pipelines
- **Example**: Agent says "The product costs $99" - faithfulness checks if this is in the retrieved context

**Answer Relevancy** - Measures response relevance to the query:
```yaml
- type: rag
  metric_type: answer_relevancy
  threshold: 0.7
```
- **What it measures**: How well the response addresses the user's question
- **When to use**: General quality assurance for any agent
- **Example**: User asks "How do I reset my password?" - checks if response actually explains password reset

**Contextual Relevancy** - Measures relevance of retrieved chunks:
```yaml
- type: rag
  metric_type: contextual_relevancy
  threshold: 0.75
```
- **What it measures**: Whether retrieved documents are relevant to the query
- **When to use**: Diagnosing retrieval quality issues

**Contextual Precision** - Evaluates chunk ranking quality:
```yaml
- type: rag
  metric_type: contextual_precision
  threshold: 0.8
```
- **What it measures**: Whether the most relevant chunks are ranked highest
- **When to use**: Optimizing retrieval ranking algorithms

**Contextual Recall** - Measures retrieval completeness:
```yaml
- type: rag
  metric_type: contextual_recall
  threshold: 0.7
```
- **What it measures**: Whether all information needed for the expected answer was retrieved
- **When to use**: Ensuring comprehensive retrieval coverage

---

## NLP Metrics (Standard)

NLP metrics compare response text to expected output using algorithms. They're fast, free (no LLM calls), and deterministic.

### F1 Score

Measures precision and recall of token overlap.

```yaml
- type: standard
  metric: f1_score
  threshold: 0.8
```

**Scale**: 0.0-1.0 (higher is better)

**What it measures**:
- Token-level match with ground truth
- Balanced precision/recall

**When to use**: When exact word matching is important

### BLEU (Bilingual Evaluation Understudy)

Measures n-gram overlap with reference translation.

```yaml
- type: standard
  metric: bleu
  threshold: 0.6
```

**Scale**: 0.0-1.0 (higher is better)

**What it measures**:
- N-gram similarity to reference
- Penalizes brevity

**When to use**: For translation, paraphrase evaluation

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

Measures recall of n-grams with reference.

```yaml
- type: standard
  metric: rouge
  threshold: 0.7
```

**Scale**: 0.0-1.0 (higher is better)

**What it measures**:
- Recall of n-grams
- Coverage of reference content

**When to use**: For summarization tasks

### METEOR (Metric for Evaluation of Translation with Explicit Ordering)

Similar to BLEU but with better handling of synonyms.

```yaml
- type: standard
  metric: meteor
  threshold: 0.65
```

**Scale**: 0.0-1.0 (higher is better)

**What it measures**:
- N-gram match with synonyms
- Word order

**When to use**: For translation, paraphrase with synonyms

---

## Legacy AI Metrics (Deprecated)

> **DEPRECATED**: Azure AI-based metrics (groundedness, relevance, coherence, safety) are deprecated and will be removed in a future version.
>
> **Migration Guide**:
> - For custom criteria evaluation → Use `type: geval` with natural language criteria
> - For RAG hallucination detection → Use `type: rag` with `metric_type: faithfulness`
> - For relevance checking → Use `type: rag` with `metric_type: answer_relevancy`

These legacy metrics are still supported for backwards compatibility:

### Groundedness (Deprecated)

```yaml
# DEPRECATED - Use type: rag with metric_type: faithfulness instead
- type: standard
  metric: groundedness
  threshold: 0.8
```

**Migration**: Replace with:
```yaml
- type: rag
  metric_type: faithfulness
  threshold: 0.8
```

### Relevance (Deprecated)

```yaml
# DEPRECATED - Use type: rag with metric_type: answer_relevancy instead
- type: standard
  metric: relevance
  threshold: 0.75
```

**Migration**: Replace with:
```yaml
- type: rag
  metric_type: answer_relevancy
  threshold: 0.75
```

### Coherence (Deprecated)

```yaml
# DEPRECATED - Use type: geval with custom criteria instead
- type: standard
  metric: coherence
  threshold: 0.7
```

**Migration**: Replace with:
```yaml
- type: geval
  name: "Coherence"
  criteria: "Evaluate whether the response is clear, well-structured, and easy to understand."
  threshold: 0.7
```

### Safety (Deprecated)

```yaml
# DEPRECATED - Use type: geval with custom criteria instead
- type: standard
  metric: safety
  threshold: 0.9
```

**Migration**: Replace with:
```yaml
- type: geval
  name: "Safety"
  criteria: |
    Evaluate whether the response is safe and appropriate:
    - No harmful or dangerous content
    - No PII leakage
    - Appropriate tone and language
  threshold: 0.9
```

---

## Metric Configuration Options

### Threshold

- **Type**: Float
- **Purpose**: Minimum score for test to pass
- **Scale**: 0-1 for all metrics
- **Optional**: Yes (default: no threshold, metric is informational)

```yaml
- type: geval
  name: "Quality"
  criteria: "..."
  threshold: 0.8
```

### Enabled

- **Type**: Boolean
- **Default**: `true`
- **Purpose**: Temporarily disable metric without removing it

```yaml
- type: rag
  metric_type: answer_relevancy
  enabled: false  # Metric runs but doesn't fail test
```

### Fail on Error

- **Type**: Boolean
- **Default**: `false` (soft failure)
- **Purpose**: Whether to fail test if evaluation errors

```yaml
- type: geval
  name: "Quality"
  criteria: "..."
  fail_on_error: false  # Continues even if LLM evaluation fails
```

---

## Complete Examples

### Basic DeepEval Setup

```yaml
evaluations:
  model:
    provider: ollama
    name: llama3.2:latest
    temperature: 0.0

  metrics:
    - type: geval
      name: "Coherence"
      criteria: "Evaluate whether the response is clear and well-structured."
      threshold: 0.7

    - type: rag
      metric_type: answer_relevancy
      threshold: 0.7
```

### RAG Pipeline Evaluation

```yaml
evaluations:
  model:
    provider: ollama
    name: llama3.2:latest
    temperature: 0.0

  metrics:
    # Detect hallucinations
    - type: rag
      metric_type: faithfulness
      threshold: 0.85
      include_reason: true

    # Check response relevance
    - type: rag
      metric_type: answer_relevancy
      threshold: 0.75

    # Evaluate retrieval quality
    - type: rag
      metric_type: contextual_relevancy
      threshold: 0.7

    # Check retrieval completeness
    - type: rag
      metric_type: contextual_recall
      threshold: 0.7
```

### Mixed Metrics (DeepEval + NLP)

```yaml
evaluations:
  model:
    provider: ollama
    name: llama3.2:latest
    temperature: 0.0

  metrics:
    # DeepEval metrics (primary)
    - type: geval
      name: "Response Quality"
      criteria: "Evaluate if the response is helpful and accurate."
      threshold: 0.75

    - type: rag
      metric_type: faithfulness
      threshold: 0.8

    # NLP metrics (secondary)
    - type: standard
      metric: f1_score
      threshold: 0.7

    - type: standard
      metric: rouge
      threshold: 0.6
```

### Enterprise Setup with Model Overrides

```yaml
evaluations:
  model:
    provider: ollama
    name: llama3.2:latest  # Default: free, local
    temperature: 0.0

  metrics:
    # Critical metrics - use powerful model
    - type: rag
      metric_type: faithfulness
      threshold: 0.9
      model:
        provider: openai
        name: gpt-4

    - type: geval
      name: "Safety"
      criteria: "Evaluate response safety and appropriateness."
      threshold: 0.95
      model:
        provider: openai
        name: gpt-4

    # Standard metrics - use default local model
    - type: geval
      name: "Coherence"
      criteria: "Evaluate response clarity."
      threshold: 0.75

    - type: rag
      metric_type: answer_relevancy
      threshold: 0.7

    # NLP metrics - no LLM needed
    - type: standard
      metric: f1_score
      threshold: 0.7
```

### Per-Test Case Evaluation

Test cases can specify which metrics to run:

```yaml
test_cases:
  - name: "Fact check test"
    input: "What's our company's founding date?"
    expected_tools: [search_kb]
    ground_truth: "Founded in 2010"
    evaluations:
      - type: rag
        metric_type: faithfulness
        threshold: 0.85
      - type: rag
        metric_type: answer_relevancy
        threshold: 0.7

  - name: "Creative task"
    input: "Generate a company tagline"
    evaluations:
      - type: geval
        name: "Creativity"
        criteria: "Evaluate if the tagline is creative and memorable."
        threshold: 0.7
      # Skip faithfulness since no retrieval context
```

---

## Test Execution

When running tests, HoloDeck:

1. Executes agent with test input
2. Records which tools were called
3. Validates tool usage (`expected_tools`)
4. Runs each enabled metric
5. Compares results against thresholds
6. Reports pass/fail per metric

### Example Output

```
Test: "Password Reset"
Input: "How do I reset my password?"
Tools called: [search_kb] ✓
Metrics:
  ✓ Faithfulness: 0.92 (threshold: 0.8)
  ✓ Answer Relevancy: 0.88 (threshold: 0.75)
  ✓ Coherence: 0.85 (threshold: 0.7)
  ✓ F1 Score: 0.81 (threshold: 0.7)
Result: PASS
```

---

## Cost Optimization

### Use Local Models for Development

```yaml
evaluations:
  model:
    provider: ollama           # Free, local
    name: llama3.2:latest
```

### Use Paid Models Only for Critical Metrics

```yaml
evaluations:
  model:
    provider: ollama           # Default: free
    name: llama3.2:latest

  metrics:
    - type: rag
      metric_type: faithfulness
      model:
        provider: openai
        name: gpt-4            # Expensive override only for critical metric
```

### Use NLP Metrics When Possible

NLP metrics are free (no LLM calls):

```yaml
- type: standard
  metric: f1_score  # No LLM cost
- type: standard
  metric: rouge     # No LLM cost
```

---

## Model Configuration Details

When specifying a model for evaluation:

```yaml
model:
  provider: ollama|openai|azure_openai|anthropic  # Required
  name: model-identifier                          # Required
  temperature: 0.0-2.0                            # Optional (recommend 0.0 for evaluation)
  max_tokens: integer                             # Optional
  top_p: 0.0-1.0                                  # Optional
```

### Provider-Specific Models

**Ollama (Recommended for Development)**
- `llama3.2:latest` - Fast, capable
- `llama3.1:latest` - More capable
- Any model available in your Ollama installation

**OpenAI**
- `gpt-4o` - Latest, best quality
- `gpt-4o-mini` - Fast, cheap
- `gpt-4-turbo` - Previous generation

**Azure OpenAI**
- `gpt-4` - Standard
- `gpt-4-32k` - Extended context

**Anthropic**
- `claude-3-opus` - Most capable
- `claude-3-sonnet` - Balanced
- `claude-3-haiku` - Fast, cheap

### Recommended Settings for Evaluation

```yaml
model:
  provider: ollama
  name: llama3.2:latest
  temperature: 0.0  # Deterministic for consistency
```

---

## Troubleshooting

### Error: "invalid metric type"

- Check metric type is valid
- Valid types: `geval`, `rag`, `standard`
- For standard metrics: f1_score, bleu, rouge, meteor
- For RAG metrics: faithfulness, answer_relevancy, contextual_relevancy, contextual_precision, contextual_recall

### Metric always fails

- Check evaluation model is working
- Try without threshold first
- Test evaluation model manually
- For RAG metrics, ensure required parameters are available

### LLM evaluation too slow

- Use local Ollama model instead of API
- Use faster model: `gpt-4o-mini` instead of `gpt-4`
- Use NLP metrics instead (free and fast)

### Inconsistent evaluation results

- Set temperature to 0.0 for deterministic results
- Use more powerful model for complex evaluations
- Add `evaluation_steps` to GEval for better consistency

### RAG metric missing retrieval_context

- Ensure your agent uses a vectorstore tool
- The test runner automatically extracts retrieval_context from tool results
- Or provide manual retrieval_context in test case

---

## Best Practices

1. **Start with DeepEval**: Use GEval and RAG metrics as primary evaluation
2. **Use Local Models**: Start with Ollama for free development, upgrade for production
3. **Mix Metric Types**: Combine DeepEval (semantic) with NLP (keyword-based)
4. **Cost-Aware**: Use cheaper/local models by default, expensive models only for critical metrics
5. **Realistic Thresholds**: Set thresholds based on actual agent performance
6. **Monitor**: Run metrics on sample of tests first
7. **Iterate**: Adjust thresholds and metrics based on results
8. **Migrate from Legacy**: Replace deprecated Azure AI metrics with DeepEval equivalents

---

## Next Steps

- See [Agent Configuration Guide](agent-configuration.md) for how to set up evaluations
- See [Examples](../examples/) for complete evaluation configurations
- See [Global Configuration](global-config.md) for shared settings
- See [API Reference](../api/evaluators.md) for evaluator class details
