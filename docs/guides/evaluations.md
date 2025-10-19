# Evaluations Guide

This guide explains HoloDeck's evaluation system for measuring agent quality.

## Overview

Evaluations measure how well your agent performs. You define metrics in `agent.yaml` to automatically grade agent responses against test cases.

HoloDeck supports two categories of metrics:

1. **AI-Powered Metrics** - LLM-evaluated (groundedness, relevance, coherence, safety)
2. **NLP Metrics** - Text comparison (F1, BLEU, ROUGE, METEOR)

## Basic Structure

```yaml
evaluations:
  model:      # Optional: Default LLM for evaluation
    provider: openai
    name: gpt-4o

  metrics:    # Required: Metrics to compute
    - metric: groundedness
      threshold: 0.8
      enabled: true
```

## Configuration Levels

Model configuration for evaluations works at three levels (priority order):

### Level 1: Per-Metric Override (Highest Priority)

Override model for a specific metric:

```yaml
evaluations:
  metrics:
    - metric: groundedness
      model:                    # Uses this model for this metric only
        provider: openai
        name: gpt-4
```

### Level 2: Evaluation-Wide Model

Default for all metrics without override:

```yaml
evaluations:
  model:                        # Uses for all metrics
    provider: openai
    name: gpt-4

  metrics:
    - metric: groundedness
      # Uses evaluation.model above
    - metric: relevance
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
    - metric: groundedness
      # Falls back to agent.model above
```

## AI-Powered Metrics

These metrics use an LLM to evaluate responses.

### Groundedness

Measures how well the response is supported by the source material.

```yaml
- metric: groundedness
  threshold: 0.8
  enabled: true
```

**Scale**: 1-5 (higher is better)

**What it measures**:
- Factual accuracy
- No hallucinations
- Claims are verifiable from sources

**When to use**: When accuracy is critical

**Example**:
- ✅ PASS: Agent cites specific knowledge base articles
- ❌ FAIL: Agent makes up company policies

### Relevance

Measures whether the response addresses the user's question.

```yaml
- metric: relevance
  threshold: 0.75
```

**Scale**: 1-5 (higher is better)

**What it measures**:
- Response answers the question
- On-topic content
- Minimal tangents

**When to use**: For general quality assurance

**Example**:
- ✅ PASS: "How do I reset my password?" → Password reset instructions
- ❌ FAIL: "How do I reset my password?" → Company history

### Coherence

Measures how well the response flows and makes sense.

```yaml
- metric: coherence
  threshold: 0.7
```

**Scale**: 1-5 (higher is better)

**What it measures**:
- Clear writing
- Logical flow
- Proper grammar

**When to use**: For content quality

### Safety

Measures whether response is appropriate and avoids harm.

```yaml
- metric: safety
  threshold: 0.9
```

**Scale**: 1-5 (higher is better)

**What it measures**:
- No harmful content
- Appropriate tone
- No PII leakage

**When to use**: For user safety, PII protection

## NLP Metrics

These metrics compare response to expected output using text algorithms.

### F1 Score

Measures precision and recall of token overlap.

```yaml
- metric: f1_score
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
- metric: bleu
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
- metric: rouge
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
- metric: meteor
  threshold: 0.65
```

**Scale**: 0.0-1.0 (higher is better)

**What it measures**:
- N-gram match with synonyms
- Word order

**When to use**: For translation, paraphrase with synonyms

## Metric Configuration

### Threshold

- **Type**: Float
- **Purpose**: Minimum score for test to pass
- **Scale**: 1-5 for AI metrics, 0-1 for NLP metrics
- **Optional**: Yes (default: no threshold, metric is informational)

```yaml
- metric: groundedness
  threshold: 0.8
```

### Enabled

- **Type**: Boolean
- **Default**: `true`
- **Purpose**: Temporarily disable metric without removing it

```yaml
- metric: relevance
  enabled: false  # Metric runs but doesn't fail test
```

### Scale

- **Type**: Integer
- **Purpose**: Scoring scale (e.g., 5 for 1-5 scale)
- **Default**: 5 for AI metrics
- **Optional**: Yes

```yaml
- metric: groundedness
  scale: 10  # 1-10 scale instead of 1-5
```

### Fail on Error

- **Type**: Boolean
- **Default**: `false` (soft failure)
- **Purpose**: Whether to fail test if evaluation errors

```yaml
- metric: groundedness
  fail_on_error: false  # Continues even if LLM evaluation fails
```

### Retry on Failure

- **Type**: Integer
- **Default**: 0
- **Range**: 1-3
- **Purpose**: Retry LLM evaluation on failure

```yaml
- metric: groundedness
  retry_on_failure: 2  # Retry up to 2 times
```

### Timeout

- **Type**: Integer (milliseconds)
- **Purpose**: Maximum time for evaluation
- **Default**: No timeout

```yaml
- metric: groundedness
  timeout_ms: 30000  # 30 second timeout
```

### Custom Prompt

- **Type**: String
- **Purpose**: Custom evaluation prompt (advanced)
- **Default**: Built-in prompt per metric

```yaml
- metric: groundedness
  custom_prompt: |
    Evaluate groundedness on scale 1-5:
    {{response}}
    Sources: {{sources}}
```

## Complete Examples

### Basic Evaluation

```yaml
evaluations:
  metrics:
    - metric: groundedness
      threshold: 0.8
    - metric: relevance
      threshold: 0.75
```

### With Custom Evaluation Model

```yaml
evaluations:
  model:
    provider: openai
    name: gpt-4  # Use better model for evaluation
    temperature: 0.2

  metrics:
    - metric: groundedness
      threshold: 0.85
    - metric: relevance
      threshold: 0.8
```

### With Per-Metric Overrides

```yaml
evaluations:
  model:
    provider: openai
    name: gpt-4o-mini  # Default: faster, cheaper

  metrics:
    - metric: groundedness
      threshold: 0.85
      model:  # Override for critical metric
        provider: openai
        name: gpt-4  # Use powerful model

    - metric: relevance
      threshold: 0.75
      # Uses evaluation.model above

    - metric: safety
      threshold: 0.9
      model:  # Override for critical metric
        provider: anthropic
        name: claude-3-opus
```

### Mixed AI and NLP Metrics

```yaml
evaluations:
  model:
    provider: openai
    name: gpt-4o

  metrics:
    # AI metrics
    - metric: groundedness
      threshold: 0.8

    - metric: relevance
      threshold: 0.75

    # NLP metrics
    - metric: f1_score
      threshold: 0.7

    - metric: rouge
      threshold: 0.6
```

### Comprehensive Enterprise Setup

```yaml
evaluations:
  model:
    provider: openai
    name: gpt-4o-mini
    temperature: 0.1  # Consistent evaluation

  metrics:
    # Critical metrics - use powerful model
    - metric: safety
      threshold: 0.95
      model:
        provider: openai
        name: gpt-4

    - metric: groundedness
      threshold: 0.9
      model:
        provider: openai
        name: gpt-4

    # Standard metrics - use default
    - metric: relevance
      threshold: 0.8

    - metric: coherence
      threshold: 0.75

    # NLP metrics - no LLM needed
    - metric: f1_score
      threshold: 0.7

    - metric: rouge
      threshold: 0.65

    # Disabled metrics - monitoring only
    - metric: meteor
      enabled: false
      threshold: 0.6

    # Soft failure metric
    - metric: custom_metric
      fail_on_error: false
      timeout_ms: 10000
      retry_on_failure: 1
```

### Per-Test Case Evaluation

Test cases can specify which metrics to run:

```yaml
test_cases:
  - name: "Fact check test"
    input: "What's our company's founding date?"
    expected_tools: [search-kb]
    ground_truth: "Founded in 2010"
    evaluations:
      - groundedness
      - relevance

  - name: "Creative task"
    input: "Generate a company tagline"
    evaluations:
      - coherence
      - safety
      # Skip groundedness since no ground truth
```

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
Tools called: [search-kb] ✓
Metrics:
  ✓ Groundedness: 0.92 (threshold: 0.8)
  ✓ Relevance: 0.88 (threshold: 0.75)
  ✓ F1 Score: 0.81 (threshold: 0.7)
Result: PASS
```

## Cost Optimization

### Use Right Model per Metric

```yaml
evaluations:
  model:
    provider: openai
    name: gpt-4o-mini  # Cheap default

  metrics:
    - metric: groundedness
      # Uses gpt-4o-mini
    - metric: safety
      model:
        provider: openai
        name: gpt-4  # Expensive override only for critical metric
```

### Disable Unnecessary Metrics

```yaml
- metric: meteor
  enabled: false  # Don't evaluate
```

### Use NLP Metrics When Possible

NLP metrics are free (no LLM calls):

```yaml
- metric: f1_score  # No LLM cost
- metric: rouge     # No LLM cost
```

## Model Configuration Details

When specifying a model for evaluation:

```yaml
model:
  provider: openai|azure_openai|anthropic  # Required
  name: model-identifier                    # Required
  temperature: 0.0-2.0                      # Optional
  max_tokens: integer                       # Optional
  top_p: 0.0-1.0                            # Optional
```

### Provider-Specific Models

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
  provider: openai
  name: gpt-4o-mini
  temperature: 0.1  # Low temperature for consistency
  max_tokens: 2000  # Enough for explanation
```

## Troubleshooting

### Error: "invalid metric type"

- Check metric name is valid
- Valid AI metrics: groundedness, relevance, coherence, safety
- Valid NLP metrics: f1_score, bleu, rouge, meteor

### Error: "threshold must be valid for scale"

- For 1-5 scale: Use values like 0.8, 1.5, 2.0, etc.
- For 0-1 scale: Use values like 0.5, 0.75, 0.9

### Metric always fails

- Check evaluation model is working
- Try without threshold first
- Test evaluation model manually

### LLM evaluation too slow

- Use faster model: `gpt-4o-mini` instead of `gpt-4`
- Add timeout: `timeout_ms: 10000`
- Use NLP metrics instead (free)

### Inconsistent evaluation results

- Increase temperature precision: `temperature: 0.1`
- Use more powerful model: `gpt-4` instead of `gpt-4o-mini`
- Add retry: `retry_on_failure: 2`

## Best Practices

1. **Start Simple**: Begin with 1-2 metrics, add more after understanding
2. **Use Defaults**: Let HoloDeck choose model/scale unless you have specific needs
3. **Mix Metric Types**: Combine AI metrics (semantic) with NLP (keyword-based)
4. **Cost-Aware**: Use cheaper models by default, expensive models only for critical metrics
5. **Realistic Thresholds**: Set thresholds based on actual agent performance
6. **Monitor**: Run metrics on sample of tests first
7. **Iterate**: Adjust thresholds and metrics based on results

## Next Steps

- See [Agent Configuration Guide](agent-configuration.md) for how to set up evaluations
- See [Examples](../examples/) for complete evaluation configurations
- See [Global Configuration](global-config.md) for shared settings
