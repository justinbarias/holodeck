# Quickstart: Test Execution Framework

**Date**: 2025-11-01
**Feature**: Execute Agent Against Test Cases
**Phase**: 1 - Design

## Overview

This quickstart guide shows developers how to use the `holodeck test` command to validate agent behavior through test execution. It covers basic usage, common patterns, and troubleshooting.

---

## Prerequisites

- HoloDeck CLI installed (`holodeck` command available)
- Agent configuration file (`agent.yaml`) with test cases defined
- Azure OpenAI or Anthropic API keys configured (for agent execution and evaluation metrics)

---

## Basic Usage

### 1. Run All Tests

Execute all test cases defined in agent.yaml:

```bash
holodeck test agent.yaml
```

**Output**:

```
🧪 Running HoloDeck Tests...

✅ Test 1/3: What are your business hours? [PASSED] (3.5s)
✅ Test 2/3: Where is my order #12345? [PASSED] (4.2s)
✅ Test 3/3: How do I return an item? [PASSED] (3.8s)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Summary: 3/3 tests passed (100% pass rate)
Duration: 11.5s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

### 2. Generate Test Report

Save test results to a JSON file:

```bash
holodeck test agent.yaml --output report.json
```

Or generate a Markdown report:

```bash
holodeck test agent.yaml --output report.md
```

**report.json** (excerpt):

```json
{
  "agent_name": "Customer Support Agent",
  "results": [
    {
      "test_name": "Business hours query",
      "test_input": "What are your business hours?",
      "agent_response": "We're open Monday-Friday 9AM-5PM EST",
      "metric_results": [
        {
          "metric_name": "groundedness",
          "score": 0.95,
          "threshold": 0.7,
          "passed": true
        }
      ],
      "passed": true
    }
  ],
  "summary": {
    "total_tests": 3,
    "passed": 3,
    "failed": 0,
    "pass_rate": 100.0
  }
}
```

---

### 3. Run Specific Tests

Run only tests matching a name:

```bash
holodeck test agent.yaml --test "Business hours query"
```

Run tests matching a pattern:

```bash
holodeck test agent.yaml --filter "order_*"
```

---

## Agent Configuration Example

### Minimal Configuration

**agent.yaml**:

```yaml
name: Customer Support Agent
model:
  provider: azure_openai
  name: gpt-4o
  temperature: 0.7

instructions:
  inline: |
    You are a helpful customer support agent.
    Answer questions about business hours, orders, and returns.

evaluations:
  metrics:
    - metric: groundedness
      threshold: 0.7
    - metric: relevance
      threshold: 0.7

test_cases:
  - name: Business hours query
    input: "What are your business hours?"
    ground_truth: "We're open Monday-Friday 9AM-5PM EST"

  - name: Order status query
    input: "Where is my order #12345?"
    expected_tools: ["get_order_status"]
```

**Run tests**:

```bash
holodeck test agent.yaml
```

---

### Configuration with Execution Settings

**agent.yaml** (with execution config):

```yaml
name: Customer Support Agent
model:
  provider: azure_openai
  name: gpt-4o

instructions:
  file: instructions/system.md

evaluations:
  metrics:
    - metric: groundedness
      threshold: 0.7
      model:
        provider: azure_openai
        name: gpt-4o # Use expensive model for critical metric

    - metric: relevance
      threshold: 0.7
      model:
        provider: azure_openai
        name: gpt-4o-mini # Use cheaper model for general metric

test_cases:
  - name: Business hours query
    input: "What are your business hours?"
    ground_truth: "We're open Monday-Friday 9AM-5PM EST"

# Execution configuration
execution:
  file_timeout: 60 # Allow 60s for large files
  llm_timeout: 120 # Allow 120s for complex LLM calls
  cache_enabled: true # Cache remote files
  verbose: false # Standard output
```

**Run tests** (uses execution config from agent.yaml):

```bash
holodeck test agent.yaml
```

**Override settings** via CLI:

```bash
holodeck test agent.yaml --llm-timeout 180 --verbose
```

---

## Multimodal Test Cases

### Test with PDF Files

**agent.yaml**:

```yaml
test_cases:
  - name: Analyze contract terms
    input: "What is the cancellation policy?"
    files:
      - path: tests/fixtures/contract.pdf
        type: pdf
        pages: [1, 2, 3] # Extract only first 3 pages
    ground_truth: "30-day cancellation with full refund"
    evaluations:
      - groundedness
      - relevance
```

**Run test**:

```bash
holodeck test agent.yaml
```

The PDF content is automatically extracted via markitdown and provided as context to the agent.

---

### Test with Images

**agent.yaml**:

```yaml
test_cases:
  - name: Identify product from photo
    input: "What product is shown in this image?"
    files:
      - path: tests/fixtures/product-photo.jpg
        type: image
        description: "Product photo from customer"
    expected_tools: ["identify_product"]
```

markitdown processes the image with OCR and LLM-based description.

---

### Test with Excel Data

**agent.yaml**:

```yaml
test_cases:
  - name: Analyze Q4 sales data
    input: "What were the total sales in Q4?"
    files:
      - path: tests/fixtures/sales-data.xlsx
        type: excel
        sheet: "Q4 Sales"
        range: "A1:E100" # Extract specific range
    ground_truth: "$2.5M in Q4 sales"
```

---

### Test with Remote Files (URLs)

**agent.yaml**:

```yaml
test_cases:
  - name: Analyze public report
    input: "What are the key findings?"
    files:
      - url: "https://example.com/annual-report.pdf"
        type: pdf
        cache: true # Cache the file in .holodeck/cache/
```

Files are downloaded, cached, and reused in subsequent runs (unless `--no-cache` is used).

---

## CI/CD Integration

### GitHub Actions Example

**.github/workflows/test-agent.yml**:

```yaml
name: Test Agent

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Install HoloDeck
        run: pip install holodeck-ai

      - name: Run Agent Tests
        env:
          AZURE_OPENAI_API_KEY: ${{ secrets.AZURE_OPENAI_API_KEY }}
        run: |
          holodeck test agent.yaml --quiet --output report.json

      - name: Upload Test Report
        uses: actions/upload-artifact@v3
        with:
          name: test-report
          path: report.json
```

---

### GitLab CI Example

**.gitlab-ci.yml**:

```yaml
test-agent:
  image: python:3.10
  script:
    - pip install holodeck-ai
    - holodeck test agent.yaml --quiet --output report.json
  artifacts:
    reports:
      junit: report.json
    paths:
      - report.json
  only:
    - merge_requests
    - main
```

---

## Common Patterns

### Pattern 1: Per-Metric Model Configuration

Use expensive models (GPT-4o) for critical metrics, cheaper models (GPT-4o-mini) for general metrics:

```yaml
evaluations:
  metrics:
    - metric: groundedness
      threshold: 0.8
      model:
        provider: azure_openai
        name: gpt-4o # Critical: factual accuracy

    - metric: safety
      threshold: 0.9
      model:
        provider: azure_openai
        name: gpt-4o # Critical: harmful content detection

    - metric: relevance
      threshold: 0.7
      model:
        provider: azure_openai
        name: gpt-4o-mini # General: response relevance

    - metric: coherence
      threshold: 0.7
      model:
        provider: azure_openai
        name: gpt-4o-mini # General: logical flow
```

**Cost Optimization**: This approach reduces evaluation costs by 70-80% while maintaining quality for critical metrics.

---

### Pattern 2: Tool Call Validation

Validate that the agent calls expected tools:

```yaml
test_cases:
  - name: Order status requires tool call
    input: "Where is my order #12345?"
    expected_tools: ["get_order_status"]
    ground_truth: "Order #12345 shipped on Jan 5th"
```

Test fails if agent doesn't call `get_order_status` tool.

---

### Pattern 3: Mixed Text and Multimodal Tests

Combine simple text tests with complex multimodal tests:

```yaml
test_cases:
  # Simple text tests (fast)
  - name: Business hours
    input: "What are your business hours?"
    ground_truth: "Monday-Friday 9AM-5PM EST"

  # Complex multimodal tests (slower)
  - name: Contract analysis
    input: "What is the cancellation policy?"
    files:
      - path: tests/fixtures/contract.pdf
        type: pdf
    ground_truth: "30-day cancellation with full refund"
```

Run simple tests first for quick validation, then comprehensive tests with files.

---

### Pattern 4: Custom Execution Timeouts

Configure timeouts for different agent types:

**Fast agent (simple Q&A)**:

```yaml
execution:
  llm_timeout: 30 # Quick responses expected
  file_timeout: 15
```

**Complex agent (document analysis)**:

```yaml
execution:
  llm_timeout: 180 # Allow time for analysis
  file_timeout: 90 # Large documents take time
```

---

## Troubleshooting

### Issue 1: Test Timeout

**Symptom**:

```
ERROR: Test case "Analyze contract" failed
  Cause: LLM API timeout after 60s
```

**Solution**:
Increase timeout in agent.yaml:

```yaml
execution:
  llm_timeout: 120
```

Or via CLI:

```bash
holodeck test agent.yaml --llm-timeout 120
```

---

### Issue 2: File Processing Failure

**Symptom**:

```
WARNING: Failed to process file
  File: tests/fixtures/large-document.pdf
  Cause: Timeout after 30s
```

**Solution**:
Increase file timeout:

```yaml
execution:
  file_timeout: 60
```

Or extract specific pages to reduce processing time:

```yaml
files:
  - path: tests/fixtures/large-document.pdf
    type: pdf
    pages: [1, 2, 3] # Only first 3 pages
```

---

### Issue 3: Metric Evaluation Fails

**Symptom**:

```
WARNING: Metric evaluation failed
  Metric: groundedness
  Cause: LLM API rate limit exceeded
  Retry: Attempt 1/3
```

**Solution**:

- Wait for rate limit reset (automatic retry up to 3 times with exponential backoff)
- Check Azure OpenAI quota/limits
- Consider using `fail_on_error: false` for non-critical metrics

```yaml
evaluations:
  metrics:
    - metric: relevance
      fail_on_error: false # Don't fail test if metric errors
```

---

### Issue 4: Remote File Download Fails

**Symptom**:

```
ERROR: Failed to download remote file
  URL: https://example.com/report.pdf
  Cause: Connection timeout after 30s
```

**Solution**:
Increase download timeout:

```yaml
execution:
  download_timeout: 60
```

Or use a local copy of the file:

```yaml
files:
  - path: tests/fixtures/report.pdf # Local file instead of URL
    type: pdf
```

---

### Issue 5: Test Fails with Tool Mismatch

**Symptom**:

```
❌ Test: Order status query [FAILED]
  Errors: Tool mismatch: Expected get_order_status but got []
```

**Solution**:

1. Verify agent has the `get_order_status` tool configured in agent.yaml
2. Check tool implementation is working correctly
3. Review agent instructions to ensure tool usage is clear
4. Consider adding more context to the test input

---

## Best Practices

### 1. Start Simple, Then Add Complexity

**Phase 1**: Basic text tests

```yaml
test_cases:
  - name: Simple Q&A
    input: "What are your hours?"
    ground_truth: "9AM-5PM EST"
```

**Phase 2**: Add evaluation metrics

```yaml
test_cases:
  - name: Simple Q&A
    input: "What are your hours?"
    ground_truth: "9AM-5PM EST"
    evaluations:
      - groundedness
      - relevance
```

**Phase 3**: Add multimodal files

```yaml
test_cases:
  - name: Document analysis
    input: "Summarize the key points"
    files:
      - path: tests/fixtures/report.pdf
        type: pdf
    evaluations:
      - groundedness
      - relevance
```

---

### 2. Use Ground Truth for Critical Tests

Always provide `ground_truth` for tests that validate factual accuracy:

```yaml
test_cases:
  - name: Business hours
    input: "What are your business hours?"
    ground_truth: "Monday-Friday 9AM-5PM EST" # GOOD: Enables groundedness check

  - name: General greeting
    input: "Hello!"
    # No ground_truth needed for greetings
```

---

### 3. Configure Per-Metric Models for Cost Optimization

```yaml
evaluations:
  model: # Global default (cheapest)
    provider: azure_openai
    name: gpt-4o-mini

  metrics:
    - metric: groundedness
      model: # Override for critical metric
        provider: azure_openai
        name: gpt-4o

    - metric: relevance
      # Uses global default (gpt-4o-mini)
```

---

### 4. Use Execution Config for Consistent Behavior

Define execution settings in agent.yaml for consistent behavior across team:

```yaml
execution:
  file_timeout: 60
  llm_timeout: 120
  cache_enabled: true
  verbose: false
```

This ensures all developers and CI/CD runs use the same timeouts.

---

### 5. Test Tool Usage with expected_tools

Validate agent calls the right tools:

```yaml
test_cases:
  - name: Order lookup
    input: "Where is order #12345?"
    expected_tools: ["get_order_status"] # Validates tool call
```

---

## Next Steps

1. **Define Test Cases**: Add test cases to your agent.yaml
2. **Run Tests**: Execute `holodeck test agent.yaml`
3. **Review Results**: Check which tests pass/fail
4. **Iterate**: Refine agent instructions and test cases based on results
5. **Integrate CI/CD**: Add test execution to your deployment pipeline

---

## Additional Resources

- **CLI Reference**: See `contracts/cli-api.md` for complete CLI documentation
- **Data Models**: See `data-model.md` for TestResult and TestReport structures
- **Research**: See `research.md` for technology integration details
