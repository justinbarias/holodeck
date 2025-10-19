# Quickstart: US1 Testing & Development

**Feature**: User Story 1 - Define Agent Configuration (Priority: P1)
**Focus**: Configuration loading and validation

## What's Testable

After US1 implementation, developers can:

1. **Create valid agent.yaml** → loads without errors
2. **Edit agent configuration** → parse and apply correctly
3. **Catch invalid configurations** → with clear error messages
4. **Specify different LLM providers** → configuration is used correctly

## Test Scenarios

### Scenario 1: Minimal Valid Agent

```yaml
# minimal_agent.yaml
name: hello-agent
model:
  provider: openai
  name: gpt-4o-mini
instructions:
  inline: "You are a helpful assistant."
```

**Test**:

```python
from holodeck.config.loader import ConfigLoader

loader = ConfigLoader()
agent = loader.load("minimal_agent.yaml")

assert agent.name == "hello-agent"
assert agent.model.provider == "openai"
assert agent.model.name == "gpt-4o-mini"
assert agent.instructions.inline == "You are a helpful assistant."
```

**Acceptance**: ✅ Loads successfully, no errors

---

### Scenario 2: Full Agent with Tools

```yaml
# full_agent.yaml
name: customer-support
description: Handles customer inquiries
model:
  provider: openai
  name: gpt-4o
  temperature: 0.7
instructions:
  file: instructions/support.md
tools:
  - name: search_faq
    type: vectorstore
    description: Search FAQ database
    source: data/faqs.md
    embedding_model: text-embedding-3-small
  - name: check_order
    type: function
    description: Check order status
    file: tools/orders.py
    function: get_order_status
evaluations:
  metrics:
    - metric: groundedness
      threshold: 4.0
    - metric: relevance
      threshold: 4.0
test_cases:
  - input: What's your return policy?
    expected_tools:
      - search_faq
  - input: Where is my order #123?
    expected_tools:
      - check_order
    ground_truth: Your order is in transit
```

**Test**:

```python
agent = loader.load("full_agent.yaml")

assert len(agent.tools) == 2
assert agent.tools[0].name == "search_faq"
assert agent.tools[0].type == "vectorstore"
assert agent.tools[1].type == "function"

assert len(agent.evaluations.metrics) == 2
assert len(agent.test_cases) == 2
```

**Acceptance**: ✅ All nested structures loaded correctly

---

### Scenario 3: Invalid Config - Missing Required Field

```yaml
# invalid_missing_field.yaml
name: broken-agent
# MISSING: model section
instructions:
  inline: "Test"
```

**Test**:

```python
with pytest.raises(ValidationError) as exc:
    loader.load("invalid_missing_field.yaml")

assert "model" in str(exc.value).lower()
assert "required" in str(exc.value).lower()
```

**Acceptance**: ✅ Clear error indicating missing `model` field

---

### Scenario 4: Invalid Config - Tool Reference

```yaml
# invalid_tool_ref.yaml
name: test-agent
model:
  provider: openai
  name: gpt-4o
instructions:
  inline: Test
test_cases:
  - input: Query
    expected_tools:
      - non_existent_tool # ERROR: tool not defined
```

**Test**:

```python
with pytest.raises(ValidationError) as exc:
    loader.load("invalid_tool_ref.yaml")

assert "non_existent_tool" in str(exc.value)
assert "not found in tools" in str(exc.value).lower()
```

**Acceptance**: ✅ Clear error about missing tool reference

---

### Scenario 5: Invalid Config - Missing File Reference

```yaml
# invalid_file_ref.yaml
name: test-agent
model:
  provider: openai
  name: gpt-4o
instructions:
  file: missing/instructions.md # ERROR: file doesn't exist
```

**Test**:

```python
with pytest.raises(ConfigFileNotFoundError) as exc:
    loader.load("invalid_file_ref.yaml")

assert "missing/instructions.md" in str(exc.value)
assert "does not exist" in str(exc.value).lower()
```

**Acceptance**: ✅ Clear error about missing file with suggestion

---

### Scenario 6: Environment Variable Interpolation

```yaml
# env_vars.yaml
name: api-agent
model:
  provider: openai
  name: gpt-4o
  # This would use an env var in a real LLM call (US3+)
instructions:
  inline: "You have access to API key: ${OPENAI_API_KEY}"
```

**Test**:

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-test123"

agent = loader.load("env_vars.yaml")
# Env var is resolved during loading
assert "${OPENAI_API_KEY}" not in agent.instructions.inline
```

**Acceptance**: ✅ Environment variables interpolated correctly

---

### Scenario 7: Instructions File vs Inline

```yaml
# Test 7a: File-based instructions
instructions:
  file: instructions/system-prompt.md

# Test 7b: Inline instructions
instructions:
  inline: "You are helpful"

# Test 7c: Invalid - both provided
instructions:
  file: instructions/prompt.md
  inline: "Also inline"  # ERROR
```

**Test 7c**:

```python
with pytest.raises(ValidationError) as exc:
    loader.load("invalid_both_instructions.yaml")

assert "file" in str(exc.value).lower()
assert "inline" in str(exc.value).lower()
assert "exactly one" in str(exc.value).lower() or "mutually exclusive" in str(exc.value).lower()
```

**Acceptance**: ✅ Mutual exclusivity enforced

---

### Scenario 8: Tool Type Specifics

```yaml
# Test vectorstore tool
tools:
  - name: search
    type: vectorstore
    source: data/docs.md
    vector_field: content

# Test function tool
tools:
  - name: execute
    type: function
    file: tools/runner.py
    function: run_task
    parameters:
      task_id:
        type: string

# Test MCP tool
tools:
  - name: files
    type: mcp
    server: "@modelcontextprotocol/server-filesystem"
    config:
      allowed_directories:
        - /tmp

# Test prompt tool
tools:
  - name: summarize
    type: prompt
    template: "Summarize this: {{input}}"
    parameters:
      input:
        type: string
    model:
      provider: openai
      name: gpt-4o-mini
```

**Test**:

```python
agent = loader.load("tools_all_types.yaml")

assert agent.tools[0].type == "vectorstore"
assert agent.tools[1].type == "function"
assert agent.tools[2].type == "mcp"
assert agent.tools[3].type == "prompt"
```

**Acceptance**: ✅ All tool types load with type-specific validation

---

### Scenario 9: Evaluation Model Overrides

```yaml
evaluations:
  model:
    provider: openai
    name: gpt-4o-mini # Default for all metrics
  metrics:
    - metric: groundedness
      threshold: 4.0
      model: # Override for this metric
        provider: openai
        name: gpt-4o # Use expensive model for critical metric
    - metric: relevance
      threshold: 4.0
      # Uses default gpt-4o-mini
```

**Test**:

```python
agent = loader.load("evaluation_overrides.yaml")

# Global model
assert agent.evaluations.model.name == "gpt-4o-mini"

# Metric 1 override
assert agent.evaluations.metrics[0].model.name == "gpt-4o"

# Metric 2 uses global
assert agent.evaluations.metrics[1].model is None  # Fallback to global
```

**Acceptance**: ✅ Model overrides work correctly

---

### Scenario 10: Load Time Performance

**Test**:

```python
import time

start = time.perf_counter()
agent = loader.load("complex_agent.yaml")  # 50 tools, 100 test cases
elapsed = time.perf_counter() - start

assert elapsed < 0.1  # < 100ms
```

**Acceptance**: ✅ Loads in under 100ms

---

## Test Execution

Run tests with:

```bash
# All US1 tests
pytest tests/unit/test_config_*.py tests/integration/test_config_*.py -v

# With coverage
pytest tests/unit/test_config_*.py --cov=holodeck.config --cov-report=html

# Watch mode
pytest-watch tests/unit/test_config_*.py
```

## Fixtures Required

```
tests/fixtures/agents/
├── valid_agent.yaml              # Scenario 1
├── full_agent.yaml               # Scenario 2
├── invalid_missing_field.yaml    # Scenario 3
├── invalid_tool_ref.yaml         # Scenario 4
├── invalid_file_ref.yaml         # Scenario 5
├── env_vars.yaml                 # Scenario 6
├── invalid_both_instructions.yaml # Scenario 7c
├── tools_all_types.yaml          # Scenario 8
├── evaluation_overrides.yaml     # Scenario 9
└── complex_agent.yaml            # Scenario 10

tests/fixtures/agents/instructions/
├── system-prompt.md
└── support.md

tests/fixtures/agents/data/
├── faqs.md
└── docs.md

tests/fixtures/agents/tools/
└── orders.py
```

## Independent Test Criteria (from spec.md)

✅ **Acceptance Scenario 1**: YAML parses without errors, all sections loaded

- Tests: Scenario 1, 2, 8

✅ **Acceptance Scenario 2**: Missing fields → clear validation errors

- Tests: Scenario 3, 4, 5, 7c

✅ **Acceptance Scenario 3**: Configuration applied correctly (provider & model used)

- Tests: Scenario 6, 9 (model selection)

**Definition of Done for US1**: All 3 acceptance scenarios passing with clear error messages.
