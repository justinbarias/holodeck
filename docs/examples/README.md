# Agent Configuration Examples

This directory contains example agent.yaml files demonstrating different features and patterns. Start with `basic_agent.yaml` and progress to more complex configurations as needed.

## Quick Reference

| Example | Use Case | Features Demonstrated |
|---------|----------|----------------------|
| [`basic_agent.yaml`](#basic_agent) | Getting started | Minimal valid config, inline instructions, no tools |
| [`with_tools.yaml`](#with-tools) | Real-world workflows | All 4 tool types: vectorstore, function, MCP, prompt |
| [`with_evaluations.yaml`](#with-evaluations) | Quality assurance | DeepEval metrics, NLP metrics, per-metric model override |
| [`with_global_config.yaml`](#with-global-config) | Multi-environment setups | Config precedence, env var substitution, inheritance |

---

## Examples

### `basic_agent.yaml`

**Purpose**: Minimal valid agent configuration

**Features**:
- Simple agent metadata (name, description)
- OpenAI model provider (gpt-4o-mini)
- Inline system instructions (no external files)
- No tools (can run standalone for chat)

**When to use**:
- Learning basic agent.yaml structure
- Testing configuration loading without tool complexity
- Building a simple chatbot or Q&A assistant

**Try it**:
```bash
# Set your API key
export OPENAI_API_KEY=your-key-here

# Run the agent
holodeck run basic_agent.yaml

# Or validate configuration
holodeck validate basic_agent.yaml
```

**Key Concepts**:
- `model` → Required. Specifies LLM provider, model name, and generation settings
- `instructions` → Required. Can be inline (shown here) or from a file
- Minimal config is valid and functional

---

### `with_tools.yaml`

**Purpose**: Comprehensive tool integration example

**Features**:
- **Vectorstore tool**: Semantic search over documentation
- **Function tool**: Custom Python function execution
- **MCP tool**: Standardized integrations (filesystem, databases, APIs)
- **Prompt tool**: LLM-powered semantic functions
- Test cases with expected tool validation

**When to use**:
- Building agents that need external data access
- Executing custom business logic
- Integrating standardized tools (GitHub, Slack, databases)
- AI-powered data processing

**Prerequisites**:
```bash
# Create the required files this example references:
mkdir -p ./data/docs ./tools

# Create a simple documentation file
echo "Installation: Run 'pip install holodeck'" > ./data/docs/getting_started.txt

# Create a Python tool file (tools/discount_calculator.py)
cat > ./tools/discount_calculator.py << 'EOF'
def calculate_discount(customer_tier: str, order_amount: float, applied_coupon: str = None) -> dict:
    """Calculate order discount based on tier and amount."""
    tier_discounts = {
        'bronze': 0.05,
        'silver': 0.10,
        'gold': 0.15,
        'platinum': 0.25
    }

    base_discount = tier_discounts.get(customer_tier.lower(), 0)
    coupon_discount = 0.05 if applied_coupon else 0

    total_discount = min(base_discount + coupon_discount, 0.50)  # Cap at 50%
    discount_amount = order_amount * total_discount

    return {
        'original_amount': order_amount,
        'discount_percent': int(total_discount * 100),
        'discount_amount': discount_amount,
        'final_amount': order_amount - discount_amount
    }
EOF

# Create instructions file (instructions.txt)
cat > ./instructions.txt << 'EOF'
You are a customer service agent with access to:
- Documentation database (docs-search tool)
- Discount calculation system (calculate-discount tool)
- File system access (file-browser tool)
- Sentiment analysis (sentiment-analyzer tool)

Use the most appropriate tool for each customer request.
EOF
```

**Try it**:
```bash
export OPENAI_API_KEY=your-key-here
export ANTHROPIC_API_KEY=your-key-here

holodeck run with_tools.yaml
```

**Tool Types Explained**:
1. **Vectorstore**: Semantic search—find relevant documents based on meaning, not keywords
2. **Function**: Execute Python code—calculate, transform, validate data
3. **MCP**: Standardized integrations—filesystem, GitHub, databases, Slack
4. **Prompt**: LLM-powered—use AI to process data (sentiment analysis, summarization)

**Key Concepts**:
- Tool types are discriminated by the `type` field
- Each tool type has specific required fields
- Tools are composable—agents can use multiple tool types together
- File paths are relative to the agent.yaml location

---

### `with_evaluations.yaml`

**Purpose**: Quality assurance and evaluation framework

**Features**:
- **DeepEval GEval metrics**: Custom criteria with chain-of-thought evaluation (recommended)
- **DeepEval RAG metrics**: Faithfulness, answer relevancy (recommended)
- **NLP metrics**: F1 score, ROUGE (standard)
- **Legacy AI metrics**: Deprecated Azure AI metrics (backwards compatibility)
- Per-metric model overrides
- Threshold-based pass/fail criteria

**When to use**:
- Validating agent response quality
- Ensuring responses are grounded in data (faithfulness)
- Measuring accuracy against ground truth
- Running quality gates in production pipelines

**Metric Types** (in order of recommendation):

| Tier | Type | Metrics | Use Case |
|------|------|---------|----------|
| **1 (Recommended)** | DeepEval GEval | Custom criteria | Flexible semantic evaluation with natural language |
| **1 (Recommended)** | DeepEval RAG | `faithfulness`, `answer_relevancy`, `contextual_relevancy`, `contextual_precision`, `contextual_recall` | RAG pipeline evaluation |
| **2 (Standard)** | NLP | `f1_score`, `bleu`, `rouge`, `meteor` | Token-level comparison with ground truth |
| **3 (Deprecated)** | Legacy AI | `groundedness`, `relevance`, `coherence`, `safety` | Azure AI-based (migrate to DeepEval) |

**DeepEval Metrics Example**:
```yaml
evaluations:
  model:
    provider: ollama
    name: llama3.2:latest
    temperature: 0.0

  metrics:
    # GEval: Custom criteria
    - type: geval
      name: "Coherence"
      criteria: "Evaluate whether the response is clear and well-structured."
      threshold: 0.7

    # RAG: Hallucination detection
    - type: rag
      metric_type: faithfulness
      threshold: 0.8

    # RAG: Response relevance
    - type: rag
      metric_type: answer_relevancy
      threshold: 0.7
```

**Try it**:
```bash
# For local evaluation (free, no API keys needed)
# Make sure Ollama is running with llama3.2:latest

# Run evaluations
holodeck test with_evaluations.yaml

# Run with verbose output
holodeck test with_evaluations.yaml --verbose
```

**Configuration Precedence**:
```yaml
evaluations:
  model:                              # Global model (applies to all metrics)
    provider: ollama
    name: llama3.2:latest
  metrics:
    - type: geval
      name: "Quality"
      criteria: "..."
      model:                          # Per-metric override (highest precedence)
        provider: openai
        name: gpt-4                   # Use powerful model for critical metric
```

**Key Concepts**:
- Evaluations run after agent execution completes
- Each metric can override the evaluation model
- `threshold` defines minimum passing score (0-1 scale)
- `fail_on_error: false` = soft failure (evaluation error doesn't block)
- `fail_on_error: true` = hard failure (evaluation error stops test)
- DeepEval metrics support local models via Ollama (free)

**Legacy Metric Migration**:

| Legacy Metric | Recommended Replacement |
|--------------|------------------------|
| `groundedness` | `type: rag`, `metric_type: faithfulness` |
| `relevance` | `type: rag`, `metric_type: answer_relevancy` |
| `coherence` | `type: geval` with custom criteria |
| `safety` | `type: geval` with custom criteria |

---

### `with_global_config.yaml`

**Purpose**: Configuration precedence and environment-specific setup

**Features**:
- Environment variable substitution (`${VAR_NAME}`)
- Configuration inheritance from global config
- Agent-specific overrides
- Multi-environment setup (dev/staging/prod)

**Global Config Location**: `~/.holodeck/config.yaml`

**Sample Global Config**:
```yaml
# ~/.holodeck/config.yaml
model:
  provider: openai
  name: gpt-4o-mini
  temperature: 0.7

deployment:
  endpoint_prefix: /api/v1

providers:
  openai:
    api_key: ${OPENAI_API_KEY}
  azure:
    api_key: ${AZURE_API_KEY}
    endpoint: ${AZURE_ENDPOINT}
```

**Configuration Precedence** (highest to lowest):
1. **Agent-specific settings** (this file): Explicit values in agent.yaml
2. **Environment variables**: `${VAR_NAME}` resolved at runtime
3. **Global config**: `~/.holodeck/config.yaml` applied as defaults

**Try it**:
```bash
# Set environment variables
export AZURE_API_KEY=your-key-here
export AZURE_ENDPOINT=https://your-instance.openai.azure.com/

# Create global config
mkdir -p ~/.holodeck
cat > ~/.holodeck/config.yaml << 'EOF'
model:
  provider: openai
  name: gpt-4o-mini
  temperature: 0.7
EOF

# Run agent (uses merged config)
holodeck run with_global_config.yaml
```

**Key Concepts**:
- Global config provides defaults for all agents
- Agent.yaml overrides global settings
- Environment variables fill sensitive values (API keys)
- Substitution pattern: `${VARIABLE_NAME}`
- Missing environment variables cause errors at config load time

**Multi-Environment Example**:
```bash
# Development
export OPENAI_API_KEY=sk-dev-...
export ENV=development

# Staging
export OPENAI_API_KEY=sk-staging-...
export ENV=staging

# Production
export OPENAI_API_KEY=sk-prod-...
export ENV=production

# Same agent.yaml works in all environments
holodeck run with_global_config.yaml
```

---

## Common Patterns

### Pattern 1: Development vs. Production

```yaml
# Use global config for dev defaults
# Override in agent.yaml for production

# agent.yaml
model:
  provider: openai
  name: ${MODEL_NAME}  # env: gpt-4o-mini (dev) or gpt-4o (prod)
  temperature: ${TEMPERATURE}  # env: 0.7 (dev) or 0.3 (prod)
```

### Pattern 2: Sensitive Data

```yaml
# Never commit API keys
# Use environment variables or global config

instructions:
  inline: |
    Use the API token from environment variable for authentication.

tools:
  - name: api-client
    type: function
    file: ./tools/api.py
    # API key injected via ${API_KEY} at runtime
```

### Pattern 3: Modular Configurations

```yaml
# Split large configurations

# main_agent.yaml
name: multi-step-agent
instructions:
  file: ./instructions.md  # Separate file

tools:
  # Reference tool configs in separate files (if using advanced tooling)
  - name: tool1
    type: vectorstore
    source: ./data/kb/
```

### Pattern 4: Cost-Effective Evaluation

```yaml
# Use local models for development, paid APIs for production

evaluations:
  model:
    provider: ollama           # Free, local (development)
    name: llama3.2:latest

  metrics:
    - type: geval
      name: "Quality"
      criteria: "..."
      # Uses local model by default

    - type: rag
      metric_type: faithfulness
      model:
        provider: openai       # Override for critical metric
        name: gpt-4
```

---

## Next Steps

1. **Start with `basic_agent.yaml`**: Understand structure
2. **Progress to `with_tools.yaml`**: Add tool integration
3. **Explore `with_evaluations.yaml`**: Add quality gates with DeepEval
4. **Deploy with `with_global_config.yaml`**: Production setup

For more information:
- See [docs/guides/agent-configuration.md](../guides/agent-configuration.md) for schema reference
- See [docs/guides/tools.md](../guides/tools.md) for tool type details
- See [docs/guides/evaluations.md](../guides/evaluations.md) for evaluation configuration
- See [docs/guides/global-config.md](../guides/global-config.md) for precedence rules

---

## Troubleshooting

**Q: ConfigError when loading agent.yaml**
- Check file paths (relative to agent.yaml location)
- Verify all required fields are present
- Ensure YAML syntax is valid

**Q: Tool execution fails**
- Verify tool files exist and are readable
- Check Python function names match tool configuration
- Ensure vectorstore source path contains data

**Q: Environment variable not substituted**
- Use `${VARIABLE_NAME}` syntax
- Set variable before running: `export VARIABLE_NAME=value`
- Check for typos in variable names

**Q: Evaluations run but show errors**
- If `fail_on_error: false`, errors are logged but don't block
- Check model API keys are set (or use Ollama for local evaluation)
- Verify ground_truth and test input are clear and specific

**Q: DeepEval metrics not working**
- Ensure Ollama is running if using local models
- Check that required parameters are available (e.g., retrieval_context for faithfulness)
- Try with a simpler model first to debug

**Q: Legacy metrics deprecated warning**
- Migrate to DeepEval equivalents (see migration table above)
- Legacy metrics still work but will be removed in future versions

---

**Created**: 2025-10-19 | **Updated**: 2025-11-30 | **Version**: 0.2.0
