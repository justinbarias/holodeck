# 🧪 HoloDeck

**Build, Test, and Deploy AI Agents — No Code Required**

HoloDeck is an open-source experimentation platform that enables teams to create, evaluate, and deploy AI agents through simple YAML configuration. Go from hypothesis to production API in minutes, not weeks.

[![PyPI version](https://badge.fury.io/py/holodeck-ai.svg)](https://badge.fury.io/py/holodeck-ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## ✨ Features

- **🎯 No-Code Agent Definition** - Define agents using simple YAML configuration
- **🧪 Hypothesis-Driven Testing** - Test agent behaviors against structured test cases
- **📊 Integrated Evaluations** - Built-in AI and NLP metrics (Groundedness, Relevance, F1, BLEU, ROUGE)
- **🔌 Plugin Ecosystem** - Extend agents with tools, APIs, and custom functions
- **💾 RAG Support** - Native vector database integration for grounding data
- **🚀 One-Click Deployment** - Deploy agents as production-ready FastAPI endpoints
- **🔒 Enterprise-Ready** - Authentication, rate limiting, monitoring, and logging built-in
- **☁️ Cloud-Native** - Deploy to Azure, AWS, or GCP with single command

---

## 🚀 Quick Start

### Installation

```bash
pip install holodeck-ai
```

### Create Your First Agent

```bash
# Initialize a new agent workspace
holodeck init customer-support --template conversational

cd customer-support
```

This creates:

```
customer-support/
├── agent.yaml              # Agent configuration
├── instructions/
│   └── system-prompt.md   # Agent instructions
├── data/                  # Grounding data (optional)
├── tools/                 # Custom tools/plugins
└── tests/
    └── test-cases.yaml    # Test scenarios
```

### Define Your Agent

Edit `agent.yaml`:

```yaml
name: "customer-support-agent"
description: "Handles customer inquiries with empathy and accuracy"

model:
  provider: openai
  name: gpt-4o-mini
  temperature: 0.7

instructions:
  file: instructions/system-prompt.md

tools:
  - name: search_knowledge_base
    type: vectorstore
    source: data/faqs.md
    description: "Search customer FAQ database"

  - name: check_order_status
    type: function
    file: tools/orders.py
    description: "Retrieve order status by order ID"

evaluations:
  - metric: groundedness
    threshold: 4.0
  - metric: relevance
    threshold: 4.0
  - metric: response_time
    max_ms: 2000

test_cases:
  - input: "What are your business hours?"
    expected_tools: ["search_knowledge_base"]

  - input: "Where is my order #12345?"
    expected_tools: ["check_order_status"]
    ground_truth: "Your order is in transit"
```

### Test Your Agent

```bash
# Run test cases with evaluations
holodeck test agent.yaml

# Interactive testing
holodeck chat agent.yaml
```

**Output:**

```
🧪 Running HoloDeck Tests...

✅ Test 1/2: What are your business hours?
   Groundedness: 4.2/5.0 ✓
   Relevance: 4.5/5.0 ✓
   Response Time: 1,234ms ✓
   Tools Used: [search_knowledge_base] ✓

✅ Test 2/2: Where is my order #12345?
   Groundedness: 4.8/5.0 ✓
   Relevance: 4.7/5.0 ✓
   F1 Score: 0.89 ✓
   Response Time: 987ms ✓

📊 Overall Results: 2/2 passed (100%)
```

### Deploy as API

```bash
# Deploy locally
holodeck deploy agent.yaml --port 8000

# Agent is now live at http://localhost:8000
```

**API Endpoints:**

```bash
# Chat with agent
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are your business hours?",
    "session_id": "user-123"
  }'

# Streaming response
curl -X POST http://localhost:8000/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about your products"}'

# Health check
curl http://localhost:8000/health
```

### Deploy to Cloud

```bash
# Deploy to Azure Container Apps
holodeck deploy agent.yaml --cloud azure --region westus2

# Deploy to AWS Lambda
holodeck deploy agent.yaml --cloud aws --region us-east-1

# Deploy to Cloud Run (GCP)
holodeck deploy agent.yaml --cloud gcp --region us-central1
```

---

## 📖 Core Concepts

### Agent Definition

Agents are defined using declarative YAML configuration:

```yaml
name: "research-agent"
model:
  provider: openai
  name: gpt-4o
instructions: |
  You are a research assistant that helps users find
  accurate information from trusted sources.
tools:
  - search_web
  - search_papers
  - summarize_document
```

### Tools & Plugins

Extend agent capabilities with a rich ecosystem of tools and plugins:

#### 1. Vector Search Tools

```yaml
tools:
  - name: search_docs
    type: vectorstore
    provider: redis
    connection: "localhost:6379"
    source: data/documents/
    embedding_model: text-embedding-3-small
```

#### 2. Custom Function Tools

```yaml
tools:
  - name: calculate_shipping
    type: function
    file: tools/shipping.py
    function: calculate_cost
    description: "Calculate shipping cost based on weight and destination"
    parameters:
      weight:
        type: float
        description: "Package weight in kg"
      destination:
        type: string
        description: "Destination country code"
```

#### 3. MCP (Model Context Protocol) Tools

HoloDeck supports the Model Context Protocol for standardized tool integration:

```yaml
tools:
  - name: filesystem_tools
    type: mcp
    server: "@modelcontextprotocol/server-filesystem"
    config:
      allowed_directories: ["/workspace/data"]

  - name: github_tools
    type: mcp
    server: "@modelcontextprotocol/server-github"
    config:
      token: "${GITHUB_TOKEN}"
      repositories: ["owner/repo"]

  - name: postgres_tools
    type: mcp
    server: "@modelcontextprotocol/server-postgres"
    config:
      connection_string: "${DATABASE_URL}"
```

#### 4. Prompt-Based Tools (Semantic Functions)

Define AI-powered tools using natural language prompts:

```yaml
tools:
  - name: summarize_text
    type: prompt
    description: "Summarize long text into key points"
    template: |
      Summarize the following text into 3-5 bullet points:

      {{$input}}

      Focus on the main ideas and key takeaways.
    parameters:
      input:
        type: string
        description: "Text to summarize"
    model:
      provider: openai
      name: gpt-4o-mini
      temperature: 0.3
```

#### 5. Plugin Packages

Install pre-built plugin packages from the HoloDeck registry:

```yaml
plugins:
  - package: "@holodeck/plugins-web"
    tools:
      - web_search
      - web_scrape
      - html_to_markdown

  - package: "@holodeck/plugins-data"
    tools:
      - csv_query
      - excel_read
      - json_transform

  - package: "@holodeck/plugins-communication"
    tools:
      - send_email
      - send_slack
      - create_jira_ticket
```

### Evaluations

Built-in evaluation metrics with configurable AI models:

**AI-Powered Metrics:**

- **Groundedness** - Is the response grounded in provided context?
- **Relevance** - Is the response relevant to the user's query?
- **Coherence** - Is the response logically coherent?
- **Safety** - Does the response avoid harmful content?

**NLP Metrics:**

- **F1 Score** - Precision and recall balance
- **BLEU** - Translation/generation quality
- **ROUGE** - Summarization quality
- **METEOR** - Semantic similarity

**Configuration:**

```yaml
evaluations:
  default_model:
    provider: openai
    name: gpt-4o-mini
    temperature: 0.0

  metrics:
    - metric: groundedness
      threshold: 4.0
      model:
        provider: openai
        name: gpt-4o # Use more powerful model for critical metrics
        temperature: 0.0

    - metric: relevance
      threshold: 4.0
      # Uses default_model

    - metric: f1_score
      threshold: 0.85
      # NLP metric, doesn't use AI model
```

### Test Cases

Define structured test scenarios with support for multimodal inputs:

#### Basic Text Test Cases

```yaml
test_cases:
  - name: "Basic FAQ handling"
    input: "What is your return policy?"
    expected_tools: ["search_knowledge_base"]

  - name: "Order status check"
    input: "Where is order #12345?"
    ground_truth: "Your order shipped on Jan 15 and arrives Jan 18"
    expected_tools: ["check_order_status"]
    evaluations:
      - f1_score
      - bleu
```

#### Multimodal Test Cases with Files

**Image Input:**

```yaml
test_cases:
  - name: "Product image analysis"
    input: "What product is shown in this image?"
    files:
      - path: tests/fixtures/product-photo.jpg
        type: image
        description: "Product photograph"
    ground_truth: "The image shows a MacBook Pro laptop"
```

**PDF Document Input:**

```yaml
test_cases:
  - name: "Contract analysis"
    input: "Summarize the key terms in this contract"
    files:
      - path: tests/fixtures/contract.pdf
        type: pdf
        description: "Service agreement contract"
    expected_tools: ["summarize_document"]
```

**Multiple Files (Mixed Media):**

```yaml
test_cases:
  - name: "Insurance claim processing"
    input: "Process this insurance claim with supporting documents"
    files:
      - path: tests/fixtures/claim-form.pdf
        type: pdf
        description: "Claim form"
      - path: tests/fixtures/damage-photo1.jpg
        type: image
        description: "Damage photo 1"
      - path: tests/fixtures/damage-photo2.jpg
        type: image
        description: "Damage photo 2"
    expected_tools: ["claim_processor", "image_analyzer"]
```

---

## 🔄 Experiments & Multi-Agent Orchestration

Group related agents and coordinate their execution using `experiment.yaml`. Experiments enable hypothesis testing, multi-agent workflows, and comparative agent evaluation.

### Single-Agent Experiments

Run multiple agent variants in a single experiment for A/B testing:

```yaml
name: "customer-support-experiment"
description: "A/B test different customer support agent configurations"

agents:
  - name: "support-basic"
    path: agents/support-basic/agent.yaml
    description: "Basic support agent with FAQ search"

  - name: "support-advanced"
    path: agents/support-advanced/agent.yaml
    description: "Advanced support agent with order lookup and ticket creation"

test_cases:
  - name: "Basic FAQ handling"
    input: "What is your return policy?"
    expected_tools: ["search_knowledge_base"]

evaluations:
  - metric: groundedness
    threshold: 4.0
  - metric: relevance
    threshold: 4.0
```

**Run the experiment:**

```bash
# Run all agents in the experiment against all test cases
holodeck experiment run experiment.yaml

# Compare results across all agents
holodeck experiment results experiment.yaml --compare

# Generate report
holodeck experiment report experiment.yaml --format html
```

### Multi-Agent Orchestration

Coordinate multiple agents working together using orchestration patterns from the [Microsoft Agent Framework](https://learn.microsoft.com/en-us/agent-framework/user-guide/workflows/orchestrations/overview).

#### Supported Orchestration Patterns

**1. Sequential Pattern** - Agents execute one after another

```yaml
name: "document-processing-experiment"
description: "Sequential multi-agent document processing workflow"

orchestration:
  pattern: sequential
  agents:
    - name: "document-parser"
      path: agents/document-parser/agent.yaml
      description: "Extract text and structure from documents"

    - name: "entity-extractor"
      path: agents/entity-extractor/agent.yaml
      description: "Extract named entities and relationships"

    - name: "summarizer"
      path: agents/summarizer/agent.yaml
      description: "Generate summary of extracted information"
```

**2. Concurrent (Parallel) Pattern** - Agents execute simultaneously

```yaml
name: "multi-aspect-analysis-experiment"
description: "Parallel analysis of different document aspects"

orchestration:
  pattern: concurrent
  agents:
    - name: "sentiment-analyzer"
      path: agents/sentiment/agent.yaml
      description: "Analyze sentiment and tone"

    - name: "keyword-extractor"
      path: agents/keywords/agent.yaml
      description: "Extract key topics and themes"

    - name: "compliance-checker"
      path: agents/compliance/agent.yaml
      description: "Check for regulatory compliance issues"

  aggregation:
    strategy: merge
    output_format: json
```

**3. Handoff Pattern** - Route to specialized agents based on task type

```yaml
name: "customer-service-system-experiment"
description: "Handoff-based customer service routing"

orchestration:
  pattern: handoff

  router:
    name: "service-router"
    path: agents/service-router/agent.yaml
    description: "Analyzes inquiries and routes to specialists"

  specialists:
    - name: "billing-specialist"
      path: agents/billing-specialist/agent.yaml
      description: "Handles billing and payment inquiries"

    - name: "technical-support"
      path: agents/technical-support/agent.yaml
      description: "Handles technical issues"
```

**4. Group Chat Pattern** - Multiple agents collaborate in discussion

```yaml
name: "research-team-experiment"
description: "Group chat for collaborative research analysis"

orchestration:
  pattern: group_chat

  participants:
    - name: "literature-reviewer"
      path: agents/literature-reviewer/agent.yaml
      role: "Finds and summarizes relevant research papers"

    - name: "data-analyst"
      path: agents/data-analyst/agent.yaml
      role: "Analyzes datasets and validates findings"

  chat_config:
    max_rounds: 10
    termination_condition: "consensus_reached"
    moderator: "literature-reviewer"
```

---

## 📊 Competitive Analysis

HoloDeck fills a critical gap: **the only open-source, self-hosted platform designed specifically for building, testing, and orchestrating AI agents through pure YAML configuration.** Built for software engineers with native CI/CD integration.

### vs. **LangSmith** (LangChain Team)

| Aspect                  | HoloDeck                                                                                     | LangSmith                              |
| ----------------------- | -------------------------------------------------------------------------------------------- | -------------------------------------- |
| **Deployment Model**    | Self-hosted (open-source)                                                                    | **SaaS only** (cloud-dependent)        |
| **CI/CD Integration**   | **Native CLI** - integrates in any CI/CD pipeline (GitHub Actions, GitLab CI, Jenkins, etc.) | API-based, requires cloud connectivity |
| **Agent Definition**    | Pure YAML (no code)                                                                          | Python code + LangChain SDK            |
| **Primary Focus**       | Agent experimentation & deployment                                                           | Production observability & tracing     |
| **Agent Orchestration** | Multi-agent patterns (sequential, concurrent, handoff)                                       | Not designed for multi-agent workflows |
| **Use Case**            | Build agents fast, test hypotheses, deploy locally                                           | Monitor & debug production LLM apps    |
| **Vendor Lock-in**      | None (MIT open-source)                                                                       | Complete (SaaS dependency)             |

---

### vs. **MLflow GenAI** (Databricks)

| Aspect                      | HoloDeck                                                                 | MLflow GenAI                                                 |
| --------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------ |
| **CI/CD Integration**       | **CLI-native** - single commands for test, validate, deploy in pipelines | Python SDK + REST API, requires infrastructure setup         |
| **Infrastructure**          | Lightweight, portable                                                    | **Heavy infrastructure** (ML tracking, Databricks-dependent) |
| **Agent Support**           | Purpose-built for agents                                                 | Not designed for agents; focuses on model evaluation         |
| **Focus**                   | Build and deploy agents                                                  | ML experiment tracking and model comparison                  |
| **Multi-Agent**             | Native orchestration patterns                                            | Single model/variant comparison focus                        |
| **Complexity**              | Minimal (YAML)                                                           | High (ML engineering mindset required)                       |

---

### vs. **Microsoft PromptFlow**

| Aspect                  | HoloDeck                                                                          | PromptFlow                                                  |
| ----------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **CI/CD Integration**   | **CLI-first design** - test, validate, deploy via shell commands in any CI system | Python SDK + Azure-centric tooling, requires infrastructure |
| **Scope**               | **Full agent lifecycle** (build, test, deploy agents)                             | **Individual tools & functions only** (not agent-level)     |
| **Design Target**       | Multi-agent workflows & orchestration                                             | Single tool/AI function development                         |
| **Configuration**       | Pure YAML (100% no-code)                                                          | Visual flow graphs + low-code Python                        |
| **Agent Orchestration** | Native multi-agent patterns (sequential, concurrent, handoff, group chat)         | Not designed for multi-agent orchestration                  |
| **Self-Hosted**         | ✅ Full support                                                                   | ⚠️ Limited (designed for Azure)                             |

---

### Why HoloDeck is Unique

**HoloDeck solves a problem none of these platforms address:**

```
┌──────────────────────────────────────────────────────────┐
│  The Agent Development Gap                               │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  LangSmith    → Production observability (SaaS-only)    │
│  MLflow       → Model tracking (heavy infrastructure)    │
│  PromptFlow   → Function/tool development (not agents)  │
│                                                          │
│  ❌ None support multi-agent orchestration              │
│  ❌ None enable pure no-code agent definition            │
│  ❌ None designed for CI/CD pipeline integration        │
│  ❌ None combine testing + evaluation + deployment      │
│                                                          │
│  ✅ HoloDeck fills ALL these gaps                       │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    HOLODECK PLATFORM                     │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Agent      │  │  Evaluation  │  │  Deployment  │
│   Engine     │  │  Framework   │  │  Engine      │
└──────────────┘  └──────────────┘  └──────────────┘
        │                  │                  │
        ├─ LLM Providers   ├─ AI Metrics     ├─ FastAPI
        ├─ Tool System     ├─ NLP Metrics    ├─ Docker
        ├─ Memory          ├─ Custom Evals   ├─ Cloud Deploy
        └─ Vector Stores   └─ Reporting      └─ Monitoring
```

---

## 🎯 Use Cases

### Customer Support Agent

```bash
holodeck init support --template customer-support
# Pre-configured with: FAQ search, ticket creation, sentiment analysis
```

### Research Assistant

```bash
holodeck init research --template research-assistant
# Pre-configured with: Web search, paper search, summarization
```

### Code Assistant

```bash
holodeck init coder --template code-assistant
# Pre-configured with: Code search, documentation lookup, testing
```

### Sales Agent

```bash
holodeck init sales --template sales-agent
# Pre-configured with: Product search, CRM integration, lead qualification
```

---

## 📊 Monitoring & Observability

HoloDeck provides comprehensive observability with native **OpenTelemetry** support and **Semantic Conventions for Generative AI**.

### OpenTelemetry Integration

HoloDeck automatically instruments your agents with OpenTelemetry traces, metrics, and logs following the [OpenTelemetry Semantic Conventions for Generative AI](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

**Basic Configuration:**

```yaml
# agent.yaml
observability:
  enabled: true
  service_name: "customer-support-agent"

  opentelemetry:
    enabled: true
    endpoint: "http://localhost:4318" # OTLP endpoint
    protocol: grpc # or http/protobuf

    traces:
      enabled: true
      sample_rate: 1.0 # Sample 100% of traces

    metrics:
      enabled: true
      interval: 60 # Export metrics every 60s

    logs:
      enabled: true
      level: info
```

**Export to Observability Platforms:**

```yaml
observability:
  opentelemetry:
    enabled: true
    exporters:
      # Jaeger
      - type: otlp
        endpoint: "http://jaeger:4318"

      # Prometheus (metrics)
      - type: prometheus
        port: 8889

      # Datadog
      - type: otlp
        endpoint: "https://api.datadoghq.com"
        headers:
          DD-API-KEY: "${DATADOG_API_KEY}"
```

### Built-in Metrics

**Request Metrics:**

- `gen_ai.client.operation.duration` - Operation duration histogram
- `gen_ai.client.token.usage` - Token usage counter
- `gen_ai.client.request.count` - Request counter
- `gen_ai.client.error.count` - Error counter

**Agent-Specific Metrics:**

- `holodeck.agent.requests.total` - Total agent requests
- `holodeck.agent.requests.duration` - Request duration histogram
- `holodeck.agent.tokens.total` - Total tokens used
- `holodeck.agent.cost.total` - Total cost (USD)
- `holodeck.tools.invocations.total` - Tool invocation count
- `holodeck.evaluations.score` - Evaluation scores gauge

### Cost Tracking

HoloDeck automatically tracks costs based on token usage and model pricing:

```yaml
observability:
  cost_tracking:
    enabled: true

    # Custom pricing (overrides defaults)
    pricing:
      openai:
        gpt-4o:
          input: 0.0025 # per 1K tokens
          output: 0.0100
        gpt-4o-mini:
          input: 0.00015
          output: 0.00060

    # Cost alerts
    alerts:
      - threshold: 100.00 # USD
        period: daily
        notify: "${ALERT_EMAIL}"
```

---

## 🗺️ Roadmap

- [ ] **v0.1** - Core agent engine + CLI
- [ ] **v0.2** - Evaluation framework
- [ ] **v0.3** - API deployment
- [ ] **v0.4** - Web UI (no-code editor)
- [ ] **v0.5** - Multi-agent orchestration
- [ ] **v0.6** - Enterprise features (SSO, audit logs, RBAC)
- [ ] **v1.0** - Production-ready release

---

## 📚 Documentation

- **[Full Documentation](https://holodeck.dev/docs)**
- **[API Reference](https://holodeck.dev/api)**
- **[Examples](https://github.com/holodeck/holodeck/tree/main/examples)**
- **[Tutorials](https://holodeck.dev/tutorials)**

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

Built with:

- [Semantic Kernel](https://github.com/microsoft/semantic-kernel) - Agent framework, Vector Store abstractions
- [FastAPI](https://fastapi.tiangolo.com/) - API deployment
- [Azure AI Evaluation](https://github.com/Azure/azure-sdk-for-python/tree/azure-ai-evaluation_1.11.2/sdk/evaluation/azure-ai-evaluation) - Evaluation metrics
- [Redis](https://redis.io/) - Vector storage

Inspired by:

- Microsoft PromptFlow
- OpenAI Evals
- LlamaIndex

---

## 💬 Community

- **Discord**: [Join our community](https://discord.gg/holodeck)
- **Twitter**: [@holodeckdev](https://twitter.com/holodeckdev)
- **GitHub Discussions**: [Ask questions](https://github.com/holodeck/holodeck/discussions)

---

<p align="center">
  Made with ❤️ by the HoloDeck team
</p>

<p align="center">
  <a href="https://holodeck.dev">Website</a> •
  <a href="https://holodeck.dev/docs">Docs</a> •
  <a href="https://github.com/holodeck/holodeck/examples">Examples</a> •
  <a href="https://discord.gg/holodeck">Discord</a>
</p>
