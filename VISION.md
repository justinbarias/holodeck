# 🧪 AgentLab

**Build, Test, and Deploy AI Agents — No Code Required**

AgentLab is an open-source experimentation platform that enables teams to create, evaluate, and deploy AI agents through simple YAML configuration. Go from hypothesis to production API in minutes, not weeks.

[![PyPI version](https://badge.fury.io/py/agentlab.svg)](https://badge.fury.io/py/agentlab)
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
pip install agentlab
```

### Create Your First Agent

```bash
# Initialize a new agent workspace
agentlab init customer-support --template conversational

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
agentlab test agent.yaml

# Interactive testing
agentlab chat agent.yaml
```

**Output:**
```
🧪 Running AgentLab Tests...

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
agentlab deploy agent.yaml --port 8000

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
agentlab deploy agent.yaml --cloud azure --region westus2

# Deploy to AWS Lambda
agentlab deploy agent.yaml --cloud aws --region us-east-1

# Deploy to Cloud Run (GCP)
agentlab deploy agent.yaml --cloud gcp --region us-central1
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
AgentLab supports the Model Context Protocol for standardized tool integration:

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

**Custom MCP Server:**
```yaml
tools:
  - name: custom_mcp
    type: mcp
    server: ./tools/my_mcp_server.py
    transport: stdio
    config:
      api_key: "${CUSTOM_API_KEY}"
```

#### 4. Prompt-Based Tools (Semantic Functions)
Define AI-powered tools using natural language prompts:

**Inline Prompt Tool:**
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

**File-Based Prompt Tool:**
```yaml
tools:
  - name: extract_entities
    type: prompt
    file: tools/prompts/extract_entities.md
    description: "Extract named entities from text"
    parameters:
      text: string
      entity_types: array
```

**Example Prompt File (`tools/prompts/extract_entities.md`):**
```markdown
---
name: extract_entities
description: Extract named entities from text
parameters:
  text:
    type: string
    description: Text to analyze
  entity_types:
    type: array
    description: Types of entities to extract
    default: ["person", "organization", "location"]
model:
  temperature: 0.1
  max_tokens: 500
---

Extract the following types of entities from the text: {{entity_types}}

Text:
{{text}}

Return the results as a JSON object with entity types as keys and lists of entities as values.
```

#### 5. Plugin Packages
Install pre-built plugin packages from the AgentLab registry:

```yaml
plugins:
  - package: "@agentlab/plugins-web"
    tools:
      - web_search
      - web_scrape
      - html_to_markdown

  - package: "@agentlab/plugins-data"
    tools:
      - csv_query
      - excel_read
      - json_transform

  - package: "@agentlab/plugins-communication"
    tools:
      - send_email
      - send_slack
      - create_jira_ticket
```

**Install plugins:**
```bash
agentlab plugin install @agentlab/plugins-web
agentlab plugin install @agentlab/plugins-data
agentlab plugin list
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

**Basic Configuration:**
```yaml
evaluations:
  - metric: groundedness
    threshold: 4.0
    enabled: true

  - metric: relevance
    threshold: 4.0
    enabled: true

  - metric: f1_score
    threshold: 0.85
    enabled: true
    requires: ground_truth
```

**Specify Evaluation Model:**
```yaml
evaluations:
  model:
    provider: openai
    name: gpt-4o-mini
    temperature: 0.0

  metrics:
    - metric: groundedness
      threshold: 4.0
      enabled: true

    - metric: relevance
      threshold: 4.0
      enabled: true

    - metric: coherence
      threshold: 3.5
      enabled: true
```

**Per-Metric Model Configuration:**
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
        name: gpt-4o  # Use more powerful model for groundedness
        temperature: 0.0

    - metric: relevance
      threshold: 4.0
      # Uses default_model

    - metric: safety
      threshold: 4.5
      model:
        provider: azure_openai
        name: gpt-4o
        temperature: 0.0

    - metric: f1_score
      threshold: 0.85
      # NLP metric, doesn't use AI model
```

**Advanced Configuration:**
```yaml
evaluations:
  model:
    provider: azure_openai
    name: gpt-4o-mini
    temperature: 0.0
    max_tokens: 1000
    endpoint: "https://your-resource.openai.azure.com/"

  metrics:
    - metric: groundedness
      threshold: 4.0
      scale: 5  # 1-5 scale
      enabled: true
      fail_on_error: true

    - metric: relevance
      threshold: 4.0
      scale: 5
      enabled: true
      custom_prompt: |
        Evaluate how relevant the response is to the user's query.
        Consider: topic alignment, completeness, directness.
        Rate from 1 (not relevant) to 5 (highly relevant).

    - metric: coherence
      threshold: 3.5
      enabled: true
      retry_on_failure: 3
      timeout_ms: 5000
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

  - name: "Receipt OCR and validation"
    input: "Extract the total amount and date from this receipt"
    files:
      - path: tests/fixtures/receipt.png
        type: image
    expected_tools: ["ocr_tool"]
    ground_truth: "Total: $42.99, Date: 2024-01-15"
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

  - name: "Multi-page invoice processing"
    input: "What is the total amount due across all pages?"
    files:
      - path: tests/fixtures/invoice-2024-01.pdf
        type: pdf
        pages: [1, 2, 3]  # Specific pages to process
    ground_truth: "Total amount due: $1,234.56"
```

**Text File Input:**
```yaml
test_cases:
  - name: "Code review"
    input: "Review this code for potential bugs"
    files:
      - path: tests/fixtures/sample-code.py
        type: text
        language: python
    expected_tools: ["code_analyzer"]

  - name: "Log file analysis"
    input: "Identify errors in this log file"
    files:
      - path: tests/fixtures/application.log
        type: text
        encoding: utf-8
    expected_tools: ["log_analyzer"]
```

**Office Document Input:**
```yaml
test_cases:
  - name: "Excel data analysis"
    input: "What is the total revenue in Q4?"
    files:
      - path: tests/fixtures/sales-data.xlsx
        type: excel
        sheet: "Q4 Summary"
    expected_tools: ["excel_analyzer"]
    ground_truth: "Q4 revenue: $2,450,000"

  - name: "Word document summarization"
    input: "Summarize this product specification document"
    files:
      - path: tests/fixtures/product-spec.docx
        type: word
    expected_tools: ["document_summarizer"]

  - name: "PowerPoint slide extraction"
    input: "What are the key points from this presentation?"
    files:
      - path: tests/fixtures/quarterly-review.pptx
        type: powerpoint
        slides: [1, 5, 10]  # Specific slides
    expected_tools: ["presentation_analyzer"]
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
      - path: tests/fixtures/police-report.pdf
        type: pdf
        description: "Police report"
    expected_tools: ["claim_processor", "image_analyzer"]
    ground_truth: "Claim approved for $5,000 based on damage assessment"

  - name: "Research paper analysis"
    input: "Compare findings across these research papers"
    files:
      - path: tests/fixtures/paper1.pdf
        type: pdf
      - path: tests/fixtures/paper2.pdf
        type: pdf
      - path: tests/fixtures/supplementary-data.xlsx
        type: excel
    expected_tools: ["document_compare", "data_analyzer"]
```

**File Configuration Options:**
```yaml
test_cases:
  - name: "Advanced file processing"
    input: "Analyze these documents"
    files:
      - path: tests/fixtures/report.pdf
        type: pdf
        pages: [1, 3, 5]           # Specific pages
        extract_images: true       # Extract embedded images
        ocr: true                  # Apply OCR to scanned pages

      - path: tests/fixtures/data.xlsx
        type: excel
        sheet: "Summary"           # Specific sheet
        range: "A1:E100"          # Cell range

      - path: tests/fixtures/photo.jpg
        type: image
        resize: [800, 600]        # Resize before processing
        format: jpeg              # Convert format

      - path: tests/fixtures/large-log.txt
        type: text
        encoding: utf-8
        max_lines: 1000           # Limit lines read
        tail: true                # Read from end

      - path: tests/fixtures/presentation.pptx
        type: powerpoint
        slides: [1, 2, 3]
        extract_notes: true       # Include speaker notes
        extract_images: true      # Extract slide images
    expected_tools: ["multi_format_processor"]
```

**File Input from URLs:**
```yaml
test_cases:
  - name: "Remote file analysis"
    input: "Analyze this remote document"
    files:
      - url: "https://example.com/reports/annual-report.pdf"
        type: pdf
        cache: true              # Cache for reuse

      - url: "https://example.com/data/sales.csv"
        type: csv

      - url: "https://example.com/images/product.jpg"
        type: image
    expected_tools: ["document_analyzer"]
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    AGENTLAB PLATFORM                     │
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
agentlab init support --template customer-support
# Pre-configured with: FAQ search, ticket creation, sentiment analysis
```

### Research Assistant
```bash
agentlab init research --template research-assistant
# Pre-configured with: Web search, paper search, summarization
```

### Code Assistant
```bash
agentlab init coder --template code-assistant
# Pre-configured with: Code search, documentation lookup, testing
```

### Sales Agent
```bash
agentlab init sales --template sales-agent
# Pre-configured with: Product search, CRM integration, lead qualification
```

---

## 🔧 Configuration

### Global Configuration

Create `~/.agentlab/config.yaml`:

```yaml
providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
    organization: "${OPENAI_ORG_ID}"

  azure_openai:
    endpoint: "https://your-resource.openai.azure.com/"
    api_key: "${AZURE_OPENAI_KEY}"

vectorstores:
  redis:
    connection: "localhost:6379"

  postgres:
    connection: "postgresql://user:pass@localhost:5432/vectors"

deployment:
  default_port: 8000
  rate_limit: 100  # requests per minute
  auth:
    type: bearer
    token: "${API_TOKEN}"
```

### Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export AZURE_OPENAI_KEY="..."
export REDIS_URL="redis://localhost:6379"
```

---

## 📊 Monitoring & Observability

AgentLab provides comprehensive observability with native **OpenTelemetry** support and **Semantic Conventions for Generative AI**.

### OpenTelemetry Integration

AgentLab automatically instruments your agents with OpenTelemetry traces, metrics, and logs following the [OpenTelemetry Semantic Conventions for Generative AI](https://opentelemetry.io/docs/specs/semconv/gen-ai/).

**Basic Configuration:**
```yaml
# agent.yaml
observability:
  enabled: true
  service_name: "customer-support-agent"

  opentelemetry:
    enabled: true
    endpoint: "http://localhost:4318"  # OTLP endpoint
    protocol: grpc  # or http/protobuf

    traces:
      enabled: true
      sample_rate: 1.0  # Sample 100% of traces

    metrics:
      enabled: true
      interval: 60  # Export metrics every 60s

    logs:
      enabled: true
      level: info
```

**Export to Observability Platforms:**
```yaml
observability:
  opentelemetry:
    enabled: true

    # Jaeger
    exporters:
      - type: otlp
        endpoint: "http://jaeger:4318"
        headers:
          api-key: "${JAEGER_API_KEY}"

    # Prometheus (metrics)
      - type: prometheus
        endpoint: "http://prometheus:9090"
        port: 8889  # Expose metrics on this port

    # Datadog
      - type: otlp
        endpoint: "https://api.datadoghq.com"
        headers:
          DD-API-KEY: "${DATADOG_API_KEY}"

    # Honeycomb
      - type: otlp
        endpoint: "https://api.honeycomb.io"
        headers:
          x-honeycomb-team: "${HONEYCOMB_API_KEY}"
          x-honeycomb-dataset: "agentlab"
```

### Semantic Conventions for Generative AI

AgentLab automatically captures standard GenAI attributes according to OpenTelemetry semantic conventions:

**Trace Attributes:**
```yaml
# Automatically captured for every LLM call
gen_ai.system: "openai"              # LLM provider
gen_ai.request.model: "gpt-4o"       # Model name
gen_ai.request.temperature: 0.7
gen_ai.request.max_tokens: 1000
gen_ai.request.top_p: 1.0

gen_ai.response.id: "chatcmpl-123"   # Response ID
gen_ai.response.model: "gpt-4o-2024-05-13"
gen_ai.response.finish_reasons: ["stop"]

gen_ai.usage.prompt_tokens: 50       # Token usage
gen_ai.usage.completion_tokens: 120
gen_ai.usage.total_tokens: 170

gen_ai.prompt.0.role: "system"       # Prompt messages
gen_ai.prompt.0.content: "You are..."
gen_ai.prompt.1.role: "user"
gen_ai.prompt.1.content: "What is..."

gen_ai.completion.0.role: "assistant"
gen_ai.completion.0.content: "The answer is..."
```

**Custom Attributes:**
```yaml
observability:
  opentelemetry:
    custom_attributes:
      deployment.environment: "production"
      service.version: "1.2.3"
      agent.type: "customer-support"
      business.unit: "sales"
```

### Built-in Metrics

**Request Metrics:**
- `gen_ai.client.operation.duration` - Operation duration histogram
- `gen_ai.client.token.usage` - Token usage counter
- `gen_ai.client.request.count` - Request counter
- `gen_ai.client.error.count` - Error counter

**Agent-Specific Metrics:**
- `agentlab.agent.requests.total` - Total agent requests
- `agentlab.agent.requests.duration` - Request duration histogram
- `agentlab.agent.tokens.total` - Total tokens used
- `agentlab.agent.cost.total` - Total cost (USD)
- `agentlab.tools.invocations.total` - Tool invocation count
- `agentlab.tools.duration` - Tool execution duration
- `agentlab.evaluations.score` - Evaluation scores gauge

**Custom Metrics:**
```yaml
observability:
  custom_metrics:
    - name: customer_satisfaction
      type: gauge
      description: "Customer satisfaction score"
      unit: "score"

    - name: resolution_time
      type: histogram
      description: "Time to resolve customer issue"
      unit: "seconds"
      buckets: [1, 5, 10, 30, 60, 120, 300]
```

### Distributed Tracing

AgentLab creates detailed trace spans for every operation:

```yaml
# Trace hierarchy example
customer-support-agent
├── gen_ai.request (OpenAI GPT-4o)
│   ├── duration: 1.2s
│   ├── tokens: 170
│   └── cost: $0.0034
├── tool.invocation (search_knowledge_base)
│   ├── duration: 0.3s
│   └── results: 5
├── tool.invocation (check_order_status)
│   ├── duration: 0.15s
│   └── order_id: "12345"
├── evaluation (groundedness)
│   ├── duration: 0.8s
│   ├── score: 4.2
│   └── model: gpt-4o-mini
└── evaluation (relevance)
    ├── duration: 0.7s
    └── score: 4.5
```

**Configure Span Details:**
```yaml
observability:
  opentelemetry:
    traces:
      capture_content: true     # Capture prompt/completion content
      capture_tokens: true      # Capture token counts
      capture_tools: true       # Capture tool calls
      capture_evaluations: true # Capture evaluation results

      # Control content size
      max_content_length: 1000  # Truncate long content

      # Sensitive data filtering
      redact_patterns:
        - pattern: '\b\d{3}-\d{2}-\d{4}\b'  # SSN
          replacement: "[REDACTED-SSN]"
        - pattern: '\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
          replacement: "[REDACTED-EMAIL]"
```

### Logs with Context

AgentLab integrates logs with traces for full observability:

```yaml
observability:
  opentelemetry:
    logs:
      enabled: true
      level: info

      # Structured logging
      format: json

      # Include trace context
      include_trace_context: true

      # Log levels per component
      levels:
        agent: info
        tools: debug
        evaluations: info
        llm: debug
```

**Example Log Output:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "info",
  "message": "Agent request completed",
  "trace_id": "a1b2c3d4e5f6",
  "span_id": "f6e5d4c3b2a1",
  "service.name": "customer-support-agent",
  "gen_ai.system": "openai",
  "gen_ai.request.model": "gpt-4o",
  "gen_ai.usage.total_tokens": 170,
  "duration_ms": 2500,
  "tools_used": ["search_knowledge_base"],
  "evaluation_scores": {
    "groundedness": 4.2,
    "relevance": 4.5
  }
}
```

### Dashboards & Visualization

**CLI Monitoring:**
```bash
# View live metrics
agentlab monitor agent.yaml

# View traces
agentlab monitor agent.yaml --traces

# View specific metrics
agentlab monitor agent.yaml --metric gen_ai.client.token.usage

# Export to file
agentlab monitor agent.yaml --export metrics.json
```

**Grafana Dashboards:**
```bash
# Export Grafana dashboard
agentlab observability export-dashboard --format grafana

# Export Prometheus config
agentlab observability export-config --format prometheus
```

**Pre-built Dashboard Templates:**
- Agent Performance Overview
- Token Usage & Cost Analysis
- Tool Invocation Analytics
- Evaluation Score Trends
- Error Rate & Latency Monitoring

### Integration Examples

**Jaeger (Distributed Tracing):**
```yaml
observability:
  opentelemetry:
    enabled: true
    exporters:
      - type: otlp
        endpoint: "http://jaeger:4318"
        protocol: grpc
```

**Prometheus + Grafana (Metrics):**
```yaml
observability:
  opentelemetry:
    exporters:
      - type: prometheus
        port: 8889
```

**Datadog (Full Stack):**
```yaml
observability:
  opentelemetry:
    exporters:
      - type: otlp
        endpoint: "https://api.datadoghq.com"
        headers:
          DD-API-KEY: "${DATADOG_API_KEY}"
```

**LangSmith:**
```yaml
observability:
  langsmith:
    enabled: true
    api_key: "${LANGSMITH_API_KEY}"
    project: "customer-support-agent"

  opentelemetry:
    enabled: true  # Can use both simultaneously
```

**Custom OTLP Collector:**
```yaml
observability:
  opentelemetry:
    exporters:
      - type: otlp
        endpoint: "http://otel-collector:4318"
        protocol: grpc
        compression: gzip
        timeout: 10s
        retry:
          enabled: true
          max_attempts: 3
```

### Cost Tracking

AgentLab automatically tracks costs based on token usage and model pricing:

```yaml
observability:
  cost_tracking:
    enabled: true

    # Custom pricing (overrides defaults)
    pricing:
      openai:
        gpt-4o:
          input: 0.0025   # per 1K tokens
          output: 0.0100
        gpt-4o-mini:
          input: 0.00015
          output: 0.00060

    # Cost alerts
    alerts:
      - threshold: 100.00  # USD
        period: daily
        notify: "${ALERT_EMAIL}"

      - threshold: 1000.00
        period: monthly
        notify: "${ALERT_EMAIL}"
```

### Performance Monitoring

```yaml
observability:
  performance:
    # Automatic performance tracking
    latency_targets:
      p50: 1000   # ms
      p95: 3000
      p99: 5000

    # Alerts
    alerts:
      - metric: gen_ai.client.operation.duration
        threshold_p99: 5000  # ms
        notify: "${ONCALL_EMAIL}"

      - metric: gen_ai.client.error.count
        threshold: 10
        period: 5m
        notify: "${ONCALL_EMAIL}"
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Clone repository
git clone https://github.com/agentlab/agentlab.git
cd agentlab

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .
```

---

## 📚 Documentation

- **[Full Documentation](https://agentlab.dev/docs)**
- **[API Reference](https://agentlab.dev/api)**
- **[Examples](https://github.com/agentlab/agentlab/tree/main/examples)**
- **[Tutorials](https://agentlab.dev/tutorials)**

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

- **Discord**: [Join our community](https://discord.gg/agentlab)
- **Twitter**: [@agentlabdev](https://twitter.com/agentlabdev)
- **GitHub Discussions**: [Ask questions](https://github.com/agentlab/agentlab/discussions)

---

<p align="center">
  Made with ❤️ by the AgentLab team
</p>

<p align="center">
  <a href="https://agentlab.dev">Website</a> •
  <a href="https://agentlab.dev/docs">Docs</a> •
  <a href="https://github.com/agentlab/agentlab/examples">Examples</a> •
  <a href="https://discord.gg/agentlab">Discord</a>
</p>
