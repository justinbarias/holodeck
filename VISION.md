# 🧪 HoloDeck

**Build, Test, and Deploy AI Agents — No Code Required**

HoloDeck is an open-source experimentation platform that enables teams to create, evaluate, and deploy AI agents through simple YAML configuration. Go from hypothesis to production API in minutes, not weeks.

[![PyPI version](https://badge.fury.io/py/holodeck.svg)](https://badge.fury.io/py/holodeck)
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

**Install plugins:**

```bash
holodeck plugin install @holodeck/plugins-web
holodeck plugin install @holodeck/plugins-data
holodeck plugin list
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
        name: gpt-4o # Use more powerful model for groundedness
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
      scale: 5 # 1-5 scale
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
        pages: [1, 2, 3] # Specific pages to process
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
        slides: [1, 5, 10] # Specific slides
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
        pages: [1, 3, 5] # Specific pages
        extract_images: true # Extract embedded images
        ocr: true # Apply OCR to scanned pages

      - path: tests/fixtures/data.xlsx
        type: excel
        sheet: "Summary" # Specific sheet
        range: "A1:E100" # Cell range

      - path: tests/fixtures/photo.jpg
        type: image
        resize: [800, 600] # Resize before processing
        format: jpeg # Convert format

      - path: tests/fixtures/large-log.txt
        type: text
        encoding: utf-8
        max_lines: 1000 # Limit lines read
        tail: true # Read from end

      - path: tests/fixtures/presentation.pptx
        type: powerpoint
        slides: [1, 2, 3]
        extract_notes: true # Include speaker notes
        extract_images: true # Extract slide images
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
        cache: true # Cache for reuse

      - url: "https://example.com/data/sales.csv"
        type: csv

      - url: "https://example.com/images/product.jpg"
        type: image
    expected_tools: ["document_analyzer"]
```

---

## 🔄 Experiments & Multi-Agent Orchestration

Group related agents and coordinate their execution using `experiment.yaml`. Experiments enable hypothesis testing, multi-agent workflows, and comparative agent evaluation.

### Single-Agent Experiments

Run multiple agent variants in a single experiment for A/B testing and comparative analysis:

**experiment.yaml:**

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

# Option 1: Inline test cases
test_cases:
  - name: "Basic FAQ handling"
    input: "What is your return policy?"
    expected_tools: ["search_knowledge_base"]

  - name: "Order status check"
    input: "Where is my order #12345?"
    ground_truth: "Your order shipped on Jan 15 and arrives Jan 18"
    expected_tools: ["check_order_status"]
    evaluations:
      - f1_score
      - bleu

evaluations:
  - metric: groundedness
    threshold: 4.0
  - metric: relevance
    threshold: 4.0
  - metric: latency
    max_ms: 2000

reporting:
  compare_metrics: true
  output_format: html
  save_to: results/experiment-run-1.html
```

**Option 2: Test cases from file:**

```yaml
name: "customer-support-experiment"
description: "A/B test different customer support agent configurations"

agents:
  - name: "support-basic"
    path: agents/support-basic/agent.yaml
  - name: "support-advanced"
    path: agents/support-advanced/agent.yaml

# Reference external test cases file
test_cases:
  file: tests/comprehensive-tests.yaml

evaluations:
  - metric: groundedness
    threshold: 4.0
  - metric: relevance
    threshold: 4.0
```

**Project structure:**

```
customer-support-experiment/
├── experiment.yaml
├── agents/
│   ├── support-basic/
│   │   ├── agent.yaml
│   │   └── instructions/
│   └── support-advanced/
│       ├── agent.yaml
│       └── instructions/
├── tests/
│   └── comprehensive-tests.yaml
└── results/
    └── experiment-run-1.html
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

**1. Sequential Pattern**

Agents execute one after another. Each agent receives the output of the previous agent as input. Ideal for workflows where later steps depend on earlier results.

**Use case:** Document processing pipeline (parse → extract entities → summarize)

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

test_cases:
  - name: "Simple document"
    input: "Parse this contract and summarize key terms"
    files:
      - path: tests/fixtures/simple-contract.pdf
        type: pdf
    ground_truth: "Key terms: 2-year term, $10k annual fee, auto-renewal"
    evaluations:
      - coherence
      - groundedness

evaluations:
  - metric: coherence
    threshold: 4.0
  - metric: accuracy
    threshold: 0.85
```

**2. Concurrent (Parallel) Pattern**

Agents execute simultaneously with independent contexts. Results are aggregated from all agents. Ideal for scenarios where tasks are independent and can run in parallel.

**Use case:** Multi-aspect analysis (sentiment + keywords + compliance checks)

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

test_cases:
  file: tests/analysis-test-cases.yaml
```

**3. Handoff Pattern**

Agents pass work to specialized agents based on task type or content characteristics. One agent analyzes the input and routes it to the most appropriate specialist agent. Ideal for routing to domain experts.

**Use case:** Customer service routing (billing specialist vs. technical support vs. sales)

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

    - name: "sales-agent"
      path: agents/sales-agent/agent.yaml
      description: "Handles sales inquiries and upsell opportunities"

test_cases:
  - name: "Route billing inquiry"
    input: "Why was I charged twice for my subscription?"
    expected_tools: ["billing-specialist"]
    ground_truth: "Routed to billing specialist and issue resolved"

  - name: "Route technical issue"
    input: "The app keeps crashing on my phone"
    expected_tools: ["technical-support"]
    ground_truth: "Routed to technical support and solution provided"

evaluations:
  - metric: routing_accuracy
    threshold: 0.95
  - metric: resolution_time
    max_ms: 5000
```

**4. Group Chat Pattern**

Multiple agents collaborate in a discussion to solve problems together. Agents can see all previous messages and contribute ideas iteratively. Ideal for brainstorming, debate, and collaborative problem-solving.

**Use case:** Research team collaboration (literature reviewer + data analyst + methodologist)

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

    - name: "methodology-expert"
      path: agents/methodology-expert/agent.yaml
      role: "Ensures research methodology is sound"

  chat_config:
    max_rounds: 10
    termination_condition: "consensus_reached"
    moderator: "literature-reviewer"

test_cases:
  file: tests/research-collaboration-cases.yaml

evaluations:
  - metric: solution_quality
    threshold: 4.5
  - metric: collaboration_score
    threshold: 0.85
```

**5. Magentic Pattern**

A specialized pattern for creating emergent AI behaviors through dynamic agent orchestration. Agents adapt their behavior based on context and feedback. Ideal for complex, unpredictable domains requiring adaptive strategies.

**Use case:** Adaptive problem-solving (agent adjusts strategy based on obstacles)

```yaml
name: "adaptive-problem-solving-experiment"
description: "Magentic pattern for adaptive AI agent coordination"

orchestration:
  pattern: magentic

  primary_agent:
    name: "problem-solver"
    path: agents/problem-solver/agent.yaml
    description: "Main problem-solving agent"

  adaptation_agents:
    - name: "strategy-advisor"
      path: agents/strategy-advisor/agent.yaml
      description: "Suggests alternative strategies when stuck"

    - name: "validator"
      path: agents/validator/agent.yaml
      description: "Validates solutions and provides feedback"

test_cases:
  file: tests/adaptive-problem-solving-cases.yaml

evaluations:
  - metric: adaptability
    threshold: 0.8
  - metric: solution_success_rate
    threshold: 0.9
```

### Experiment Features

**Test Variants:**

```yaml
test_cases:
  - name: "Basic query"
    input: "What is your return policy?"
    expected_tools: ["search_knowledge_base"]

  - name: "Complex query"
    input: "Can I return items after 60 days if I have a receipt and they're in original packaging?"
    expected_tools: ["search_knowledge_base"]

test_variants:
  # Run same test cases for each agent
  - variant: baseline
    description: "Standard test execution"

  - variant: stress
    description: "High-load test execution"
    parameters:
      concurrent_requests: 100
```

**Experiment Versioning & Tags:**

```yaml
name: "customer-support-experiment"
version: "1.2.0"
tags:
  - "production-ready"
  - "cost-optimized"
  - "low-latency"

metadata:
  author: "support-team"
  created: "2024-01-15"
  baseline_experiment: "v1.1.0"
```

**Conditional Execution:**

```yaml
orchestration:
  pattern: sequential
  agents:
    - name: "agent-1"
      path: agents/agent-1/agent.yaml
      condition: "always"

    - name: "agent-2"
      path: agents/agent-2/agent.yaml
      condition: "if_succeeded"

    - name: "agent-3"
      path: agents/agent-3/agent.yaml
      condition: "if_failed"
```

**CLI Commands:**

```bash
# Run experiment
holodeck experiment run experiment.yaml

# Validate orchestration configuration
holodeck experiment run experiment.yaml --validate-orchestration

# Test individual agents in orchestration
holodeck experiment debug experiment.yaml --agent document-parser

# Stream results from parallel execution
holodeck experiment run experiment.yaml --stream

# Generate comparison report
holodeck experiment report experiment.yaml --format html --output results/report.html
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
| **Evaluation**          | Built-in AI + NLP metrics, integrated                                                        | Custom evals, trace-based monitoring   |
| **Use Case**            | Build agents fast, test hypotheses, deploy locally                                           | Monitor & debug production LLM apps    |
| **Vendor Lock-in**      | None (MIT open-source)                                                                       | Complete (SaaS dependency)             |
| **Cost**                | Free (self-hosted)                                                                           | Per-request pricing for tracing        |
| **Infrastructure**      | Your machine (CLI) or simple cloud deployment                                                | Requires LangSmith cloud subscription  |

**Key Difference**: LangSmith is production monitoring/observability as a managed service. HoloDeck is agent development and experimentation as self-hosted infrastructure with native CI/CD.

---

### vs. **MLflow GenAI** (Databricks)

| Aspect                      | HoloDeck                                                                 | MLflow GenAI                                                 |
| --------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------ |
| **CI/CD Integration**       | **CLI-native** - single commands for test, validate, deploy in pipelines | Python SDK + REST API, requires infrastructure setup         |
| **Infrastructure**          | Lightweight, portable                                                    | **Heavy infrastructure** (ML tracking, Databricks-dependent) |
| **Deployment Model**        | Open-source, self-hosted CLI                                             | Enterprise ML platform (often requires Databricks)           |
| **Agent Support**           | Purpose-built for agents                                                 | Not designed for agents; focuses on model evaluation         |
| **Focus**                   | Build and deploy agents                                                  | ML experiment tracking and model comparison                  |
| **Vectorstore Integration** | Native first-class support                                               | External integrations required                               |
| **Multi-Agent**             | Native orchestration patterns                                            | Single model/variant comparison focus                        |
| **Evaluation**              | Integrated, no-code                                                      | Modular evaluation metrics (Python-heavy)                    |
| **Deployment**              | Single CLI command                                                       | Requires ML infrastructure setup                             |
| **Complexity**              | Minimal (YAML)                                                           | High (ML engineering mindset required)                       |
| **Best For**                | Software engineers building agents                                       | Data science teams with ML infrastructure                    |

**Key Difference**: MLflow is a bloated ML infrastructure platform. HoloDeck is a lightweight, CLI-first agent platform designed for CI/CD integration.

---

### vs. **Microsoft PromptFlow**

| Aspect                  | HoloDeck                                                                          | PromptFlow                                                  |
| ----------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **CI/CD Integration**   | **CLI-first design** - test, validate, deploy via shell commands in any CI system | Python SDK + Azure-centric tooling, requires infrastructure |
| **Scope**               | **Full agent lifecycle** (build, test, deploy agents)                             | **Individual tools & functions only** (not agent-level)     |
| **Design Target**       | Multi-agent workflows & orchestration                                             | Single tool/AI function development                         |
| **Configuration**       | Pure YAML (100% no-code)                                                          | Visual flow graphs + low-code Python                        |
| **Agent Orchestration** | Native multi-agent patterns (sequential, concurrent, handoff, group chat)         | Not designed for multi-agent orchestration                  |
| **Evaluation**          | Integrated with agent execution                                                   | Per-tool evaluation nodes                                   |
| **Deployment**          | Local CLI, cloud plugins (simple)                                                 | Azure Container Apps (cloud-focused)                        |
| **Self-Hosted**         | ✅ Full support                                                                   | ⚠️ Limited (designed for Azure)                             |
| **Open Source**         | ✅ MIT (true open-source)                                                         | ✅ MIT (but Azure-first philosophy)                         |
| **Use Case**            | Build complete agents, integrate in CI/CD pipelines                               | Debug and test individual AI functions                      |

**Key Difference**: PromptFlow is a tool development environment. HoloDeck is an agent development and deployment platform with CI/CD as a first-class concern.

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

### Decision Matrix: Choose HoloDeck When You Need...

| Requirement                       | HoloDeck | LangSmith | MLflow | PromptFlow |
| --------------------------------- | -------- | --------- | ------ | ---------- |
| **Self-hosted agent development** | ✅       | ❌        | ⚠️     | ⚠️         |
| **Zero-code agent definition**    | ✅✅     | ❌        | ❌     | ❌         |
| **Multi-agent orchestration**     | ✅✅     | ❌        | ❌     | ❌         |
| **CI/CD pipeline integration**    | ✅✅     | ❌        | ⚠️     | ⚠️         |
| **Local deployment (no cloud)**   | ✅✅     | ❌        | ❌     | ⚠️         |
| **Lightweight (<50MB footprint)** | ✅✅     | N/A       | ❌     | ✅         |
| **Built-in evaluation framework** | ✅       | ✅        | ✅     | ✅         |
| **Production observability**      | ⚠️       | ✅✅      | ✅     | ✅         |
| **Open-source & vendor-free**     | ✅       | ❌        | ✅     | ✅         |
| **Individual tool debugging**     | ⚠️       | ✅        | ✅     | ✅✅       |
| **Enterprise features**           | 🔮 v0.5  | ✅        | ✅     | ✅         |

---

### When to Use HoloDeck

✅ **Choose HoloDeck if you want to**:

- Build AI agents **without writing code** (pure YAML, no Python required)
- Orchestrate **multiple agents** in coordinated workflows (sequential, concurrent, handoff, group chat, magentic patterns)
- Integrate agent testing & validation into **CI/CD pipelines** (GitHub Actions, GitLab CI, Jenkins, etc.)
- Deploy agents instantly to local FastAPI endpoints or cloud
- Test hypotheses rapidly with integrated evaluation metrics
- Use vector search and structured data as first-class tools
- Connect via MCP protocol for standardized integrations
- Stay 100% open-source and self-hosted (no vendor lock-in)
- Iterate from prototype to production in minutes, with version control

❌ **Consider alternatives if you need to**:

- Monitor production LLM apps in real-time → **LangSmith** (but accept SaaS dependency + cost)
- Track ML experiments across teams → **MLflow** (but accept heavy infrastructure + Databricks)
- Debug individual AI functions/tools → **PromptFlow** (but accept limited to single-tool scope)

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

## 🔧 Configuration

### Global Configuration

Create `~/.holodeck/config.yaml`:

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
  rate_limit: 100 # requests per minute
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

    # Jaeger
    exporters:
      - type: otlp
        endpoint: "http://jaeger:4318"
        headers:
          api-key: "${JAEGER_API_KEY}"

      # Prometheus (metrics)
      - type: prometheus
        endpoint: "http://prometheus:9090"
        port: 8889 # Expose metrics on this port

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
          x-honeycomb-dataset: "holodeck"
```

### Semantic Conventions for Generative AI

HoloDeck automatically captures standard GenAI attributes according to OpenTelemetry semantic conventions:

**Trace Attributes:**

```yaml
# Automatically captured for every LLM call
gen_ai.system: "openai" # LLM provider
gen_ai.request.model: "gpt-4o" # Model name
gen_ai.request.temperature: 0.7
gen_ai.request.max_tokens: 1000
gen_ai.request.top_p: 1.0

gen_ai.response.id: "chatcmpl-123" # Response ID
gen_ai.response.model: "gpt-4o-2024-05-13"
gen_ai.response.finish_reasons: ["stop"]

gen_ai.usage.prompt_tokens: 50 # Token usage
gen_ai.usage.completion_tokens: 120
gen_ai.usage.total_tokens: 170

gen_ai.prompt.0.role: "system" # Prompt messages
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

- `holodeck.agent.requests.total` - Total agent requests
- `holodeck.agent.requests.duration` - Request duration histogram
- `holodeck.agent.tokens.total` - Total tokens used
- `holodeck.agent.cost.total` - Total cost (USD)
- `holodeck.tools.invocations.total` - Tool invocation count
- `holodeck.tools.duration` - Tool execution duration
- `holodeck.evaluations.score` - Evaluation scores gauge

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

HoloDeck creates detailed trace spans for every operation:

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
      capture_content: true # Capture prompt/completion content
      capture_tokens: true # Capture token counts
      capture_tools: true # Capture tool calls
      capture_evaluations: true # Capture evaluation results

      # Control content size
      max_content_length: 1000 # Truncate long content

      # Sensitive data filtering
      redact_patterns:
        - pattern: '\b\d{3}-\d{2}-\d{4}\b' # SSN
          replacement: "[REDACTED-SSN]"
        - pattern: '\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b' # Email
          replacement: "[REDACTED-EMAIL]"
```

### Logs with Context

HoloDeck integrates logs with traces for full observability:

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
holodeck monitor agent.yaml

# View traces
holodeck monitor agent.yaml --traces

# View specific metrics
holodeck monitor agent.yaml --metric gen_ai.client.token.usage

# Export to file
holodeck monitor agent.yaml --export metrics.json
```

**Grafana Dashboards:**

```bash
# Export Grafana dashboard
holodeck observability export-dashboard --format grafana

# Export Prometheus config
holodeck observability export-config --format prometheus
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
    enabled: true # Can use both simultaneously
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
      p50: 1000 # ms
      p95: 3000
      p99: 5000

    # Alerts
    alerts:
      - metric: gen_ai.client.operation.duration
        threshold_p99: 5000 # ms
        notify: "${ONCALL_EMAIL}"

      - metric: gen_ai.client.error.count
        threshold: 10
        period: 5m
        notify: "${ONCALL_EMAIL}"
```

---

## 📚 Documentation

- **[Full Documentation](https://holodeck.dev/docs)**
- **[API Reference](https://holodeck.dev/api)**
- **[Examples](https://github.com/holodeck/holodeck/tree/main/examples)**
- **[Tutorials](https://holodeck.dev/tutorials)**

---

## 🗺️ Roadmap & Delivery Status

### ✅ Shipped

**v0.1 — Core agent engine & CLI** (2025-11)

- YAML agent configuration with Pydantic v2 validation and env var substitution
- `holodeck init` with templates and interactive wizard
- `holodeck chat` with streaming and session management
- Tool system: vectorstore, function, MCP (stdio), prompt
- ChromaDB, PostgreSQL (pgvector), Pinecone, Qdrant vector stores
- Ollama provider for local LLMs
- Structured data ingestion (CSV/JSON)
- `holodeck mcp` CLI for discovering and managing MCP servers

**v0.2 — Evaluation framework**

- NLP metrics: F1, BLEU, ROUGE, METEOR
- Azure AI metrics: Groundedness, Relevance, Coherence
- DeepEval metrics: G-Eval, Faithfulness, Answer Relevancy, Contextual Relevancy/Precision/Recall
- Global, per-evaluation, and per-metric model configuration
- Multimodal test inputs (images, PDFs, Office documents)
- Markdown and JSON report generation

**v0.3 — API deployment** (2026-01)

- `holodeck serve` — FastAPI REST + AG-UI endpoints for local agent hosting
- `holodeck deploy build` — container builds via Docker SDK with Jinja2 Dockerfile templates
- `holodeck deploy run/status/destroy` — Azure Container Apps deployment
- Published base image `ghcr.io/justinbarias/holodeck-base:latest` (linux/amd64, linux/arm64)
- Cross-architecture builds (amd64 target from Apple Silicon)
- OpenTelemetry observability with OTLP export (traces, metrics, logs)
- Anthropic tool filtering to reduce token usage

**v0.4 — Hierarchical document search** (2026-02)

- `HierarchicalDocumentTool` with H1-H6 heading-chain tracking
- LLM-based contextual embeddings (~49% retrieval accuracy improvement)
- Domain-aware subsection recognition (legal, legislative, academic, technical, medical, patent)
- Tiered keyword search with Reciprocal Rank Fusion (k=60)
- Native hybrid search for azure-ai-search, weaviate, qdrant, mongodb, azure-cosmos-nosql
- BM25 fallback (rank_bm25) for other providers
- External OpenSearch keyword backend with configurable auth/TLS

**v0.5 — Multi-backend architecture** (2026-02 → 2026-03)

- Claude Agent SDK as a first-class native backend
- Protocol-driven abstraction: `AgentBackend`, `AgentSession`, `ExecutionResult`, `ContextGenerator`
- `BackendSelector` auto-routing by `model.provider`
- Extended thinking, web search, bash, file system, subagents, and permission modes
- `auth_provider` support: `api_key`, `oauth_token`, `bedrock`, `vertex`, `foundry`
- Claude support in `holodeck serve` with pre-flight validation and session actors
- OpenTelemetry GenAI semantic-convention instrumentation via `otel-instrumentation-claude-agent-sdk`
- Real-time tool streaming via Claude SDK `PreToolUse`/`PostToolUse` hooks
- Node.js-aware Dockerfile generation for Claude agent containers

**v0.6 — Platform extensions & remote sources** (2026-03)

- Async tool initialization REST endpoints: `POST`/`GET /tools/{name}/init`, `GET /tools`
- RFC 7807 `ProblemDetail` error responses
- Custom Anthropic-compatible endpoint support (Ollama, LiteLLM, etc.) via `AuthProvider.custom`
- Remote source resolution for S3, Azure Blob, and HTTPS
- Auto-detect deploy extras and pin Azure OpenAI API version

### 🚧 In progress — specs authored, implementation pending

- **#016 — GraphRAG integration** — knowledge graph retrieval with entity extraction and cross-document synthesis
- **#023 — Additional backends** — Google ADK and Microsoft Agent Framework (beyond SK + Claude today)
- **#024 — Claude serve/deploy parity** — remaining user stories beyond Claude `serve` + Node.js Dockerfile support
- **#026 — Claude SDK config additions** — effort level, budget cap, fallback model, disallowed tools
- **#027 — MCP HTTP/SSE transport** — remote MCP servers beyond stdio
- **#028 — YAML hooks system** — declarative tool interception, logging, rejection, modification, webhooks
- **#029 — Subagent orchestration** — multi-agent teams with isolated prompts/tools/MCP per subagent
- **#030 — Skills support** — load custom skills from `.claude/skills/` via YAML

### 📋 Planned

- **Deployment engine completion** — `holodeck deploy push` (registry push) and cloud providers beyond Azure (AWS App Runner, GCP Cloud Run)
- **Experiments & multi-agent orchestration** — `holodeck experiment` CLI with sequential, concurrent, handoff, group chat, and magentic patterns (documented above as aspirational)
- **Plugin ecosystem** — pre-built plugin packages (`@holodeck/plugins-web`, `-data`, `-communication`)
- **Cost tracking & alerts** — token-based cost computation with configurable thresholds and notification hooks
- **`holodeck monitor`** — live CLI metrics view with Grafana/Prometheus dashboard exports
- **Enterprise features** — SSO, audit logs, RBAC
- **Web UI** — no-code visual editor for agents, tools, and evaluations
- **v1.0** — Production-ready release

> **Note:** Features illustrated elsewhere in this document as code examples (e.g., `holodeck experiment`, plugin packages, `holodeck monitor`, cost alerts, Web UI, some multi-agent orchestration patterns) describe the target vision. Consult this status matrix for the authoritative shipped/pending view, or [docs/CHANGELOG.md](docs/CHANGELOG.md) for release-level detail.

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
