# Quick Start: Global Settings and Response Format

**Feature**: Global Settings and Response Format Configuration
**Date**: 2025-10-25

## Overview

HoloDeck now supports:
1. **Three-level configuration hierarchy** for centralizing LLM provider and deployment settings
2. **Response format constraints** to ensure agents generate structured, parseable output
3. **Inheritance and override** mechanisms to reduce duplication across agents

---

## Getting Started: 5 Minutes

### Step 1: Create User-Level Configuration (Optional)

Create `~/.holodeck/config.yml` to share defaults across all projects:

```yaml
providers:
  openai:
    provider: openai
    name: gpt-4o
    temperature: 0.7
    max_tokens: 2000
    api_key: "${OPENAI_API_KEY}"

vectorstores:
  knowledge_base:
    provider: postgres
    connection_string: "${DATABASE_URL}"

deployment:
  type: docker
```

**Note**: Use environment variables (e.g., `${OPENAI_API_KEY}`) for secrets, never commit them.

### Step 2: Create Project-Level Configuration (Optional)

Create `config.yml` in your project root to override user-level defaults:

```yaml
providers:
  openai:
    temperature: 0.5  # Lower temperature for this project
```

### Step 3: Add Response Format to Agent

Edit `agent.yaml` to constrain response structure:

```yaml
name: qa-agent
description: "Question and answer agent with structured responses"

providers:
  default: openai

# Define response format (inline)
response_format:
  type: object
  properties:
    question: { type: string }
    answer: { type: string }
    confidence: { type: number, minimum: 0, maximum: 1 }
    sources: { type: array, items: { type: string } }
  required: [question, answer, confidence]

tools:
  - name: search
    type: vectorstore
    source: knowledge_base
```

**Tip**: Store response format in a separate file for reusability:
```yaml
response_format: schemas/qa_response.json
```

---

## Configuration Inheritance Examples

### Example 1: Full Inheritance

Agent inherits everything from global settings:

```yaml
# ~/.holodeck/config.yml
providers:
  openai:
    temperature: 0.7
    max_tokens: 2000

# agent.yaml
name: simple-agent
# No providers specified → inherits openai defaults
```

**Result**: Agent uses temperature=0.7, max_tokens=2000

### Example 2: Project Override

Project-level config overrides user defaults:

```yaml
# ~/.holodeck/config.yml
providers:
  openai:
    temperature: 0.7

# config.yml (project root)
providers:
  openai:
    temperature: 0.5  # Override for this project

# agent.yaml
name: project-agent
# No providers specified → inherits from project config
```

**Result**: Agent uses temperature=0.5 (project overrides user)

### Example 3: Agent Override

Agent-level config overrides both global levels:

```yaml
# ~/.holodeck/config.yml
providers:
  openai:
    temperature: 0.7

# agent.yaml
name: strict-agent
providers:
  openai:
    temperature: 0.1  # Very low temperature
```

**Result**: Agent uses temperature=0.1

### Example 4: Disable Global Inheritance

Agent ignores all global settings:

```yaml
# ~/.holodeck/config.yml
providers:
  openai:
    temperature: 0.7

# agent.yaml
name: independent-agent
inherit_global: false
providers:
  openai:
    temperature: 0.9
    max_tokens: 1000
```

**Result**: Agent uses ONLY specified config (temperature=0.9, max_tokens=1000)

---

## Response Format Examples

### Example 1: Simple Structured Response

Ensure agent returns consistent object with required fields:

```yaml
response_format:
  type: object
  properties:
    status: { type: string }
    message: { type: string }
    timestamp: { type: string }
  required: [status, message]
```

### Example 2: Array Response

Ensure agent returns list of objects:

```yaml
response_format:
  type: array
  items:
    type: object
    properties:
      id: { type: integer }
      name: { type: string }
      email: { type: string }
    required: [id, name]
```

### Example 3: External Schema File

Reference schema from file (relative to project root):

```yaml
response_format: schemas/search_results.json
```

**schemas/search_results.json**:
```json
{
  "type": "array",
  "items": {
    "type": "object",
    "properties": {
      "title": { "type": "string" },
      "url": { "type": "string" },
      "snippet": { "type": "string" }
    },
    "required": ["title", "url"]
  }
}
```

---

## Common Patterns

### Pattern 1: Multi-Agent Project with Shared LLM Config

```
project/
├── config.yml                # Shared LLM provider
├── agent1.yaml              # Uses shared provider
├── agent2.yaml              # Uses shared provider
└── schemas/
    ├── agent1_response.json
    └── agent2_response.json
```

**config.yml**:
```yaml
providers:
  openai:
    provider: openai
    name: gpt-4o
    api_key: "${OPENAI_API_KEY}"
```

**agent1.yaml**:
```yaml
name: agent1
providers:
  default: openai
response_format: schemas/agent1_response.json
```

### Pattern 2: Development vs. Production Configs

**~/.holodeck/config.yml** (user machine):
```yaml
providers:
  openai:
    temperature: 0.9  # Exploratory
```

**config.yml** (production project):
```yaml
providers:
  openai:
    temperature: 0.1  # Conservative
```

Agents in production inherit stricter config; developers use exploratory defaults locally.

### Pattern 3: Provider Fallback

Define multiple providers in global config, agents choose:

```yaml
# ~/.holodeck/config.yml
providers:
  openai:
    provider: openai
    api_key: "${OPENAI_API_KEY}"
  anthropic:
    provider: anthropic
    api_key: "${ANTHROPIC_API_KEY}"

# agent1.yaml
name: agent1
providers:
  default: openai

# agent2.yaml (uses different provider)
name: agent2
providers:
  default: anthropic
```

---

## Configuration File Locations

### User-Level

```
~/.holodeck/config.yml      # Primary (checked first)
~/.holodeck/config.yaml     # Fallback (checked if .yml not found)
```

**Scope**: Applies to all agents across all projects on this machine

### Project-Level

```
config.yml                  # Primary (checked first)
config.yaml                 # Fallback (checked if .yml not found)
```

**Scope**: Applies to all agents in this project; overrides user-level config

**Both file extensions supported**; `.yml` preferred if both exist.

---

## Error Messages & Troubleshooting

### Error: "Invalid JSON in response_format"

```
Error: Invalid JSON in response_format: Expecting property name enclosed in double quotes at line 3, column 5
```

**Fix**: Check response_format JSON syntax. Use a JSON validator (e.g., jsonlint.com).

### Error: "Response format file not found"

```
Error: Response format file not found: schemas/qa_response.json
```

**Fix**: Check file path is relative to project root and file exists.

### Error: "Unknown JSON Schema keyword"

```
Error: Unknown JSON Schema keyword: patternProperties
```

**Fix**: Use only supported keywords: `type`, `properties`, `required`, `additionalProperties`

### Warning: "LLM provider may not support structured output"

```
Warning: LLM provider 'custom-llm' may not support structured output. Structured responses may not be enforced.
```

**Fix**: Use OpenAI-API compliant provider or update provider configuration.

---

## Next Steps

- **For Teams**: Create `~/.holodeck/config.yml` on development machines for shared LLM provider config
- **For Projects**: Add `config.yml` to project root for project-specific overrides
- **For Structured Output**: Add `response_format` to agents that need consistent response structures
- **For Complex Schemas**: Store response formats in separate files under `schemas/`

---

## Reference

- **Configuration Hierarchy**: user-level → project-level → agent-level (explicit always wins)
- **Response Format Support**: Basic JSON Schema (type, properties, required, additionalProperties)
- **File Discovery**: Both `.yml` and `.yaml` supported; `.yml` preferred
- **Inheritance Control**: Set `inherit_global: false` in agent.yaml to use only agent config
