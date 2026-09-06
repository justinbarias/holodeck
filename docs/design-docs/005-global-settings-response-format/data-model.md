# Data Model: Configuration Entities & Relationships

**Feature**: Global Settings and Response Format Configuration
**Phase**: 1 (Design)
**Date**: 2025-10-25

## Overview

Configuration uses existing `GlobalConfig` model (src/holodeck/models/config.py) for both user-level and project-level settings. Response format and tools are agent-specific only, not global settings.

---

## Configuration Entity Model

### 1. GlobalConfig (User-Level & Project-Level)

**File Locations**:
- User-level: `~/.holodeck/config.yml` or `~/.holodeck/config.yaml`
- Project-level: `config.yml` or `config.yaml` (at project/folder root)

**Purpose**: Shared LLM provider, vectorstore, and deployment defaults

**Existing Fields** (from src/holodeck/models/config.py):
- `providers` (dict[str, LMProvider]) - Named LLM provider configurations
  - Each provider includes authentication (api_key, credentials, etc.)
- `vectorstores` (dict[str, VectorstoreConfig]) - Named vectorstore configurations
- `deployment` (DeploymentConfig) - Deployment platform settings

**Inheritance Precedence**: Project-level overrides user-level

**Example User-Level Config**:
```yaml
providers:
  openai:
    provider: openai
    name: gpt-4o
    temperature: 0.7
    max_tokens: 2000
    api_key: "sk-..."

vectorstores:
  knowledge_base:
    provider: postgres
    connection_string: "postgresql://..."

deployment:
  type: docker
  settings:
    image: holodeck-agent:latest
```

**Example Project-Level Config**:
```yaml
providers:
  openai:
    temperature: 0.5  # Override user default
  anthropic:
    provider: anthropic
    name: claude-3-opus
    api_key: "sk-ant-..."

vectorstores:
  project_kb:
    provider: postgres
    connection_string: "postgresql://project-db/..."
```

---

### 2. AgentConfiguration (agent.yaml)

**File Location**: `agent.yaml` (in project root or subdirectory)

**Purpose**: Individual agent configuration with optional response format and tools

**Key Fields** (relevant to this feature):

- `inherit_global` (boolean, default: true) - Whether to inherit global settings
  - If `false`: Ignore all global settings; use only explicit agent configuration

- `providers` (dict[str, str] OR object, optional) - Agent-specific provider selection/overrides
  - String format: Reference to named provider from global config (e.g., `"openai"`)
  - Object format: Full provider definition (overrides global)

- `response_format` (object OR string OR null, optional) - Agent-specific response format constraint
  - If object: Inline JSON Schema (Basic JSON Schema features only)
  - If string: File path to external schema file (relative to project root)
  - If null or `~`: Explicitly disable response format constraints despite global setting
  - If omitted: No response format constraint

- `tools` (array, optional) - Agent-specific tools

**Inheritance Logic**:
```
if agent.inherit_global == false:
    use only agent.* configuration
else:
    for each field in agent.yaml:
        if specified (not null/omitted):
            use agent.field (overrides global)
        else:
            check project-level global settings
            if found, use project-level value
            else check user-level global settings
            if found, use user-level value
            else use default or remain unset
```

**Example**:
```yaml
name: qa-agent
description: "Question and answer agent"

# Reference global provider
providers:
  default: openai

# Agent-specific response format
response_format:
  type: object
  properties:
    question: { type: string }
    answer: { type: string }
    sources: { type: array, items: { type: string } }
  required: [question, answer]

# Agent-specific tools
tools:
  - name: search_knowledge_base
    type: vectorstore
    source: knowledge_base
  - name: format_response
    type: function
    file: tools/format.py
```

---

## Response Format Schema

### Type & Constraints

**Type**: JSON object conforming to Basic JSON Schema specification

**Supported Keywords**:
- `type` - Data type (string, number, integer, boolean, array, object, null)
- `properties` - Object property definitions
- `required` - List of required property names
- `additionalProperties` - Boolean; whether additional properties allowed (default: true)

**Unsupported Keywords** (not allowed):
- `$ref`, `$defs`, `definitions` - External schema references
- `anyOf`, `oneOf`, `allOf` - Complex logic operators
- `patternProperties`, `minLength`, `maxLength` - String/pattern constraints
- Custom extensions or unknown keywords

### Validation Rules

1. Schema must be valid JSON/YAML syntax
2. Schema must conform to Basic JSON Schema specification
3. All schema keywords must be recognized (error on unknown keyword)
4. Schema must be loadable and valid at agent configuration load time

### Error Handling

| Error Type | Message | Action |
|-----------|---------|--------|
| Invalid JSON syntax | `Invalid JSON in response_format: [error] at line [N]` | Fail config load |
| Unknown schema keyword | `Unknown JSON Schema keyword: [keyword]` | Fail config load |
| Schema validation error | `Invalid JSON Schema: [constraint violation]` | Fail config load |
| Missing schema file | `Response format file not found: [path]` | Fail config load |
| LLM provider unsupported | `Warning: LLM provider [name] may not support structured output` | Log warning, continue |

### Valid Schema Examples

**Simple Object**:
```yaml
response_format:
  type: object
  properties:
    status: { type: string }
    data: { type: object }
  required: [status]
```

**Array of Objects**:
```yaml
response_format:
  type: array
  items:
    type: object
    properties:
      id: { type: integer }
      name: { type: string }
    required: [id, name]
```

**External File Reference**:
```yaml
response_format: schemas/qa_response.json
```

---

## Configuration Loading & Merging

### File Discovery Sequence

**User-Level Settings**:
1. Check `~/.holodeck/config.yml`
2. If not found, check `~/.holodeck/config.yaml`
3. If both exist, use `.yml` and log info message
4. If neither exist, skip (optional)

**Project-Level Settings**:
1. Check `config.yml` at project root
2. If not found, check `config.yaml` at project root
3. If both exist, use `.yml` and log info message
4. If neither exist, skip (optional)

### Precedence Chain

```
User-Level GlobalConfig (lowest priority)
    ↓ Overridden by
Project-Level GlobalConfig
    ↓ Overridden by
Agent-Level Configuration (highest priority)
```

### Merge Rules

1. **If `agent.inherit_global == false`**: Use ONLY agent.* fields
2. **If `agent.inherit_global == true` (default)**:
   - For each field specified in agent.yaml: Use agent value
   - For each field NOT specified in agent.yaml:
     - Check project-level GlobalConfig
     - If found, use project value
     - Else check user-level GlobalConfig
     - If found, use user value
     - Else use default or remain unset

### Merge Examples

**Example 1: Full Inheritance**
```
User-level GlobalConfig:
  providers:
    openai: (temperature: 0.7, max_tokens: 2000)

Agent config: (no providers specified)

Result: temperature=0.7, max_tokens=2000
```

**Example 2: Project Override**
```
User-level GlobalConfig:
  providers:
    openai: (temperature: 0.7, max_tokens: 2000)

Project-level GlobalConfig:
  providers:
    openai: (temperature: 0.5)

Agent config: (no providers specified)

Result: temperature=0.5, max_tokens=2000 (inherited from user)
```

**Example 3: Agent Override**
```
User-level GlobalConfig:
  providers:
    openai: (temperature: 0.7)

Agent config:
  providers:
    openai: (temperature: 0.2)

Result: temperature=0.2
```

**Example 4: Disable Inheritance**
```
User-level GlobalConfig:
  providers:
    openai: (temperature: 0.7, max_tokens: 2000)

Agent config:
  inherit_global: false
  providers:
    openai: (temperature: 0.9)

Result: temperature=0.9 (max_tokens unset, no inheritance from global)
```

---

## Configuration Loading Lifecycle

### Sequence

```
1. Application Startup
   ↓
2. Load User-Level GlobalConfig
   ├─ Check ~/.holodeck/config.yml|yaml
   ├─ Validate YAML syntax
   ├─ Parse into GlobalConfig model
   └─ Validate LLMProvider definitions
   ↓
3. Load Project-Level GlobalConfig
   ├─ Check config.yml|yaml at project root
   ├─ Validate YAML syntax
   ├─ Parse into GlobalConfig model
   ├─ Merge with user-level (project > user)
   └─ Validate LLMProvider definitions
   ↓
4. For Each Agent Configuration:
   ├─ Load agent.yaml
   ├─ Validate YAML syntax
   ├─ Check inherit_global flag
   ├─ If inherit_global == false:
   │  └─ Use only agent config
   ├─ Else:
   │  ├─ Merge global configs with agent config
   │  └─ Resolve provider references
   ├─ Validate response_format schema if specified
   │  ├─ Check JSON syntax
   │  ├─ Validate schema keywords
   │  └─ Load external schema file if path specified
   ├─ Log warning if LLM provider unsupported
   └─ Configuration ready for agent execution
```

### Error Handling States

| Scenario | Action | Result |
|----------|--------|--------|
| Invalid YAML syntax | Log error with file/line | Configuration load fails |
| Invalid GlobalConfig model | Log validation error | Configuration load fails |
| Missing schema file | Log error with path | Configuration load fails |
| Invalid response_format JSON | Log error with location | Configuration load fails |
| Invalid schema keyword | Log error with keyword | Configuration load fails |
| Unknown provider reference | Log error with provider name | Configuration load fails |
| LLM provider unsupported | Log warning | Configuration loaded with warning |

---

## Constraints & Assumptions

- GlobalConfig model is reused for both user-level and project-level settings
- API keys are part of LMProvider definitions (not separate)
- Response format constraints are agent-specific only
- Tools are agent-specific only
- Configuration files must be valid YAML (errors fail config load)
- File paths in response_format are relative to project root
- OpenAI-API compliant providers assumed; unsupported providers trigger warning
- Agent config completely replaces global settings (no partial inheritance)
- Configuration is immutable after load (changes require agent restart)
