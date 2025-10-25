# Configuration Schema Contract

**Feature**: Global Settings and Response Format Configuration
**Date**: 2025-10-25
**Version**: 1.0

## Global Configuration Schema

Configuration files (`~/.holodeck/config.yml` and `config.yml`) conform to this schema.

```yaml
# Global configuration schema (user-level and project-level)
GlobalConfig:
  type: object
  properties:
    providers:
      type: object
      description: "Named LLM provider configurations"
      additionalProperties:
        $ref: "#/components/schemas/LMProvider"

    vectorstores:
      type: object
      description: "Named vectorstore configurations"
      additionalProperties:
        $ref: "#/components/schemas/VectorstoreConfig"

    deployment:
      type: object
      description: "Deployment platform configuration"
      $ref: "#/components/schemas/DeploymentConfig"

  additionalProperties: false
  example:
    providers:
      openai:
        provider: openai
        name: gpt-4o
        temperature: 0.7
        max_tokens: 2000
        api_key: "sk-..."
    vectorstores:
      kb:
        provider: postgres
        connection_string: "postgresql://..."
    deployment:
      type: docker
      settings:
        image: holodeck-agent:latest
```

## Agent Configuration Schema (Response Format & Tools)

Agent configuration (`agent.yaml`) extends with response_format and tools.

```yaml
# Agent configuration (excerpt relevant to this feature)
AgentConfig:
  type: object
  properties:
    name:
      type: string
      description: "Agent name"

    inherit_global:
      type: boolean
      default: true
      description: "Whether to inherit global settings"

    providers:
      type: object
      description: "Agent-specific LLM provider selection/override"
      additionalProperties: true

    response_format:
      oneOf:
        - type: object
          description: "Inline JSON Schema (Basic JSON Schema features)"
          properties:
            type: { type: string }
            properties:
              type: object
              additionalProperties: true
            required:
              type: array
              items: { type: string }
            additionalProperties:
              type: boolean
          additionalProperties: false

        - type: string
          description: "File path to external schema file (relative to project root)"
          pattern: "^[a-zA-Z0-9._/-]+\\.(json|yaml|yml)$"

        - type: "null"
          description: "Explicitly disable response format constraints"

    tools:
      type: array
      description: "Agent-specific tools"
      items:
        type: object
        properties:
          name: { type: string }
          type: { type: string, enum: [vectorstore, function, mcp, prompt, plugin] }
          source: { type: string }
          description: { type: string }
        required: [name, type]
        additionalProperties: true

  additionalProperties: true
  example:
    name: "qa-agent"
    inherit_global: true
    providers:
      default: "openai"
    response_format:
      type: object
      properties:
        question: { type: string }
        answer: { type: string }
        confidence: { type: number }
      required: [question, answer]
    tools:
      - name: search
        type: vectorstore
        source: knowledge_base
```

## Component Schemas

### LMProvider

```yaml
LMProvider:
  type: object
  properties:
    provider:
      type: string
      description: "LLM provider name (openai, anthropic, etc.)"
      enum: [openai, anthropic, azure_openai, custom]

    name:
      type: string
      description: "Model identifier (gpt-4o, claude-3-opus, etc.)"

    temperature:
      type: number
      minimum: 0.0
      maximum: 2.0
      description: "Sampling temperature"

    max_tokens:
      type: integer
      minimum: 1
      description: "Maximum tokens per response"

    api_key:
      type: string
      description: "API key (use environment variables for secrets)"

    base_url:
      type: string
      description: "Custom base URL for provider"

  required: [provider, name]
  additionalProperties: true
  example:
    provider: openai
    name: gpt-4o
    temperature: 0.7
    max_tokens: 2000
    api_key: "${OPENAI_API_KEY}"
```

### VectorstoreConfig

```yaml
VectorstoreConfig:
  type: object
  properties:
    provider:
      type: string
      description: "Vectorstore provider (postgres, redis, etc.)"
      enum: [postgres, redis, pinecone, weaviate, milvus]

    connection_string:
      type: string
      description: "Connection string for the vectorstore"

    options:
      type: object
      description: "Provider-specific options"
      additionalProperties: true

  required: [provider, connection_string]
  additionalProperties: false
  example:
    provider: postgres
    connection_string: "postgresql://user:pass@host/db"
    options:
      pool_size: 20
```

### DeploymentConfig

```yaml
DeploymentConfig:
  type: object
  properties:
    type:
      type: string
      description: "Deployment type"
      enum: [docker, kubernetes, cloud]

    settings:
      type: object
      description: "Deployment-specific settings"
      additionalProperties: true

  required: [type]
  additionalProperties: false
  example:
    type: docker
    settings:
      image: holodeck-agent:latest
      port: 8000
```

### ResponseFormatSchema

```yaml
ResponseFormatSchema:
  type: object
  description: "Basic JSON Schema for response format constraints"
  properties:
    type:
      type: string
      enum: [string, number, integer, boolean, array, object, "null"]
      description: "Data type constraint"

    properties:
      type: object
      description: "Object property definitions"
      additionalProperties:
        $ref: "#/components/schemas/ResponseFormatSchema"

    required:
      type: array
      items: { type: string }
      description: "Required property names"

    additionalProperties:
      type: boolean
      description: "Whether additional properties allowed"

    # NOT SUPPORTED:
    # - $ref, $defs, definitions
    # - anyOf, oneOf, allOf
    # - patternProperties, minLength, maxLength
    # - Custom extensions

  additionalProperties: false
  example:
    type: object
    properties:
      question: { type: string }
      answer: { type: string }
      confidence: { type: number }
    required: [question, answer]
```

---

## Validation Rules

### GlobalConfig Validation

1. `providers[].provider` must match enum values
2. `providers[].temperature` must be 0.0-2.0
3. `providers[].max_tokens` must be positive integer
4. `vectorstores[].provider` must match enum values
5. `deployment.type` must match enum values

### AgentConfig Validation

1. `inherit_global` must be boolean (default: true)
2. `response_format` must be valid Basic JSON Schema if object
3. `response_format` file path must exist (if string)
4. `response_format` keywords must be in allowed list
5. All `response_format` references must be resolvable

### ResponseFormatSchema Validation

1. Schema must be valid JSON/YAML
2. All keywords must be in supported list (no $ref, anyOf, patternProperties, etc.)
3. `type` must match enum values
4. `properties` values must be valid ResponseFormatSchema
5. `required` items must reference existing properties

---

## File Format & Extension Handling

### Supported File Extensions

- `.yml` (preferred)
- `.yaml` (fallback)

### File Discovery

**User-Level**:
1. Check `~/.holodeck/config.yml`
2. If not found, check `~/.holodeck/config.yaml`
3. If both exist, prefer `.yml` and log info

**Project-Level**:
1. Check `config.yml`
2. If not found, check `config.yaml`
3. If both exist, prefer `.yml` and log info

### YAML Parsing Rules

- All YAML 1.2 standard syntax supported
- Environment variables supported: `${VAR_NAME}` (e.g., `${OPENAI_API_KEY}`)
- Comments supported (YAML standard)
- No restrictions on indentation (standard YAML)

---

## Configuration Precedence

```
User-Level GlobalConfig (LOWEST PRIORITY)
    ↓ Merged/overridden by
Project-Level GlobalConfig
    ↓ Merged/overridden by
Agent-Level Config (HIGHEST PRIORITY)
```

---

## Error Response Format

Configuration load errors use this format:

```yaml
ConfigurationError:
  type: object
  properties:
    error_type:
      type: string
      enum:
        - yaml_syntax_error
        - schema_validation_error
        - file_not_found
        - invalid_json_schema
        - unknown_schema_keyword
        - unsupported_provider

    message:
      type: string
      description: "Human-readable error message"

    file:
      type: string
      description: "Path to problematic configuration file"

    line:
      type: integer
      description: "Line number in file (if applicable)"

    details:
      type: string
      description: "Additional details (specific constraint violation, etc.)"

  example:
    error_type: invalid_json_schema
    message: "Invalid JSON Schema in response_format"
    file: agent.yaml
    line: 25
    details: "Unknown keyword 'patternProperties' - use 'properties' instead"
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-10-25 | Initial schema with GlobalConfig, response_format, and inheritance |
