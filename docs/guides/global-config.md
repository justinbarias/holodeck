# Global Configuration Guide

This guide explains HoloDeck's global configuration system for shared settings across agents.

## Overview

Global configuration lives at `~/.holodeck/config.yaml` and provides default settings that apply to all agents. Use global config to:

- Set default LLM providers and credentials
- Define reusable vectorstore connections
- Configure deployment defaults
- Store API keys securely
- Reduce duplication across agent.yaml files

## Basic Structure

```yaml
# ~/.holodeck/config.yaml

providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    organization: my-org

vectorstores:
  default:
    embedding_model: text-embedding-3-small

deployment:
  default_port: 8000
```

## Configuration Precedence

When multiple configuration sources define the same setting, HoloDeck applies them in priority order:

```
1. agent.yaml (Highest Priority)
   ├─ Explicit values in agent configuration
   │
2. Environment Variables (Medium Priority)
   ├─ ${VAR_NAME} patterns in agent.yaml or global config
   │
3. ~/.holodeck/config.yaml (Lowest Priority)
   └─ Global defaults
```

### Precedence Diagram

```
┌─────────────────────────┐
│   agent.yaml explicit   │  Takes precedence
├─────────────────────────┤
│  Environment variables  │  Used if agent.yaml absent
├─────────────────────────┤
│ ~/.holodeck/config.yaml │  Fallback default
└─────────────────────────┘
```

### Examples

#### Example 1: Provider Override

Global config:
```yaml
providers:
  openai:
    model: gpt-4o-mini
    temperature: 0.7
```

Agent config:
```yaml
model:
  provider: openai
  name: gpt-4o       # Overrides global default
  temperature: 0.5   # Overrides global default
```

Result: Agent uses `gpt-4o` at temperature `0.5` (agent config wins)

#### Example 2: Environment Variable

Global config:
```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
```

Agent config:
```yaml
model:
  provider: openai
```

Environment:
```bash
export OPENAI_API_KEY="sk-..."
```

Result: Uses environment variable for API key

#### Example 3: Full Precedence Chain

Global config:
```yaml
providers:
  default_model: gpt-4o-mini

deployment:
  default_port: 8000
```

Agent config:
```yaml
model:
  provider: openai
  # No explicit temperature

deployment:
  port: 8080  # Overrides global
```

Environment:
```bash
export TEMPERATURE=0.5
```

Result: Model uses `gpt-4o-mini`, port is `8080`, temperature is `0.5`

## Providers Section

Defines LLM provider credentials and defaults.

```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    organization: my-org
    model: gpt-4o
    temperature: 0.7

  azure_openai:
    api_key: ${AZURE_OPENAI_KEY}
    api_version: 2024-02-01
    deployment_id: my-deployment
    endpoint: https://my-resource.openai.azure.com/

  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
```

### OpenAI Provider

```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}              # Required
    organization: my-org                    # Optional
    model: gpt-4o                           # Optional: default model
    temperature: 0.7                        # Optional: default temperature
```

### Azure OpenAI Provider

```yaml
providers:
  azure_openai:
    api_key: ${AZURE_OPENAI_KEY}            # Required
    endpoint: https://my-resource.openai.azure.com/  # Required
    api_version: 2024-02-01                 # Required
    deployment_id: my-deployment            # Required
```

### Anthropic Provider

```yaml
providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}           # Required
    model: claude-3-opus                    # Optional: default model
```

## Vectorstores Section

Defines reusable vectorstore configurations.

```yaml
vectorstores:
  default:
    embedding_model: text-embedding-3-small
    chunk_size: 512
    chunk_overlap: 50

  large_docs:
    embedding_model: text-embedding-3-large
    chunk_size: 2048
    chunk_overlap: 256

  structured_data:
    embedding_model: text-embedding-3-small
    meta_fields: [title, source, date]
```

These can be referenced in agent config (future enhancement).

## Deployment Section

Defines deployment defaults.

```yaml
deployment:
  default_port: 8000
  host: 0.0.0.0
  workers: 4
  timeout: 30
```

### Fields

- **default_port**: Port for local API server
- **host**: Bind address (0.0.0.0 for all interfaces)
- **workers**: Number of worker processes
- **timeout**: Request timeout in seconds

## Environment Variables

Replace sensitive values with environment variables using `${VAR_NAME}` syntax:

```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}      # Reads from environment
    organization: my-org             # Literal value
```

### Setting Environment Variables

**On Linux/macOS**:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

**On Windows**:

```cmd
set OPENAI_API_KEY=sk-...
set ANTHROPIC_API_KEY=sk-ant-...
```

**In .env file** (automatic loading):

Create `.env` in project directory:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

HoloDeck automatically loads `.env` file if present.

### Variable Precedence

For `${VARIABLE_NAME}`:

1. Check environment variable
2. Check .env file
3. Return empty string if not found (error at agent runtime)

## File Locations

### Default Location

```
~/.holodeck/config.yaml
```

On different operating systems:

- **Linux**: `/home/username/.holodeck/config.yaml`
- **macOS**: `/Users/username/.holodeck/config.yaml`
- **Windows**: `C:\Users\username\.holodeck\config.yaml`

### Custom Location (Future)

```bash
holodeck --config /path/to/custom.yaml ...
```

## Complete Example

```yaml
# ~/.holodeck/config.yaml

# LLM Provider Credentials
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    organization: acme-corp
    model: gpt-4o
    temperature: 0.7

  azure_openai:
    api_key: ${AZURE_OPENAI_KEY}
    endpoint: https://acme-openai.openai.azure.com/
    api_version: 2024-02-01
    deployment_id: gpt-4-deployment

  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    model: claude-3-sonnet

# Vectorstore Configurations
vectorstores:
  default:
    embedding_model: text-embedding-3-small
    chunk_size: 512
    chunk_overlap: 50

  large:
    embedding_model: text-embedding-3-large
    chunk_size: 2048
    chunk_overlap: 256

  pdf_docs:
    embedding_model: text-embedding-3-small
    meta_fields: [source, page, date]

# Deployment Defaults
deployment:
  default_port: 8000
  host: 0.0.0.0
  workers: 4
  timeout: 30
```

## Usage Patterns

### Pattern 1: Secure API Keys

Keep secrets in global config, reference in agent:

Global config:
```yaml
# ~/.holodeck/config.yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
```

Agent config:
```yaml
# my-agent/agent.yaml
model:
  provider: openai
  name: gpt-4o
  # API key comes from global config
```

Environment:
```bash
export OPENAI_API_KEY="sk-..."
```

### Pattern 2: Organization Defaults

Set defaults for your team:

Global config:
```yaml
# ~/.holodeck/config.yaml
providers:
  openai:
    organization: acme-corp
    temperature: 0.7
    max_tokens: 2000
```

Agent config:
```yaml
# my-agent/agent.yaml
model:
  provider: openai
  name: gpt-4o
  # Uses temperature 0.7 from global config
```

### Pattern 3: Multi-Environment

Use environment variables for environment-specific settings:

Global config:
```yaml
# ~/.holodeck/config.yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY_${ENV}}
```

Set environment:
```bash
export ENV=prod
export OPENAI_API_KEY_prod="sk-prod-..."
export OPENAI_API_KEY_dev="sk-dev-..."
```

## Creating Global Config

### Step 1: Create Directory

```bash
mkdir -p ~/.holodeck
```

### Step 2: Create config.yaml

Create `~/.holodeck/config.yaml`:

```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
```

### Step 3: Set Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
```

### Step 4: Verify

Test by running an agent:

```bash
holodeck test my-agent/agent.yaml
```

## Troubleshooting

### Error: "api_key not found"

- Check global config exists at `~/.holodeck/config.yaml`
- Verify environment variable is set: `echo $OPENAI_API_KEY`
- Check variable name matches in config

### Error: "invalid provider"

- Check spelling of provider in agent.yaml
- Valid providers: `openai`, `azure_openai`, `anthropic`

### Agent ignoring global config

- Verify global config file exists
- Check file permissions: `ls -la ~/.holodeck/`
- Verify YAML syntax: `cat ~/.holodeck/config.yaml`

### Environment variable not expanding

- Check syntax: `${VAR_NAME}` (with braces)
- Verify variable exists: `env | grep VAR_NAME`
- Note: `$VAR_NAME` (without braces) is not expanded

## Best Practices

1. **Keep Secrets Secure**: Never commit API keys to version control
2. **Use Environment Variables**: Store keys in env, not YAML
3. **Global Defaults**: Use global config for shared organization settings
4. **Per-Agent Overrides**: Use agent.yaml for agent-specific settings
5. **Don't Over-Configure**: Keep global config minimal and focused
6. **Document Settings**: Add comments to explain why settings exist
7. **Version Control**: Commit `config.yaml.example` with placeholders, not real keys

## Example: Secure Setup

```bash
# 1. Create global config with placeholders
mkdir -p ~/.holodeck

cat > ~/.holodeck/config.yaml << 'EOF'
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    organization: my-org
EOF

# 2. Set environment variables in shell profile
# Add to ~/.bashrc or ~/.zshrc
export OPENAI_API_KEY="sk-..."

# 3. Create agent config
cat > my-agent/agent.yaml << 'EOF'
name: my-agent

model:
  provider: openai
  name: gpt-4o

instructions:
  inline: "You are helpful."
EOF

# 4. Run agent (global config automatically loaded)
holodeck test my-agent/agent.yaml
```

## Next Steps

- See [Agent Configuration Guide](agent-configuration.md) for agent-specific settings
- See [File References Guide](file-references.md) for path resolution
- See [Environment Variables Documentation](../guides/environment-variables.md) (future)
