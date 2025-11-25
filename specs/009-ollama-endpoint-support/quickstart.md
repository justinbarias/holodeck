# Quickstart: Ollama Endpoint Support

**Feature**: 009-ollama-endpoint-support
**Last Updated**: 2025-11-26
**Estimated Time**: 10 minutes

## Overview

This guide walks you through configuring and using Ollama as an LLM provider in HoloDeck for local, cost-effective agent development and testing.

## Prerequisites

- HoloDeck CLI installed (`holodeck` command available)
- Ollama installed and running (see [Ollama Installation](#ollama-installation))
- At least one Ollama model pulled (see [Model Setup](#model-setup))

## Quick Start (5 Minutes)

### 1. Install and Start Ollama

```bash
# macOS/Linux - Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server
ollama serve
```

Verify Ollama is running:
```bash
curl http://localhost:11434/api/version
# Should return: {"version":"0.x.x"}
```

### 2. Pull an Ollama Model

```bash
# Pull a lightweight model (phi3, ~2.3GB)
ollama pull phi3

# OR pull Llama 3 (larger, more capable, ~4.7GB)
ollama pull llama3

# List available models
ollama list
```

### 3. Create Agent Configuration

Create `agent.yaml`:

```yaml
name: my-ollama-agent
description: Agent using local Ollama model
author: Your Name

model:
  provider: ollama
  name: phi3
  endpoint: http://localhost:11434
  temperature: 0.7
  max_tokens: 1000

instructions:
  inline: |
    You are a helpful AI assistant. Provide concise, accurate responses
    to user questions. Be friendly and professional.

tools: []

test_cases:
  - input: "What is the capital of France?"
    ground_truth: "Paris"
    expected_tools: []
```

### 4. Test Your Agent

```bash
# Run interactive chat
holodeck chat

# Run test cases
holodeck test
```

**Expected Output**:
```
🤖 Starting chat session with agent: my-ollama-agent
💬 You: What is the capital of France?
🤖 Assistant: The capital of France is Paris.
```

---

## Ollama Installation

### macOS

**Option 1: Official Installer**
1. Download from https://ollama.com/download
2. Open `.dmg` file and drag to Applications
3. Run Ollama from Applications folder

**Option 2: Homebrew**
```bash
brew install ollama
ollama serve
```

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
```

### Windows

1. Download installer from https://ollama.com/download
2. Run installer
3. Ollama starts automatically as a service

### Docker (All Platforms)

```bash
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama-data:/root/.ollama \
  ollama/ollama

# Pull model
docker exec ollama ollama pull phi3
```

---

## Model Setup

### Recommended Models

| Model | Size | Use Case | Pull Command |
|-------|------|----------|--------------|
| **phi3** | 2.3GB | Lightweight, fast, good quality | `ollama pull phi3` |
| **llama3** | 4.7GB | Balanced performance and quality | `ollama pull llama3` |
| **mistral** | 4.1GB | Excellent for reasoning tasks | `ollama pull mistral` |
| **codellama** | 3.8GB | Code generation and analysis | `ollama pull codellama` |
| **gemma** | 1.7GB | Smallest, fastest, basic tasks | `ollama pull gemma` |

### Model Management Commands

```bash
# List pulled models
ollama list

# Pull a new model
ollama pull <model-name>

# Remove a model
ollama rm <model-name>

# Show model details
ollama show <model-name>

# Test a model directly
ollama run <model-name>
```

---

## Configuration Examples

### Example 1: Local Development (Basic)

```yaml
name: local-dev-agent
model:
  provider: ollama
  name: phi3
  endpoint: http://localhost:11434
instructions:
  file: instructions/general.txt
```

**Use Case**: Simple local agent for development and testing

---

### Example 2: Multiple Model Comparison

**Agent 1** (`agent-llama3.yaml`):
```yaml
name: llama3-agent
model:
  provider: ollama
  name: llama3
  endpoint: http://localhost:11434
  temperature: 0.0
instructions:
  inline: "You are a helpful assistant."
test_cases:
  - input: "Explain quantum computing"
    ground_truth: "..."
```

**Agent 2** (`agent-phi3.yaml`):
```yaml
name: phi3-agent
model:
  provider: ollama
  name: phi3
  endpoint: http://localhost:11434
  temperature: 0.0
instructions:
  inline: "You are a helpful assistant."
test_cases:
  - input: "Explain quantum computing"
    ground_truth: "..."
```

**Run Comparison**:
```bash
holodeck test agent-llama3.yaml
holodeck test agent-phi3.yaml
# Compare outputs and metrics
```

---

### Example 3: Remote Ollama Server

**Scenario**: Ollama running on a powerful server, accessed from development machine

**.env file**:
```
OLLAMA_ENDPOINT=http://192.168.1.100:11434
OLLAMA_API_KEY=your-api-key-here
```

**agent.yaml**:
```yaml
name: remote-ollama-agent
model:
  provider: ollama
  name: llama3
  endpoint: ${OLLAMA_ENDPOINT}
  api_key: ${OLLAMA_API_KEY}
  temperature: 0.7
  max_tokens: 2000
instructions:
  file: instructions/production.txt
```

**Usage**:
```bash
# Ensure .env is in same directory or parent
holodeck chat
```

---

### Example 4: Code Assistant with CodeLlama

```yaml
name: code-assistant
description: AI assistant for code review and generation
model:
  provider: ollama
  name: codellama
  endpoint: http://localhost:11434
  temperature: 0.2  # Lower temperature for code
  max_tokens: 2000
instructions:
  inline: |
    You are an expert software engineer. Help users with:
    - Code review and bug detection
    - Code generation from specifications
    - Explaining complex code
    - Best practices and optimization suggestions

test_cases:
  - input: "Write a Python function to check if a number is prime"
    expected_tools: []
  - input: "Review this code for bugs: def add(a, b): return a + b"
    expected_tools: []
```

---

## Troubleshooting

### Issue 1: "Connection refused" Error

**Error**:
```
Failed to connect to Ollama endpoint at http://localhost:11434.
Ensure Ollama is running: ollama serve
```

**Solutions**:
1. Check if Ollama is running:
   ```bash
   curl http://localhost:11434/api/version
   ```
2. Start Ollama server:
   ```bash
   ollama serve
   ```
3. Verify endpoint URL in agent.yaml is correct

---

### Issue 2: "Model not found" Error

**Error**:
```
Model 'llama3' not found on Ollama endpoint http://localhost:11434.
Pull the model first: ollama pull llama3
```

**Solutions**:
1. Pull the model:
   ```bash
   ollama pull llama3
   ```
2. List available models:
   ```bash
   ollama list
   ```
3. Update agent.yaml with available model name

---

### Issue 3: Slow Performance

**Symptoms**: Agent responses take 10+ seconds

**Solutions**:
1. **Use smaller model**: Switch from `llama3` to `phi3` or `gemma`
2. **Reduce max_tokens**: Lower `max_tokens` to 500-1000
3. **Check system resources**: Ensure sufficient RAM/CPU
   ```bash
   # macOS/Linux
   top
   ```
4. **Use GPU acceleration** (if available):
   - Ollama automatically uses GPU if detected
   - Verify: `ollama run llama3` should show GPU usage

---

### Issue 4: Environment Variables Not Resolving

**Error**:
```
Field 'model.endpoint': endpoint contains unresolved variable: ${OLLAMA_ENDPOINT}
```

**Solutions**:
1. Create `.env` file in project directory:
   ```
   OLLAMA_ENDPOINT=http://localhost:11434
   ```
2. Verify .env file is in correct location:
   ```bash
   ls -la .env
   ```
3. Alternative: Set environment variables directly:
   ```bash
   export OLLAMA_ENDPOINT=http://localhost:11434
   holodeck chat
   ```

---

### Issue 5: Model Output Quality Issues

**Symptoms**: Responses are nonsensical, repetitive, or incorrect

**Solutions**:
1. **Adjust temperature**: Try 0.7 for creative tasks, 0.2 for factual/code tasks
2. **Improve instructions**: Add more specific guidance in `instructions`
3. **Use larger model**: Switch from `gemma` → `phi3` → `llama3`
4. **Check model version**: Update model to latest version
   ```bash
   ollama pull llama3  # Re-pulls latest version
   ```

---

## Advanced Configuration

### Custom Execution Settings

```yaml
model:
  provider: ollama
  name: llama3
  endpoint: http://localhost:11434
  temperature: 0.8  # More creative (0.0 = deterministic, 2.0 = random)
  max_tokens: 2000  # Longer responses
  top_p: 0.9       # Nucleus sampling (balance between diversity and quality)
```

**Parameter Guide**:
- **temperature**: Controls randomness (0.0-2.0)
  - 0.0-0.3: Factual, consistent (recommended for code, data)
  - 0.5-0.8: Balanced (general use)
  - 1.0-2.0: Creative, diverse (writing, brainstorming)

- **max_tokens**: Maximum response length
  - 500-1000: Short responses (Q&A, chat)
  - 1000-2000: Medium responses (explanations)
  - 2000-4000: Long responses (essays, detailed code)

- **top_p**: Nucleus sampling threshold (0.0-1.0)
  - 0.9: Recommended default (good balance)
  - Lower (0.5-0.7): More focused, less diverse
  - Higher (0.95-1.0): More diverse, potentially creative

---

### Using Ollama for Evaluations

```yaml
evaluations:
  model:  # Use Ollama for evaluation metrics
    provider: ollama
    name: llama3
    endpoint: http://localhost:11434
  metrics:
    - type: groundedness
      threshold: 0.8
    - type: relevance
      threshold: 0.7
```

**Benefits**:
- No API costs for evaluations
- Faster local evaluation
- Works offline

---

## Performance Benchmarks

### Response Time (Average, phi3 model)

| Platform | Hardware | Tokens/sec | Response Time (100 tokens) |
|----------|----------|------------|---------------------------|
| macOS M1 Pro | 32GB RAM | 45-60 | ~2 seconds |
| Linux (Intel i7) | 16GB RAM | 25-35 | ~3-4 seconds |
| Linux (GPU) | NVIDIA RTX 3060 | 80-100 | ~1 second |
| Docker (CPU) | 8GB RAM | 15-25 | ~5-6 seconds |

### Model Comparison (on M1 Pro, 100 tokens)

| Model | Load Time | Response Time | Quality (Subjective) |
|-------|-----------|---------------|---------------------|
| gemma | 1-2s | 1-2s | ⭐⭐⭐ |
| phi3 | 2-3s | 2-3s | ⭐⭐⭐⭐ |
| llama3 | 3-4s | 3-4s | ⭐⭐⭐⭐⭐ |
| mistral | 3-4s | 3-4s | ⭐⭐⭐⭐⭐ |
| codellama | 2-3s | 2-3s | ⭐⭐⭐⭐ (code-focused) |

---

## Next Steps

### 1. Add Vectorstore Tools
Enhance your agent with knowledge base search:
```yaml
tools:
  - type: vectorstore
    name: search_docs
    description: "Search documentation"
    source: ./docs/
    database:
      provider: in-memory
```

### 2. Create Test Cases
Define comprehensive test scenarios:
```yaml
test_cases:
  - input: "What is HoloDeck?"
    ground_truth: "HoloDeck is an agent experimentation platform"
    expected_tools: ["search_docs"]
  - input: "How do I install Ollama?"
    ground_truth: "..."
```

### 3. Run Evaluations
Measure agent performance:
```bash
holodeck test --evaluations
```

### 4. Deploy to Production
Convert agent to API endpoint:
```bash
holodeck deploy --provider docker
```

---

## Resources

### Official Documentation
- **Ollama**: https://ollama.com/docs
- **Ollama Models**: https://ollama.com/library
- **Ollama API**: https://github.com/ollama/ollama/blob/main/docs/api.md

### HoloDeck Resources
- **Feature Specification**: `specs/009-ollama-endpoint-support/spec.md`
- **Data Model**: `specs/009-ollama-endpoint-support/data-model.md`
- **Configuration Schema**: `specs/009-ollama-endpoint-support/contracts/ollama-config-schema.md`

### Community
- **Ollama GitHub**: https://github.com/ollama/ollama
- **Ollama Discord**: https://discord.gg/ollama
- **HoloDeck Issues**: https://github.com/your-org/holodeck/issues

---

## FAQ

### Q: Can I use multiple Ollama models in the same project?
**A**: Yes! Create separate agent.yaml files with different model names and switch between them.

### Q: Does Ollama support streaming responses?
**A**: Yes, Ollama supports streaming, but HoloDeck currently processes complete responses. Streaming support is planned.

### Q: Can I use Ollama with cloud-hosted evaluations (e.g., OpenAI for groundedness)?
**A**: Yes! Use Ollama for agent chat and OpenAI for evaluation metrics:
```yaml
model:
  provider: ollama
  name: phi3
  endpoint: http://localhost:11434

evaluations:
  model:
    provider: openai
    name: gpt-4o
    api_key: ${OPENAI_API_KEY}
```

### Q: How do I update Ollama models?
**A**: Re-pull the model: `ollama pull llama3` downloads the latest version.

### Q: Can I fine-tune Ollama models?
**A**: Yes, using Ollama's Modelfile system. See: https://ollama.com/docs/modelfile

### Q: What's the difference between Ollama and OpenAI?
**A**:
- **Ollama**: Local, free, offline, requires hardware, limited to open models
- **OpenAI**: Cloud, paid, requires internet, no hardware needed, access to GPT-4

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-26 | 1.0.0 | Initial quickstart guide |

---

**Need Help?** Check the [Troubleshooting](#troubleshooting) section or open an issue on GitHub.
