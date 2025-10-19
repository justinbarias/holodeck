# Installation Guide

Get HoloDeck up and running in minutes.

## Prerequisites

- **Python 3.14+** (check with `python --version`)
- **pip** (usually included with Python)
- **Git** (for development setup)

## Standard Installation

### 1. Install from PyPI

```bash
pip install holodeck-ai
```

This installs the latest stable version and all required dependencies:

- `pydantic` - Configuration validation
- `pyyaml` - YAML parsing
- `python-dotenv` - Environment variable support
- `semantic-kernel` - Agent framework base

### 2. Verify Installation

Check that HoloDeck is installed correctly:

```bash
python -m holodeck --version
# Output: holodeck 0.1.0
```

Try importing the main module:

```python
from holodeck.config.loader import ConfigLoader
print("✓ HoloDeck installed successfully!")
```

## Development Setup

For development or contributing to HoloDeck:

### 1. Clone the Repository

```bash
git clone https://github.com/anthropics/holodeck.git
cd holodeck
```

### 2. Initialize Development Environment

```bash
# Create virtual environment and install all dev dependencies
make init
```

This command:

- Creates a Python virtual environment in `.venv/`
- Installs dependencies (`poetry install`)
- Installs pre-commit hooks
- Configures development tools (black, ruff, mypy)

### 3. Activate Virtual Environment

```bash
# On macOS / Linux
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

You'll see `(.venv)` prefix in your terminal when activated.

### 4. Verify Development Setup

```bash
# Check Python and pytest
python --version  # Should be 3.14+
pytest --version  # Should show pytest available

# Run the test suite
make test
```

## Setup Verification Checklist

After installation, verify everything works:

```bash
# ✓ Check Python version
python --version
# Expected: Python 3.14.x

# ✓ Check HoloDeck is installed
python -c "import holodeck; print(holodeck.__version__)"
# Expected: 0.1.0

# ✓ Check ConfigLoader works
python << 'EOF'
from holodeck.config.loader import ConfigLoader
loader = ConfigLoader()
print("✓ ConfigLoader imported successfully")
EOF

# ✓ Create a test agent.yaml file
cat > test-agent.yaml << 'EOF'
name: "Test Agent"
description: "Verification test agent"
model:
  provider: "openai"
  name: "gpt-4o-mini"
instructions:
  inline: "You are a helpful assistant."
EOF

# ✓ Load and validate the agent (requires OPENAI_API_KEY env var)
python << 'EOF'
from holodeck.config.loader import ConfigLoader
loader = ConfigLoader()
agent = loader.load_agent_yaml("test-agent.yaml")
print(f"✓ Agent loaded: {agent.name}")
EOF
```

## Environment Variables

HoloDeck uses environment variables for API credentials. Set these in your shell or `.env` file:

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
```

### Azure OpenAI

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-key"
export OPENAI_API_VERSION="2024-02-15-preview"
```

### Anthropic

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Managing Environment Variables with .env Files

Create a `.env` file in your project directory:

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

Then load it in your Python script:

```python
from dotenv import load_dotenv
load_dotenv()

from holodeck.config.loader import ConfigLoader
# Now all env vars from .env are available
```

⚠️ **Never commit `.env` files to version control!** Add to `.gitignore`:

```bash
# .gitignore
.env
.env.local
.env.*.local
```

## Development Dependencies (Optional)

For development, install additional tools:

```bash
# Activate your virtual environment first
source .venv/bin/activate  # macOS/Linux

# Install development dependencies
make install-dev
```

This adds:

- `pytest-cov` - Code coverage measurement
- `pytest-xdist` - Parallel test execution
- `black` - Code formatter
- `ruff` - Linter
- `mypy` - Type checker
- `bandit` - Security analyzer

## Common Development Commands

Once installed, use these commands during development:

```bash
make format             # Format code with black + ruff
make format-check       # Check formatting (without modifying)
make lint               # Run linting checks
make type-check         # Type checking with mypy
make security           # Security checks (bandit, safety)
make test               # Run test suite
make test-coverage      # Run tests with coverage report
make test-parallel      # Run tests in parallel
```

## Troubleshooting

### "Python 3.14+ required"

If you see this error, you need a newer Python version:

```bash
# Check your Python version
python --version

# On macOS with Homebrew
brew install python@3.14

# On Ubuntu/Debian
sudo apt-get install python3.14 python3.14-venv

# On Windows, download from python.org
```

### "ModuleNotFoundError: No module named 'holodeck'"

Make sure HoloDeck is installed:

```bash
# Verify installation
pip show holodeck

# Reinstall if needed
pip install --upgrade holodeck
```

### "OPENAI_API_KEY not found"

Ensure your API key is set:

```bash
# Check if env var is set
echo $OPENAI_API_KEY  # macOS/Linux
echo %OPENAI_API_KEY%  # Windows

# Set it if missing (use your actual key)
export OPENAI_API_KEY="sk-..."
```

### Virtual Environment Issues (Development)

If your virtual environment is broken:

```bash
# Remove and recreate
rm -rf .venv

# Reinitialize
make init

# Activate
source .venv/bin/activate
```

## Next Steps

- ✅ Installation complete!
- 📖 [Follow the Quickstart Guide →](quickstart.md)
- 📚 [Read Agent Configuration Guide →](../guides/agent-configuration.md)
- 💡 [Explore Examples →](../examples/README.md)

## Getting Help

- **Installation Issues**: Check [Troubleshooting](#troubleshooting) section
- **GitHub Issues**: Report bugs at [github.com/anthropics/holodeck/issues](https://github.com/anthropics/holodeck/issues)
- **Documentation**: Visit [docs.holodeck.ai](https://docs.holodeck.ai)
