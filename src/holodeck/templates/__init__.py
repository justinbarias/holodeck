"""Bundled project templates for HoloDeck agent initialization.

This package contains ready-to-use agent templates that can be scaffolded
via `holodeck init` command. Each template provides a complete project structure
with sample configurations, instructions, tools, and test cases.

**Available Templates:**

1. **Conversational Agent**
   - Multi-turn conversation with context maintenance
   - Memory management and dialogue history
   - Natural language understanding and response generation
   - Use case: Chatbots, dialogue systems, interactive assistants

2. **Customer Support Agent**
   - Knowledge base integration for FAQ answers
   - Ticket creation and escalation flows
   - Sentiment analysis and customer satisfaction tracking
   - Use case: Customer service automation, support chatbots

3. **Research Assistant Agent**
   - Document analysis and summarization
   - Multi-source information synthesis
   - Citation and reference management
   - Use case: Literature review, research automation, knowledge synthesis

**Template Structure:**

Each template includes:

```
template/
├── agent.yaml              # Main agent configuration
├── instructions/           # System prompts and instructions
│   └── *.md or *.txt      # Instruction files (referenced in agent.yaml)
├── tools/                  # Custom Python tools (optional)
│   └── *.py               # Tool implementations
├── tests/                  # Sample test cases
│   └── *.yaml             # Test case definitions
└── data/                   # Sample data and resources
    └── */                  # Data files for tools
```

**Configuration Inheritance:**

Templates can reference:
- Global config from `holodeck.yaml` in parent directory
- Environment-specific overrides
- LLM provider settings and credentials

**Example Usage:**

```bash
# Initialize from conversational template
holodeck init --template conversational --name my-agent

# The created project has:
# - Ready-to-run agent.yaml
# - Example test cases in tests/
# - Sample instructions in instructions/
# - Working tools in tools/
```

**Extending Templates:**

Users can modify templates by editing:
1. agent.yaml: Agent behavior and configuration
2. instructions/: System prompts and guidelines
3. tools/: Add custom Python functions
4. tests/: Add test cases and expected outputs

Attributes:
    TEMPLATES: Dictionary of available templates
    DEFAULT_TEMPLATE: Default template name ('conversational')

See Also:
    holodeck.cli.commands.init: Command for initializing projects
    holodeck.config.loader: Configuration loading and validation
"""
