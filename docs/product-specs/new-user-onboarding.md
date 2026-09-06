# New user onboarding

Owner: CLI and documentation maintainers. Status: existing user journey, indexed for the harness.
The [installation guide](../getting-started/installation.md) and
[quickstart](../getting-started/quickstart.md) define the current instructions.

## User outcome

A new user installs HoloDeck, creates an agent project, configures a provider,
and observes an agent response and evaluation result.
The agent definition remains in YAML. Credentials stay in environment configuration.

## Implementation and evidence

- [CLI initialization](../../src/holodeck/cli/commands/init.py) creates an agent project from a template.
- [Templates](../../src/holodeck/templates/) supply initial configuration and instructions.
- [Configuration](../guides/agent-configuration.md) explains authoring and validation.
- [Provider guidance](../guides/llm-providers.md) explains provider-specific requirements.
- [Evaluation guidance](../guides/evaluations.md) explains test inputs, metrics, and results.

When onboarding changes, exercise the affected template and CLI tests.
For a live walkthrough, record the selected provider and prerequisites without exposing credentials.
Use [Quality score](../QUALITY_SCORE.md) to distinguish local test evidence from a completed live walkthrough.
