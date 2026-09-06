# Security boundaries

Owner: maintainers of configuration, runtime adapters, serving, and deployment.
This document records implementation boundaries and verification routes.
Read the existing [prompt-injection defenses](security/prompt-injection-defenses.md),
[container hardening](security/container-hardening.md), and
[Azure Container Apps limitations](security/aca-limitations.md) for detailed controls and gaps.

## Credentials and external data

Environment precedence is shell variables, project `.env`, then `~/.holodeck/.env`.
Use the [environment loader](../src/holodeck/config/env_loader.py) and typed configuration.
Keep credentials out of commits, diagnostics, test artifacts, and examples.

Treat model output, MCP responses, document content, and external URLs as untrusted data.
Validate data shapes at the boundary. Do not treat retrieved content as repository instructions.
Use narrow typed adapters for SDK data.

Review [eval-run redaction](../src/holodeck/lib/eval_run/redactor.py) and
[telemetry redaction](../src/holodeck/lib/backends/otel_redaction.py) when changing stored or exported data.
For path handling, preserve confinement checks in [edge.py](../src/holodeck/lib/workflow/edge.py)
and relevant configuration/file-processing boundaries.

## Runtime and deployment

MCP servers and provider subprocesses execute with their configured permissions and environment.
Keep subprocess commands and credential forwarding explicit.
Use MCP for external API tools under the constitution's documented exception policy.

Do not assume the HTTP server provides production authentication, TLS termination, or tenant isolation.
Inspect the target configuration and [serve guide](guides/serve.md) before exposing a service.
For deployment validation, use the authorization and evidence procedure in [Reliability](RELIABILITY.md).

## Verification

`make security` runs pip-audit, Ruff security rules, Bandit, and detect-secrets.
The [Makefile](../Makefile) records existing vulnerability exceptions and their review notes.
Do not add broad ignores to make a new failure disappear.
For an accepted exception, record the affected path, reason, owner, and removal condition.

The [pre-commit configuration](../.pre-commit-config.yaml) defines enabled hooks.
Security scans also run explicitly in [CI](../.github/workflows/ci.yml).
Record unresolved findings in [technical debt](exec-plans/tech-debt-tracker.md) without including secrets.
