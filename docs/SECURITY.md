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

## NLTK model-artifact exception

Reviewed 2026-09-06. Owner: evaluation and dependency maintainers. Review again by 2026-10-06,
or before changing evaluator dependencies or adding NLTK model persistence.

[GHSA-8mgp-746c-j5xp](https://github.com/nltk/nltk/security/advisories/GHSA-8mgp-746c-j5xp)
(PYSEC-2026-3740 / CVE-2026-81726) affects NLTK through 3.10.3.
The maintainer lists no patched release. Model import/export APIs can bypass NLTK's
`pathsec` restrictions when callers supply paths outside permitted roots.

The reviewed HoloDeck evaluators do not call the affected APIs: `TransitionParser.train/parse`,
`AveragedPerceptron.save/load`, `PerceptronTagger.save_to_json`, or `save_maxent_params`.
HoloDeck's [NLP evaluators](../src/holodeck/lib/evaluators/nlp_metrics.py) use ROUGE scoring
and the Hugging Face METEOR metric. Installed `rouge-score` 0.1.2, `evaluate` 0.4.6,
the cached METEOR module, and Azure AI Evaluation 1.13.7 use scoring, tokenization,
stemming, or fixed corpus downloads. The inspected paths do not pass user-selected
model filenames to the affected APIs. This assessment does not cover arbitrary custom tools.

The audit excludes only this advisory and its aliases. NLTK remains unpatched;
this is a scoped exception for the inspected call paths, not a dependency fix.
Remove the exception when a patched release is available and locked, or before a changed
call path makes the affected APIs reachable. Track follow-up as H-007 in the
[debt tracker](exec-plans/tech-debt-tracker.md).
