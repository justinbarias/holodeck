# Product sense

HoloDeck helps users define, test, compare, and deploy agents through YAML.
The experiment must expose evidence: responses, tool use, evaluation results, cost, and latency.

## Product rules

- Keep the ordinary agent authoring path declarative.
- Make configuration errors actionable before expensive runtime work starts.
- Preserve evaluation flexibility across global, run, and metric model choices.
- Support document-heavy, multimodal workflows and expected tool behavior.
- Keep native Claude capabilities available alongside the OpenAI backend.
- Let developers compose Temporal workflows in Python while agents remain YAML-defined.

The [constitution](design-docs/constitution.md) records the governing principles.
[VISION.md](../VISION.md) provides product direction. It is not an implementation inventory.
[Product specs](product-specs/index.md) and source code establish feature scope and status.

## Evaluate a change

Describe the user problem and observable result before choosing an implementation.
Prefer a focused addition that fits the existing configuration and execution contracts.
For a behavior change, supply a reproducible example and useful acceptance evidence.
For dashboard changes, preserve the distinction between saved evidence and live execution.
