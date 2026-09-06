# Frontend and interaction design

HoloDeck has a Python Dash dashboard, a Click CLI, and HTTP protocol surfaces.
There is no separate JavaScript application in the current architecture.

## Dashboard

Start at [dashboard/app.py](../src/holodeck/dashboard/app.py).
The [views](../src/holodeck/dashboard/views/) provide summary, explorer, and comparison pages.
[Data loading](../src/holodeck/dashboard/data_loader.py) reads saved evaluation results.
The dashboard does not execute an evaluation when it reads a result file.

Reuse [design tokens](../src/holodeck/dashboard/assets/01-tokens.css) and
[component styles](../src/holodeck/dashboard/assets/02-holodeck.css).
The [design-system reference](references/design-system-reference-llms.txt) points to the same sources.

For interaction changes, exercise loading, empty results, error states, filters, and comparisons that the change affects.
For visual changes, inspect the rendered page at relevant window sizes.
Use committed fixtures or synthetic data without credentials.
Record the observed result in the task or execution plan.

## CLI and protocols

Use Click output in CLI commands and logging in library code.
Keep actionable error messages near the invalid configuration or failed operation.
Use typed AG-UI requests and events at the HTTP boundary.
See the [dashboard guide](guides/dashboard.md), [server guide](guides/serve.md), and [quality map](QUALITY_SCORE.md).
