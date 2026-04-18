"""EvalRun persistence package.

Contains the atomic writer, redactor, slugifier, and metadata builder for
persisting `EvalRun` JSON artifacts at `<agent_base_dir>/results/<slug>/<ts>.json`.

Public API is populated incrementally as modules land. See
`specs/031-eval-runs-dashboard/plan.md` for the full design.
"""
