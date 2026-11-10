# Baseline measurements (T0)

Measured 2026-11-10 on macOS (Apple silicon), Python 3.14.0, uv 0.10.12, before any 042 change other than T1. Later tasks repeat these measurements and record old against new (plan task T23).

## Method

- Wheels built with `uv build --wheel` from `git archive` copies of each commit, so no untracked work in the worktree is included.
- Each wheel installed into a new `uv venv --python 3.14` with `uv pip install`, constrained to the versions in `uv.lock` (`uv export --frozen --no-dev --no-hashes`) and the `pytz>=2024` override from `[tool.uv]`. Installs are therefore the locked versions, not the newest resolvable.
- Base is `holodeck-ai` with no extras. All-extras is every runtime extra in `pyproject.toml` (all except `dev`): pinecone, postgres, qdrant, chromadb, vectorstores, otel-prometheus, otel-azmon, otel-all, claude-otel, openai-agents, dashboard, deploy, deploy-aws, s3, azure-blob, all-sources, deploy-gcp, deploy-azure, deploy-all, temporal.
- Package count is the number of `*.dist-info` directories in site-packages (includes `holodeck-ai`). Installed bytes is the sum of file sizes in site-packages, taken before the first run (no `__pycache__`).
- `holodeck --help`: wall time of `venv/bin/holodeck --help`; one warm-up run, then the median of 5.
- Test import: `python -X importtime -c "import holodeck.cli.commands.test"`, the module `holodeck test` loads. The value is the cumulative time of that module in the import tree; one warm-up, then the median of 5.
- Idle machine for the timing runs (the first timing pass overlapped other work and was repeated).

## Results

Commit `3454ea6` is `main` before T1. Commit `7d36a70` is this branch after T1 (unused declarations removed, HTTPX replaces Requests).

| Install | Commit | Packages | site-packages (bytes) | site-packages (MiB) | `holodeck --help` (s) | `holodeck test` import (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base | `3454ea6` | 264 | 1,252,412,970 | 1,194 | 5.65 | 4,773 |
| All extras | `3454ea6` | 318 | 1,551,471,472 | 1,480 | 5.54 | 4,816 |
| Base | `7d36a70` | 262 | 1,249,699,684 | 1,192 | 5.47 | 4,725 |
| All extras | `7d36a70` | 316 | 1,548,758,186 | 1,477 | 5.62 | 4,806 |

Raw runs (seconds for `--help`, milliseconds for import), in the order listed above:

- `--help`: 5.66 5.65 5.69 5.62 5.51 / 5.51 5.54 5.70 5.60 5.54 / 5.55 5.47 5.50 5.46 5.45 / 5.68 5.62 5.51 5.64 5.59
- import: 4899 4778 4773 4764 4739 / 4807 4831 4817 4816 4725 / 4725 4771 4705 4713 4733 / 4840 4802 4772 4822 4806

## Observations

- `holodeck --help` takes about as long as importing the `test` command: the help listing appears to load every command module (not checked in code), so lazy loading (PR #430) does not shorten `--help`.
- The `test` import is dominated by `holodeck.lib.evaluators.azure_ai`, which imports `azure.ai.evaluation` (about 3.9 s cumulative), inside `holodeck.lib.test_runner.executor` (about 5.5 s cumulative). Removing Microsoft AI evaluation (FR-030) is the largest import-time item.
- T1 removed 2 packages and 2.7 MB; it did not change startup time.
- Base and all-extras differ by 54 packages and about 286 MiB. The base install already carries the evaluation, SDK and Semantic Kernel stacks.
