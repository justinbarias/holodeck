# Temporal hardship demo

This demo runs two HoloDeck agents as Temporal activities. Workflow code
extracts evidence, evaluates the versioned affordability table without an LLM,
and asks the second agent to write the gated decision letter.

From the repository root, install the Temporal extra and provide Claude OAuth:

```bash
source .venv/bin/activate
uv sync --extra temporal
export CLAUDE_CODE_OAUTH_TOKEN=your-token
```

Start a local Temporal server. Leave it running in a separate terminal, or
background it before using the two application terminals below:

```bash
temporal server start-dev
```

Terminal 1 hosts the two HoloDeck activities:

```bash
cd sample/temporal-hardship
env -u CLAUDECODE holodeck worker --config worker.yaml
```

Terminal 2 hosts the user-authored workflow, starts one execution, and prints
the result:

```bash
cd sample/temporal-hardship
python run_workflow.py
```

For a Temporal server or queue with different settings, export the same values
in both application terminals before running their commands:

```bash
export TEMPORAL_ADDRESS=localhost:7233
export TEMPORAL_NAMESPACE=default
export TEMPORAL_TASK_QUEUE=hardship
```

The scripts use top-level `policy` and `workflow` modules. Run them from this
directory as shown; they do not import the repository's `tests` package.

The wording varies, but the output has this shape and names the deterministic
verdict and policy version:

```json
{
  "letter": "... affordable ... policy version 2026-06-01.1 ...",
  "tone": "neutral"
}
```
