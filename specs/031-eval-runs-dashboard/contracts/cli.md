# CLI Contract: `holodeck test` and `holodeck test view`

**Feature**: 031-eval-runs-dashboard

This document defines the externally-observable behaviour of the CLI surface. All behaviour here is verified by integration tests under `tests/integration/cli/`.

---

## `holodeck test` (modified)

No new flags. No removed flags. No changed semantics for existing flags.

### New side effect

After tests execute (regardless of pass/fail), the command MUST persist an `EvalRun` to:

```
results/<slugify(agent.name)>/<iso_ts>.json
```

- Emitted even when the user passes `--output` (additive, FR-006).
- Emitted when at least one test result is produced (pass OR fail). If `test_cases` is empty, no run is written.
- Emitted after `--output` report generation, inside the same process.
- Persistence failures (permission denied, disk full) log WARNING, surface a single-line CLI notice, and MUST NOT change the existing exit code (FR-009).
- On success, emit one informational line:
  ```
  EvalRun persisted: results/<slug>/<iso_ts>.json
  ```
  in the normal (non-`--quiet`) output stream. Under `--quiet`, suppress.

### Exit codes

Unchanged:
| Code | Meaning |
|---|---|
| 0 | All tests passed |
| 1 | One or more tests failed |
| 2 | Configuration error |
| 3 | Execution error |
| 4 | Evaluation error |

---

## `holodeck test view` (new)

Registered as a Click subcommand of the `test` command group.

```
Usage: holodeck test view [OPTIONS] [AGENT_CONFIG]

  Launch the Streamlit dashboard scoped to the experiment identified by
  AGENT_CONFIG's `name`. Requires the optional `dashboard` extra.

Arguments:
  AGENT_CONFIG  Path to agent.yaml. [default: agent.yaml]

Options:
  --port INTEGER          Port for the Streamlit server. [default: 8501]
  --no-browser            Do not auto-open the browser.
  --help                  Show this message and exit.
```

### Pre-flight checks (in order)

1. **Resolve `AGENT_CONFIG`**: load via `ConfigLoader`; extract `agent.name` and capture `agent_base_dir` (the directory containing the supplied agent.yaml). On validation error → exit 2.
2. **Check extra**: `importlib.util.find_spec("streamlit")`. If `None`:
   ```
   Dashboard not installed. Install the optional extra:
     uv add 'holodeck-ai[dashboard]'   # or: pip install 'holodeck-ai[dashboard]'
   ```
   Exit code 2. No Python traceback (FR-022, SC-007).
3. **Resolve results dir**: `<agent_base_dir>/results/<slugify(agent.name)>/`. If missing, do **not** error — the empty state is rendered by the app (FR-023). Creation is deferred to the Streamlit app since the dir may legitimately not exist on first launch.

### Network notice

On successful launch, emit a one-line warning before handing off to Streamlit (FR-020):

```
Warning: Streamlit binds to 0.0.0.0 by default. results/ may contain
sensitive conversation/tool data — firewall this port on shared infrastructure.
```

### Subprocess behaviour

- Launched via `subprocess.Popen([sys.executable, "-m", "streamlit", "run", <app_path>, "--server.port=<port>", ...])` (see research.md R2).
- Environment variables passed to the child:
  - `HOLODECK_DASHBOARD_RESULTS_DIR=<absolute path: agent_base_dir/results/<slug>>`
  - `HOLODECK_DASHBOARD_AGENT_NAME=<slugified agent name>`
  - `HOLODECK_DASHBOARD_AGENT_DISPLAY_NAME=<raw agent.name>` (for header rendering)
- Signal handling: Ctrl+C in the parent forwards SIGINT to the child; on child-exit timeout (5 s), SIGKILL.
- Exit code: propagate the Streamlit child's exit code.

### Error taxonomy

| Condition | Exit code | Output |
|---|---|---|
| `streamlit` not installed | 2 | Install hint (stderr); no traceback |
| `AGENT_CONFIG` invalid | 2 | ConfigError message |
| Port in use | 3 | Streamlit's own error; we propagate |
| Ctrl+C | 130 | (conventional) |

---

## Behavioural Invariants (test-enforceable)

| Invariant | Verified by |
|---|---|
| Running `holodeck test` with a valid `agent.yaml` creates exactly one file under `results/<slug>/`. | `test_test_persists_eval_run.py` |
| Running `holodeck test` with `--output` still writes both artifacts. | `test_test_persists_eval_run.py` |
| Persistence failure does not mask the test exit code. | `test_persistence_failure_does_not_mask_exit.py` (chmod 0o400) |
| `holodeck test view` with missing `streamlit` prints a hint and exits 2. | `test_view_command.py` (patched `find_spec`) |
| `holodeck test view` with `streamlit` installed spawns a subprocess with the documented argv. | `test_view_command.py` (patched `subprocess.Popen`) |
| `EvalRun.model_validate_json(path.read_text()) == written_instance`. | `test_eval_run.py` round-trip |
| Auto-version is stable across two runs with identical prompt body. | `test_prompt_version.py` |
| Auto-version differs by at least one character edit. | `test_prompt_version.py` |
| Writer is atomic: interrupted write leaves no `.tmp` in final position and no partial JSON. | `test_writer_atomic.py` (patched `os.replace` to raise) |
