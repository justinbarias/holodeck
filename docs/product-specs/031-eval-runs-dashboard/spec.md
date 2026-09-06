# Feature Specification: Eval Runs, Prompt Versioning, and Test View Dashboard

**Feature Branch**: `031-eval-runs-dashboard`
**Spec ID**: 031-eval-runs-dashboard
**Created**: 2026-04-18
**Status**: Draft
**Input**: Extend HoloDeck's evaluation system to (1) persist every `holodeck test` invocation as a strongly-typed `EvalRun` JSON document containing both test results and the full agent configuration snapshot, (2) add YAML frontmatter to system prompt files for prompt versioning via the `python-frontmatter` library, and (3) introduce a new `holodeck test view` command that launches a Streamlit dashboard for tracking experiment runs over time, summarizing performance by metric type with breakdowns for RAG and G-Eval sub-metrics, and exploring individual test cases (agent config, conversation thread, tool calls/args/results, expected tools, evaluations).

## Clarifications

### Session 2026-04-18

- Q: Relationship between `EvalRun` and existing `TestReport`? → A: `EvalRun` wraps `TestReport` and adds a `metadata` field containing the full `Agent` config snapshot plus prompt version info. `TestReport` is unchanged.
- Q: What defines an "experiment" for grouping in the dashboard? → A: One experiment = one `agent.yaml` file, identified by `agent.name`. No new config field required.
- Q: Prompt versioning approach? → A: Use `python-frontmatter` (MIT) to read optional YAML frontmatter from instruction markdown files. Version derived automatically from a content hash; an optional manual `version:` key in frontmatter overrides the auto-version.
- Q: Results directory layout? → A: `results/<agent-name>/<ISO-timestamp>.json` — one JSON file per run, no automatic cleanup or rotation.
- Q: How does the dashboard ship? → A: Optional install extra (`uv add holodeck[dashboard]`). The `holodeck test view` command surfaces a friendly install hint when the extra is missing.
- Q: Redaction allowlist scope? → A: Minimal name-based allowlist (`api_key`, `password`, `secret`) combined with type-driven redaction of any Pydantic `SecretStr` field. Non-matching fields are persisted as-is.
- Q: Dashboard network binding? → A: Bind to `0.0.0.0` (Streamlit default). Network exposure is the user's responsibility; document that `results/` data should be considered sensitive and users should firewall the port if running on shared infrastructure.
- Q: Multimodal test-case file capture in snapshot? → A: Path only (as declared in `agent.yaml`). No hash, no bytes, no extracted text. Reproducibility depends on the user preserving the referenced files.
- Q: Atomic write guarantees for `EvalRun` files? → A: Write to `<path>.tmp`, `fsync`, then `os.replace()` to the final path. Readers see either the old file or the fully-written new file, never a partial one.
- Q: Dashboard filtering for high run counts? → A: Full faceted filtering on the Summary view: date range, prompt version, model name, pass/fail threshold, and frontmatter tag.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Persist Strongly-Typed EvalRun Per Test Invocation (Priority: P1)

A developer runs `holodeck test agent.yaml` and expects every invocation to be persisted automatically as a strongly-typed JSON document under `results/<agent-name>/<ISO-timestamp>.json`, capturing both test outcomes and the exact agent configuration that produced them. This enables historical analysis, reproducibility, and feeds the dashboard in Story 4.

**Why this priority**: Persistence is the foundation for every other story. Without an `EvalRun` artifact on disk, prompt versioning cannot be tied to outcomes, and the dashboard has nothing to read. This story alone delivers value: users can diff two run files in their editor to spot regressions even before the dashboard exists.

**Independent Test**: Run `holodeck test agent.yaml` in a project with no prior runs; verify a file appears at `results/<agent-name>/<ISO-timestamp>.json`; load it back via `EvalRun.model_validate_json(...)` and confirm the test results, summary, and embedded agent configuration round-trip without loss.

**Acceptance Scenarios**:

1. **Given** an agent.yaml with at least one test case, **When** the developer runs `holodeck test agent.yaml`, **Then** a JSON file is written to `results/<agent.name>/<ISO-8601-timestamp>.json` containing a serialized `EvalRun`.
2. **Given** an `EvalRun` JSON file on disk, **When** it is loaded back via the `EvalRun` model, **Then** all fields validate and equal the original run's data (round-trip equivalence).
3. **Given** an `EvalRun` is being constructed, **When** the agent configuration includes secrets in environment-substituted fields (e.g., API keys), **Then** the persisted snapshot redacts those fields before writing to disk.
4. **Given** the `results/` directory does not exist, **When** the test command runs, **Then** the directory tree is created automatically with appropriate permissions.
5. **Given** the user passed `--output report.md`, **When** the test command runs, **Then** the `EvalRun` JSON is still written to `results/` in addition to honoring the `--output` flag (no regression of existing behavior).
6. **Given** an agent name contains characters unsafe for filesystem paths (e.g., `/`, spaces), **When** the run is persisted, **Then** the directory name is sanitized predictably (slugified) without collision risk.

---

### User Story 2 - Prompt Versioning via Frontmatter (Priority: P1)

A developer wants every test run to record exactly which prompt version produced the results so they can attribute regressions to prompt changes. They add YAML frontmatter to their instructions markdown file (e.g., `instructions.md`) with optional metadata (`version`, `author`, `description`, `tags`). HoloDeck reads the frontmatter, stores it on the `EvalRun.metadata.prompt_version`, and uses an automatic content-hash version when no manual version is provided.

**Why this priority**: Without prompt versioning, two runs with identical agent configs but different prompt text would appear indistinguishable in the dashboard. This is critical for the experiment-tracking value proposition. P1 because it must land before the dashboard surfaces "version" as a column.

**Independent Test**: Create an `instructions.md` with a YAML frontmatter block including `version: "1.2"`; run `holodeck test`; open the produced `EvalRun` JSON and verify `metadata.prompt_version.version == "1.2"`. Then remove the manual version and re-run; verify `version` is auto-derived from the content hash and stable across reruns when the prompt body is unchanged.

**Acceptance Scenarios**:

1. **Given** an instructions file with frontmatter `version: "1.2"`, **When** the agent is loaded, **Then** the `EvalRun.metadata.prompt_version.version` field equals `"1.2"`.
2. **Given** an instructions file with frontmatter but no `version` key, **When** the agent is loaded, **Then** the `version` is auto-derived from a deterministic SHA-256 hash of the prompt body (stripped of frontmatter), prefixed with `auto-` (e.g., `auto-3f9a1c`).
3. **Given** an instructions file with no frontmatter at all, **When** the agent is loaded, **Then** the system MUST NOT error; `prompt_version.version` is set to the auto-hash value, and other fields (`author`, `description`, `tags`) are `None`/empty.
4. **Given** the prompt body content is unchanged across two runs, **When** both runs persist `EvalRun` files, **Then** the auto-derived `version` is identical between them.
5. **Given** a user edits one character of the prompt body, **When** the agent is reloaded, **Then** the auto-derived `version` changes.
6. **Given** instructions are provided inline (`instructions.inline`) rather than via file, **When** the agent is loaded, **Then** frontmatter parsing is skipped and the version is auto-derived from the inline content hash.
7. **Given** the frontmatter contains keys outside the documented schema, **When** the agent is loaded, **Then** unknown keys are preserved in `prompt_version.extra` for forward compatibility (no validation error).

---

### User Story 3 - EvalRun Captures Full Agent Configuration Snapshot (Priority: P1)

A developer needs every persisted `EvalRun` to embed a complete, faithful snapshot of the agent configuration that produced the run — model provider, model name, temperature, max_tokens, embedding provider, tools (full configuration), evaluations, claude-specific settings, and the resolved instructions text. This enables exact reproducibility and lets the dashboard display configuration alongside results without consulting the live `agent.yaml` (which may have changed).

**Why this priority**: P1 because reproducibility is the single most valuable property of stored eval data. A run without the originating config is just a number; a run with the config is a scientific record.

**Independent Test**: Run `holodeck test agent.yaml`; modify `agent.yaml` (e.g., change `temperature` from 0.7 to 0.2); load the previously written `EvalRun` and verify `metadata.agent_config.model.temperature == 0.7` (the original, not the current).

**Acceptance Scenarios**:

1. **Given** an `agent.yaml` is loaded and validated into an `Agent` Pydantic model, **When** a test run executes, **Then** the persisted `EvalRun.metadata.agent_config` field contains a deep copy of that `Agent` model serialized to JSON.
2. **Given** the agent config references environment variables (e.g., `${OPENAI_API_KEY}`), **When** the snapshot is persisted, **Then** secret-bearing fields (api_key, auth tokens, connection strings) are redacted to `"***"` while non-secret env-substituted values (model names, endpoints) are preserved as resolved values.
3. **Given** the agent uses tools (vectorstore, MCP, function), **When** the run is persisted, **Then** every tool's full configuration appears in `metadata.agent_config.tools` exactly as it was at run time.
4. **Given** the agent configures Claude SDK options (`claude:` block with extended_thinking, web_search, subagents, etc.), **When** the run is persisted, **Then** the entire `claude` configuration is captured under `metadata.agent_config.claude`.
5. **Given** the `agent.yaml` is later modified, **When** an old `EvalRun` JSON is loaded, **Then** the snapshot reflects the historical state (not the current state on disk).
6. **Given** an `EvalRun` JSON file from a prior run, **When** the user wants to reproduce it, **Then** they have enough information in `metadata.agent_config` to reconstruct the exact agent that ran (modulo redacted secrets and external state like vector store contents).

---

### User Story 4 - Test View Dashboard: Summary Trend Analysis (Priority: P2)

A developer runs `holodeck test view` (or `holodeck test view agent.yaml`) and is presented with a Streamlit dashboard in their browser showing a Summary view: all runs for the current experiment (= the agent identified by the supplied `agent.yaml`), trends over time for pass rate and metric scores, and breakdowns by evaluation type (`standard`, `rag`, `geval`) with further drill-downs for RAG sub-metrics (faithfulness, answer_relevancy, contextual_precision, contextual_recall, contextual_relevancy) and individual G-Eval custom metrics by name.

**Why this priority**: P2 because while the dashboard is the most visible feature, it depends on Stories 1-3 for data. Once persistence is in place, value compounds quickly: developers can spot regressions, validate prompt experiments, and compare model variants visually.

**Independent Test**: With at least 3 prior `EvalRun` JSON files for the same agent in `results/<agent-name>/`, run `holodeck test view`; verify the dashboard launches in a browser, displays a Summary tab listing all 3 runs sorted by timestamp, shows a line chart of pass rate over time, and renders separate breakdown panels for `standard`, `rag` (with sub-metric stacks), and `geval` (with per-custom-name stacks).

**Acceptance Scenarios**:

1. **Given** the user runs `holodeck test view` in a directory with `agent.yaml`, **When** the dashboard launches, **Then** it auto-discovers `results/<agent.name>/*.json`, loads them, and displays a Summary tab.
2. **Given** the Summary tab is displayed, **When** the user views trend charts, **Then** there is a pass-rate-over-time line chart and per-metric average-score lines, with the x-axis showing run timestamp.
3. **Given** runs include both RAG and G-Eval evaluations, **When** the Summary view renders, **Then** there are distinct breakdown panels for each evaluation type, with RAG further split by sub-metric and G-Eval split by custom metric `name`.
4. **Given** the user has runs for multiple agents in `results/`, **When** they launch `holodeck test view agent.yaml`, **Then** only runs for the agent identified by `agent.yaml`'s `name` field are shown (other agents excluded).
5. **Given** no `results/<agent-name>/` directory exists, **When** the dashboard launches, **Then** an empty-state panel explains "No runs found yet — execute `holodeck test` to generate one" with no errors.
6. **Given** the `holodeck[dashboard]` extra is not installed, **When** the user runs `holodeck test view`, **Then** the CLI prints an actionable install hint (`uv add holodeck[dashboard]` or `pip install holodeck[dashboard]`) and exits non-zero without a Python traceback.
7. **Given** the user closes the browser tab, **When** they Ctrl+C in the terminal, **Then** the Streamlit subprocess terminates cleanly.

---

### User Story 5 - Test View Dashboard: Run Explorer with Per-Test Detail (Priority: P2)

From the dashboard, a developer drills into an individual run and then into an individual test case to inspect: the agent configuration that produced it (model, temperature, prompt version, tool list), the full conversation thread (user input, agent response), all tool calls highlighted with name, arguments, and results, the configured expected tools, the test case's evaluations with scores and reasoning, and the ground-truth comparison.

**Why this priority**: P2 because deep inspection is the second-order value after the trend overview. Without it, developers can identify *which* run regressed but not *why*. With it, they can read transcripts, review tool args, and form hypotheses without leaving the dashboard.

**Independent Test**: From the Summary tab, click into a single run; verify a list of test cases is shown; click a test case; verify the side panel displays the agent config snapshot, the rendered conversation, every tool call with collapsible JSON for args and results, the expected-tools list with match indicators, and the per-metric evaluation results (score, threshold, pass/fail, reasoning when present).

**Acceptance Scenarios**:

1. **Given** the user clicks a run in the Summary view, **When** the Explorer view opens, **Then** it lists every test case with name, pass/fail badge, and headline metric scores.
2. **Given** the user clicks a test case in the Explorer, **When** the detail panel renders, **Then** it shows the agent configuration snapshot (model provider/name/temperature, embedding provider, prompt version + frontmatter metadata, tool names).
3. **Given** the test case has a conversation, **When** the detail panel renders, **Then** the user input and agent response are displayed in a chat-style layout.
4. **Given** the test case includes tool calls, **When** the detail panel renders, **Then** each tool call is highlighted (visually distinct from chat messages) with the tool name, arguments shown as formatted JSON, and the tool result shown as formatted JSON.
5. **Given** the test case has `expected_tools` configured, **When** the detail panel renders, **Then** each expected tool is shown with a check/cross indicating whether it was actually called.
6. **Given** the test case has `metric_results`, **When** the detail panel renders, **Then** every metric is shown with name, score, threshold, pass/fail status, and (where present) the LLM reasoning explanation.
7. **Given** a test case had errors during execution, **When** the detail panel renders, **Then** errors are displayed prominently in an error block with the full message.
8. **Given** a tool call result is large (e.g., 100KB+ JSON), **When** the detail panel renders, **Then** the result is collapsed by default with an expand option to avoid overwhelming the UI.

---

### Edge Cases

- What happens when two `holodeck test` invocations finish in the same second? → Use ISO-8601 with millisecond precision in the filename to avoid collision; if a collision still occurs, append a short random suffix.
- What happens when `agent.name` is missing or empty? → Configuration validation already rejects empty names; persistence relies on validated config so this case cannot reach persistence.
- What happens when an instructions file has malformed YAML frontmatter (invalid YAML)? → Surface a clear `ConfigError` at agent load time pointing to the file and the YAML parse error; do not silently fall through.
- What happens when the dashboard reads a corrupt or partial `EvalRun` JSON file (e.g., interrupted write)? → Skip the file with a logged warning; the dashboard renders the rest. Does not crash.
- What happens when the `EvalRun` schema evolves and an older file lacks newly added fields? → Pydantic's optional/default-bearing fields handle additive changes gracefully; the dashboard tolerates missing optional fields.
- What happens when a user has 1000+ runs in `results/<agent-name>/`? → Dashboard loads in under 5 seconds (P95) and paginates the run list; trend charts aggregate efficiently.
- What happens when redaction encounters a secret-bearing field whose name is unknown to the redactor? → The redactor uses a minimal name-based allowlist (`api_key`, `password`, `secret`) plus type-driven redaction of any `SecretStr`-typed field. Fields outside both rules are persisted as-is. This relies on Pydantic models correctly typing secret fields as `SecretStr`; secret fields typed as plain `str` with non-allowlisted names will leak and must be fixed at the model layer.
- What happens when the `instructions.file` path is relative? → Resolved against the agent.yaml's directory (existing behavior); frontmatter parsing follows the same resolution.

## Requirements *(mandatory)*

### Functional Requirements

#### Persistence (`EvalRun`)

- **FR-001**: System MUST define an `EvalRun` Pydantic model that wraps the existing `TestReport` and adds a `metadata` field of a new `EvalRunMetadata` type.
- **FR-002**: `EvalRunMetadata` MUST contain (a) a deep snapshot of the validated `Agent` configuration, (b) a `PromptVersion` describing the system prompt at run time, and (c) `created_at` (ISO-8601 timestamp), `holodeck_version`, `cli_args` (sanitized command-line invocation), and `git_commit` (when available).
- **FR-003**: System MUST persist an `EvalRun` JSON file at `results/<slugified-agent-name>/<ISO-8601-timestamp>.json` for every successful or failed `holodeck test` invocation that produced at least one test result.
- **FR-004**: System MUST create the `results/` directory tree if it does not exist.
- **FR-005**: System MUST redact secret-bearing fields from the embedded agent configuration snapshot before writing to disk using a two-rule policy: (a) name-based allowlist of exactly `api_key`, `password`, `secret`; (b) type-driven redaction of any field declared as Pydantic `SecretStr` in the agent models. Fields that match neither rule are persisted as-is. The allowlist MUST be centralized in a single module-level constant for discoverability.
- **FR-006**: Persistence MUST occur in addition to (not instead of) any existing `--output` behavior, preserving backward compatibility with the markdown/JSON report flag.
- **FR-007**: `EvalRun` MUST round-trip via `model_dump_json()` and `model_validate_json()` without data loss (excluding redacted secrets).
- **FR-008**: Filename timestamp MUST use ISO-8601 with millisecond precision (e.g., `2026-04-18T14-22-09.812Z.json`) using filesystem-safe separators.
- **FR-009**: When persistence fails (e.g., permission denied, disk full), the test run MUST still complete and emit its console summary; the persistence error MUST be logged at WARNING level and surfaced in the CLI output but MUST NOT mask the test exit code.
- **FR-009a**: For test cases that reference multimodal files via `files:`, the persisted snapshot MUST record only the declared path, type, and any inline metadata (e.g., `sheet`, `range`, `pages`, `description`) from `agent.yaml`. The snapshot MUST NOT embed file bytes, content hashes, or extracted text. Reproducibility of multimodal runs relies on the user preserving the referenced files out-of-band.
- **FR-009b**: `EvalRun` files MUST be written atomically: serialize to a sibling temp file (e.g., `<target>.tmp`), `fsync` the file descriptor, then `os.replace()` the temp file to the final target path. Readers (including the dashboard) MUST only ever observe either the previous complete file or the new complete file, never a partially-written one. The dashboard's skip-on-corruption behavior (FR-024) remains as a defense-in-depth measure for files produced outside this write path.

#### Prompt Versioning

- **FR-010**: System MUST add `python-frontmatter` as a core dependency.
- **FR-011**: When loading `instructions.file`, system MUST parse YAML frontmatter via `python-frontmatter` and expose the parsed metadata to the agent loader.
- **FR-012**: System MUST define a `PromptVersion` Pydantic model with fields: `version: str`, `author: str | None`, `description: str | None`, `tags: list[str]`, `source: Literal["file", "inline"]`, `file_path: str | None`, `body_hash: str` (SHA-256 of body), `extra: dict[str, Any]`.
- **FR-013**: When frontmatter contains a `version` key, system MUST use it verbatim. Otherwise, system MUST set `version` to `"auto-" + first 8 hex chars of SHA-256(body)`.
- **FR-014**: When `instructions.inline` is used, frontmatter parsing MUST be skipped and `PromptVersion` populated with `source="inline"`, `file_path=None`, `version=auto-hash`.
- **FR-015**: System MUST NOT alter the prompt body presented to the LLM — frontmatter is metadata only and MUST be stripped before the body reaches the agent.
- **FR-016**: Unknown frontmatter keys MUST be preserved under `PromptVersion.extra` rather than rejected.
- **FR-017**: Malformed YAML frontmatter MUST raise a `ConfigError` at agent load time with file path and parse error context.

#### `holodeck test view` Command

- **FR-018**: System MUST register `holodeck test view` as a Click subcommand of the existing `test` command.
- **FR-019**: `holodeck test view` MUST accept an optional `agent_config` positional argument (default `agent.yaml`) used to determine the experiment (= `agent.name`).
- **FR-020**: `holodeck test view` MUST launch a Streamlit application in the user's default browser, serving a dashboard scoped to the resolved experiment. The Streamlit server binds to `0.0.0.0` (Streamlit default); the CLI MUST print a one-line warning at launch reminding users that `results/` data may contain sensitive conversations/tool args and that the port should be firewalled on shared infrastructure.
- **FR-021**: The dashboard MUST auto-discover and load all `EvalRun` JSON files under `results/<slugified-agent-name>/`.
- **FR-022**: Streamlit and dashboard-specific dependencies MUST be packaged as an optional install extra named `dashboard`. The CLI MUST detect the missing extra at command invocation and emit a one-line install hint without a stack trace.
- **FR-023**: Dashboard MUST render an empty-state message when no runs exist for the resolved experiment.
- **FR-024**: Dashboard MUST handle corrupt or partially-written `EvalRun` files by skipping them with a logged warning and continuing.

#### Dashboard - Summary View

- **FR-025**: Summary view MUST display a sortable table of all runs for the experiment with columns: timestamp, pass rate, total tests, prompt version, model name.
- **FR-026**: Summary view MUST display a pass-rate-over-time line chart, x-axis = run timestamp.
- **FR-027**: Summary view MUST display per-metric average-score trend lines.
- **FR-028**: Summary view MUST display three breakdown panels — one each for `standard`, `rag`, and `geval` evaluation types — with averages and trends segmented as follows:
  - `standard` → by metric name (bleu, rouge, meteor)
  - `rag` → by sub-metric (faithfulness, answer_relevancy, contextual_relevancy, contextual_precision, contextual_recall)
  - `geval` → by custom metric name (each unique G-Eval `name` in the data)
- **FR-028a**: Summary view MUST provide faceted filtering controls that compose (AND-combined) and apply to both the run table and all trend/breakdown charts:
  - Date-range picker (bounded by earliest/latest run timestamp in the current experiment)
  - Prompt-version multi-select (populated from distinct `metadata.prompt_version.version` values)
  - Model-name multi-select (populated from distinct `metadata.agent_config.model.name` values)
  - Pass-rate threshold slider (e.g., "show only runs with pass rate ≥ X%")
  - Frontmatter-tag multi-select (populated from the union of `metadata.prompt_version.tags` across runs)
- **FR-028b**: Active filters MUST be reflected in the URL query string so filtered views are shareable and survive page reload.

#### Dashboard - Explorer View

- **FR-029**: Explorer view MUST be reachable by clicking a run from the Summary view.
- **FR-030**: Explorer view MUST list all test cases in the selected run with name, pass/fail badge, and a one-line metric summary.
- **FR-031**: Selecting a test case MUST display a detail panel containing:
  - Agent config snapshot (provider, model name, temperature, embedding provider, prompt version + frontmatter metadata, tool names)
  - Conversation thread (user input + agent response in chat-style layout)
  - Tool calls highlighted distinctly with tool name, arguments (formatted JSON), and result (formatted JSON, collapsed when large)
  - Expected tools with check/cross match indicators
  - Per-metric evaluation results (name, score, threshold, pass/fail, reasoning if present)
  - Ground truth (when present)
  - Errors (when present)
- **FR-032**: Large tool-call results (over 4KB serialized) MUST be collapsed by default with an expand affordance.

### Key Entities *(include if feature involves data)*

- **EvalRun**: Top-level persisted artifact. Wraps a `TestReport` and adds `metadata: EvalRunMetadata`. One JSON file per `holodeck test` invocation.
- **EvalRunMetadata**: Captures the agent configuration snapshot, prompt version, run timestamp, holodeck version, sanitized CLI args, and optional git commit.
- **PromptVersion**: Captures system prompt identity at run time — `version`, `author`, `description`, `tags`, `source` (file or inline), `file_path`, `body_hash`, and a permissive `extra` dict for forward compatibility with future frontmatter keys.
- **Experiment**: An implicit grouping in the dashboard, identified by `agent.name`. No new entity in the Pydantic model layer; expressed only as a directory naming convention (`results/<slugified-agent-name>/`).
- **Run File**: ISO-8601-timestamped JSON file in `results/<slugified-agent-name>/`; the on-disk representation of an `EvalRun`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: After running `holodeck test agent.yaml`, a developer can locate the persisted run file at a predictable path within 10 seconds of test completion, without consulting documentation.
- **SC-002**: A developer can fully reconstruct the exact agent configuration that produced any historical run by inspecting only the corresponding `EvalRun` JSON file (modulo redacted secrets and vector store contents).
- **SC-003**: Two runs against an identical agent configuration and an identical prompt body produce `EvalRun` files with the same `metadata.prompt_version.version` value 100% of the time.
- **SC-004**: Editing one character of the prompt body causes the next run's `metadata.prompt_version.version` to differ from the previous run's value 100% of the time (when no manual `version:` key is present).
- **SC-005**: A user with 50 prior runs for one agent can launch `holodeck test view`, see the Summary view rendered, and identify the most recent regression in pass rate within 60 seconds of running the command.
- **SC-006**: A user can drill from the Summary view into any individual test case detail in three clicks or fewer.
- **SC-007**: The `holodeck test view` command emits a clear, single-line install hint (no Python traceback) when the `dashboard` extra is not installed.
- **SC-008**: Adding `EvalRun` persistence imposes less than 200ms of overhead per `holodeck test` invocation regardless of test count (measured: persistence wall time only, excluding test execution).
- **SC-009**: Existing `--output` JSON/markdown report consumers see no breaking changes; outputs remain byte-equivalent to the prior implementation when no other changes are made.
- **SC-010**: Dashboard loads runs and renders the Summary view in under 5 seconds (P95) for experiments with up to 1000 runs.

## Assumptions

- The `python-frontmatter` library (MIT licensed) is acceptable as a new core dependency. It is mature, has minimal transitive dependencies, and is widely used in static-site generators.
- Streamlit is acceptable as an optional dependency under the `dashboard` extra. Users who do not want it can omit it without losing any other HoloDeck capability.
- "Experiment" requires no new YAML field — it is implicit in `agent.name`. If users later need cross-agent experiments or named experiment tags, that is a follow-up spec.
- Redaction combines a minimal name-based allowlist (`api_key`, `password`, `secret`) with type-driven redaction of `SecretStr` fields defined in the agent Pydantic models. This assumes all provider-specific secret fields (Anthropic `auth_token`, Azure `api_key`, OAuth tokens, AWS credentials, connection strings) are already typed as `SecretStr` in their respective models; any that are not MUST be migrated to `SecretStr` as part of this feature's implementation. Users who need stricter redaction should treat the entire `results/` directory as sensitive.
- `EvalRun` files are intended to be checked into version control or shared with teammates after redaction. They are not designed to hold large transcripts or binary attachments — file sizes are expected to remain under 1MB per run for typical configurations. Multimodal test-case `files:` are captured by path only (no bytes, no hash, no extracted text), so file-size bounds are independent of attachment size.
- Existing `TestReport` and `TestResult` models remain unchanged. `EvalRun` is purely additive.
- The dashboard is read-only. It does not write back to `results/` or modify agent configurations; users edit `agent.yaml` and re-run `holodeck test` to iterate.
- The Streamlit server binds to `0.0.0.0` by default (Streamlit's native behavior). Network exposure is treated as the user's responsibility — HoloDeck prints a launch-time warning but does not restrict binding or add authentication in v1. Users on shared infrastructure are expected to firewall the port or use SSH tunneling.
- Filesystem path slugification follows a documented rule (lowercase, alphanumerics + hyphens, with non-conforming characters replaced by `-`); collisions are rare in practice but the system will not silently overwrite — a colliding directory name produced by two different agent names is left as a known limitation for the v1 of this feature.
- The git commit field is a best-effort capture (read from `git rev-parse HEAD` if a repo exists in CWD); absence is acceptable and not an error.
