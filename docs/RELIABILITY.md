# Reliability

Owner: runtime and deployment maintainers. Review this document when lifecycle or deployment behavior changes.

## Runtime contracts

Use the backend lifecycle from [base.py](../src/holodeck/lib/backends/base.py).
Preserve initialization, session preparation, cancellation, streaming, and teardown behavior.
Use async clients or thread boundaries for blocking work.
For timeout, retry, or session changes, exercise the affected lifecycle tests.

The [server](../src/holodeck/serve/server.py) provides `/health`, `/health/agent`, AG-UI `/awp`, and REST endpoints.
Health probes establish readiness. A representative request establishes useful agent behavior.
Temporal retries must preserve the distinction between retryable model failures and configuration errors.
See [retry classification tests](../tests/unit/temporal/test_retry_classification.py).

## Local reproduction and evidence

1. Select a committed fixture or a minimal agent configuration that reproduces the problem.
2. Use separate result directories, ports, and container names for concurrent worktrees.
3. Capture the command, configuration identity, response, and relevant redacted logs or traces.
4. Exercise the failed behavior again after the change.
5. Record the observed result and any unavailable external dependency in the execution plan.

Use the [observability guide](guides/observability.md) for configured OpenTelemetry exporters and GenAI signals.
Do not assume a collector exists in every checkout.
The repository does not provision the article's per-worktree observability stack automatically.
That gap is tracked in [technical debt](exec-plans/tech-debt-tracker.md).

## Verify suite

`make verify` runs the three offline tiers. Run `make verify-live` separately when you have credentials.
The `/verify` skill ([SKILL.md](../.claude/skills/verify/SKILL.md)) selects tiers from the changed files.

| Tier | Command | Proves | Credentials |
| --- | --- | --- | --- |
| fast | `make verify-fast` | Harness, lock, Ruff, Black, MyPy and schema checks pass, and the full pytest suite passes | None |
| deps | `make verify-deps` | The lock is current. No unreviewed pip-audit advisory exists. npm audit of `sample/copilotkit` passes (skipped when that sample is absent) | None (network to the advisory databases) |
| smoke | `make verify-smoke` | Both backends complete real `holodeck test run` and `holodeck serve` AG-UI turns, including a tool call and thread isolation | None |
| live | `make verify-live` | Every tool type each backend supports, plus serve AG-UI turns, against real providers; traces in a local Aspire dashboard | `CLAUDE_CODE_OAUTH_TOKEN` (Claude), `OPENAI_API_KEY` (OpenAI, and embeddings for both); Docker for Qdrant and Aspire |

**deps.** [audit_deps.py](../scripts/verify/audit_deps.py) audits every locked package (all extras and groups).
Accepted advisories live in [audit-ignores.toml](../security/audit-ignores.toml).
Each entry has `id`, `package`, `reason`, `owner` and `expires`. An `id` can be the advisory id or any alias.
The audit fails on an unignored advisory, an expired entry, or an entry that matches no current advisory.

**smoke.** [smoke.py](../scripts/verify/smoke.py) starts [fake_provider.py](../scripts/verify/fake_provider.py).
That process implements the OpenAI Responses and Anthropic Messages APIs on a local port.
The script points the [verify fixtures](../tests/fixtures/verify/smoke-openai.yaml) at it with a scrubbed environment and fake keys.
For each backend, it runs `holodeck test run`, then 2 threads x 2 turns over `POST /awp`.
It asserts `RUN_STARTED` first, `RUN_FINISHED` last, no `RUN_ERROR`, scripted text, and `TOOL_CALL_*` events on the tool turn.
It also asserts that no request for thread B carries thread A's text.
Artifacts, including the provider request log, go to `.verify/smoke/<UTC timestamp>/`.
On Linux the Claude backend needs `bubblewrap` (`bwrap`): the Claude Code CLI will not start without it while HoloDeck's default subprocess env scrubbing is on. Smoke fails fast with an install hint when it is missing.
CI runs this tier on Python 3.11. A weekly [scheduled workflow](../.github/workflows/scheduled-verify.yml) runs `verify-deps` and `verify-smoke`.

**live.** [live.py](../scripts/verify/live.py) runs the [live validation loop](../AGENTS.md) without the CopilotKit browser step.
It runs each backend whose credential is set. Put credentials in `.verify/live.env` (gitignored; copy [live.env.example](../tests/fixtures/verify/live.env.example)). Shell variables win.
Claude uses [live-full-claude.yaml](../tests/fixtures/verify/live-full-claude.yaml) and OpenAI uses [live-full-openai.yaml](../tests/fixtures/verify/live-full-openai.yaml).

| Tool type | Claude | OpenAI |
| --- | --- | --- |
| `function` (`smoke_echo`, `leak_probe`) | yes | yes |
| `mcp` stdio (local FastMCP server, started with `uvx`) | yes | yes |
| `vectorstore` on Qdrant, ChromaDB (server) and PostgreSQL (pgvector), `hierarchical_document` on Qdrant | needs `OPENAI_API_KEY` | yes |
| subagents (`claude.agents` / `openai.agents` handoff) | yes | yes |
| `skill` | no: the Claude backend does not adapt skills yet | yes |
| `prompt` | no: no runtime adapter on any backend | no |

Each tool type has one multi-turn test case with `expected_tools` and a `type: code` grader that checks the tool result or reply for a fixed marker. No case uses an LLM judge.
The script reports PASS, FAIL or SKIP for each test case from the results JSON under the run copy's `results/`.
Without `OPENAI_API_KEY` (embeddings), Qdrant, Chroma or Postgres, it removes the vector tools from the run copy and reports their cases as SKIP. The other cases still run.
Then it starts `holodeck serve` and sends 2 threads x 2 turns. Thread A turn 2 must emit `TOOL_CALL_START` for `smoke_echo`. Thread B turn 2 must contain B's code word and not A's.
It also fails when a credential value occurs in any file of the run directory.
Each run uses an empty `HOME` and `CLAUDE_CONFIG_DIR`, so `~/.holodeck/.env` and a stored Claude login cannot replace the credential under test.

Docker containers (left running after the run; `--no-infra` skips them, and vector and trace checks then SKIP):

| Container | Image | Host ports |
| --- | --- | --- |
| `holodeck-verify-qdrant` | `qdrant/qdrant:v1.17.1` | 6333 |
| `holodeck-verify-chroma` | `chromadb/chroma:1.5.9` (matches the locked client) | 8765 |
| `holodeck-verify-postgres` | `pgvector/pgvector:pg17` (throwaway credentials, loopback only) | 54329 |
| `holodeck-verify-aspire` | `mcr.microsoft.com/dotnet/aspire-dashboard:13.5.2` (anonymous) | 18888 UI, 4317 OTLP/gRPC, 4318 OTLP/HTTP |

If another Aspire dashboard, Qdrant, Chroma or Postgres already serves those ports, the script reuses it and does not stop it.
Stop the managed containers with `docker rm -f holodeck-verify-qdrant holodeck-verify-chroma holodeck-verify-postgres holodeck-verify-aspire`. Removing Qdrant, Chroma or Postgres also clears the ingested fixture vectors.
Traces are for human review at <http://127.0.0.1:18888/traces>. Look for the services `holodeck-verify-live-claude` and `holodeck-verify-live-openai`.
The `leak_probe` tool returns a fake `ghp_` token. Its span must show `[REDACTED:github-token]`, never the token.

Pass flags through `LIVE_ARGS`, for example `make verify-live LIVE_ARGS="--backend claude --no-infra"`. `--limit N` runs only the first N test cases.
Cost: each backend makes about 12 small model calls (`claude-haiku-5-5`, `gpt-6-luna`) plus a few embedding calls.
Artifacts (SSE transcripts, logs, the run copy of the agent) go to `.verify/live/<UTC timestamp>/`.
Exit code 2 means every backend was skipped. Report that as skipped, not passed.
A backend with its credential set and no passing case or turn fails.

## End-to-end deployment validation

Run this loop only when the user explicitly requests deployment validation or authorizes the target deployment.
This loop publishes images and changes a live service.
Reuse authorization already supplied in the conversation.
The [deployment guide](guides/deployment.md) describes supported configuration and CLI options.

The former local sample, `sample/financial-assistant/claude`, is git-ignored and may not exist in another checkout.
Use the user-selected agent, registry, and target. Do not assume a personal cloud endpoint.

1. Build the working-tree wheel with `uv build --wheel`. Keep exactly the intended wheel in the Docker build context.
2. Build [docker/Dockerfile.local](../docker/Dockerfile.local) with that wheel, using `--no-cache` and the target platform.
3. Inspect the installed wheel version inside the image.
4. Account for [builder.py](../src/holodeck/deploy/builder.py), which currently passes `pull=True` to Docker.
5. Build the agent image with `holodeck deploy build`, then publish the verified tag to the authorized registry.
6. Deploy with `holodeck deploy run` for the selected agent configuration.
7. Wait for health with a bounded timeout, then send a deterministic request with a known expected answer.
8. Record the image digest, target, response, and relevant telemetry.

`pull=True` can replace a locally tagged base image with its published registry version.
If validation requires a temporary local override, keep that edit isolated and restore only that edit afterward.
Do not commit an override that disables normal base-image pulling.
Azure Container Apps requires a compatible target image, commonly `linux/amd64` for this repository's deployment loop.
Readiness and cold-start timings vary. Measure them instead of relying on the old sample's timings.
