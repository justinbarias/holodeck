# Prompt-Injection Defenses (Spec 034 P2b)

HoloDeck ships three independent defense layers against credential leakage and
prompt injection for Claude agents. All defaults are on; no `agent.yaml`
changes are required.

| Layer | Default | Enforcement point | How to disable |
|---|---|---|---|
| PostToolUse credential redaction | On for all Claude agents | `claude_hooks.py` | `claude.disable_default_hooks: true` (warning emitted) |
| OTel trace attribute redaction | On for all Claude agents | `otel_redaction.py` | Cannot be disabled by `disable_default_hooks` |
| Subprocess env scrubbing | On for all Claude agents | `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1` + `CLAUDE_CODE_MCP_ALLOWLIST_ENV=1` | `claude.disable_subprocess_env_scrub: true` (warning emitted) |

---

## 1. PostToolUse credential redaction

A `PostToolUse` hook runs on every tool invocation for every Claude agent.
Before the tool output re-enters the model context, the hook applies five
regex patterns in order:

| Pattern name | Marker written into output |
|---|---|
| `anthropic-key` | `[REDACTED:anthropic-key]` |
| `aws-access-key` | `[REDACTED:aws-access-key]` |
| `github-token` | `[REDACTED:github-token]` |
| `jwt` | `[REDACTED:jwt]` |
| `bearer` | `Bearer [REDACTED]` |

The redaction walks strings, dicts, lists, and tuples recursively (bounded to
200 levels). Non-string scalars pass through unchanged.

**Opt-out.** Set `claude.disable_default_hooks: true` in `agent.yaml`. A
`WARNING`-level log line is emitted at agent load time so the choice is visible
in observability tooling. Use this only when the agent's tool outputs are known
to contain credential-shaped strings that are safe to expose (for example,
synthetic test tokens).

Note: removing the Bash AST deny hook from this layer was an intentional
design decision — Bash hardening is owned by SDK permission rules and P1b
`auto_disallow` semantics, which are more precise and harder to bypass than a
regex hook. See spec 034 plan Task 2 for the rationale.

---

## 2. OTel `RedactingSpanProcessor`

The `RedactingSpanProcessor` sits before all exporting span processors on the
tracer provider. Every exporter (OTLP, Console, Azure Monitor) therefore sees
scrubbed attributes. The processor scrubs any span attribute whose key starts
with one of the following prefixes:

- `tool.input.*`
- `tool.output.*`
- `gen_ai.tool.input.*`
- `gen_ai.tool.output.*`
- `gen_ai.prompt`
- `gen_ai.completion`

These are the namespaces that the `otel-instrumentation-claude-agent-sdk`
GenAI instrumentor populates with tool I/O content.

**Independence from `disable_default_hooks`.** The OTel processor runs
regardless of the `disable_default_hooks` flag. Operators cannot accidentally
disable trace redaction by opting out of the PostToolUse hook. The two layers
use the same `redact_credentials()` helper from `claude_hooks.py`.

---

## 3. Subprocess env scrubbing

By default, two environment variables are injected into the Claude subprocess
before it starts:

- `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1` — instructs the SDK's Bash tool to
  strip secrets from the subprocess environment before each shell invocation.
- `CLAUDE_CODE_MCP_ALLOWLIST_ENV=1` — restricts which env vars are forwarded
  to stdio MCP servers.

These are set via `ClaudeAgentOptions.env` and spread into the subprocess
environment by the SDK. HoloDeck sets them with `setdefault` so any explicit
override in the agent config takes precedence.

**Opt-out.** Set `claude.disable_subprocess_env_scrub: true` in `agent.yaml`.
A `WARNING`-level log line is emitted at agent load time. Use this only when a
tool explicitly needs access to env vars that would otherwise be scrubbed.

### Runtime dependency: `bubblewrap`

The Claude Agent SDK enforces `bubblewrap` (`bwrap`) for the subprocess
isolation backing `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1`. HoloDeck's
generated container images include it by default (installed in
`docker/Dockerfile`).

If you bake your own base image, ensure `bubblewrap` is on `PATH`. Without
it, the SDK exits with `error: bubblewrap is required for subprocess env
scrubbing and isolation` at the first tool turn. Set
`claude.disable_subprocess_env_scrub: true` if you have a legitimate reason
to run without bwrap (e.g. distroless base) — the agent will then inherit
the parent env in tool subprocesses.

### macOS local development

`bubblewrap` is Linux-only. When running `holodeck serve` locally on
macOS, the SDK may surface a similar error. If so, export
`CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=0` in your shell or set
`claude.disable_subprocess_env_scrub: true` in the agent.yaml for local
runs. Production container images include bwrap.

---

## What is NOT defended against

- **The SDK subprocess itself inherits the parent env at startup.** The
  `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB` flag reduces exposure at shell invocation
  time, but the Claude Code binary that *is* the subprocess launches with the
  full agent container environment. The structural fix (credential proxy /
  Envoy injector pattern) is tracked as spec 034 P3.
- **Exfiltration over allowed tool channels.** If an agent has network-capable
  tools (HTTP, MCP servers), a successful injection that stays within the
  allowed tool set can still exfiltrate data. Restrict tool grants and MCP
  allowlists at the `agent.yaml` level.
- **Model-context-window extraction.** The PostToolUse hook scrubs tool
  *outputs*; it does not scan tool *inputs* or the system prompt. System
  prompt confidentiality is outside this scope.

---

## Cross-references

- [`container-hardening.md`](container-hardening.md) — image-layer and ACA
  ingress hardening (non-root user, read-only corpus, ephemeral scratch).
- [Anthropic secure-deployment guide](https://code.claude.com/docs/en/agent-sdk/secure-deployment)
  — upstream recommendations this implementation follows.
