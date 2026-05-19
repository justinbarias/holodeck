# Production Hardening for Claude-Backed Agents — Design

**Status:** Draft for review
**Date:** 2026-05-18
**Author:** justinbarias (with Claude)
**Related:** `specs/021-claude-agent-sdk`, `specs/024-claude-serve-deploy`, `specs/028-yaml-hooks-system`
**External refs:** [Anthropic — Hosting the Agent SDK](https://code.claude.com/docs/en/agent-sdk/hosting), [Anthropic — Securely Deploying AI Agents](https://code.claude.com/docs/en/agent-sdk/secure-deployment)

## Motivation

HoloDeck ships a Claude backend that wraps the Claude Agent SDK and exposes it over AG-UI for chat and over REST for evaluation. In practice, the SDK is not a stateless API call — it spawns a long-running Claude Code CLI subprocess per session, holds shell state, and consumes ~150–300 MiB resident per process. We have been sizing and securing it as if it were a stateless API call.

The financial-assistant sample exposed three failure modes in production:

1. **Stability.** Two concurrent AG-UI sessions on a 1 GiB / 0.5 CPU replica are enough to cgroup-OOM the container. Each session spawns its own SDK subprocess and `WorkingSetBytes` spikes faster than ACA's 1-minute metric resolution can observe. We had been treating this as a transient bug; it is structural.
2. **Permission posture.** HoloDeck's `permission_mode: acceptAll` maps to the SDK's `bypassPermissions`, which disables the SDK permission system entirely. The agent has Bash, Write, Edit, WebFetch available by default. A prompt-injection payload in any document the agent reads can — today — issue arbitrary shell commands inside the container.
3. **Credential surface.** OAuth token, Azure key, Qdrant JWT live in container env vars and are inherited by every SDK subprocess and every tool subprocess. There is no credential boundary; there is no egress allowlist; there is no redaction on tool outputs that enter the model context or OTel traces.

Anthropic's [hosting guide](https://code.claude.com/docs/en/agent-sdk/hosting) and [secure-deployment guide](https://code.claude.com/docs/en/agent-sdk/secure-deployment) describe the patterns and isolation primitives needed to fix all three. This spec aligns HoloDeck with those guides and ships secure-by-default container deployments.

The design is structured in four phases so that the parts that stop the bleeding ship first, and the parts that require multi-week infra work are scoped and isolated.

The key constraints driving the design:

1. **Stability fix is non-negotiable and ships first.** Everything else waits behind the OOM fix.
2. **Secure-by-default, opt-out for permissive.** Operators inherit a permission posture that fails closed. The escape hatch exists, but its name signals risk and the warning is loud.
3. **No required user YAML changes for P1 and P2.** New defaults are inferred from existing fields. Operators who want to override get a new schema field; they do not have to use it.
4. **Defense in depth.** Container hardening, permission allowlists, hook-based redaction, and an opt-in egress proxy each address a different layer. None replaces the others.
5. **The egress-proxy / hardened-profile work is genuinely larger.** It is gated behind `deployment.security_profile: hardened` so the simpler P1/P2 work can ship without waiting on it.

## What's broken today (concrete)

| Layer | Today | What that means |
|---|---|---|
| Replica sizing | ACA defaults from operator YAML; financial-assistant sample is `cpu: 0.5 / memory: 1Gi` | Two concurrent SDK subprocesses can exceed the cgroup memory limit between metric samples; exit-137 with no metric warning |
| Concurrent session cap | Static default `10` in `server.py:121` | Decoupled from replica resources; cap doesn't help small replicas and doesn't scale with large ones |
| `max_turns` | Defaults to whatever the SDK does when not set | SDK hosting FAQ explicitly recommends bounding this |
| Permission mode | `acceptAll` → `bypassPermissions` (`validators.py:307`) | Permission system disabled entirely; Bash/Write/Edit/WebFetch unconditionally allowed |
| Built-in tool allowlist | Whatever the SDK ships with by default | The SDK ships with Bash/Write/Edit/WebFetch; we do not restrict them |
| Container user | `holodeck` UID 1000 (non-root) ✓ | Good — keep |
| Container fs | RW root, RW agent corpus | Agent can persist changes anywhere under `/app`; no read-only enforcement |
| Container caps | Default | No `--cap-drop ALL`; no `no-new-privileges` |
| Node.js | Always installed in generated Dockerfile | SDK ships its own bundled CLI; the only reason to install Node.js is for stdio MCP servers that need it |
| Readiness | Liveness only | Replica accepts traffic before tool init completes; first request races init |
| Ingress | `ingress_external` per YAML; samples set `true` | Easy to leave external by accident; no warning at deploy time |
| Hooks | YAML hook surface exists (spec 028); no HoloDeck-provided defaults | Prompt-injection defenses are entirely on the operator |
| Credentials | All in container env, inherited by every subprocess | No proxy; no allowlist; no redaction layer |
| OTel | `capture_content: true` captures everything including any credential-shaped strings in tool outputs | Traces leak whatever the tool returned |

The plan replaces each row of "Today" with a specific change, grouped into four phases.

## Phase overview

```
┌─────────────────────────────────────────────────────────────────────┐
│ P1 — Stop the OOM                              (days, ships first)  │
│   • ACA defaults: 1 CPU / 2 GiB for Claude agents                   │
│   • max_concurrent_sessions = floor(cpu_cores × 2)                  │
│   • 429 with Retry-After on overflow                                │
│   • max_turns = 20 when unset                                       │
│   • Readiness probe distinct from liveness                          │
│   • Echo resolved limits at `deploy run`                            │
├─────────────────────────────────────────────────────────────────────┤
│ P1 — Permission posture                        (days, ships with ↑) │
│   • auto-disallow Bash/Write/Edit/WebFetch unless declared          │
│   • manual → SDK `plan` mode in serve context                       │
│   • deprecate acceptAll; require explicit unsafe opt-in             │
│   • new `claude.permissions` schema field                           │
├─────────────────────────────────────────────────────────────────────┤
│ P2 — Container runtime                         (1–2 sprints)        │
│   • generated Dockerfile drops Node.js when not needed              │
│   • read-only corpus dirs; writable scratch via tmpfs               │
│   • ACA securityContext: drop caps, readOnlyRootFilesystem          │
│   • ingress defaults to internal; opt-in for external               │
├─────────────────────────────────────────────────────────────────────┤
│ P2 — Prompt-injection defenses                 (1–2 sprints)        │
│   • PreToolUse Bash AST deny list (HoloDeck-provided hook)          │
│   • PostToolUse credential redaction in tool outputs                │
│   • OTel attribute redaction independent of hooks                   │
├─────────────────────────────────────────────────────────────────────┤
│ P3 — Credential boundary, opt-in               (multi-week)         │
│   • `deployment.security_profile: hardened`                         │
│   • Envoy sidecar holds credentials; agent has none                 │
│   • Domain allowlist derived from agent.yaml                        │
│   • ANTHROPIC_BASE_URL → localhost sidecar                          │
├─────────────────────────────────────────────────────────────────────┤
│ P4 — Hybrid sessions / SDK subprocess pooling   (1–2 sprints)       │
│   • Switch claude_backend to per-turn `query(resume=session_id)`    │
│   • Memory scales with concurrent **turns**, not open sessions      │
│   • Transcript on disk is the durable state; subprocess is ephemeral│
│   • Aligns with SDK's documented "Pattern 3 — Hybrid Sessions"      │
└─────────────────────────────────────────────────────────────────────┘
```

P1 is one focused PR (and probably ships under a feature branch within days). P2 and P3 land independently after P1 stabilises trunk. P4 is the biggest memory-headroom win after P1 but requires reshaping `claude_backend.py`'s session model.

### Status tracker

| Phase | Status | Branch / PR | Notes |
|-------|--------|-------------|-------|
| P1a — OOM stability | ✅ shipped | `feature/034-p1-stability-permissions` (commits `a391c7d`, `b344684`, `777ece1`) | ACA defaults bumped, memory-derived session cap, 429 backpressure, readiness probe. Memory derivation (not CPU) confirmed in ACA via `Claude cgroup memory limit: 2147483648 bytes` startup log. `max_concurrent_sessions` upper bound widened to 500 as an explicit-override escape hatch for warm-only workloads (P4 makes this a concurrent-turn cap). |
| P1b — Permission posture | ✅ shipped | same branch | `manual → acceptEdits` mapping in serve, auto-disallow risky built-ins, `i_understand_this_is_unsafe` gate on `acceptAll`, silent test-mode escalation removed. |
| P2a — Container hardening | ⏳ not started | — | — |
| P2b — Prompt-injection defenses | ⏳ not started | — | — |
| P3 — Credential boundary | ⏳ not started | — | — |
| P4 — Hybrid sessions / pooling | ✅ shipped | `feature/034-p4-hybrid-sessions` (commits `6a2ad9c` → `72bba40`) | 9 surgical commits, 124 unit tests green. `ClaudeSession` now uses top-level `query(resume=sdk_session_id)` per turn; `_send_lock` serialises concurrent sends; `close()` deletes the on-disk JSONL transcript. End-to-end ACA deploy validated 2026-05-19: AG-UI single-turn + multi-turn `resume=` + 3 concurrent threads all 200 with no OOM. **Follow-up still open:** SessionStore cap remains binding on open-session count (not concurrent turns), so 5-concurrent cold burst against the 2 GiB / cap-4 replica returned 2 × 429 as expected; the structural-pool win lands when the cap is repurposed (separate PR). |

## Phase 1a — Stop the OOM

### Architecture: sizing follows the SDK's documented unit

The hosting guide is explicit: "Recommended: 1 GiB RAM, 5 GiB of disk, and 1 CPU. Vary this based on your task as needed." That recommendation describes one SDK instance, not one container. With Pattern 2 (long-running, multi-process) — which is what HoloDeck does today — the container has to fit one or more SDK subprocesses plus the serve process plus tool subprocesses. The financial-assistant 0.5 CPU / 1 GiB sizing is below the documented per-instance minimum, before you even count the serve overhead.

The fix is to make sizing-driven decisions the default:

```
replica_cpu (from agent.yaml or ACA default)
       │
       ▼
default replica_memory = max(2 GiB, 2 × replica_cpu Gi)   # ACA constraint
       │
       ▼
default max_concurrent_sessions = max(1, floor(replica_cpu × 2))
       │
       ▼
echo resolved values at `holodeck deploy run` time
```

For a 1 CPU replica → 2 GiB memory, 2 concurrent sessions max. For a 2 CPU replica → 4 GiB, 4 sessions. The "× 2" is calibrated against the ~300 MiB-per-subprocess observation from the financial-assistant investigation, plus headroom for the serve process and tool subprocesses. If real-world numbers diverge materially the multiplier is one constant to tune.

### Why "× 2" and not something smarter

The right answer is a per-replica memory accountant that reads cgroup pressure and gates new sessions. That is a separate feature. For now, "× 2" is honest: it is a heuristic, it is documented, and it is checkable at deploy time because we echo the resolved cap.

### Backpressure: 429 with Retry-After

Today the serve layer accepts a request, forwards to `agui.py:_get_or_create_session()`, and if a new SDK subprocess can't be created the failure surfaces as an OOM mid-request. The fix is an `asyncio.Semaphore(max_concurrent_sessions)` wrapping session creation. When the cap is hit:

```
HTTP/1.1 429 Too Many Requests
Retry-After: 5
Content-Type: application/problem+json

{
  "type": "https://holodeck.dev/errors/session-cap-exceeded",
  "title": "Session limit reached",
  "status": 429,
  "detail": "This replica is serving the maximum of 2 concurrent sessions. Retry shortly.",
  "session_cap": 2,
  "in_flight_sessions": 2
}
```

The semaphore is per-replica. ACA's autoscaler ramps replicas; the operator sees backpressure as 429s, not OOM crashes.

### `max_turns` default

The hosting FAQ: *"An agent session will not timeout, but consider setting a 'maxTurns' property to prevent Claude from getting stuck in a loop."* When unset, we default to 20 in `build_options()`. Operators can override per-agent. The number is conservative enough that legitimate multi-step ConvFinQA reasoning is unaffected and low enough to bound runaway tool-call loops.

### Readiness probe distinct from liveness

Today the generated ACA template emits a single HTTP probe at `/health` used for both liveness and readiness implicitly. Tool init (qdrant connection, payload index check, OpenSearch wipe-and-rebuild for relevant providers) runs at startup before any traffic should be routed.

The change:

- Liveness probe stays at `/health` — confirms the process is alive.
- New readiness probe at `/ready` — returns 200 only after `tool_init_manager.py` reports all tools initialized.

ACA routes traffic only to ready replicas. First requests no longer race tool init.

### Operator-visible verification

`holodeck deploy run` prints the resolved limits:

```
Deploying financial-assistant to Container App 'financial-assistant'
  Replica: 1.0 CPU / 2 GiB memory
  Concurrent sessions per replica: 4 (derived from 2048 MiB memory limit @ 400 MiB/session)
  Max turns per session: 20 (default)
  Ingress: internal
  Readiness probe: /ready (initial delay 5s)
```

If the operator's sizing is silently below the SDK guidance — e.g. they set `cpu: 0.25` — we emit a warning:

```
WARNING: replica CPU (0.25) is below Anthropic's recommended minimum of 1 CPU per SDK instance.
  Concurrent sessions will be capped at 1. Consider increasing `deployment.target.azure.cpu`.
```

### What "stop the OOM" does *not* fix

- Burst traffic past `max_replicas × max_concurrent_sessions`. The 429 surfaces this honestly; raising replica count is on the operator.
- One absurdly large prompt that single-handedly OOMs a replica. The semaphore caps process count, not per-process memory.
- Cold-start latency. P1 surfaces a readiness probe but doesn't make init faster. That's separate work folded into P2.
- **Idle sessions costing one full SDK subprocess each.** P1 caps concurrent *open sessions* because each one holds a persistent `ClaudeSDKClient`. A user who opens 5 chat windows and walks away pins 5 × ~330 MiB even though nothing is running. Fixing this structurally is **Phase 4** (hybrid sessions — see below). P1 is a ceiling, not an architectural fix.

## Phase 1b — Permission posture

### The current default is unsafe

`validators.py:_build_permission_mode()` maps:

| HoloDeck mode | SDK literal | Meaning |
|---|---|---|
| `manual` | `default` | SDK prompts the user before each tool call |
| `acceptEdits` | `acceptEdits` | SDK auto-approves edits, prompts for others |
| `acceptAll` | `bypassPermissions` | SDK permission system disabled entirely |

In `serve` mode there is no operator at the terminal, so `manual` → `default` effectively wedges the SDK on the first tool call. That has pushed operators toward `acceptAll`, which disables permissions entirely. The current path of least resistance is the least safe one.

### The fix

```
                  ┌───────────────────────────────────┐
                  │ HoloDeck permission_mode          │
                  └───────────────────────────────────┘
                            │
        ┌───────────────────┼────────────────────────┐
        │                   │                        │
        ▼                   ▼                        ▼
   `manual`            `acceptEdits`             `acceptAll`
        │                   │                        │
   serve mode?              ▼                   has top-level
        ├─ yes → SDK   SDK `acceptEdits`        `claude.i_understand_
        │  `plan`                                this_is_unsafe: true`?
        └─ no  → SDK                            ┌─ yes → SDK `bypassPermissions`
           `default`                            │        + loud warning at load
                                                └─ no  → load fails with migration error
```

In parallel, regardless of `permission_mode`, we auto-populate `disallowed_tools` with `{Bash, Write, Edit, WebFetch}` for any not declared in `agent.tools`. The SDK ships with these tools available; HoloDeck removes them unless the operator explicitly asked for them.

### New `claude.permissions` schema block

Operators who want explicit control get a typed field instead of having to read the validator source to learn what's auto-derived:

```yaml
claude:
  permissions:
    allowed_tools: [Read, Grep]
    disallowed_tools: [Bash, Write, Edit, WebFetch]
    permission_mode: plan         # plan | acceptEdits | manual
  # Legacy escape hatch — required when permission_mode resolves to bypassPermissions
  i_understand_this_is_unsafe: true
```

When `claude.permissions.*` is set, the explicit values win over the auto-derivation. When the block is absent, the auto-derivation runs.

### Migration path

The change is breaking for one specific case: an agent that today uses `acceptAll` and does not declare its built-in tools. On load:

```
ERROR loading agent 'my-agent':
  permission_mode 'acceptAll' is deprecated because it disables the Claude SDK
  permission system entirely. This is the most direct path from prompt-injection
  to arbitrary tool execution.

  To migrate:
    1. List the tools your agent actually needs in `agent.tools` (recommended), or
    2. Add `claude.i_understand_this_is_unsafe: true` to keep the legacy behavior.

  See: docs/security/permissions.md
```

The legacy escape hatch keeps working; using it requires opting in by name.

### What permission posture does *not* fix

- The SDK still runs with whatever filesystem and network the container gives it. Permissions are an SDK-level allowlist, not an OS-level sandbox. Container hardening (P2) and the egress proxy (P3) layer in on top.
- Prompt-injection payloads that target the *allowed* tools (e.g. an agent that legitimately has `Read` and gets instructed to read `/etc/passwd`) are not stopped by the allowlist. The hook layer (P2) is the next line of defense.

## Phase 2a — Container runtime hardening

### Generated Dockerfile changes

`src/holodeck/deploy/dockerfile.py` produces the per-agent Dockerfile that `holodeck deploy build` consumes. The current template:

- Always installs Node.js (40–80 MiB image bloat).
- COPYs the agent corpus owned by `holodeck:holodeck` with default permissions, so the runtime user can write to it.
- Sets `WORKDIR /app` with no explicit read-only enforcement.

The hosting doc is explicit: *"Both SDK packages bundle a native Claude Code binary for the host platform, so no separate Claude Code or Node.js install is needed for the spawned CLI."* The only reason HoloDeck needs Node.js is when an MCP server's `command` requires it (typically `node` or `npx`).

Changes:

```dockerfile
# Conditional: omit when agent.yaml has no Node-requiring stdio MCP server
RUN apt-get update && apt-get install -y nodejs npm   # ← gated

# Corpus dirs become root-owned and writable only via tmpfs scratch
COPY --chown=root:root data /app/data
COPY --chown=root:root instructions /app/instructions
RUN chmod -R a-w /app/data /app/instructions

# SDK scratch directory — writable, mounted as tmpfs by ACA
RUN mkdir -p /var/holodeck/work && chown holodeck:holodeck /var/holodeck/work
```

The runtime user (`holodeck`) can read corpus files but not modify them. Anything the SDK or its tools need to write goes under `/var/holodeck/work`, which ACA mounts as tmpfs.

### ACA template changes

`src/holodeck/deploy/deployers/azure_containerapps.py` emits the Container App spec. Today it sets `cpu`/`memory`/`min_replicas`/`max_replicas` from agent.yaml. The new template adds:

```python
container_spec = {
    "name": agent_name,
    "image": image_ref,
    "resources": {"cpu": cpu, "memory": memory},
    "probes": [
        {"type": "Liveness",   "httpGet": {"path": "/health"}, ...},
        {"type": "Readiness",  "httpGet": {"path": "/ready"},  ...},  # NEW
    ],
    # NEW securityContext block — degrades gracefully to what ACA supports
    "securityContext": {
        "runAsNonRoot": True,
        "readOnlyRootFilesystem": True,
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
    },
    # NEW writable volumes mounted as tmpfs by ACA
    "volumeMounts": [
        {"volumeName": "tmp",         "mountPath": "/tmp"},
        {"volumeName": "sdk-scratch", "mountPath": "/var/holodeck/work"},
    ],
}
```

ACA's surface is a subset of full Kubernetes; not every property maps 1:1. Where ACA degrades gracefully (e.g. `capabilities.drop`), we use the supported form. Where ACA doesn't expose a primitive (e.g. seccomp profiles), we document the gap and accept it. The opt-in `security_profile: hardened` path (P3) covers the cases ACA can't, by adding an Envoy sidecar.

### Ingress defaults to internal

Today `deployment.target.azure.ingress_external` defaults to whatever the operator writes. Sample agents set it to `true`. New default:

```
ingress_external: false
```

If the operator sets it to `true`, `holodeck deploy run` emits:

```
WARNING: deploying with PUBLIC ingress (ingress_external: true).
  This Container App will be reachable from the public internet at:
    https://<fqdn>
  If this was unintentional, set `deployment.target.azure.ingress_external: false`
  in agent.yaml and redeploy.
```

The warning is non-fatal; it just makes the choice deliberate.

## Phase 2b — Prompt-injection defenses

### The hook surface is already there

Spec 028 (`yaml-hooks-system`) shipped the YAML surface for user-defined hooks. `_options_with_hooks()` in `claude_backend.py:601` merges them into `ClaudeAgentOptions.hooks`. P2 plumbs HoloDeck-provided default hooks into that pipeline ahead of any user hooks.

```
SDK PreToolUse event
   │
   ▼
┌────────────────────────────────────┐
│ HoloDeck default hooks (this spec) │
│  • Bash AST deny list              │
└────────────────────────────────────┘
   │
   ▼
┌────────────────────────────────────┐
│ User-defined hooks (spec 028)       │
└────────────────────────────────────┘
   │
   ▼
Tool call proceeds (or is denied)
```

The default hook chain is opt-out via `claude.disable_default_hooks: true`. We do not let it be silently disabled — disabling it emits a load-time warning naming the agent and the hooks being disabled.

### Default hook 1 — Bash AST deny list (PreToolUse)

The secure-deployment doc describes the SDK's built-in command parsing: *"Before executing bash commands, Claude Code parses them into an AST and matches the result against your permission rules."* HoloDeck adds a default rule set:

| Pattern | Reason |
|---|---|
| `eval` (always requires approval per SDK) | We emit a structured rejection rather than letting the SDK prompt |
| Piped `curl ... | sh`, `curl ... | bash`, `wget ... | sh` | Remote-code-execution vector |
| `bash -c` with command substitution `$(...)` or backticks | Dynamic command construction defeats the deny list |
| Raw `sudo` | No legitimate use inside a non-root container; defends against suid bugs |
| `nc`, `ncat`, `socat` | Reverse-shell vector |

When the AST matches, the hook returns a structured rejection that surfaces in OTel as a `tool.denied` span with the matched pattern:

```json
{
  "decision": "block",
  "reason": "Bash command rejected by HoloDeck default deny list",
  "matched_rule": "curl_pipe_sh",
  "stop_reason": "default_hook"
}
```

This is opt-out, not configurable in-depth. Operators who need a different rule set use the user-hook surface (spec 028) and set `claude.disable_default_hooks: true`.

### Default hook 2 — Credential redaction (PostToolUse)

Tool outputs flow back into the model context. Today nothing redacts them. The default PostToolUse hook scans `tool_result.content` for credential-shaped patterns and replaces them with `[REDACTED:<kind>]`:

| Pattern | Replaced with |
|---|---|
| `sk-ant-api03-…` | `[REDACTED:anthropic-key]` |
| `AKIA[0-9A-Z]{16}` | `[REDACTED:aws-access-key]` |
| `ghp_[A-Za-z0-9]{36}` | `[REDACTED:github-token]` |
| JWT (`eyJ…\.…\.…`) | `[REDACTED:jwt]` |
| `Bearer [A-Za-z0-9_\-\.]+` in HTTP headers | `Bearer [REDACTED]` |

These are heuristics — a false positive risk exists. Operators can opt out per-tool or globally via the user-hook surface. The patterns themselves are a versioned constant; we don't expose them as configuration in v1.

### OTel attribute redaction (independent of hooks)

The hook above only protects the model context. OTel traces with `capture_content: true` capture tool outputs separately, before any hook fires. We add a redaction pass inside `otel_bridge.py` that applies the same patterns to span attributes whose names start with `tool.output` or `tool.input`.

This is independent of `disable_default_hooks` — operators cannot accidentally disable trace redaction by disabling hooks.

### What hooks do *not* fix

- A determined attacker who base64-encodes their payload defeats both the Bash deny list (no AST match) and the credential redaction (no pattern match). Defense in depth means the next layer (P3 egress proxy) is what stops the post-exploitation step.
- A legitimate tool output that contains a credential-shaped string the agent needs. Operators with this case opt out per-tool or move to the hardened profile (P3) where the credential never enters the agent in the first place.

## Phase 3 — Credential boundary (opt-in)

### When this phase applies

Phases 1–2 close the most common failure modes for single-tenant deployments. They do not move credentials out of the agent's environment. The secure-deployment doc's "proxy pattern" is the answer when:

- The agent processes content from multiple tenants in the same container.
- The agent reads untrusted documents whose contents could include prompt-injection payloads designed to exfiltrate credentials.
- Compliance requires that credentials never appear in the agent's process environment.

Operators who don't have those constraints stay on the default profile.

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│ Azure Container App (single revision)                          │
│                                                                │
│  ┌─────────────────────┐         ┌──────────────────────────┐ │
│  │ agent container      │         │ envoy sidecar             │ │
│  │  • holodeck serve    │         │  • credential_injector    │ │
│  │  • no creds in env   │  HTTP   │  • domain allowlist       │ │
│  │  • ANTHROPIC_BASE_URL├────────►│  • secret-mounted          │ │
│  │    = localhost:8081  │ localhost│   credentials             │ │
│  │  • HTTP_PROXY=       │         │                           │ │
│  │    localhost:8081    │         │                           │ │
│  └─────────────────────┘         └────────────┬──────────────┘ │
│                                                │                │
└────────────────────────────────────────────────┼────────────────┘
                                                 ▼
                                       api.anthropic.com
                                       configured MCP HTTP endpoints
                                       embedding provider endpoint
                                       (everything else: rejected)
```

The agent container has zero secret-bearing env vars. The Envoy sidecar holds them (mounted as Azure Container App secrets, mounted only to the sidecar). The sidecar enforces a domain allowlist derived from the agent.yaml at deploy time:

- `api.anthropic.com` (always)
- Embedding provider endpoint (from `agent.embedding_provider.endpoint`)
- Each MCP server's HTTP endpoint (from `agent.tools[*].connection_url` for MCP HTTP/SSE)
- Nothing else

### Opt-in: `deployment.security_profile`

```yaml
deployment:
  security_profile: hardened       # default | hardened
  target:
    provider: azure
    azure:
      cpu: 1.0
      memory: 2Gi
```

When `security_profile: hardened`, `holodeck deploy run`:

1. Reads the agent.yaml and builds the domain allowlist.
2. Emits an Envoy config (`envoy.yaml`) as a Container App config map.
3. Provisions the Container App with two containers (agent + envoy sidecar) sharing localhost.
4. Moves credentials from agent container env to Container App secrets mounted only to the sidecar.
5. Sets `ANTHROPIC_BASE_URL=http://localhost:<envoy-port>` and `HTTPS_PROXY=http://localhost:<envoy-port>` in the agent container.

The Claude backend, on detecting `security_profile: hardened`, requires `ANTHROPIC_BASE_URL` to be set — there is no silent fallback to direct Anthropic API calls.

### What the proxy pattern does *not* fix

- **TLS interception of arbitrary HTTPS.** The doc is clear: for non-Anthropic HTTPS without a TLS-terminating proxy, the sidecar only sees opaque TLS tunnels. We don't ship a CA-injection setup in v1. MCP servers needed at the application layer are handled via the `custom tool` pattern from the secure-deployment doc — they expose an HTTP endpoint inside the sidecar and the agent calls them via plain HTTP.
- **Multi-tenant-per-container isolation.** This profile assumes one tenant per replica. Multiple end-customers concurrently inside the same container with hard isolation between them is a separate spec (would require per-session ephemeral containers, Pattern 1 in the hosting doc).

## Phase 4 — Hybrid sessions / SDK subprocess pooling

P1 caps concurrent SDK subprocesses to whatever fits in cgroup memory. That cap is binding on **open sessions**, not on **active turns** — a 2 GiB replica with 5 idle chat sessions is at the cap even though the CPU and most of the memory is idle. Phase 4 removes that cliff by aligning HoloDeck with the SDK's documented **Pattern 3 — Hybrid Sessions**: each turn is its own subprocess, conversation state lives in a transcript file, idle sessions cost ~zero resident memory.

### What the SDK actually supports

From the SDK sessions doc (verified during P1 validation):

1. **Session state is a JSONL transcript on disk**, written automatically to `~/.claude/projects/<encoded-cwd>/<session-id>.jsonl`. Not opaque in-subprocess state.
2. **`ClaudeAgentOptions.resume=<session_id>`** rehydrates a subprocess from that transcript at spawn time. Fully supported on the top-level `query()` function.
3. **`ClaudeSDKClient`** (what HoloDeck uses today) keeps one subprocess for its lifetime and locks to one session_id internally. **Cannot be reused across distinct sessions** — confirmed both by the docs and by the existing codebase comment in `claude_backend.py` (rotating session_id wedges the CLI).
4. There is **no public API for subprocess pooling** in the conventional warm-worker sense. The SDK owns subprocess lifecycle; the only pooling lever the SDK gives you is "make subprocesses ephemeral via `resume=`."

### Current model vs. Phase 4 model

```
Today (P1):                       Phase 4 (Hybrid):
1 session = 1 ClaudeSDKClient     1 session = 1 transcript file
        = 1 persistent subprocess         (cheap, on disk)
        = ~330 MiB resident       1 turn   = spawn subprocess
                                          + query(resume=session_id)
N idle sessions: N × 330 MiB              + run turn, exit
                                  Resident memory tracks *active turns*.
```

For typical chat duty cycle (sessions spend ≥90% of wall time waiting on the user), this is 5–10× less resident memory.

### Architecture sketch

- Replace `ClaudeSession._ensure_client()` and `ClaudeSession.send()` so each `send()` calls the top-level `query(prompt, options=ClaudeAgentOptions(resume=session_id, ...))` and drains the async-iterable.
- On first turn, capture `session_id` from `ResultMessage` and store it on the HoloDeck `ServerSession`.
- The HoloDeck `SessionStore` cap changes meaning: it now tracks **concurrent turns** in flight, not open sessions. Probably rename to `max_concurrent_turns` to be honest. Cap can be much higher (e.g. 20–50 on a 2 GiB replica) because idle sessions are free.
- Transcript cleanup: on `ServerSession.close()` and on an idle TTL (e.g. 1h), delete the JSONL. Otherwise `~/.claude/projects/` grows unboundedly.
- AG-UI streaming: the top-level `query()` async-iterable yields the same SDK message types as `client.send_streaming()`, so the AG-UI bridge mapping should be near-identical. Worth confirming on the spike.

### Risks to validate before committing

| Risk | Why it matters | Validation | Status |
|------|----------------|------------|--------|
| **MCP server reinit cost per turn** | qdrant client reconnect + hierarchical-doc tool warmup + ingestion idempotency check on every spawn. Could push P50 mid-conversation latency from ~30s to ~40s. | Five sequential turns of the same session under both models on the financial-assistant. | ✅ **Resolved by spike** (2026-05-19). P4 total 101.13s vs baseline 112.72s over 5 turns; P4 was *2.32s/turn faster*. Per-turn variance dominated by LLM + retrieval, not subprocess spawn. |
| **`resume=session_id` context fidelity** | If the transcript replay loses tool-use/tool-result blocks, the agent would re-retrieve every turn and lose conversation continuity. | Conversation-dependent follow-up turns ("and the prior year?") that can only be answered using prior context. | ✅ **Resolved by spike v2** (2026-05-19). Five-turn conversation with three follow-up turns: P4 correctly referenced ALXN/rental-payments and JKHY/capex anchors across spawns, including prior tool-result blocks ("from the **same retrieved JKHY 2008 filing**, the answer is already in context"). |
| **AG-UI stream-shape compat** | The current AG-UI bridge consumes SDK messages emitted via `client.send_streaming()`. | No separate spike needed — `query()` yields the same SDK message types. Validate by keeping `ClaudeSession.send_streaming()` as the stable public surface and swapping its guts; the AG-UI bridge above is unchanged. Cover via the existing deploy validation loop. | ⏳ Validate during implementation. |
| **String-prompt + SDK MCP tool callbacks deadlock** | Discovered in spike v1: when `query()` is called with a `str` prompt, it calls `end_input()` after writing the prompt, which closes stdin. Subsequent control-channel messages (the SDK's reverse path used by SDK MCP tool callbacks) fail with `ProcessTransport is not ready for writing`. | Use streaming-mode prompts (`AsyncIterable[dict]`) for every P4 turn — the SDK keeps stdin open via the background `stream_input` task. | ✅ **Resolved by spike** — required for any implementation using `holodeck_tools` SDK MCP server. |
| **Subprocess spawn jitter under burst** | Spawning 5 subprocesses at once = 5 simultaneous Node.js CLI starts + 5 MCP server inits. Could cause CPU saturation despite memory headroom. | Re-run the 5-concurrent burst test from P1 validation; measure end-to-end latency distribution. | ⏳ Validate during implementation. |
| **Transcript disk pressure** | Long-lived deployments accumulate JSONLs forever without cleanup. | Implement TTL job; size estimate per session before committing. | ⏳ Implementation detail. |
| **Lost session state on transcript corruption** | A corrupted transcript bricks the session permanently (no in-memory fallback as we have today). | Catch decode errors at resume; surface as "session lost, please reconnect" rather than 500. | ⏳ Implementation detail. |

#### Spike data (2026-05-19, financial-assistant sample, sequential turns)

Spike v1 — latency, independent turns:

| turn | baseline | P4 (`query` + `resume`) | delta |
|------|---------:|------------------------:|------:|
| 1 (cold) | 47.05s | 45.34s | -1.71s |
| 2 | 8.67s | 10.86s | +2.19s |
| 3 | 20.84s | 14.69s | -6.15s |
| 4 | 16.99s | 15.23s | -1.76s |
| 5 | 19.16s | 15.01s | -4.15s |
| **TOTAL** | **112.72s** | **101.13s** | **-11.59s** |

Spike v2 — context preservation, conversation-dependent turns:

| turn | baseline | P4 | delta |
|------|---------:|---:|------:|
| 1 (cold) | 38.69s | 40.38s | +1.69s |
| 2 ("prior year") | 18.80s | 20.90s | +2.10s |
| 3 ("percentage change") | 5.17s | 12.04s | +6.87s |
| 4 (context switch) | 15.78s | 9.21s | -6.58s |
| 5 ("year before") | 11.37s | 5.31s | -6.06s |
| **TOTAL** | **89.81s** | **87.84s** | **-1.97s** |

Both legs answered every conversation-dependent turn correctly, demonstrating that `resume=` faithfully replays user messages, assistant responses, and tool-use/tool-result blocks across spawned subprocesses.

### Why this didn't ship in v1

The two-week diagnostic loop that produced P1 surfaced this as the natural next move, but P1 was already a bounded scope with clear success criteria (no more OOM, cap is binding, permissions tightened). Stuffing a subprocess-lifecycle rewrite into the same PR would have made it un-reviewable. P1 gets us to "no OOM at our current scale" — P4 gets us to "no OOM as scale grows."

### Open questions

1. **Does `query(resume=session_id)` round-trip every option** (hooks, MCP servers, allowed_tools, permission_mode) cleanly? ✅ Spike confirmed MCP servers, allowed_tools, and prompt envelope all round-trip — same `options` dataclass reused per turn with only `resume` mutated. Hooks and `permission_mode` round-trip implicitly because they're carried on the same options object.
2. **Does the SDK serialize tool-use blocks faithfully in the transcript?** ✅ Spike v2 confirmed — turn 5 ("And the year before?") reproduced the $34,202 figure from turn 4's tool result *without re-running the retrieval tool*, proving tool-use AND tool-result blocks are replayed from the transcript.
3. **Streaming-mode vs string-mode `query()`** — spike v1 surfaced this: string-mode closes stdin after writing the prompt, deadlocking any SDK MCP tool callback. P4 implementation **must** use streaming-mode (`AsyncIterable[dict]`) prompts. Documented as a hard requirement above.
4. **Transcript bloat on long tool outputs** — still open. The hierarchical_document tool can return 8KB-ish per retrieval; 50 turns × 5 retrievals × 8KB = 2MB transcript that gets read+parsed every turn. Need a size-budget knob or rolling-summary mechanism before exposing P4 to long-running chat sessions.
5. **OTel context propagation across spawned subprocesses** — today the OTel instrumentor patches `ClaudeSDKClient` instances. With `query()` we get a fresh client each turn; need to verify the instrumentor still wires the trace context, or move the patch to the top-level `query` function.

## What we are deliberately *not* doing in v1

- **Per-session ephemeral containers (Pattern 1).** That's the strongest isolation pattern in the hosting doc but requires a new deployer (Modal, Fly Machines, etc.) and per-session billing infrastructure. Documented as a follow-up spec.
- **gVisor / Firecracker runtime.** ACA doesn't support gVisor. Self-hosted Docker users could opt into `runsc`, but we don't template it.
- **Statistical anomaly detection on tool calls.** Out of scope; the hook layer is the right abstraction for layering this later.
- **Auto-derived domain allowlist from prompt content.** P3 derives the allowlist from explicit YAML fields. Inferring it from prompt text or runtime behavior is brittle and is not what the secure-deployment doc recommends.
- **Per-tool credential mapping in default profile.** P3 moves credentials behind the sidecar. The default profile keeps the existing env-var model; we don't try to half-implement credential separation.
- **Replacing the existing YAML hook surface.** Spec 028 stays as-is. P2's defaults plug into the same chain.

## Cost reality

P1 changes are nearly free at runtime: a semaphore acquisition, a probe endpoint, a default in `build_options()`. Operator-visible cost is the bumped replica spec (1 CPU → 2 GiB), which roughly doubles the per-replica ACA bill from ~$0.05/hr to ~$0.10/hr on consumption profile. ACA scale-to-zero is preserved.

P2 changes are also free at runtime: dropping Node.js shrinks the image (~50 MiB → faster pulls, faster cold starts), hook execution is a few microseconds per tool call, and OTel redaction is a regex pass per span attribute.

P3 adds the Envoy sidecar to every replica, roughly doubling per-replica memory (~50–80 MiB for Envoy) and adding ~1 ms latency per upstream request. This is why it is opt-in.

## How HoloDeck's structure changes

### Schema additions

New fields under `claude:`:

```yaml
claude:
  permissions:                    # NEW (P1b)
    allowed_tools: [...]
    disallowed_tools: [...]
    permission_mode: plan | acceptEdits | manual
  disable_default_hooks: false    # NEW (P2b)
  i_understand_this_is_unsafe: false  # NEW (P1b) — required for bypassPermissions
  max_concurrent_sessions: null   # existing — default now derived from cgroup memory
  session_memory_estimate_mib: 200  # NEW (P1a) — per-session memory budget for derivation
```

New field under `deployment:`:

```yaml
deployment:
  security_profile: default | hardened   # NEW (P3)
```

Reuses spec 028's `claude.hooks` surface for user-defined hooks; no new field needed.

### Backend changes

| File | Change |
|---|---|
| `src/holodeck/lib/backends/claude_backend.py` | `build_options()` derives `disallowed_tools` from `agent.tools`; sets `max_turns` default; plumbs default hooks |
| `src/holodeck/lib/backends/validators.py` | `_build_permission_mode()` rewritten per the decision tree above; emits migration error for legacy `acceptAll` without opt-in |
| `src/holodeck/lib/backends/claude_hooks.py` | NEW — bundled default hooks (Bash AST deny, credential redaction) |
| `src/holodeck/lib/backends/otel_bridge.py` | Add redaction pass on `tool.input`/`tool.output` attributes |

### Serve changes

| File | Change |
|---|---|
| `src/holodeck/serve/server.py` | `max_concurrent_sessions` default becomes `max(1, floor(cpu_cores × 2))` derived from replica env (`HOLODECK_REPLICA_CPU` env var emitted by deployer) |
| `src/holodeck/serve/protocols/agui.py` | Wrap `_get_or_create_session()` in `asyncio.Semaphore`; return 429 on overflow |
| `src/holodeck/serve/server.py` | Add `/ready` endpoint distinct from `/health`; returns 200 only after `tool_init_manager` reports ready |

### Deploy changes

| File | Change |
|---|---|
| `src/holodeck/deploy/dockerfile.py` | Conditional Node.js install; chown corpus to root + chmod a-w; create `/var/holodeck/work` |
| `src/holodeck/deploy/deployers/azure_containerapps.py` | Emit `securityContext`; readiness probe; `ingress_external` defaults to `false`; echo resolved limits at deploy time |
| `src/holodeck/deploy/deployers/azure_containerapps.py` | When `security_profile: hardened`, emit two containers + Envoy config + Container App secrets |
| `src/holodeck/deploy/envoy.py` | NEW — generate Envoy config from agent.yaml allowlist |

### Model changes

| File | Change |
|---|---|
| `src/holodeck/models/claude_config.py` | Add `permissions: ClaudePermissionsConfig`, `disable_default_hooks`, `i_understand_this_is_unsafe`; remove the implicit `acceptAll` mapping |
| `src/holodeck/models/deployment.py` | Add `security_profile`; default Azure `cpu`/`memory` to `1.0`/`2Gi` for Claude agents; default `ingress_external` to `false` |

## Validation at startup

Phases 1–2 add the following load-time checks. Phase 3 adds two more.

1. Existing agent.yaml validation (unchanged).
2. **(P1b)** If `permission_mode: acceptAll` and `i_understand_this_is_unsafe: false`: load fails with migration error.
3. **(P1b)** If `claude.permissions` and any legacy permission fields are both set on the same agent: load fails with a "choose one" error.
4. **(P2b)** If `claude.disable_default_hooks: true`: load succeeds with a loud warning naming the agent.
5. **(P3)** If `security_profile: hardened` and any credential-bearing env var is set on the agent container: load fails — the operator has half-migrated.
6. **(P3)** If `security_profile: hardened` and `agent.embedding_provider.endpoint` or any MCP HTTP endpoint is missing: load fails — the allowlist would be incomplete.

All validation errors collected in a single pass, surfaced together.

## CLI surface

No new top-level subcommands. Two existing commands grow output:

```
$ holodeck deploy run sample/financial-assistant/claude
Building image…                              done (32s)
Pushing to ghcr.io/justinbarias/...          done (12s)
Resolving deployment configuration:
  Replica: 1.0 CPU / 2 GiB memory
  Concurrent sessions per replica: 2
  Max turns per session: 20
  Permission mode: plan (auto-mapped from `manual` in serve context)
  Disallowed tools (auto): Bash, Write, Edit, WebFetch
  Default hooks: enabled
  Ingress: internal
  Security profile: default
Deploying to Azure Container Apps…           done (28s)
Container App URL (internal):
  https://financial-assistant.internal.nicemoss-50caf9f5.eastus.azurecontainerapps.io
```

When the security profile is hardened, the resolution block expands:

```
  Security profile: hardened
    Envoy sidecar: enabled
    Domain allowlist:
      • api.anthropic.com
      • holodeckai.openai.azure.com
      • <qdrant-cluster>.qdrant.tech
    Credentials in agent container: none
    Credentials in envoy sidecar: 3 (CLAUDE_CODE_OAUTH_TOKEN, AZURE_OPENAI_API_KEY, QDRANT_REMOTE_URL)
```

```
$ holodeck serve agent.yaml
Loading agent…
  Permission mode: plan (auto-mapped from `manual`)
  Disallowed tools (auto): Bash, Write, Edit, WebFetch
  Max concurrent sessions: 4 (derived from 2048 MiB memory limit @ 400 MiB/session)
  Max turns per session: 20
  Default hooks: enabled
Listening on 0.0.0.0:8080…
```

## Phase-by-phase ship criteria

| Phase | Ship criteria |
|---|---|
| **P1a — OOM** | Financial-assistant sample handles 3 concurrent AG-UI sessions on default sizing without exit-137; overflow returns 429 with Retry-After; replica trace IDs stay constant. Unit tests for the semaphore + 429 path; integration test against a real ACA deploy. |
| **P1b — Permissions** | Default-built agents cannot Bash unless declared; loading an agent with legacy `acceptAll` fails closed unless explicit opt-in; new `claude.permissions` schema documented in `docs/`. Unit tests on the decision tree; CI grep-test asserting no code path maps to `bypassPermissions` except the explicit opt-in. |
| **P2a — Container** | Generated Dockerfile omits Node.js for agents without Node-needing MCP servers; ACA template emits `securityContext` and dual probes; ingress defaults to internal. Unit tests on Dockerfile generation + ACA template generation. |
| **P2b — Hooks** | Default hooks ship enabled; synthetic prompt-injection test case verifies Bash deny + credential redaction in both context and OTel; opt-out emits warning. Integration test against a real AG-UI session. |
| **P3 — Hardened** | Hardened profile deploys two containers; agent container env has zero credential-bearing vars; calls to non-allowlisted domains rejected by sidecar with structured error. End-to-end test against a real ACA deploy with the hardened profile. |

Each phase merges independently. P1 is one PR. P2a and P2b can be parallel PRs. P3 is a sequence (Envoy generator → ACA deployer changes → backend integration).

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Existing agents break on the new `disallowed_tools` defaults | Auto-derivation emits a warning at load time naming the auto-disallowed tools and pointing at the new schema; sample agents updated as part of P1b |
| Operators flip `i_understand_this_is_unsafe: true` to dodge the migration | The flag's name signals risk; load-time warning is loud; `holodeck serve` and `deploy run` echo the flag's effect in their resolved-config output |
| `floor(cpu_cores × 2)` is wrong for some workloads | Override remains available via `claude.max_concurrent_sessions`; we surface the resolved value at deploy time so operators can tune |
| Readiness probe wedged → ACA replaces replicas in a loop | Tool init has its own timeout from spec 025; readiness reflects but doesn't change that timeout |
| Default hooks block a legitimate command (false positive) | Opt-out per-agent via `disable_default_hooks: true`; users-hooks surface still available for finer control |
| Default hooks redact a credential the agent genuinely needs in context | Same as above — opt-out path is explicit |
| `security_profile: hardened` rejects a domain the agent needs | Allowlist is derived from explicit YAML fields; if a tool needs an undeclared domain, the operator declares it; if a domain is not declarable in YAML, the operator opens an issue rather than disabling the proxy |
| Envoy sidecar fails to start → replica unhealthy | Liveness probe on the agent container fails fast; ACA replaces; operator sees normal ACA failure mode |
| Phases ship out of order and create inconsistent defaults | Phase gates: P2 cannot ship without P1; P3 cannot ship without P2a (container hardening is a prerequisite for the sidecar's volume model) |

## Out of scope for v1

| Out-of-scope | Why deferred | Path forward |
|---|---|---|
| Ephemeral per-session containers (Pattern 1) | New deployer (Modal / Fly Machines) + per-session billing | Separate spec; the right pattern for hostile multi-tenant workloads |
| gVisor runtime templating | ACA doesn't support it; self-hosted users can adopt without HoloDeck help | Docs note in `docs/security/` |
| TLS-terminating proxy with CA injection | Significant cert management; only needed for arbitrary HTTPS service auth | Follow-up to P3; document the custom-tool pattern for now |
| Per-replica memory accountant gating sessions on cgroup pressure | Better than `floor(cpu × 2)` but requires reading cgroup pressure | v2 of P1a; static heuristic is honest enough for v1 |
| Auto-discovery of required MCP domains | Hard to do correctly without false positives | Stays explicit in YAML |
| Per-tool credential mapping in default profile | Half-implementing the credential boundary is worse than not having one | Operators who need credential isolation opt into P3 |
| Statistical-anomaly detection on tool call patterns | Useful but unbounded scope | Hook surface is the right abstraction; out of v1 |

## Open questions resolved during design

1. **Default replica sizing for non-Claude agents?** No change. SK-backed agents (OpenAI/Azure/Ollama) don't spawn per-session subprocesses and don't have the same memory footprint. The 1 CPU / 2 GiB default applies only when `model.provider` routes to the Claude backend.
2. **What about agents that legitimately need Bash?** Declare it in `agent.tools` like any other tool. The auto-disallow only fires for tools not declared.
3. **What about `manual` permission mode in non-serve contexts (e.g. `holodeck test run`)?** Unchanged. The mapping to `plan` only fires when running in `serve` mode where there's no operator at the terminal. `holodeck test run` and `holodeck chat` keep SDK `default` (interactive).
4. **Do the default hooks count toward the user's `claude.hooks` slot count?** No. They run as a separate, earlier stage and are not visible in `claude.hooks` config.
5. **Where do credentials in P3 actually live?** Azure Container App secrets, mounted only to the sidecar via the ACA `volumes` mechanism. Not in the image, not in the agent container env, not in the cluster's general secret store.
6. **What if an agent uses both stdio MCP servers (needing Node) and `security_profile: hardened`?** Node still installs for stdio MCP servers; the sidecar still proxies HTTPS egress. They are orthogonal axes.

## v1 contract

- Five named phases, ship-independent: **P1a (OOM)**, **P1b (permissions)**, **P2a (container)**, **P2b (hooks)**, **P3 (hardened profile)**.
- One new schema field per phase, all gated behind explicit YAML keys.
- No required user YAML changes for P1 and P2; defaults are inferred from existing fields.
- One escape hatch for legacy `acceptAll` permission mode (`i_understand_this_is_unsafe`), keyed by a name that signals risk.
- One opt-in for credential boundary (`security_profile: hardened`), gated behind the multi-week egress-proxy work.
- Resolved limits and security posture echoed at `serve` and `deploy run` time so operators can see what's actually in effect.
- All validation errors surfaced together at load time.
- No changes to evaluation runs (`holodeck test run`) except inheriting P1 permission defaults.

## Implementation surface — modules to create / modify

**New modules:**

- `src/holodeck/lib/backends/claude_hooks.py` — bundled default hooks (Bash AST deny, credential redaction).
- `src/holodeck/deploy/envoy.py` — Envoy config generator for the hardened profile.

**Modified modules:**

- `src/holodeck/lib/backends/claude_backend.py` — `build_options()` derives defaults; plumbs default hooks.
- `src/holodeck/lib/backends/validators.py` — `_build_permission_mode()` decision tree; migration errors.
- `src/holodeck/lib/backends/otel_bridge.py` — credential redaction on span attributes.
- `src/holodeck/serve/server.py` — `/ready` endpoint; CPU-derived `max_concurrent_sessions` default.
- `src/holodeck/serve/protocols/agui.py` — semaphore + 429 on overflow.
- `src/holodeck/deploy/dockerfile.py` — conditional Node.js; corpus read-only; tmpfs scratch dir.
- `src/holodeck/deploy/deployers/azure_containerapps.py` — securityContext; dual probes; ingress default; hardened-profile sidecar.
- `src/holodeck/models/claude_config.py` — `ClaudePermissionsConfig`; `disable_default_hooks`; `i_understand_this_is_unsafe`.
- `src/holodeck/models/deployment.py` — `security_profile`; new Azure defaults.
- `src/holodeck/cli/commands/deploy.py` — echo resolved limits and security posture.
- `src/holodeck/cli/commands/serve.py` — same.

**Tests:**

- `tests/unit/serve/test_session_semaphore.py` — overflow → 429 with Retry-After; semaphore correctness.
- `tests/unit/serve/test_readiness_probe.py` — `/ready` only returns 200 after tool init completes.
- `tests/unit/lib/backends/test_permissions.py` — decision tree; auto-disallow; migration error.
- `tests/unit/lib/backends/test_default_hooks.py` — Bash deny patterns; credential redaction patterns.
- `tests/unit/lib/backends/test_otel_redaction.py` — span attribute redaction.
- `tests/unit/deploy/test_dockerfile_generation.py` — conditional Node.js; corpus permissions; tmpfs paths.
- `tests/unit/deploy/test_aca_template.py` — securityContext; dual probes; ingress default; resolved-config output.
- `tests/unit/deploy/test_envoy_generator.py` — allowlist derivation; sidecar config; credential mounts.
- `tests/integration/security/test_bash_deny_e2e.py` — synthetic prompt-injection test against a live AG-UI endpoint.
- `tests/integration/security/test_hardened_profile_e2e.py` — agent container has no creds; non-allowlisted domain rejected; allowlisted domain works.

## Dependencies to add

- No new Python deps.
- New runtime image dep (P3 only): Envoy. Pulled as a separate Container App container image (`envoyproxy/envoy:v1.31-latest`), not bundled into the agent image.
