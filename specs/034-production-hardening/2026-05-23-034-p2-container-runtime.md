# Spec 034 Phase 2 — Container Runtime Hardening + Prompt-Injection Defenses Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land spec 034 Phase 2 — harden the generated container runtime (P2a) and add HoloDeck-provided default hooks for prompt-injection defense plus OTel attribute redaction (P2b).

**Architecture:** Two parallel sub-phases that can ship as independent PRs. P2a touches the Dockerfile generator + Azure Container Apps deployer. P2b adds a new bundled-hooks module that plugs into the existing `ClaudeAgentOptions.hooks` chain ahead of user hooks, plus a span-processor that scrubs credential-shaped strings out of trace attributes regardless of `disable_default_hooks`.

**Tech Stack:** Python 3.10+, Pydantic v2, `claude-agent-sdk` (HookEvent / HookMatcher / HookContext / PermissionDecision), `azure.mgmt.appcontainers` models (Container, ContainerAppProbe, Volume, VolumeMount, SecurityContext), OpenTelemetry SDK (SpanProcessor, ReadableSpan), pytest + `-n auto`.

---

## Coverage check: P2 vs. https://code.claude.com/docs/en/agent-sdk/secure-deployment

The user asked whether P2 covers everything in the Anthropic secure-deployment doc. Short answer: **P2 closes the container/runtime + prompt-injection rows; everything credential- or network-related is out of P2 scope by design and is gated to P3 (`security_profile: hardened`).**

| Anthropic recommendation | Where it lands in HoloDeck | Plan task |
|---|---|---|
| Permissions system / Bash AST parsing | **Fully handled by the SDK + P1b** — the SDK already parses Bash commands into an AST and matches against `allowed_tools` glob rules; P1b auto-disallows `Bash` for any agent that doesn't declare it. No P2b layer needed (an earlier draft added a regex-based deny hook on top; dropped 2026-05-23 as redundant — see Task 2 note for rationale). | — |
| Web search summarization | SDK built-in, nothing to do | — |
| Sandbox mode (`sandbox-runtime` / bubblewrap / sandbox-exec) | Out of v1; spec 034 §"What we are deliberately *not* doing in v1" | — |
| `--cap-drop ALL` | **Not exposed by ACA.** Verified against ARM API 2026-01-01 + `azure-mgmt-appcontainers==4.0.0` + the [Microsoft ACA security baseline](https://learn.microsoft.com/en-us/security/benchmark/azure/baselines/azure-container-apps-security-baseline) (no `securityContext` surface at any API version). Platform-default cap set is Microsoft-controlled; customers cannot further restrict. **Documented gap.** | Task 14 |
| `--security-opt no-new-privileges` | **Not exposed by ACA.** ACA never grants privileged mode in the first place, so privilege escalation is implicitly bounded by the platform — but there is no field to set. **Documented gap.** | Task 14 |
| `--read-only` root filesystem | **Not exposed by ACA.** No `readOnlyRootFilesystem` field on any API version. Best available approximation: Dockerfile `chmod -R a-w` on the corpus (Task 10) + tmpfs scratch (Task 11). | Tasks 10, 11, 14 |
| `--tmpfs /tmp` and writable scratch | **P2a** via ACA `Volume(storage_type=EMPTY_DIR)` + `VolumeMount` (the one securityContext-adjacent thing ACA *does* expose) | Task 11 |
| Run as non-root (`--user 1000:1000`) | **Enforced at image layer** by Dockerfile `USER holodeck` (UID 1000). ACA has no `runAsNonRoot` field to assert it in the manifest. | — (already in Dockerfile) |
| `--network none` + Unix-socket proxy | Out of P2 — covered by P3 (Envoy sidecar) | — |
| Mount code read-only | **P2a** — `chown root:root` on `/app/data` and `/app/instructions`, `chmod -R a-w` on those dirs | Task 11 |
| Avoid mounting `.env`, `~/.ssh`, `~/.aws`, `~/.git-credentials`, etc. | HoloDeck never mounts host secrets; the COPY surface is `data_directories` + `instruction_files` resolved from `agent.yaml`. **No new work in P2.** (Could add an opt-in lint that flags credential-shaped filenames in the COPY surface; folded into Task 16 as a low-priority hint.) | Task 16 (lint) |
| `--security-opt seccomp=…` custom profile | **ACA does not expose seccomp.** Documented gap; out of v1. | — |
| `--userns-remap`, `--ipc private` | **ACA does not expose these.** Documented gap; out of v1. | — |
| `--pids-limit` | **ACA does not expose pid limits.** Note in `docs/security/`. | Task 16 |
| gVisor / Firecracker / VMs | Out of v1 (ACA can't host them); documented in spec 034 §"Out of scope" | — |
| Cloud private subnet + egress firewall + proxy | Out of P2 — covered by P3 (`security_profile: hardened` Envoy sidecar) | — |
| Proxy pattern for credentials (`credential_injector`) | Out of P2 — P3 | — |
| `ANTHROPIC_BASE_URL` / `HTTP_PROXY` / `HTTPS_PROXY` | Out of P2 — P3 | — |
| Custom-tool / MCP pattern for non-Anthropic creds | Existing MCP surface already supports this | — |
| TLS-terminating proxy + CA injection | Out of v1 (documented in spec) | — |
| Read-only code mount warning about secret files | **P2a** — Dockerfile already only COPYs the agent corpus + instructions, not host directories. Add a CLI lint that warns when filenames matching `.env`/`*.pem`/`*.key`/`credentials*.json` are inside the COPY surface. | Task 16 |
| Overlay filesystem for review-before-persist | Out of scope | — |
| Web search summarization (built-in) | SDK already enforces | — |
| OWASP recommendations on prompt injection (defense in depth) | **P2b** — credential redaction in tool outputs + OTel attribute redaction + default-on subprocess env scrubbing. Bash hardening is owned by SDK permission rules + P1b auto-disallow. | Tasks 3–8 |
| Subprocess env inheritance (tool processes inherit container env) | **P2b** — sets `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1` and `CLAUDE_CODE_MCP_ALLOWLIST_ENV=1` on every Claude agent's `ClaudeAgentOptions.env`. Bash tool subprocesses no longer see Anthropic/cloud creds; stdio MCP servers see only their declared env. **The SDK subprocess itself still inherits the full parent env** — no SDK lever for that; P3 sidecar is the structural fix. | Task 4.5 |

**Gaps the plan does not close (deferred or environment-bound):**

1. Network egress allowlist + credential boundary — deferred to **P3** by design.
2. `seccomp`, `--userns-remap`, `--ipc private`, `--pids-limit` — ACA limitations; documented in `docs/security/aca-limitations.md` (Task 16).
3. Sandbox-runtime / bubblewrap / sandbox-exec — out of v1 per spec.
4. gVisor / Firecracker — ACA can't host these.

These omissions are *consistent with the spec's stated v1 scope*, not surprises. The plan flags them in the docs task so an operator can read what is and isn't enforced.

---

## File map

**New files:**

- `src/holodeck/lib/backends/claude_hooks.py` — bundled default PostToolUse hook (credential redaction) + the `redact_credentials` helper reused by the OTel span processor.

(Task 4.5 also modifies `src/holodeck/models/claude_config.py` to add `disable_subprocess_env_scrub` and `src/holodeck/lib/backends/claude_backend.py` to set the two CLI-level scrub flags inside `ClaudeAgentOptions.env`.)
- `src/holodeck/lib/backends/otel_redaction.py` — `RedactingSpanProcessor` for trace attributes. (Spec §"Implementation surface" puts redaction in `otel_bridge.py`, but that module only translates env vars; the actual span attributes are set by the external `ClaudeAgentSdkInstrumentor` package and need a `SpanProcessor` on the tracer provider. New module is honest about that.)
- `tests/unit/lib/backends/test_default_hooks.py`
- `tests/unit/lib/backends/test_otel_redaction.py`
- `tests/unit/deploy/test_dockerfile_hardening.py`
- `tests/unit/deploy/test_aca_volumes.py` (renamed from `test_aca_security_context.py` — ACA does not expose securityContext primitives; see research note in Task 11)
- `docs/security/container-hardening.md` (operator-facing).
- `docs/security/aca-limitations.md` (documented gaps).

**Modified files:**

- `src/holodeck/models/claude_config.py` — add `disable_default_hooks: bool = False`.
- `src/holodeck/lib/backends/claude_backend.py` — plumb default hooks into `build_options()`; activate redaction span processor in `_activate_instrumentation()`.
- `src/holodeck/deploy/dockerfile.py` — corpus to `root:root` + `chmod a-w`; create `/var/holodeck/work` owned by `holodeck`; Node.js gated on stdio-MCP detection rather than provider==anthropic.
- `src/holodeck/cli/commands/deploy.py` — derive `needs_nodejs` from MCP tool `command` fields; lint COPY surface for credential-shaped filenames.
- `src/holodeck/deploy/deployers/azure_containerapps.py` — add `Volume(storage_type=EMPTY_DIR)` declarations on `Template`; add `VolumeMount`s on the `Container` for `/tmp` and `/var/holodeck/work`. **No** `securityContext` block — ACA does not expose it (see Task 11 research note).

---

## Phase split

- **P2a — Container runtime** (Tasks 9–14, 16, 17): Dockerfile changes, ACA template changes, docs.
- **P2b — Prompt-injection defenses** (Tasks 1–8, 15, 17): `disable_default_hooks` schema field, default hooks module, OTel redaction span processor, integration test, docs.

The two sub-phases share no code; they can land as independent PRs. The plan orders P2b first (smaller blast radius, ships under a feature branch) but the order is not load-bearing — they can be parallelised.

---

# P2b — Prompt-injection defenses

## Task 1: Add `disable_default_hooks` schema field

**Files:**
- Modify: `src/holodeck/models/claude_config.py` (`ClaudeConfig`, after `i_understand_this_is_unsafe`)
- Test: `tests/unit/models/test_claude_config.py` (existing file — append)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/models/test_claude_config.py`:

```python
def test_claude_config_disable_default_hooks_defaults_false():
    """ClaudeConfig.disable_default_hooks defaults to False (defaults on)."""
    config = ClaudeConfig()
    assert config.disable_default_hooks is False


def test_claude_config_disable_default_hooks_accepts_true():
    """ClaudeConfig.disable_default_hooks accepts True."""
    config = ClaudeConfig(disable_default_hooks=True)
    assert config.disable_default_hooks is True


def test_claude_config_disable_default_hooks_rejects_non_bool():
    """ClaudeConfig.disable_default_hooks rejects non-bool input."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ClaudeConfig(disable_default_hooks="yes")  # type: ignore[arg-type]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source .venv/bin/activate
pytest tests/unit/models/test_claude_config.py::test_claude_config_disable_default_hooks_defaults_false -v
```
Expected: FAIL with `AttributeError: 'ClaudeConfig' object has no attribute 'disable_default_hooks'`.

- [ ] **Step 3: Add the field**

Edit `src/holodeck/models/claude_config.py`, inside `ClaudeConfig`, immediately after the `i_understand_this_is_unsafe` field (currently around line 327):

```python
    disable_default_hooks: bool = Field(
        default=False,
        description=(
            "Disable HoloDeck-provided default hooks (credential redaction "
            "PostToolUse hook). When True, the agent runs with ONLY "
            "user-defined hooks. Loud warning emitted at load time. Spec "
            "034 P2b. Note: OTel attribute redaction runs independently and "
            "is NOT disabled by this flag."
        ),
    )
```

- [ ] **Step 4: Run the three tests to verify they pass**

```bash
pytest tests/unit/models/test_claude_config.py -v -k disable_default_hooks
```
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/holodeck/models/claude_config.py tests/unit/models/test_claude_config.py
git commit -m "feat(claude): add ClaudeConfig.disable_default_hooks field (spec 034 P2b)"
```

---

## Task 2: ~~Bash AST deny hook~~ — REMOVED (redundant with SDK + P1b)

**Status:** Dropped from P2b on 2026-05-23 after research into the SDK's existing Bash handling.

### Why this was removed

An earlier draft of P2b created a `PreToolUse` hook with six regex patterns (`curl|sh`, `wget|sh`, `bash -c $(...)`, `sudo`, `nc/ncat`, `socat`) intended as a "Bash AST deny list". Closer reading of the [Anthropic secure-deployment doc](https://code.claude.com/docs/en/agent-sdk/secure-deployment) and HoloDeck's P1b implementation showed it was redundant:

1. **The SDK already does Bash AST parsing.** From the doc: *"Before executing bash commands, Claude Code parses them into an AST and matches the result against your permission rules. Commands that cannot be parsed cleanly, or that do not match an allow rule, require explicit approval. A small set of constructs such as `eval` always require approval regardless of allow rules."* The SDK does *real* AST parsing (not regex) against operator-supplied permission rules.
2. **P1b auto-disallows `Bash`** unless the operator explicitly declares it in `agent.tools`. Most HoloDeck agents will never have Bash available at all.
3. **The deny hook was regex-based**, not AST-based. The name "Bash AST deny" was a misnomer — it overlapped with what the SDK genuinely does (parse + match permission rules) without adding parsing of its own. It only fired in the narrow case where an operator (a) granted broad `Bash` access without granular allow rules **and** (b) prompt injection drove a payload matching one of six well-known shapes. The SDK's permission system is the right defense for case (a); the operator must write granular rules.
4. **Base64-encoded payloads defeat regex matching anyway.** The actual defense against post-exploitation is P3's egress proxy, not a string-match deny list.

### What replaces it

Nothing — the gap doesn't exist. Bash hardening for HoloDeck agents is:

- **P1b** — `Bash` is auto-disallowed unless the operator declares it (already shipped).
- **SDK** — when `Bash` is declared, the SDK parses commands into an AST and matches against the operator's `allowed_tools` rules (e.g. `Bash(npm install:*)`); unparseable commands require approval; `eval` always requires approval.
- **Operator responsibility** — if you allow broad `Bash`, you must write granular allow rules. The plan does not try to compensate for under-configured agents.

If an agent legitimately needs Bash and gets attacked, the next defense layer is **P3's egress proxy** (network egress allowlist + credential injection), not a hook on the command string. A determined attacker base64-encodes the payload and defeats any regex deny list; only the network boundary stops the post-exploitation step.

### Plan structure impact

- `claude_hooks.py` is created in Task 3 (not Task 2) and contains only the credential-redaction hook + `redact_credentials` helper.
- `build_default_hooks()` returns a single PostToolUse matcher (not Pre+Post).
- Task 4's `build_options()` merge handles a single event (PostToolUse) instead of two.
- Task 7's integration test targets the credential-redaction path instead of the Bash deny path.

Task numbers 3–8 are unchanged to keep cross-references stable.

_(Task 2 body removed — see rationale above. Task 3 below now owns the `claude_hooks.py` file creation.)_

---

## Task 3: Create `claude_hooks.py` with credential-redaction PostToolUse hook

**Files:**
- Create: `src/holodeck/lib/backends/claude_hooks.py`
- Create: `tests/unit/lib/backends/test_default_hooks.py`

This is the file-creation task. The Bash deny hook that originally lived here was removed (see Task 2). The module ships only the credential-redaction PostToolUse hook and the `redact_credentials` helper (the helper is also reused by Task 5's `RedactingSpanProcessor`).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/lib/backends/test_default_hooks.py`:

```python
"""Tests for HoloDeck-provided default hooks (spec 034 P2b)."""

from __future__ import annotations

import pytest

from holodeck.lib.backends.claude_hooks import (
    CREDENTIAL_PATTERNS,
    build_default_hooks,
    redact_credentials,
)


@pytest.mark.parametrize(
    "raw,expected_marker",
    [
        ("auth=sk-ant-api03-" + "a" * 95, "[REDACTED:anthropic-key]"),
        ("key=AKIA" + "B" * 16, "[REDACTED:aws-access-key]"),
        ("token: ghp_" + "c" * 36, "[REDACTED:github-token]"),
        (
            "jwt: eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ.signaturepart",
            "[REDACTED:jwt]",
        ),
        ("Authorization: Bearer abc.def-123_xyz", "Bearer [REDACTED]"),
    ],
)
def test_redact_credentials_replaces_known_shapes(raw, expected_marker):
    """Each credential shape is replaced by its marker."""
    redacted = redact_credentials(raw)
    assert expected_marker in redacted


def test_redact_credentials_leaves_clean_text_unchanged():
    """No false positives on plain output."""
    text = "Retrieved 42 rows from table customers"
    assert redact_credentials(text) == text


def test_redact_credentials_handles_nested_structures():
    """Redacts strings inside dicts/lists recursively."""
    payload = {
        "auth": {"token": "ghp_" + "x" * 36},
        "rows": ["AKIA" + "Y" * 16, "ok"],
    }
    redacted = redact_credentials(payload)
    assert "[REDACTED:github-token]" in redacted["auth"]["token"]
    assert "[REDACTED:aws-access-key]" in redacted["rows"][0]
    assert redacted["rows"][1] == "ok"


def test_credential_patterns_documented_count():
    """Five credential shapes per spec 034 §'Default hook 2'."""
    assert len(CREDENTIAL_PATTERNS) == 5


def test_build_default_hooks_returns_post_tool_use_only():
    """build_default_hooks wires only the PostToolUse credential hook.

    Bash deny hook removed 2026-05-23 — see Task 2 note. Bash hardening
    is handled by SDK permission rules + P1b auto-disallow.
    """
    hooks = build_default_hooks()
    assert "PreToolUse" not in hooks
    assert "PostToolUse" in hooks
    assert len(hooks["PostToolUse"]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/lib/backends/test_default_hooks.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'holodeck.lib.backends.claude_hooks'`.

- [ ] **Step 3: Create the module**

Create `src/holodeck/lib/backends/claude_hooks.py`:

```python
"""HoloDeck-provided default hooks for Claude Agent SDK.

Ships a single PostToolUse hook that scrubs credential-shaped substrings
out of tool outputs before they re-enter the model context. The
`redact_credentials` helper is also imported by the OTel
RedactingSpanProcessor (spec 034 P2b, Task 5) so trace attributes are
scrubbed independently.

Opt-out via ``claude.disable_default_hooks: true`` — emits a loud
load-time warning. OTel attribute redaction is NOT disabled by that
flag and runs independently.

Spec 034 P2b — Prompt-injection defenses.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from claude_agent_sdk import HookMatcher
from claude_agent_sdk.types import HookContext, HookEvent, SyncHookJSONOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Credential redaction patterns. Order matters — earlier patterns win on
# overlap. Each entry is (replacement_marker, compiled_pattern).
# ---------------------------------------------------------------------------

CREDENTIAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "[REDACTED:anthropic-key]",
        re.compile(r"sk-ant-api03-[A-Za-z0-9_-]{90,}"),
    ),
    (
        "[REDACTED:aws-access-key]",
        re.compile(r"AKIA[0-9A-Z]{16}"),
    ),
    (
        "[REDACTED:github-token]",
        re.compile(r"ghp_[A-Za-z0-9]{36}"),
    ),
    (
        "[REDACTED:jwt]",
        re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    ),
    (
        "Bearer [REDACTED]",
        re.compile(r"Bearer\s+[A-Za-z0-9_\-\.=]+"),
    ),
)


def _redact_string(text: str) -> str:
    """Apply every credential pattern to *text*. Order matters."""
    for replacement, pattern in CREDENTIAL_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_credentials(value: Any) -> Any:
    """Recursively redact credential-shaped substrings inside *value*.

    Walks strings, dicts, lists, and tuples. Non-string scalars (int, float,
    bool, None) are returned unchanged. Returns a new structure — does not
    mutate *value* in place. The OTel span processor (Task 5) reuses this
    helper, so the behavior must stay safe for arbitrary JSON-shaped input.

    Args:
        value: Tool output, span attribute, or arbitrary JSON-shaped data.

    Returns:
        The same structure with credential-shaped strings replaced by
        ``[REDACTED:<kind>]`` markers.
    """
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, dict):
        return {k: redact_credentials(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_credentials(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_credentials(v) for v in value)
    return value


async def _post_tool_credential_redaction(
    input_data: Any,
    tool_use_id: str | None,
    context: HookContext,
) -> SyncHookJSONOutput:
    """PostToolUse hook: scrub credentials out of tool_response before context.

    Returns a hook output that REPLACES tool_response with the redacted
    payload. The SDK feeds the updated output into the model context.
    """
    tool_response = input_data.get("tool_response")
    if tool_response is None:
        return SyncHookJSONOutput()

    redacted = redact_credentials(tool_response)
    if redacted == tool_response:
        return SyncHookJSONOutput()

    return SyncHookJSONOutput(
        hookSpecificOutput={
            "hookEventName": "PostToolUse",
            "additionalContext": None,
        },
        updatedOutput=redacted,
    )


def build_default_hooks() -> dict[HookEvent, list[HookMatcher]]:
    """Build the HoloDeck-provided default hook chain.

    Returns:
        Hook dict suitable for merging into ``ClaudeAgentOptions.hooks``.
        Currently contains only the PostToolUse credential-redaction
        matcher; Bash hardening is owned by SDK permission rules + P1b
        auto-disallow (see spec 034 P2 plan Task 2 for rationale).
    """
    return {
        "PostToolUse": [HookMatcher(hooks=[_post_tool_credential_redaction])],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/lib/backends/test_default_hooks.py -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/holodeck/lib/backends/claude_hooks.py tests/unit/lib/backends/test_default_hooks.py
git commit -m "feat(claude): add credential-redaction PostToolUse hook (spec 034 P2b)"
```

---

## Task 4: Plumb default hooks into `build_options()` with opt-out warning

**Files:**
- Modify: `src/holodeck/lib/backends/claude_backend.py` (`build_options`, near the top of the function around line 367–470)
- Test: `tests/unit/lib/backends/test_claude_backend_default_hooks.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/lib/backends/test_claude_backend_default_hooks.py`:

```python
"""Default hooks are merged into ClaudeAgentOptions by default."""

from __future__ import annotations

import logging

import pytest

from holodeck.lib.backends.claude_backend import build_options
from holodeck.models.agent import Agent
from holodeck.models.claude_config import ClaudeConfig
from holodeck.models.llm import LLMProvider, ProviderEnum


def _minimal_agent(claude: ClaudeConfig | None = None) -> Agent:
    return Agent(
        name="test",
        description="test",
        model=LLMProvider(provider=ProviderEnum.ANTHROPIC, name="claude-sonnet-4-6"),
        instructions=None,  # type: ignore[arg-type]
        claude=claude or ClaudeConfig(),
    )


def test_build_options_includes_default_hooks_by_default():
    """Credential-redaction PostToolUse hook is merged in by default."""
    agent = _minimal_agent()
    options = build_options(agent)
    assert options.hooks is not None
    assert "PostToolUse" in options.hooks
    # Bash deny hook was removed 2026-05-23 — see plan Task 2 note.
    assert "PreToolUse" not in options.hooks
    # The default redaction matcher should be FIRST under PostToolUse so it
    # short-circuits before any user matchers added via spec 028.
    post = options.hooks["PostToolUse"]
    assert len(post) >= 1


def test_build_options_omits_default_hooks_when_disabled(caplog):
    """disable_default_hooks=True drops them and logs a warning."""
    agent = _minimal_agent(ClaudeConfig(disable_default_hooks=True))
    with caplog.at_level(logging.WARNING):
        options = build_options(agent)
    assert options.hooks is None or "PostToolUse" not in (options.hooks or {})
    assert any(
        "disable_default_hooks" in record.message for record in caplog.records
    )


def test_build_options_merges_default_hooks_before_user_hooks():
    """User hooks (spec 028) sit after the HoloDeck defaults."""
    # NOTE: user hooks come in via agent.claude.hooks (spec 028). For this
    # unit test, simulate by populating a hooks dict directly on the agent
    # and asserting the default matcher precedes the user matcher.
    # If `claude.hooks` isn't wired into build_options yet, this test is
    # a spec-compliance proof rather than a regression guard.
    pass  # leave as a documentation test; implementation arrives with spec 028 surface
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/lib/backends/test_claude_backend_default_hooks.py -v
```
Expected: FAIL — `build_options` does not currently inject default hooks.

- [ ] **Step 3: Modify `build_options`**

Open `src/holodeck/lib/backends/claude_backend.py`. Inside `build_options()`, before the final `return ClaudeAgentOptions(**opts_kwargs)` (around line 514), add:

```python
    # Spec 034 P2b — merge HoloDeck default hooks (credential-redaction
    # PostToolUse) ahead of any user-supplied hooks. Opt-out by setting
    # `claude.disable_default_hooks: true` in agent.yaml. Bash hardening
    # is handled by SDK permission rules + P1b auto-disallow.
    from holodeck.lib.backends.claude_hooks import build_default_hooks

    user_hooks: dict[Any, list[Any]] = opts_kwargs.get("hooks") or {}
    if claude is not None and claude.disable_default_hooks:
        logger.warning(
            "HoloDeck default hooks DISABLED for agent '%s' (credential "
            "redaction PostToolUse hook will NOT run). OTel attribute "
            "redaction is unaffected. To re-enable, remove "
            "`claude.disable_default_hooks: true` from agent.yaml.",
            agent.name,
        )
        if user_hooks:
            opts_kwargs["hooks"] = user_hooks
    else:
        default_hooks = build_default_hooks()
        merged: dict[Any, list[Any]] = {}
        # Default hooks first so they short-circuit when they deny.
        for event, matchers in default_hooks.items():
            merged[event] = list(matchers)
        for event, matchers in user_hooks.items():
            if event in merged:
                merged[event] = merged[event] + list(matchers)
            else:
                merged[event] = list(matchers)
        if merged:
            opts_kwargs["hooks"] = merged
```

(Add `logger = logging.getLogger(__name__)` at module top if not present — it should already exist.)

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/lib/backends/test_claude_backend_default_hooks.py -v
```
Expected: 2 PASS (the third is a pass-through doc test).

- [ ] **Step 5: Run the broader claude_backend test suite to catch regressions**

```bash
pytest tests/unit/lib/backends/ -n auto -v
```
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/holodeck/lib/backends/claude_backend.py tests/unit/lib/backends/test_claude_backend_default_hooks.py
git commit -m "feat(claude): wire HoloDeck default hooks into build_options (spec 034 P2b)"
```

---

## Task 4.5: Default-on subprocess env scrubbing

**Files:**
- Modify: `src/holodeck/models/claude_config.py` (add opt-out field)
- Modify: `src/holodeck/lib/backends/claude_backend.py` (`build_options`, same block as Task 4)
- Test: append to `tests/unit/lib/backends/test_claude_backend_default_hooks.py`

### Research note: discovered 2026-05-23

The SDK subprocess inherits the **full parent process env** (minus `CLAUDECODE`) — verified in `claude_agent_sdk._internal.transport.subprocess_cli.connect()`. `ClaudeAgentOptions.env` is *additive*, not a whitelist. There's no SDK API to give the subprocess a smaller env than its parent. P3's Envoy sidecar is the only structural fix for *that* surface.

**However**, Claude Code itself exposes two CLI-level env flags that scrub credentials from *downstream subprocesses* the agent spawns:

| Flag | What it does | Scope |
|---|---|---|
| [`CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1`](https://code.claude.com/docs/en/sandboxing#scope) | *"strip Anthropic and cloud provider credentials from subprocesses"* (per the sandboxing doc) | Bash tool subprocesses (and likely other Claude-Code-spawned subprocesses) |
| [`CLAUDE_CODE_MCP_ALLOWLIST_ENV=1`](https://code.claude.com/docs/en/env-vars) | *"spawn stdio MCP servers with only a safe baseline environment plus the server's configured env, instead of inheriting your shell environment"* | stdio MCP server subprocesses |

Both are off by default in the SDK. HoloDeck can flip them on for every agent — they sit inside `ClaudeAgentOptions.env`, so they're set in the SDK subprocess's env and propagate to its children. **Free defense in depth**: a prompt-injection payload that reads `/proc/self/environ` inside a tool subprocess gets a sanitized view; stdio MCP servers see only their declared env from `agent.yaml`.

### Opt-out field

Add to `ClaudeConfig` (parallel to `disable_default_hooks`):

```python
disable_subprocess_env_scrub: bool = Field(
    default=False,
    description=(
        "Disable HoloDeck's default-on subprocess env scrubbing. When True, "
        "tool subprocesses (Bash, stdio MCP servers) inherit the full "
        "parent env including Anthropic and cloud provider credentials. "
        "Only set this if a legitimate tool needs the inherited creds — "
        "the safer pattern is to declare the required env on the tool's "
        "`command` block instead. Spec 034 P2b. Disables both "
        "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB and CLAUDE_CODE_MCP_ALLOWLIST_ENV."
    ),
)
```

The naming mirrors `disable_default_hooks` — opt-out is loud and named after the risk it surfaces.

- [ ] **Step 1: Add the schema field**

Edit `src/holodeck/models/claude_config.py`, append after `disable_default_hooks`:

```python
disable_subprocess_env_scrub: bool = Field(
    default=False,
    description=(
        "Disable HoloDeck's default-on subprocess env scrubbing "
        "(CLAUDE_CODE_SUBPROCESS_ENV_SCRUB + CLAUDE_CODE_MCP_ALLOWLIST_ENV). "
        "When True, Bash tool subprocesses and stdio MCP servers inherit "
        "the full agent container env including Anthropic and cloud "
        "provider credentials. Spec 034 P2b."
    ),
)
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/unit/lib/backends/test_claude_backend_default_hooks.py`:

```python
def test_build_options_sets_subprocess_env_scrub_by_default():
    """CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1 is set on every Claude agent."""
    agent = _minimal_agent()
    options = build_options(agent)
    assert options.env.get("CLAUDE_CODE_SUBPROCESS_ENV_SCRUB") == "1"
    assert options.env.get("CLAUDE_CODE_MCP_ALLOWLIST_ENV") == "1"


def test_build_options_omits_subprocess_env_scrub_when_disabled(caplog):
    """disable_subprocess_env_scrub=True drops both flags + logs warning."""
    agent = _minimal_agent(ClaudeConfig(disable_subprocess_env_scrub=True))
    with caplog.at_level(logging.WARNING):
        options = build_options(agent)
    assert "CLAUDE_CODE_SUBPROCESS_ENV_SCRUB" not in options.env
    assert "CLAUDE_CODE_MCP_ALLOWLIST_ENV" not in options.env
    assert any(
        "disable_subprocess_env_scrub" in record.message
        for record in caplog.records
    )


def test_build_options_does_not_override_explicit_operator_env():
    """If operator sets CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=0 explicitly, respect it.

    The default-on behavior shouldn't override an operator's explicit
    contrary setting. (This rarely matters — operators flip the schema
    field, not the env var directly — but be a good citizen.)
    """
    # Construct an agent that already has the env var set to "0".
    # Implementation detail: where this hook lives in build_options
    # determines how this is set up. Adjust the test to whatever path
    # operators actually use to inject custom env.
    pass  # documentation test; depends on the env-injection path
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/unit/lib/backends/test_claude_backend_default_hooks.py -v -k "subprocess_env_scrub"
```
Expected: FAIL — `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB` not in `options.env`.

- [ ] **Step 4: Wire the env vars into `build_options()`**

Edit `src/holodeck/lib/backends/claude_backend.py`. In the same block where Task 4 added default-hook merging (just before the `return ClaudeAgentOptions(**opts_kwargs)`), add:

```python
    # Spec 034 P2b — default-on subprocess env scrubbing. These two flags
    # tell the Claude CLI to strip Anthropic/cloud creds from tool
    # subprocesses (Bash, etc.) and to spawn stdio MCP servers with only
    # their declared env (not the full inherited shell env). They don't
    # affect the SDK subprocess's own env (which inherits everything from
    # this serve process — see P3 for the structural fix). Opt out via
    # `claude.disable_subprocess_env_scrub: true` in agent.yaml.
    env_overrides = dict(opts_kwargs.get("env") or {})
    if claude is not None and claude.disable_subprocess_env_scrub:
        logger.warning(
            "Subprocess env scrubbing DISABLED for agent '%s' "
            "(tool subprocesses and stdio MCP servers will inherit the "
            "full agent container env including credentials). To re-enable, "
            "remove `claude.disable_subprocess_env_scrub: true` from "
            "agent.yaml.",
            agent.name,
        )
    else:
        # Don't override an operator's explicit setting — only set the
        # default when the key isn't already present.
        env_overrides.setdefault("CLAUDE_CODE_SUBPROCESS_ENV_SCRUB", "1")
        env_overrides.setdefault("CLAUDE_CODE_MCP_ALLOWLIST_ENV", "1")
    if env_overrides:
        opts_kwargs["env"] = env_overrides
```

This block must sit **after** any other code path that populates `opts_kwargs["env"]` (e.g. the OTel env-var translation from `otel_bridge.translate_observability()`), so the scrub flags layer on top of operator-set env rather than getting overwritten.

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/unit/lib/backends/test_claude_backend_default_hooks.py -v
```
Expected: all PASS (the two new assertions + the existing default-hooks ones).

- [ ] **Step 6: Run the broader test suite for regressions**

```bash
pytest tests/unit/ -n auto -v
```
Expected: green. The OTel translation tests in particular should still pass — the scrub flags are additive on top of the existing env block.

- [ ] **Step 7: Commit**

```bash
git add src/holodeck/models/claude_config.py src/holodeck/lib/backends/claude_backend.py tests/unit/lib/backends/test_claude_backend_default_hooks.py
git commit -m "feat(claude): default-on subprocess env scrubbing (spec 034 P2b)"
```

### What this does NOT defend against

- **The SDK subprocess inheriting the full parent process env.** This is structural — see the research note above and P3 in the spec.
- **Tool subprocesses that genuinely *need* the inherited env.** If a custom function tool reads `AZURE_OPENAI_API_KEY` from `os.environ` to call its own API directly (rather than going through HoloDeck's tool registry), it'll find an empty value after this change. The opt-out field is the escape hatch; the better fix is to make the tool read its credentials from its declared `command`/`env` block instead.
- **Tools that read env from `/proc/self/environ` before fork.** The scrub happens at subprocess spawn, so a tool spawned *after* the scrub flag is set sees the cleaned env. Anything reading the SDK subprocess's own env is unaffected — same surface as the parent-env-inheritance issue.

---

## Task 5: Add `RedactingSpanProcessor` for OTel attribute scrubbing

**Files:**
- Create: `src/holodeck/lib/backends/otel_redaction.py`
- Create: `tests/unit/lib/backends/test_otel_redaction.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/lib/backends/test_otel_redaction.py`:

```python
"""Trace attributes are scrubbed by RedactingSpanProcessor (spec 034 P2b)."""

from __future__ import annotations

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from holodeck.lib.backends.otel_redaction import RedactingSpanProcessor


def _new_provider_with_redaction() -> tuple[TracerProvider, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    # RedactingSpanProcessor must run BEFORE the exporting processor in the
    # chain so the exporter sees redacted attributes.
    provider.add_span_processor(RedactingSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def test_redacting_processor_scrubs_tool_output():
    provider, exporter = _new_provider_with_redaction()
    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span("execute_tool") as span:
        span.set_attribute("tool.output", "key=ghp_" + "x" * 36)
    span_data = exporter.get_finished_spans()[0]
    assert "[REDACTED:github-token]" in span_data.attributes["tool.output"]


def test_redacting_processor_leaves_unrelated_attributes_alone():
    provider, exporter = _new_provider_with_redaction()
    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span("execute_tool") as span:
        span.set_attribute("tool.name", "Bash")
        span.set_attribute("rows.returned", 42)
    span_data = exporter.get_finished_spans()[0]
    assert span_data.attributes["tool.name"] == "Bash"
    assert span_data.attributes["rows.returned"] == 42


def test_redacting_processor_handles_tool_input_namespace():
    provider, exporter = _new_provider_with_redaction()
    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span("execute_tool") as span:
        span.set_attribute("tool.input.headers", "Authorization: Bearer abc.def-1")
    span_data = exporter.get_finished_spans()[0]
    assert "Bearer [REDACTED]" in span_data.attributes["tool.input.headers"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/lib/backends/test_otel_redaction.py -v
```
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create the module**

Create `src/holodeck/lib/backends/otel_redaction.py`:

```python
"""OTel span processor that redacts credential-shaped trace attributes.

Sits *before* exporting span processors on the tracer provider so any
exporter (OTLP, Console, Azure Monitor) sees scrubbed attributes. Runs
independently of ``claude.disable_default_hooks`` — operators cannot
accidentally disable trace redaction by disabling user-facing hooks.

Spec 034 P2b §"OTel attribute redaction (independent of hooks)".
"""

from __future__ import annotations

import logging
from typing import Any

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from holodeck.lib.backends.claude_hooks import redact_credentials

logger = logging.getLogger(__name__)

# Span attribute name prefixes whose values are scrubbed. Anything with a
# different prefix is left alone — these namespaces are the ones the GenAI
# instrumentor (`otel-instrumentation-claude-agent-sdk`) populates with
# tool I/O content.
_REDACTED_PREFIXES: tuple[str, ...] = (
    "tool.input",
    "tool.output",
    "gen_ai.tool.input",
    "gen_ai.tool.output",
    "gen_ai.prompt",
    "gen_ai.completion",
)


def _should_redact(attribute_key: str) -> bool:
    return any(attribute_key.startswith(prefix) for prefix in _REDACTED_PREFIXES)


class RedactingSpanProcessor(SpanProcessor):
    """SpanProcessor that scrubs credential-shaped strings on span end.

    The OTel SDK exposes span attributes on ``ReadableSpan`` via the
    ``_attributes`` dict, which the SDK mutates in place during the span's
    lifetime. Mutating it on ``on_end`` is the documented mechanism used by
    e.g. the Baggage span processor. Exporters registered AFTER this
    processor see the redacted payload.
    """

    def on_start(  # type: ignore[override]
        self, span: Span, parent_context: Context | None = None
    ) -> None:
        return None

    def on_end(self, span: ReadableSpan) -> None:  # type: ignore[override]
        attributes = getattr(span, "_attributes", None)
        if not attributes:
            return
        for key in list(attributes.keys()):
            if not _should_redact(key):
                continue
            try:
                attributes[key] = redact_credentials(attributes[key])
            except Exception:  # noqa: BLE001 — never break tracing on redact failure
                logger.warning(
                    "RedactingSpanProcessor: failed to redact attribute %s; "
                    "leaving original value",
                    key,
                )

    def shutdown(self) -> None:  # type: ignore[override]
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:  # type: ignore[override]
        return True
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/lib/backends/test_otel_redaction.py -v
```
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/holodeck/lib/backends/otel_redaction.py tests/unit/lib/backends/test_otel_redaction.py
git commit -m "feat(otel): add RedactingSpanProcessor for trace attributes (spec 034 P2b)"
```

---

## Task 6: Wire `RedactingSpanProcessor` into the tracer provider

**Files:**
- Modify: `src/holodeck/lib/backends/claude_backend.py` (`_activate_instrumentation`, around line 1316)
- Test: `tests/integration/observability/test_redaction_e2e.py` (new — integration test that runs a real instrumented `query` against a stub model)

The right insertion point depends on where the tracer provider is constructed. The codebase exposes it through `get_observability_context()` (already used in `_activate_instrumentation`). The processor must be registered before any exporter so exporters see scrubbed attributes.

- [ ] **Step 1: Locate the tracer provider construction**

```bash
grep -n "TracerProvider\|add_span_processor\|tracer_provider" \
    src/holodeck/lib/observability/*.py src/holodeck/lib/backends/*.py 2>&1 | head -20
```

Expected: surfaces the module that builds the `TracerProvider`. If the construction site lives in `holodeck.lib.observability.context` (`get_observability_context`), patch is straightforward — add the processor there. If it lives elsewhere, follow the call chain from `_activate_instrumentation`.

- [ ] **Step 2: Write the failing test**

Create `tests/integration/observability/test_redaction_e2e.py`:

```python
"""End-to-end: tool output with a fake credential is redacted in spans."""

from __future__ import annotations

import pytest

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from holodeck.lib.backends.otel_redaction import RedactingSpanProcessor


@pytest.mark.integration
def test_holodeck_tracer_provider_includes_redaction_processor():
    """Constructed tracer provider has RedactingSpanProcessor registered."""
    # Replace this with the real factory call used by claude_backend.
    from holodeck.lib.observability.context import (
        build_tracer_provider,  # actual symbol name to be confirmed in Step 1
    )

    provider = build_tracer_provider(
        # minimal config flags; adjust to the real signature
    )
    processors = getattr(provider, "_active_span_processor", None)
    # In SDK >=1.20 processors are wrapped in a MultiSpanProcessor. Pull
    # the underlying list:
    inner = getattr(processors, "_span_processors", [processors])
    assert any(
        isinstance(p, RedactingSpanProcessor) for p in inner
    ), "RedactingSpanProcessor must be on the tracer provider"
```

(This step intentionally pseudo-codes the factory call — the engineer doing the implementation completes the exact API in Step 3 based on what Step 1 reveals.)

- [ ] **Step 3: Run test to verify it fails (as expected, on the API name)**

```bash
pytest tests/integration/observability/test_redaction_e2e.py -v
```
Expected: FAIL.

- [ ] **Step 4: Register the processor in the tracer provider factory**

Patch the tracer provider construction site identified in Step 1. The change is:

```python
from holodeck.lib.backends.otel_redaction import RedactingSpanProcessor

# Add as the FIRST processor on the provider — runs before exporters.
provider.add_span_processor(RedactingSpanProcessor())
# … existing add_span_processor calls for OTLP exporter follow.
```

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/integration/observability/test_redaction_e2e.py -v
```
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/holodeck/lib/observability tests/integration/observability/test_redaction_e2e.py
git commit -m "feat(otel): register RedactingSpanProcessor on tracer provider (spec 034 P2b)"
```

---

## Task 7: Synthetic prompt-injection integration test

**Files:**
- Create: `tests/integration/security/test_credential_redaction_e2e.py`

- [ ] **Step 1: Write the test**

Create `tests/integration/security/test_credential_redaction_e2e.py`:

```python
"""Prompt-injection scenario: credentials in tool output are scrubbed before
they reach the model context.

Runs against the hook in isolation (no live SDK call — quota-free CI).
The hook is invoked with realistic SDK-shaped payloads that simulate a
tool returning a response containing a credential-shaped substring.
"""

from __future__ import annotations

import pytest

from holodeck.lib.backends.claude_hooks import (
    _post_tool_credential_redaction,
    redact_credentials,
)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_github_token_redacted_in_tool_response():
    """A GitHub token in tool_response is replaced before model context."""
    payload = {
        "tool_name": "Bash",
        "tool_use_id": "t1",
        "tool_response": (
            "exported GH_TOKEN=ghp_" + "a" * 36 + " for the workflow"
        ),
    }
    output = await _post_tool_credential_redaction(
        payload, "t1", None,  # type: ignore[arg-type]
    )
    updated = output.get("updatedOutput")
    assert updated is not None
    assert "[REDACTED:github-token]" in updated
    assert "ghp_" + "a" * 36 not in updated


@pytest.mark.integration
@pytest.mark.asyncio
async def test_clean_response_is_passthrough():
    """No-op when the tool response has no credential-shaped content."""
    payload = {
        "tool_name": "Read",
        "tool_use_id": "t2",
        "tool_response": "Retrieved 42 rows from table customers",
    }
    output = await _post_tool_credential_redaction(
        payload, "t2", None,  # type: ignore[arg-type]
    )
    # No-op output: no updatedOutput section.
    assert "updatedOutput" not in output


@pytest.mark.integration
def test_nested_tool_response_redacted():
    """Direct test of the helper on a JSON-shaped payload."""
    payload = {
        "logs": [
            {"line": "auth: Bearer eyJfoo.eyJbar.signature"},
            {"line": "ok"},
        ],
        "headers": {"Authorization": "Bearer abc.def-1"},
    }
    redacted = redact_credentials(payload)
    assert "Bearer [REDACTED]" in redacted["logs"][0]["line"] or \
        "[REDACTED:jwt]" in redacted["logs"][0]["line"]
    assert redacted["logs"][1]["line"] == "ok"
    assert "Bearer [REDACTED]" in redacted["headers"]["Authorization"]
```

- [ ] **Step 2: Run test**

```bash
pytest tests/integration/security/test_credential_redaction_e2e.py -v
```
Expected: 3 PASS (no implementation needed; tests exercise the hook + helper from Task 3).

- [ ] **Step 3: Commit**

```bash
git add tests/integration/security/test_credential_redaction_e2e.py
git commit -m "test(security): synthetic prompt-injection coverage for credential redaction"
```

---

## Task 8: P2b verification — full local CI pass

- [ ] **Step 1: Run formatters, linters, type-checker, security checks**

```bash
source .venv/bin/activate
make format
make lint
make type-check
make security
make test-unit
```

Expected: green across the board. Fix any issues introduced by P2b code (mypy strict mode is enforced).

- [ ] **Step 2: Commit fixes if any**

```bash
git add -p
git commit -m "chore: P2b lint + type fixes"
```

---

# P2a — Container runtime hardening

## Task 9: Refine Node.js install gating to MCP stdio detection

**Files:**
- Modify: `src/holodeck/cli/commands/deploy.py` (around line 622)
- Test: `tests/unit/cli/test_deploy_nodejs_gating.py` (new)

The current heuristic — `needs_nodejs = agent.model.provider == ProviderEnum.ANTHROPIC` — over-installs Node. The SDK bundles its own Claude Code binary; Node is only needed when an MCP server's `command` resolves to `node`/`npx`. Refining this shrinks the image by ~50 MiB per agent that has no Node MCP servers.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/cli/test_deploy_nodejs_gating.py`:

```python
"""needs_nodejs is derived from stdio MCP server `command` fields, not provider."""

from __future__ import annotations

import pytest

from holodeck.cli.commands.deploy import _agent_needs_nodejs
from holodeck.models.agent import Agent
from holodeck.models.llm import LLMProvider, ProviderEnum
from holodeck.models.tool import McpServerTool, ToolType


def _claude_agent_with_tools(tools: list) -> Agent:
    return Agent(
        name="t",
        description="t",
        model=LLMProvider(provider=ProviderEnum.ANTHROPIC, name="claude-sonnet-4-6"),
        instructions=None,  # type: ignore[arg-type]
        tools=tools,
    )


def test_no_mcp_tools_means_no_nodejs():
    agent = _claude_agent_with_tools([])
    assert _agent_needs_nodejs(agent) is False


def test_python_mcp_server_means_no_nodejs():
    agent = _claude_agent_with_tools([
        McpServerTool(
            type=ToolType.MCP,
            name="qdrant",
            command=["python", "-m", "qdrant_mcp"],
        ),
    ])
    assert _agent_needs_nodejs(agent) is False


def test_node_mcp_server_means_nodejs_required():
    agent = _claude_agent_with_tools([
        McpServerTool(
            type=ToolType.MCP,
            name="filesystem",
            command=["node", "/srv/fs-mcp.js"],
        ),
    ])
    assert _agent_needs_nodejs(agent) is True


def test_npx_mcp_server_means_nodejs_required():
    agent = _claude_agent_with_tools([
        McpServerTool(
            type=ToolType.MCP,
            name="brave",
            command=["npx", "-y", "@modelcontextprotocol/server-brave-search"],
        ),
    ])
    assert _agent_needs_nodejs(agent) is True
```

(Adjust the `McpServerTool` constructor to match the real tool model — confirm field names from `src/holodeck/models/tool.py`.)

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/cli/test_deploy_nodejs_gating.py -v
```
Expected: FAIL on `ImportError: cannot import name '_agent_needs_nodejs'`.

- [ ] **Step 3: Replace the heuristic in `deploy.py`**

Edit `src/holodeck/cli/commands/deploy.py`. Around line 622, replace:

```python
needs_nodejs = agent.model.provider == ProviderEnum.ANTHROPIC
```

with a function call:

```python
needs_nodejs = _agent_needs_nodejs(agent)
```

And add the helper at module scope:

```python
_NODE_BIN_NAMES: frozenset[str] = frozenset({"node", "npx", "yarn", "pnpm"})


def _agent_needs_nodejs(agent: "Agent") -> bool:
    """Return True iff any MCP tool spawns Node via its stdio command.

    The Claude Agent SDK bundles its own CLI binary, so Node is no longer
    required just because the provider is Anthropic. Node is needed only
    when an MCP server's stdio `command[0]` is a Node interpreter
    (`node`, `npx`, `yarn`, `pnpm`).

    Spec 034 P2a §"Generated Dockerfile changes".
    """
    for tool in agent.tools or []:
        command = getattr(tool, "command", None)
        if not command:
            continue
        first = command[0] if isinstance(command, list) and command else None
        if isinstance(first, str) and first.split("/")[-1] in _NODE_BIN_NAMES:
            return True
    return False
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/cli/test_deploy_nodejs_gating.py -v
```
Expected: 4 PASS.

- [ ] **Step 5: Run downstream deploy tests to catch regressions**

```bash
pytest tests/unit/cli/ tests/unit/deploy/ -n auto -v
```
Expected: green.

- [ ] **Step 6: Commit**

```bash
git add src/holodeck/cli/commands/deploy.py tests/unit/cli/test_deploy_nodejs_gating.py
git commit -m "feat(deploy): gate Node.js install on stdio-MCP detection (spec 034 P2a)"
```

---

## Task 10: Dockerfile — root-own + chmod a-w on corpus directories

**Files:**
- Modify: `src/holodeck/deploy/dockerfile.py`
- Test: `tests/unit/deploy/test_dockerfile_hardening.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/deploy/test_dockerfile_hardening.py`:

```python
"""Generated Dockerfile applies P2a hardening (spec 034)."""

from __future__ import annotations

from holodeck.deploy.dockerfile import generate_dockerfile


def _gen(**overrides) -> str:
    defaults = dict(
        agent_name="t",
        port=8080,
        protocol="rest",
        instruction_files=["instructions.md"],
        data_directories=["data/"],
    )
    defaults.update(overrides)
    return generate_dockerfile(**defaults)


def test_data_directory_copied_as_root_and_chmod_a_minus_w():
    df = _gen()
    assert "COPY --chown=root:root data/" in df
    assert "chmod -R a-w /app/data" in df


def test_instruction_files_copied_as_root_and_chmod_a_minus_w():
    df = _gen()
    assert "COPY --chown=root:root instructions.md" in df
    # Instruction files are individual; the chmod walks /app/instructions/*
    # or matches each file. The spec uses a single `chmod -R a-w` on each
    # copied target — accept either /app/instructions.md or /app/instructions:
    assert "chmod a-w /app/instructions.md" in df or "chmod -R a-w /app" in df


def test_scratch_dir_created_writable_for_holodeck_user():
    df = _gen()
    assert "mkdir -p /var/holodeck/work" in df
    assert "chown holodeck:holodeck /var/holodeck/work" in df


def test_nodejs_omitted_by_default():
    df = _gen(needs_nodejs=False)
    assert "nodesource" not in df
    assert "apt-get install -y --no-install-recommends nodejs" not in df


def test_nodejs_included_when_flag_true():
    df = _gen(needs_nodejs=True)
    assert "nodesource" in df
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/deploy/test_dockerfile_hardening.py -v
```
Expected: FAIL.

- [ ] **Step 3: Modify the Dockerfile template**

Edit `src/holodeck/deploy/dockerfile.py`. Replace the data/instruction COPY blocks and the trailing chown:

```jinja
{% if instruction_files %}
# Copy instruction files (read-only at runtime)
{% for file in instruction_files %}
COPY --chown=root:root {{ file }} /app/{{ file }}
RUN chmod a-w /app/{{ file }}
{% endfor %}
{% endif %}

{% if data_directories %}
# Copy data directories (read-only at runtime)
{% for dir in data_directories %}
COPY --chown=root:root {{ dir }} /app/{{ dir }}
RUN chmod -R a-w /app/{{ dir }}
{% endfor %}
{% endif %}
```

And add (just before the final `USER holodeck` block):

```jinja
# Spec 034 P2a — writable scratch dir for SDK + tool subprocesses.
# Mounted as tmpfs by the ACA deployer.
RUN mkdir -p /var/holodeck/work && chown holodeck:holodeck /var/holodeck/work
```

Then **remove** the existing `RUN chown -R holodeck:holodeck /app` line (line 84) — it conflicts with the root-owned read-only corpus. The entrypoint script and agent.yaml are owned by root and world-readable by default, which is fine for execution.

(Keep `chmod +x /app/entrypoint.sh` after its COPY.)

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/deploy/test_dockerfile_hardening.py -v
```
Expected: all PASS.

- [ ] **Step 5: Smoke-test image build locally (optional, fast)**

```bash
cd sample/financial-assistant/claude
holodeck deploy build --tag p2a-smoke --no-push
docker run --rm --entrypoint sh ghcr.io/justinbarias/holodeck-financial-assistant:p2a-smoke \
    -c "test ! -w /app/data && echo OK"
```
Expected: prints `OK` (read-only confirmed).

- [ ] **Step 6: Commit**

```bash
git add src/holodeck/deploy/dockerfile.py tests/unit/deploy/test_dockerfile_hardening.py
git commit -m "feat(deploy): root-own corpus + tmpfs scratch in generated Dockerfile (spec 034 P2a)"
```

---

## Task 11: ACA template — EmptyDir volumes for ephemeral scratch

**Files:**
- Modify: `src/holodeck/deploy/deployers/azure_containerapps.py` (around line 170 — `Container(...)` and `Template(...)` construction)
- Test: `tests/unit/deploy/test_aca_volumes.py` (new)

### Research note: ACA does NOT expose `securityContext`

Earlier drafts of this plan called for emitting a `ContainerSecurityContext` block on the `Container` resource (`runAsNonRoot=true`, `allowPrivilegeEscalation=false`, `capabilities.drop=["ALL"]`, `readOnlyRootFilesystem=true`). **Verification on 2026-05-23 confirmed those primitives are not exposed by ACA at any API version:**

| Source | Finding |
|---|---|
| `azure-mgmt-appcontainers==4.0.0` `Container.__init__` | Accepts only `image, name, command, args, env, resources, volume_mounts, probes`. No `security_context`, no `ContainerSecurityContext` model anywhere in the SDK. The only `Security`-named class is `IpSecurityRestrictionRule` (ingress IP allowlist). |
| [ARM Bicep template ref — `Microsoft.App/containerApps@2026-01-01`](https://learn.microsoft.com/en-us/azure/templates/microsoft.app/containerapps?pivots=deployment-language-bicep) | Container resource has no `securityContext`, `runAsNonRoot`, `runAsUser`, `allowPrivilegeEscalation`, `readOnlyRootFilesystem`, `capabilities`, or `seccompProfile`. **Consistent across every API version from 2022-03-01 through 2026-01-01.** |
| [Microsoft ACA Security Baseline (updated 2026-04-01)](https://learn.microsoft.com/en-us/security/benchmark/azure/baselines/azure-container-apps-security-baseline) | Documents network/identity/encryption controls only. Zero mention of container-level security primitives. Explicitly states `Customer can access HOST / OS: No Access` — the host security boundary is owned by Microsoft and is not customer-tunable. |
| GitHub `microsoft/azure-container-apps` | No active feature request for `securityContext`. Not on the public roadmap. |

ACA is a serverless-ish platform: Microsoft owns the host kernel, capability set, and seccomp profile platform-wide. Customer security expression happens at the **image level** (Dockerfile `USER`, file permissions) and via the **surfaces ACA does expose** (VNet/NSG, internal ingress, managed identity, ACR scanning).

**The only securityContext-adjacent thing ACA exposes is `Volume`/`VolumeMount`** with storage type `EMPTY_DIR` (verified — `StorageType` enum has `AZURE_FILE`, `EMPTY_DIR`, `NFS_AZURE_FILE`, `SECRET`). This task wires that up.

The four primitives we wanted are documented as fundamental gaps in Task 14's `aca-limitations.md`; the escape hatch (for threat models that require them) is AKS with Pod Security Standards.

### What this task actually does

Add `EMPTY_DIR` volumes at the `Template` level and mount them at `/tmp` and `/var/holodeck/work` on the `Container`. That's it — no `security_context` parameter exists to populate.

- [ ] **Step 1: Confirm SDK surface against the installed version**

```bash
source .venv/bin/activate
python -c "
from azure.mgmt.appcontainers import models as m
import inspect
print('Container.__init__:', inspect.signature(m.Container.__init__))
print('Volume.__init__:', inspect.signature(m.Volume.__init__))
print('VolumeMount.__init__:', inspect.signature(m.VolumeMount.__init__))
print('StorageType values:', [v for v in dir(m.StorageType) if not v.startswith('_')])
"
```

Expected output (verified 2026-05-23 on SDK 4.0.0):

```
Container.__init__: (self, *, image, name, command, args, env, resources, volume_mounts, probes, **kwargs)
Volume.__init__: (self, *, name, storage_type, storage_name, secrets, mount_options, **kwargs)
VolumeMount.__init__: (self, *, volume_name, mount_path, sub_path, **kwargs)
StorageType values: ['AZURE_FILE', 'EMPTY_DIR', 'NFS_AZURE_FILE', 'SECRET']
```

If the installed SDK has gained a `security_context` field on `Container` since this plan was written, **stop and re-evaluate** — the four primitives may now be expressible, and Tasks 12 + 14 should be revisited.

- [ ] **Step 2: Write the failing test**

Create `tests/unit/deploy/test_aca_volumes.py`:

```python
"""ACA deployer emits EmptyDir volumes for ephemeral scratch (spec 034 P2a).

ACA does NOT expose securityContext primitives at any API version — see
the research note in Task 11 of the P2 plan. This test covers only the
volumes/volume_mounts surface, which IS exposed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from azure.mgmt.appcontainers.models import StorageType

from holodeck.deploy.deployers.azure_containerapps import (
    AzureContainerAppsDeployer,
)
from holodeck.models.deployment import AzureContainerAppsConfig


@pytest.fixture
def deployer():
    config = AzureContainerAppsConfig(
        subscription_id="00000000-0000-0000-0000-000000000000",
        resource_group="rg",
        environment_name="env",
        location="eastus",
    )
    with patch(
        "holodeck.deploy.deployers.azure_containerapps.ContainerAppsAPIClient"
    ):
        with patch(
            "holodeck.deploy.deployers.azure_containerapps.DefaultAzureCredential"
        ):
            yield AzureContainerAppsDeployer(config)


def _stub_poller_result():
    return MagicMock(
        id="x", name="x", provisioning_state="Succeeded",
        configuration=MagicMock(ingress=MagicMock(fqdn="x.example")),
    )


def test_deploy_emits_tmpfs_volumes_at_template_level(deployer):
    """Template.volumes contains two EMPTY_DIR volumes."""
    with patch.object(
        deployer._client.container_apps, "begin_create_or_update"
    ) as bcou:
        poller = MagicMock()
        poller.result.return_value = _stub_poller_result()
        bcou.return_value = poller
        deployer.deploy(
            service_name="t", image_uri="ghcr.io/foo:bar", port=8080, env_vars={},
        )
        envelope = bcou.call_args.kwargs["container_app_envelope"]
        volumes = envelope.template.volumes or []
        names = {v.name for v in volumes}
        assert names == {"tmp", "sdk-scratch"}
        for v in volumes:
            assert v.storage_type == StorageType.EMPTY_DIR


def test_deploy_emits_volume_mounts_on_container(deployer):
    """Container.volume_mounts wires /tmp and /var/holodeck/work."""
    with patch.object(
        deployer._client.container_apps, "begin_create_or_update"
    ) as bcou:
        poller = MagicMock()
        poller.result.return_value = _stub_poller_result()
        bcou.return_value = poller
        deployer.deploy(
            service_name="t", image_uri="ghcr.io/foo:bar", port=8080, env_vars={},
        )
        envelope = bcou.call_args.kwargs["container_app_envelope"]
        mounts = envelope.template.containers[0].volume_mounts or []
        paths = {m.mount_path for m in mounts}
        assert paths == {"/tmp", "/var/holodeck/work"}
        volume_names = {m.volume_name for m in mounts}
        assert volume_names == {"tmp", "sdk-scratch"}


def test_deploy_does_not_attempt_to_set_security_context(deployer):
    """Regression guard: Container does NOT carry a security_context attr.

    ACA's Container resource has no security_context field at any API
    version. If a future SDK release adds one, this test will fail and
    prompt revisiting Tasks 11/12/14 of spec 034 P2.
    """
    with patch.object(
        deployer._client.container_apps, "begin_create_or_update"
    ) as bcou:
        poller = MagicMock()
        poller.result.return_value = _stub_poller_result()
        bcou.return_value = poller
        deployer.deploy(
            service_name="t", image_uri="ghcr.io/foo:bar", port=8080, env_vars={},
        )
        envelope = bcou.call_args.kwargs["container_app_envelope"]
        container = envelope.template.containers[0]
        # Container should not have a security_context kwarg set. The
        # azure-mgmt-appcontainers SDK doesn't even define the attribute,
        # but be explicit about the regression guard.
        assert not hasattr(container, "security_context") or \
            container.security_context is None
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
pytest tests/unit/deploy/test_aca_volumes.py -v
```
Expected: FAIL on the two assertions about volumes/mounts (the third test passes incidentally — current code doesn't set security_context).

- [ ] **Step 4: Add volumes + mounts to the deployer**

Edit `src/holodeck/deploy/deployers/azure_containerapps.py`. Extend the SDK imports inside `__init__` (the existing import block around lines 70–82):

```python
from azure.mgmt.appcontainers.models import (
    Configuration,
    Container,
    ContainerApp,
    ContainerAppProbe,
    ContainerAppProbeHttpGet,
    ContainerResources,
    EnvironmentVar,
    Ingress,
    Scale,
    StorageType,            # NEW
    Template,
    TrafficWeight,
    Volume,                 # NEW
    VolumeMount,            # NEW
)
```

Add the corresponding `self._StorageType`, `self._Volume`, `self._VolumeMount` attributes mirroring the existing pattern:

```python
self._StorageType: type[StorageType] = StorageType
self._Volume: type[Volume] = Volume
self._VolumeMount: type[VolumeMount] = VolumeMount
```

In `deploy()`, immediately before the `container = self._Container(...)` block (around line 170), build:

```python
# Spec 034 P2a — ephemeral writable scratch (EmptyDir). ACA does not
# expose tmpfs directly, but EmptyDir is per-replica and cleared on
# replica restart, which gives the same operational property: tool
# outputs and SDK scratch never persist across replicas. See the
# research note in spec 034 P2 plan Task 11 for why this is the only
# securityContext-adjacent primitive ACA exposes.
volumes = [
    self._Volume(name="tmp", storage_type=self._StorageType.EMPTY_DIR),
    self._Volume(name="sdk-scratch", storage_type=self._StorageType.EMPTY_DIR),
]
volume_mounts = [
    self._VolumeMount(volume_name="tmp", mount_path="/tmp"),
    self._VolumeMount(volume_name="sdk-scratch", mount_path="/var/holodeck/work"),
]
```

Update the `Container(...)` construction to include `volume_mounts=volume_mounts`:

```python
container = self._Container(
    name=service_name,
    image=image_uri,
    resources=self._ContainerResources(
        cpu=self._config.cpu,
        memory=self._config.memory,
    ),
    env=env_list if env_list else None,
    probes=[liveness_probe, readiness_probe],
    volume_mounts=volume_mounts,
)
```

Update the `Template(...)` construction to include `volumes=volumes`:

```python
template = self._Template(
    containers=[container],
    scale=scale,
    volumes=volumes,
)
```

**Do not** add a `security_context=` kwarg — `Container.__init__` does not accept one.

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/unit/deploy/test_aca_volumes.py -v
```
Expected: 3 PASS.

- [ ] **Step 6: Run broader deploy tests for regressions**

```bash
pytest tests/unit/deploy/ -n auto -v
```
Expected: green.

- [ ] **Step 7: Commit**

```bash
git add src/holodeck/deploy/deployers/azure_containerapps.py tests/unit/deploy/test_aca_volumes.py
git commit -m "feat(deploy): emit ACA EmptyDir volumes for ephemeral scratch (spec 034 P2a)"
```

---

## Task 12: Update `_echo_resolved_config` to surface P2a posture

**Files:**
- Modify: `src/holodeck/deploy/deployers/azure_containerapps.py` (`_echo_resolved_config`, around line 338)
- Test: append to `tests/unit/deploy/test_aca_volumes.py`

The operator-facing echo at deploy time already prints CPU/memory/session cap/ingress. P2a adds two honest lines: the ephemeral scratch mounts ACA actually provides, and the image-layer enforcement points (USER directive, corpus chmod). It does **not** claim `runAsNonRoot`/`capabilities.drop`/etc. — those aren't expressible in ACA.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/deploy/test_aca_volumes.py`:

```python
def test_echo_resolved_config_mentions_p2a_posture(deployer, caplog):
    """Deploy-time echo names the ACA-enforced and image-enforced posture."""
    import logging
    with caplog.at_level(
        logging.INFO, logger="holodeck.deploy.deployers.azure_containerapps"
    ):
        deployer._echo_resolved_config(
            service_name="t", image_uri="ghcr.io/foo:bar", port=8080,
        )
    msgs = " ".join(r.message for r in caplog.records).lower()
    # ACA-enforced (EmptyDir volumes)
    assert "emptydir" in msgs or "/var/holodeck/work" in msgs
    assert "/tmp" in msgs
    # Image-enforced (Dockerfile USER + chmod)
    assert "non-root" in msgs
    assert "read-only" in msgs


def test_echo_resolved_config_does_not_falsely_claim_aca_securitycontext(
    deployer, caplog,
):
    """Don't claim runAsNonRoot/capabilities.drop — ACA can't express them."""
    import logging
    with caplog.at_level(
        logging.INFO, logger="holodeck.deploy.deployers.azure_containerapps"
    ):
        deployer._echo_resolved_config(
            service_name="t", image_uri="ghcr.io/foo:bar", port=8080,
        )
    msgs = " ".join(r.message for r in caplog.records).lower()
    assert "runasnonroot" not in msgs
    assert "allowprivilegeescalation" not in msgs
    assert "capabilities.drop" not in msgs
    assert "capabilities dropped" not in msgs
```

- [ ] **Step 2: Run test**

```bash
pytest tests/unit/deploy/test_aca_volumes.py -v -k echo
```
Expected: FAIL on the first test (missing log lines).

- [ ] **Step 3: Extend `_echo_resolved_config`**

Add the following logger lines inside `_echo_resolved_config`, after the existing probes line (around line 376):

```python
logger.info(
    "  Ephemeral scratch (ACA EmptyDir): /tmp, /var/holodeck/work"
)
logger.info(
    "  Image-layer hardening: non-root (UID 1000), "
    "corpus read-only (/app/data, /app/instructions)"
)
logger.info(
    "  Note: ACA does not expose runAsNonRoot, capabilities.drop, "
    "readOnlyRootFilesystem, or seccomp — see "
    "docs/security/aca-limitations.md"
)
```

The third line is verbose on purpose — operators reading the deploy log should see that the image-layer non-root posture is intentionally where the enforcement lives, not because we forgot to set a manifest field.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/deploy/test_aca_volumes.py -v -k echo
```
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/holodeck/deploy/deployers/azure_containerapps.py tests/unit/deploy/test_aca_volumes.py
git commit -m "feat(deploy): echo honest P2a posture at deploy time (spec 034 P2a)"
```

---

## Task 13: Lint COPY surface for credential-shaped filenames

**Files:**
- Modify: `src/holodeck/cli/commands/deploy.py` (the build command — around the section that resolves `instruction_files` + `data_directories` near line 583)

The Anthropic doc warns about mounting `.env`, `~/.ssh`, `~/.aws`, `*.pem`, `*.key`, `credentials*.json`. HoloDeck doesn't mount host directories, but a user's `data_directories` could include a `.env` file that gets baked into the image. Emit a deploy-time warning when one is detected.

- [ ] **Step 1: Write the failing test**

Append to a new file `tests/unit/cli/test_deploy_credential_lint.py`:

```python
"""Deploy-time warning when COPY surface contains credential-shaped files."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from holodeck.cli.commands.deploy import _warn_if_credential_files_in_copy_surface


def test_warn_on_env_file(tmp_path: Path, caplog):
    (tmp_path / ".env").write_text("SECRET=1")
    with caplog.at_level(logging.WARNING):
        _warn_if_credential_files_in_copy_surface(
            instruction_files=[],
            data_directories=[str(tmp_path)],
            base_dir=tmp_path.parent,
        )
    assert any(".env" in r.message for r in caplog.records)


def test_no_warning_on_clean_directory(tmp_path: Path, caplog):
    (tmp_path / "data.csv").write_text("a,b")
    with caplog.at_level(logging.WARNING):
        _warn_if_credential_files_in_copy_surface(
            instruction_files=[],
            data_directories=[str(tmp_path)],
            base_dir=tmp_path.parent,
        )
    assert not any(
        "credential" in r.message.lower() for r in caplog.records
    )


@pytest.mark.parametrize("name", ["secrets.pem", "id_rsa", "azure-credentials.json", "service-account.json"])
def test_warn_on_credential_shaped_filenames(tmp_path: Path, caplog, name):
    (tmp_path / name).write_text("x")
    with caplog.at_level(logging.WARNING):
        _warn_if_credential_files_in_copy_surface(
            instruction_files=[],
            data_directories=[str(tmp_path)],
            base_dir=tmp_path.parent,
        )
    assert any(name in r.message for r in caplog.records)
```

- [ ] **Step 2: Run test**

```bash
pytest tests/unit/cli/test_deploy_credential_lint.py -v
```
Expected: FAIL on `ImportError`.

- [ ] **Step 3: Add the lint helper**

In `src/holodeck/cli/commands/deploy.py`, add at module scope:

```python
import fnmatch

_CREDENTIAL_FILENAME_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa*",
    "*credentials*.json",
    "*service-account*.json",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
)


def _warn_if_credential_files_in_copy_surface(
    *,
    instruction_files: list[str],
    data_directories: list[str],
    base_dir: "Path",
) -> None:
    """Emit a warning when the Docker COPY surface contains files shaped
    like credentials. Filenames only — no content inspection.

    Spec 034 P2a §"Read-only code mounting / sensitive files".
    """
    from pathlib import Path

    flagged: list[str] = []
    for f in instruction_files:
        for pat in _CREDENTIAL_FILENAME_PATTERNS:
            if fnmatch.fnmatch(Path(f).name, pat):
                flagged.append(f)
                break
    for d in data_directories:
        dpath = Path(d)
        if not dpath.is_absolute():
            dpath = (base_dir / d).resolve()
        if not dpath.exists():
            continue
        for entry in dpath.rglob("*"):
            for pat in _CREDENTIAL_FILENAME_PATTERNS:
                if fnmatch.fnmatch(entry.name, pat):
                    flagged.append(str(entry))
                    break

    if flagged:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            "Found credential-shaped files in the Docker COPY surface — "
            "they will be baked into the agent image:\n  - %s\n"
            "Remove them, add them to a `.dockerignore`-equivalent exclude "
            "rule, or move them out of `data_directories`/`instruction_files`.",
            "\n  - ".join(sorted(set(flagged))),
        )
```

And call it once from the existing `build` command, just before invoking `generate_dockerfile`:

```python
_warn_if_credential_files_in_copy_surface(
    instruction_files=instruction_files or [],
    data_directories=data_directories or [],
    base_dir=Path(agent_dir),
)
```

(Replace `agent_dir` with the actual variable used at the call site.)

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/cli/test_deploy_credential_lint.py -v
```
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/holodeck/cli/commands/deploy.py tests/unit/cli/test_deploy_credential_lint.py
git commit -m "feat(deploy): lint COPY surface for credential-shaped filenames (spec 034 P2a)"
```

---

## Task 14: Documentation

**Files:**
- Create: `docs/security/container-hardening.md`
- Create: `docs/security/aca-limitations.md`

- [ ] **Step 1: Write the operator-facing hardening doc**

Create `docs/security/container-hardening.md`:

```markdown
# Container Hardening Posture (Spec 034 P2a)

When you `holodeck deploy run`, HoloDeck applies the following hardening to
the generated Container App. All defaults are on; no agent.yaml changes
are required.

| Layer | Default | Enforcement layer | How to override |
|---|---|---|---|
| Runtime user | UID 1000 (`holodeck`), non-root | Dockerfile `USER` directive | Cannot be raised to root in generated images. |
| Corpus filesystems (`/app/data`, `/app/instructions`) | Owned by root, read-only (chmod a-w) | Dockerfile | Move writable content to `/var/holodeck/work` via the tool/SDK. |
| Ephemeral scratch (`/var/holodeck/work`, `/tmp`) | ACA EmptyDir volume per replica | ACA `Volume`/`VolumeMount` | Size scales with the replica's memory limit; cleared on replica restart. |
| Container ingress | Internal (`ingress_external: false`) | ACA Ingress | Set `deployment.target.azure.ingress_external: true` (warning emitted). |
| Node.js | Installed only when an MCP server's `command` starts with `node`/`npx`/`yarn`/`pnpm` | Generated Dockerfile | Add a stdio MCP server with one of those commands. |
| Credential-shaped files in COPY surface | Warned at `deploy build` (filenames only; no content scan) | CLI lint | Remove the file, or accept the warning if the file is legitimately public. |

### Posture ACA does NOT enforce (and we don't claim to)

The Anthropic secure-deployment guide recommends `--cap-drop ALL`,
`--security-opt no-new-privileges`, `--read-only` root FS, and seccomp
profiles. **ACA does not expose any of these primitives** at any API
version (verified 2026-05-23 against `azure-mgmt-appcontainers==4.0.0`
and ARM API 2026-01-01). The ACA platform default cap set / seccomp
profile is Microsoft-controlled and not customer-tunable.

The image-layer enforcement points (non-root user, read-only corpus,
ephemeral scratch) are the closest *equivalents* HoloDeck can produce
on ACA. For threat models that require the strict Kubernetes
securityContext primitives, see [`aca-limitations.md`](aca-limitations.md)
for the AKS escape hatch.

## What hardening does NOT cover

The following recommendations from
[Anthropic's secure-deployment guide](https://code.claude.com/docs/en/agent-sdk/secure-deployment)
are outside P2a:

- **`cap-drop ALL` / `no-new-privileges` / `readOnlyRootFilesystem` / seccomp.**
  Not expressible in the ACA management surface. See
  [`aca-limitations.md`](aca-limitations.md).
- **Network egress restrictions.** Default profile allows the container to
  reach any HTTPS endpoint. The `hardened` security profile (spec 034 P3,
  not yet shipped) is the way to restrict egress.
- **Credential boundary (proxy pattern).** Credentials remain in container
  env vars in the default profile. Move to `hardened` for the Envoy
  credential-injector pattern.
- **`--userns-remap`, `--ipc private`, `--pids-limit`.** Not exposed by ACA.
- **gVisor / Firecracker / VM isolation.** ACA cannot host these.

## Verifying hardening is in effect

```bash
# After `holodeck deploy run`, check the deploy-time echo for the
# ACA-enforced + image-enforced lines:
#   Ephemeral scratch (ACA EmptyDir): /tmp, /var/holodeck/work
#   Image-layer hardening: non-root (UID 1000), corpus read-only (/app/data, /app/instructions)
#   Note: ACA does not expose runAsNonRoot, capabilities.drop, ...

# Confirm runtime non-root + read-only corpus on a deployed agent:
az containerapp exec --resource-group <rg> --name <agent> \
    --command "sh -c 'id -u; test ! -w /app/data && echo READONLY_OK'"
# Expect: 1000 \n READONLY_OK
```
```

- [ ] **Step 2: Write the ACA limitations doc**

Create `docs/security/aca-limitations.md`:

```markdown
# Azure Container Apps Security Limitations

The Anthropic secure-deployment guide recommends a set of Linux container
security primitives. **Most of them are not exposed by Azure Container
Apps' management surface at any API version.** This is verified, not
suspected:

- `azure-mgmt-appcontainers==4.0.0` Python SDK — `Container.__init__`
  accepts only `image, name, command, args, env, resources,
  volume_mounts, probes`. No `security_context`, no
  `ContainerSecurityContext` model anywhere in the package.
- [ARM Bicep template ref — `Microsoft.App/containerApps@2026-01-01`](https://learn.microsoft.com/en-us/azure/templates/microsoft.app/containerapps?pivots=deployment-language-bicep)
  — `Container` resource has no `securityContext`, `runAsNonRoot`,
  `runAsUser`, `allowPrivilegeEscalation`, `readOnlyRootFilesystem`,
  `capabilities`, or `seccompProfile`. Consistent across every API
  version from 2022-03-01 through 2025-10-02-preview to 2026-01-01.
- [Microsoft ACA Security Baseline (updated 2026-04-01)](https://learn.microsoft.com/en-us/security/benchmark/azure/baselines/azure-container-apps-security-baseline)
  — documents network/identity/encryption controls only. Profile
  explicitly: `Customer can access HOST / OS: No Access`. Host security
  boundary is Microsoft-owned and not customer-tunable.

ACA is a serverless-ish platform; the design choice is that Microsoft
owns the host kernel, capability set, and seccomp profile platform-wide.
Customer security expression happens at the **image layer** + via the
**surfaces ACA does expose** (VNet/NSG, ingress, identity, ACR).

## Fundamental gaps vs. the Anthropic guide

| Primitive | Why it matters | What HoloDeck does instead |
|---|---|---|
| `runAsNonRoot=true` | Refuses to start if the image runs as root | **Image-layer enforcement** — Dockerfile `USER holodeck` (UID 1000). Doesn't refuse-to-start guarantee, but the image cannot run as root by construction. |
| `allowPrivilegeEscalation=false` | Blocks setuid escalation paths | **Platform default** — ACA does not grant privileged mode at all. No customer-facing field to assert it, but the property holds. |
| `capabilities.drop: [ALL]` | Removes Linux capabilities like NET_ADMIN, SYS_ADMIN | **Microsoft-controlled default cap set**, not customer-tunable. If your threat model requires CAP_NET_RAW removal etc., move to AKS. |
| `readOnlyRootFilesystem=true` | Prevents writes to image FS | **Image-layer approximation** — `chmod -R a-w` on `/app/data`, `/app/instructions`. The rest of `/` is writable; this is genuinely weaker than the Kubernetes primitive. |
| `seccomp` profiles (custom) | Restricts available syscalls beyond Docker default | ACA uses the platform-default seccomp profile. Customer cannot supply one. |
| `--userns-remap` | Maps container UID 0 to unprivileged host UID | Not exposed. The non-root `holodeck` user (UID 1000) mitigates most of the risk; with no exposed UID 0 in the image, userns remap is mostly belt-and-suspenders. |
| `--ipc private` | Isolates SysV IPC namespaces | Not exposed. ACA replicas don't share an IPC namespace by default. |
| `--pids-limit` | Bounds fork bomb damage | Not exposed. The replica memory limit indirectly caps fork bomb impact (each forked process consumes RSS). |
| `--network none` + Unix socket proxy | Eliminates direct network egress | Use `deployment.security_profile: hardened` (P3, in progress) for the Envoy-sidecar equivalent. |

## Escape hatches

For threat models that genuinely require the primitives above, deploy to:

- **AKS** with Pod Security Standards (`restricted` profile). Full
  Kubernetes securityContext is available. HoloDeck doesn't ship an
  AKS deployer in v1, but the image generated by `holodeck deploy build`
  is unmodified — you can apply your own AKS manifest with the
  securityContext block populated.
- **Self-hosted Kubernetes** — same story.
- **Modal / Fly Machines** — per-session ephemeral containers; see
  spec 034 §"Out of scope for v1".

## Why this isn't a HoloDeck bug

There is no `holodeck deploy` flag that can turn these on. The ACA ARM
API doesn't accept the fields. We file no feature request because none
of these primitives are on Microsoft's public roadmap for ACA — the
[microsoft/azure-container-apps GitHub](https://github.com/microsoft/azure-container-apps/issues)
has no active issue for `securityContext` support as of 2026-05-23.

If you need them, AKS (or a non-Azure runtime) is the right deployment
target — not a different HoloDeck configuration.
```

- [ ] **Step 3: Commit**

```bash
git add docs/security/container-hardening.md docs/security/aca-limitations.md
git commit -m "docs(security): P2 container hardening posture + ACA gaps (spec 034)"
```

---

## Task 15: Update spec 034 status tracker

**Files:**
- Modify: `specs/034-production-hardening/2026-05-18-production-hardening-for-claude-agents.md` (status table, around line 100)

- [ ] **Step 1: Flip P2a + P2b rows to "✅ shipped"**

Edit the status tracker table:

```markdown
| P2a — Container hardening | ✅ shipped | `feature/034-p2-container-runtime` | Dockerfile root-own corpus + chmod a-w; ACA emits EmptyDir volumes for /tmp + /var/holodeck/work (the only securityContext-adjacent primitive ACA exposes); Node.js gated on stdio-MCP detection; deploy-time credential-filename lint. **Verified 2026-05-23**: ACA does NOT expose runAsNonRoot, allowPrivilegeEscalation, capabilities.drop, readOnlyRootFilesystem, or seccomp at any API version — non-root + read-only-corpus posture is enforced at the image layer instead. Full gap analysis + AKS escape hatch in docs/security/aca-limitations.md. |
| P2b — Prompt-injection defenses | ✅ shipped | `feature/034-p2-container-runtime` | New `claude_hooks` module: PostToolUse credential redaction (5 patterns). Default-on; opt-out via `claude.disable_default_hooks: true` with loud warning. `RedactingSpanProcessor` scrubs `tool.input.*`, `tool.output.*`, `gen_ai.*` span attributes independently of the hooks flag. **Default-on subprocess env scrubbing** (Task 4.5): `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1` + `CLAUDE_CODE_MCP_ALLOWLIST_ENV=1` set on every agent's options.env so tool/MCP subprocesses don't inherit Anthropic/cloud creds; opt-out via `claude.disable_subprocess_env_scrub: true`. **Bash AST deny hook dropped 2026-05-23** as redundant — the SDK already parses Bash commands into an AST against operator-supplied permission rules, and P1b auto-disallows `Bash` for any agent that doesn't declare it. The SDK subprocess itself still inherits the full HoloDeck-serve env — that's P3's domain. |
```

- [ ] **Step 2: Commit**

```bash
git add specs/034-production-hardening/2026-05-18-production-hardening-for-claude-agents.md
git commit -m "docs(spec-034): mark P2a + P2b as shipped"
```

---

## Task 16: Final P2 verification — full CI + end-to-end smoke

- [ ] **Step 1: Local CI**

```bash
source .venv/bin/activate
make format
make lint
make type-check
make security
make test
```

Expected: all green.

- [ ] **Step 2: End-to-end deploy validation (OPTIONAL — only if the user explicitly requests it)**

This step rebuilds the local wheel, bakes it into the base image, pushes
the agent image, and rolls a live ACA revision. Per CLAUDE.md, **do not
run this automatically.** Run it only when the user asks "do the deploy
validation loop" or similar. The sequence is in CLAUDE.md §"End-to-End
Deploy Validation Loop".

If running it: after the rollout completes, exercise the AG-UI endpoint
and verify in OTel/Aspire that:

1. A synthetic tool-call payload containing `ghp_` + 36 chars is replaced
   with `[REDACTED:github-token]` in the `tool.output` span attribute.
2. A synthetic Bash tool call containing `curl … | sh` is denied with a
   `permissionDecision: deny` hook output visible in the agent transcript.
3. The container's `/app/data` is read-only:
   `az containerapp exec --command "sh -c 'test ! -w /app/data && echo OK'"`.

- [ ] **Step 3: Open the PR**

```bash
gh pr create --title "spec 034 P2 — container runtime + prompt-injection defenses" \
    --body "$(cat <<'EOF'
## Summary
- P2a: Dockerfile root-owns corpus + chmod a-w (read-only at runtime); ACA EmptyDir volumes for /tmp + /var/holodeck/work; Node.js gated on stdio-MCP detection; deploy-time credential-shaped-filename lint. ACA does not expose `securityContext` primitives at any API version (verified against `azure-mgmt-appcontainers==4.0.0` + ARM API 2026-01-01 + the [Microsoft ACA security baseline](https://learn.microsoft.com/en-us/security/benchmark/azure/baselines/azure-container-apps-security-baseline)) — non-root + readonly-corpus enforced at the image layer instead. Gap analysis in docs/security/aca-limitations.md.
- P2b: New `claude_hooks` module (credential redaction PostToolUse hook) merged ahead of user hooks; opt-out via `claude.disable_default_hooks: true`. `RedactingSpanProcessor` scrubs OTel trace attributes independently of the hooks flag. Default-on `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB=1` + `CLAUDE_CODE_MCP_ALLOWLIST_ENV=1` so tool/MCP subprocesses don't inherit credentials (opt-out via `claude.disable_subprocess_env_scrub: true`). Bash AST deny hook from earlier drafts dropped as redundant with SDK permission rules + P1b auto-disallow — see Task 2 note for rationale.
- Coverage gaps vs. https://code.claude.com/docs/en/agent-sdk/secure-deployment documented in docs/security/aca-limitations.md; everything credential/egress-related deferred to P3 by design.

## Test plan
- [ ] `make test-unit` green
- [ ] `make lint` / `make type-check` / `make security` green
- [ ] (optional) end-to-end deploy validation against financial-assistant sample with redaction + Bash-deny verification

EOF
)"
```

---

## Self-review

I re-read the spec and the Anthropic doc with fresh eyes against this plan. Three items called out:

1. **Spec coverage:** P2a + P2b sections of the spec are mapped to Tasks 1–14. The `claude.permissions` schema block from spec §"Phase 1b" is *not* in this plan — that's P1b, already shipped per the status tracker. Confirmed not in scope.

2. **Anthropic-doc coverage:** Documented gaps in Task 14's `aca-limitations.md` — these are environment-bound (ACA can't host them) or P3-bound (network/credentials). The "Cloud deployments" section of the Anthropic doc (private subnet + Envoy `credential_injector`) maps cleanly to spec 034 P3, not P2. No silent coverage holes.

**ACA surface verification (added during plan revision, 2026-05-23):** An earlier draft of this plan called for emitting a `ContainerSecurityContext` on the `Container` resource. Verification against the installed SDK (`azure-mgmt-appcontainers==4.0.0`), the latest ARM API (2026-01-01), and the [Microsoft ACA security baseline](https://learn.microsoft.com/en-us/security/benchmark/azure/baselines/azure-container-apps-security-baseline) found that **none** of `runAsNonRoot`, `allowPrivilegeEscalation`, `capabilities.drop`, `readOnlyRootFilesystem`, or `seccompProfile` are exposed by ACA at any API version. The plan was revised in place: Task 11 now ships only `EmptyDir` `Volume`/`VolumeMount` (the one securityContext-adjacent primitive ACA does expose); the non-root + readonly-corpus posture is enforced at the Dockerfile layer instead; Task 14's `aca-limitations.md` was promoted from "gap notes" to a full gap analysis with the AKS escape hatch. Honest posture on ACA is materially weaker than the Anthropic guide describes, and the plan now says so.

3. **Type/symbol consistency:** `_post_tool_credential_redaction` and `redact_credentials` referenced in Tasks 3, 5, 7; `RedactingSpanProcessor` consistent across Tasks 5 and 6; `disable_subprocess_env_scrub` + `CLAUDE_CODE_SUBPROCESS_ENV_SCRUB` + `CLAUDE_CODE_MCP_ALLOWLIST_ENV` consistent across Task 4.5 schema, build_options, and tests; `_agent_needs_nodejs` consistent across Task 9 and (called from) `deploy.py`; `_warn_if_credential_files_in_copy_surface` consistent across Task 13 and the doc. Bash deny symbols (`_pre_tool_bash_deny`, `_bash_ast_violation`, `BASH_DENY_PATTERNS`) were removed when Task 2 was dropped 2026-05-23 — no lingering references in implementation tasks. No naming drift detected.

**Subprocess env-inheritance research (added during plan revision, 2026-05-23):** Initial reading of the SDK source confirmed that `claude_agent_sdk._internal.transport.subprocess_cli.connect()` spreads the full `os.environ` (minus `CLAUDECODE`) into the subprocess env, and that `ClaudeAgentOptions.env` is additive rather than a whitelist. Follow-up research into Anthropic's documentation surfaced two CLI-level scrub flags (`CLAUDE_CODE_SUBPROCESS_ENV_SCRUB` and `CLAUDE_CODE_MCP_ALLOWLIST_ENV`) that operate on subprocesses spawned *by* the SDK (Bash tools, stdio MCP servers) — not on the SDK subprocess itself, but valuable enough to default-on. Task 4.5 was added to wire them up. The structural problem (HoloDeck-serve env → SDK subprocess) remains a P3 concern; the plan now says so explicitly in the coverage matrix.

One assumption flagged for the implementing engineer:

- **OTel tracer-provider construction site.** Task 6 leaves the exact file path for `add_span_processor(RedactingSpanProcessor())` to be discovered via `grep` (Step 1). The processor implementation in Task 5 is provider-agnostic; the wiring in Task 6 is a 1-line patch wherever the provider is built. If the construction site turns out to be in user code outside HoloDeck (e.g. only inside the `holodeck-otel` extra), the right move is to register the processor from `_activate_instrumentation` in `claude_backend.py` immediately after the instrumentor is attached.
