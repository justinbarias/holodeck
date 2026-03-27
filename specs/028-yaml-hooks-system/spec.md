# Feature Specification: YAML-Definable Hooks System

**Feature Branch**: `feature/007-claude-agent-features`
**Spec ID**: 028-yaml-hooks-system
**Created**: 2026-03-28
**Status**: Draft
**Input**: Expose the Claude Agent SDK's full hook system to YAML users, allowing declarative hook definitions for all SDK event types with no-code actions (log, reject, modify, notify, script).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Block Dangerous Tool Calls (Priority: P1)

A security-conscious team wants to prevent their Claude agent from executing destructive shell commands. They define a `PreToolUse` hook in YAML that rejects Bash tool calls matching dangerous patterns (e.g., `rm -rf`, `DROP TABLE`).

**Why this priority**: Safety-critical. Hooks that block dangerous operations are the highest-value use case and address a real production concern where static permission modes are too coarse.

**Independent Test**: Define a `PreToolUse` hook in agent.yaml that rejects Bash calls containing `rm -rf`, run a test case that would trigger such a call, and verify the call is blocked with the configured rejection message.

**Acceptance Scenarios**:

1. **Given** a `PreToolUse` hook matching tool `Bash` with a reject action when input contains `rm -rf`, **When** the agent attempts to run `rm -rf /tmp/data`, **Then** the tool call is blocked and the agent receives the rejection message.
2. **Given** the same hook, **When** the agent runs `ls -la`, **Then** the tool call proceeds normally (no match).
3. **Given** multiple `PreToolUse` hooks with different matchers, **When** a tool call matches the first hook, **Then** only the first matching hook's action is applied.

---

### User Story 2 - Log All Tool Usage to a Webhook (Priority: P1)

An observability team wants to track all tool invocations for audit and debugging. They define a `PostToolUse` hook that sends tool call details to a webhook endpoint.

**Why this priority**: Observability is a core production need. While OTel covers some tracing, webhook-based logging allows integration with custom dashboards and alerting systems.

**Independent Test**: Define a `PostToolUse` hook with a `notify` action pointing to a webhook URL, run a test case, and verify the webhook receives the tool call payload.

**Acceptance Scenarios**:

1. **Given** a `PostToolUse` hook with a `notify` action targeting `https://hooks.example.com/log`, **When** any tool completes execution, **Then** the hook sends a payload with tool name, input, output, and duration to the webhook.
2. **Given** the webhook is unreachable, **When** a tool completes, **Then** the hook failure is logged as a warning but does not block agent execution.
3. **Given** a `PostToolUse` hook with a tool name matcher, **When** a non-matching tool completes, **Then** the webhook is not called.

---

### User Story 3 - Run a Custom Script on Session Events (Priority: P2)

A team wants to trigger custom automation when an agent session starts or stops -- for example, initializing a temporary workspace on start and cleaning it up on stop. They define `SessionStart` and `Stop` hooks that execute shell scripts.

**Why this priority**: Session lifecycle hooks enable advanced automation patterns but are less common than tool-level hooks.

**Independent Test**: Define a `Stop` hook with a `script` action, run a test session, and verify the script is executed when the session ends.

**Acceptance Scenarios**:

1. **Given** a `Stop` hook with a `script` action pointing to `./scripts/cleanup.sh`, **When** the agent session ends, **Then** the script is executed.
2. **Given** a script hook where the script returns a non-zero exit code, **When** the hook fires, **Then** the failure is logged as a warning but does not affect the session result.

---

### User Story 4 - Modify Tool Input Before Execution (Priority: P3)

A user wants to inject default parameters into tool calls -- for example, always adding a `--safe-mode` flag to Bash commands. They define a `PreToolUse` hook with a `modify` action that transforms the tool input.

**Why this priority**: Input modification is a powerful but advanced use case. Most users will start with reject/log actions before needing input transformation.

**Independent Test**: Define a `PreToolUse` hook that appends `--dry-run` to all Bash tool inputs, run a test case, and verify the modified input is passed to the tool.

**Acceptance Scenarios**:

1. **Given** a `PreToolUse` hook with a `modify` action that prepends `set -e;` to Bash command inputs, **When** the agent runs a Bash command, **Then** the SDK receives the modified command with the prefix.

---

### Edge Cases

- What happens when multiple hooks match the same event? Hooks are evaluated in declaration order; the first matching hook with a terminal action (reject) stops evaluation. Non-terminal actions (log, notify) allow subsequent hooks to fire.
- What happens when a hook's matcher regex is invalid? A validation error is raised at config load time.
- What happens when internal tool-tracking hooks conflict with user-defined hooks? User-defined hooks are merged with internal hooks; internal hooks always fire (they are non-terminal logging hooks).
- What happens when a `notify` webhook times out? The timeout is configurable (default: 5 seconds). Timeout failures are logged as warnings and do not block execution.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST accept a `hooks` section under `claude` in agent YAML containing a list of hook definitions.
- **FR-002**: Each hook definition MUST specify an `event` type matching one of the SDK's supported event types (PreToolUse, PostToolUse, PostToolUseFailure, Stop, Notification, and others as supported).
- **FR-003**: Each hook definition MUST specify an `action` type: one of `log`, `reject`, `modify`, `notify`, or `script`.
- **FR-004**: Hook definitions MAY include a `matcher` with a tool name pattern (glob or regex) to filter which tool calls trigger the hook.
- **FR-005**: The `reject` action MUST include a `message` field that is returned to the agent as the rejection reason.
- **FR-006**: The `notify` action MUST include a `url` field for the webhook endpoint and MAY include `headers` and a `timeout` (default: 5 seconds).
- **FR-007**: The `script` action MUST include a `command` field with the shell command or script path to execute.
- **FR-008**: The `modify` action MUST include a transformation specification for altering tool input.
- **FR-009**: System MUST validate hook definitions at config load time (valid event types, valid matchers, required action fields).
- **FR-010**: System MUST merge user-defined hooks with internal tool-tracking hooks, preserving internal hooks.
- **FR-011**: Hook failures (webhook timeout, script error) MUST be logged as warnings and MUST NOT block agent execution.
- **FR-012**: The `log` action MUST write hook event details to the application log at a configurable level (default: info).

### Key Entities

- **HookDefinition**: A declarative hook configuration with event type, optional matcher, and action.
- **HookAction**: One of log, reject, modify, notify, or script -- each with type-specific fields.
- **HookMatcher**: Optional filter for tool name patterns (glob or regex).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can define hooks in agent YAML that block specific tool calls, verified by the agent receiving a rejection message instead of executing the tool.
- **SC-002**: Users can define hooks that log or notify on tool events, verified by webhook receipt or log output.
- **SC-003**: Invalid hook configurations produce clear validation errors at config load time, before agent execution.
- **SC-004**: Hook failures (webhook timeout, script error) do not interrupt agent execution, verified by successful agent completion despite hook failures.
- **SC-005**: Internal tool-tracking hooks continue to function alongside user-defined hooks.

## Assumptions

- The Claude Agent SDK's hook system supports registering multiple callbacks for the same event type.
- Hook actions are evaluated synchronously within the hook callback (the SDK waits for the hook to return before proceeding).
- The `modify` action's transformation specification will follow a simple field-level patch format (e.g., prepend/append to specific input fields) rather than arbitrary code execution, to maintain the no-code principle.
- Not all 13+ SDK event types need to be supported in the first release. The spec covers the most valuable subset: PreToolUse, PostToolUse, PostToolUseFailure, Stop, and Notification. Additional event types can be added incrementally.
