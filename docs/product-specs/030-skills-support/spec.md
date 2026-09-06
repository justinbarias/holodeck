# Feature Specification: Skills Support

**Feature Branch**: `feature/007-claude-agent-features`
**Spec ID**: 030-skills-support
**Created**: 2026-03-28
**Status**: Draft
**Input**: Expose the Claude Agent SDK's skills system and setting sources in HoloDeck YAML, enabling users to load custom skills from `.claude/skills/` and control which setting sources the agent uses.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Load Project-Level Skills (Priority: P1)

A user has created custom skills in their project's `.claude/skills/` directory (e.g., a "summarize" skill that provides specialized summarization instructions and tool access). They configure their agent to load these skills by setting `setting_sources` in their YAML, and the agent can invoke the skills during execution.

**Why this priority**: Skills are the primary extensibility mechanism in the Claude SDK. They allow teams to create reusable, shareable agent capabilities without modifying the agent's core configuration. This is the highest-value use case.

**Independent Test**: Create a `.claude/skills/summarize/SKILL.md` file, set `claude.setting_sources: [project]` in agent.yaml, initialize the agent, and verify the SDK loads and can invoke the skill.

**Acceptance Scenarios**:

1. **Given** an agent.yaml with `claude.setting_sources: [project]` and a `.claude/skills/summarize/SKILL.md` file in the project directory, **When** the agent is initialized, **Then** the SDK loads skills from the project's `.claude/skills/` directory.
2. **Given** `claude.setting_sources: [project, user]`, **When** the agent is initialized, **Then** both project-level and user-level skills are available.
3. **Given** `claude.setting_sources: [project]` but no `.claude/skills/` directory exists, **When** the agent is initialized, **Then** initialization succeeds with no skills loaded (no error).

---

### User Story 2 - Control Setting Sources for Agent Configuration (Priority: P1)

A user wants to control which configuration sources the Claude SDK loads from. They may want to load only project-level settings (ignoring user-level preferences) or combine multiple sources. They set `setting_sources` in their YAML to control this.

**Why this priority**: Setting sources determine what configuration the SDK subprocess inherits. Without this, the SDK may load user-level settings that conflict with the agent's intended behavior.

**Independent Test**: Set `claude.setting_sources: [project]` in agent.yaml, ensure a `.claude/settings.json` exists at the project level, initialize the agent, and verify the SDK receives the setting sources parameter.

**Acceptance Scenarios**:

1. **Given** `claude.setting_sources: [project]`, **When** the agent is initialized, **Then** the SDK options include `setting_sources=["project"]`.
2. **Given** `claude.setting_sources: [local]`, **When** the agent is initialized, **Then** the SDK loads from local workspace settings only.
3. **Given** `claude.setting_sources: [invalid]`, **When** the config is loaded, **Then** a validation error is raised listing valid values (user, project, local).
4. **Given** no `setting_sources` field, **When** the agent is initialized, **Then** no setting sources parameter is passed to the SDK (SDK default behavior).

---

### User Story 3 - Expose Specific Skills to Subagents (Priority: P3)

A user with subagent definitions wants specific subagents to have access to specific skills. For example, a "Report Writer" subagent should have access to a "formatting" skill but not a "data-analysis" skill.

**Why this priority**: Per-subagent skill access builds on both skills support and subagent definitions. It's an advanced composition pattern that's lower priority than basic skill loading.

**Independent Test**: Define a subagent with `skills: [formatting]` in agent.yaml, initialize the agent, and verify the subagent SDK definition includes the skills list.

**Acceptance Scenarios**:

1. **Given** a subagent definition with `skills: [formatting, summarize]`, **When** the agent is initialized, **Then** the subagent's SDK `AgentDefinition` includes those skills.
2. **Given** a subagent definition with no `skills` field, **When** the agent is initialized, **Then** the subagent inherits all available skills.

---

### Edge Cases

- What happens when `setting_sources` includes "user" but the user has no `~/.claude/` directory? The SDK handles this gracefully; no error is produced.
- What happens when a skill file (`SKILL.md`) has invalid content? The SDK validates skill files; errors should be propagated to the user.
- What happens when the same skill name exists in both project and user directories? The SDK's precedence rules apply (typically project overrides user).
- What happens when `setting_sources` is an empty list? Passed through as-is; the SDK loads no settings from any source.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST accept a `setting_sources` field on `ClaudeConfig` as a list of valid source identifiers (user, project, local).
- **FR-002**: System MUST pass the `setting_sources` value through to the Claude SDK options when present.
- **FR-003**: System MUST validate that `setting_sources` contains only valid values, raising a clear error for invalid entries.
- **FR-004**: System MUST NOT pass `setting_sources` when the field is not set (preserve SDK defaults).
- **FR-005**: When `setting_sources` includes "project", the SDK MUST be configured to look for skills in the project's `.claude/skills/` directory relative to the agent's `working_directory` (or the agent YAML's directory if no working directory is set).
- **FR-006**: Subagent definitions MAY include a `skills` field (list of skill names) that is passed to the SDK `AgentDefinition`.
- **FR-007**: System MUST work correctly when `setting_sources` is set but no `.claude/` directory or skills exist (graceful no-op).

### Key Entities

- **ClaudeConfig.setting_sources**: A list of setting source identifiers controlling which configuration directories the SDK loads from.
- **SettingSource**: Enumeration of valid values (user, project, local).
- **SubagentDefinition.skills**: Optional list of skill names available to a subagent.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can configure `setting_sources` in agent YAML and the SDK loads configuration and skills from the specified sources.
- **SC-002**: Skills defined in `.claude/skills/` are available to the agent when the appropriate setting source is configured.
- **SC-003**: Invalid setting source values produce clear validation errors at config load time.
- **SC-004**: Omitting `setting_sources` preserves SDK default behavior with no errors.
- **SC-005**: Subagents can be configured with specific skill access when subagent definitions are used.

## Assumptions

- The Claude Agent SDK supports a `setting_sources` parameter on `ClaudeAgentOptions` (or equivalent) that accepts a list of source identifiers.
- Skills are defined as markdown files (`SKILL.md`) in `.claude/skills/<skill-name>/` directories, following the SDK's skill format.
- The SDK handles skill discovery, validation, and loading internally -- HoloDeck only needs to pass the `setting_sources` parameter.
- Per-subagent skill access is supported via the `skills` field on `AgentDefinition`.
