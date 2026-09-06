# Quickstart: Multi-Agent Teams in YAML

**Spec**: 029-subagent-orchestration
**Audience**: HoloDeck users defining a multi-agent team in `agent.yaml`.

This quickstart shows how to define a parent agent with three specialized
subagents — a researcher, a data analyst, and a report writer — entirely
in YAML. The parent automatically delegates to subagents based on their
`description` fields; you don't write any routing code.

---

## 1. Minimal example

```yaml
name: research-team
instructions: |
  You are a research coordinator. Delegate research, analysis, and writing
  to your subagents. Synthesize their findings into a final answer.

model:
  provider: anthropic
  name: claude-opus-4-7

claude:
  permission_mode: acceptEdits
  web_search: true

  agents:
    researcher:
      description: Gathers raw information from the web and primary sources.
      prompt: |
        You are a meticulous researcher. Search the web for primary sources.
        Cite every claim with a URL. Do not analyze — just gather facts.
      tools: [WebSearch, WebFetch]
      model: haiku

    analyst:
      description: Analyzes data and identifies patterns and outliers.
      prompt_file: ./prompts/analyst.md
      tools: [Read]
      model: sonnet

    writer:
      description: Produces a polished final report from analyzed findings.
      prompt: |
        You are an expert technical writer. Produce a clear, well-structured
        report. Use headers, bullets, and inline citations.
      # No tools list → inherits all parent tools.
      # No model → inherits parent model.
```

---

## 2. Field reference

Each entry under `claude.agents.<name>` accepts these fields:

| Field | Required | Description |
|---|---|---|
| `description` | yes | Used by the parent agent for routing. Be specific about what the subagent does and when to invoke it. |
| `prompt` | one of | Inline system prompt. |
| `prompt_file` | one of | Path to a `.md` or `.txt` file containing the prompt, resolved relative to the agent.yaml directory. |
| `tools` | no | List of tool names this subagent may use. Built-ins (`WebSearch`, `Read`, `Write`, `Bash`, …), MCP tools (`mcp__<server>__<tool>`), or HoloDeck-bridged names are valid. Omit to inherit all parent tools. |
| `model` | no | One of `sonnet`, `opus`, `haiku`, `inherit`. Omit to inherit the parent's model. |

> **Note on `tools`**: subagents share the parent's MCP server
> registrations. To restrict an MCP tool to one subagent, name it on that
> subagent's `tools` list and leave it off the others. The Claude SDK
> enforces the allowlist at runtime.

---

## 3. Restricting MCP tool access per subagent

```yaml
claude:
  agents:
    db_analyst:
      description: Runs SQL queries against the production read-replica.
      prompt: You are a SQL expert. Use mcp__db__query to answer questions.
      tools: [mcp__db__query, mcp__db__describe]

    researcher:
      description: Searches the web. Cannot touch the database.
      prompt: You are a web researcher.
      tools: [WebSearch, WebFetch]
      # ↑ no mcp__db__* — DB tools are unreachable from this subagent.
```

Both subagents share the same parent-level `mcp_servers.db` registration,
but the SDK only exposes the tools each one explicitly enumerates.

---

## 4. Capping HoloDeck-side concurrency

Use the existing top-level `execution.parallel_test_cases` field — it
caps the test runner's per-test-case semaphore. There is no
subagent-specific knob, because HoloDeck cannot intercede in the
SDK's internal subagent dispatch (the SDK manages that itself).

```yaml
execution:
  parallel_test_cases: 2   # at most 2 concurrent agent sessions

claude:
  agents:
    # ...
```

> **Note (migration)**: earlier prereleases offered a
> `claude.subagents` block with `enabled` and `max_parallel`. That
> block is removed in spec 029 — both fields were redundant. If your
> YAML still has `claude.subagents`, delete it. Loading produces a
> clear validation error pointing here.

---

## 5. Validation walkthrough

The following YAML triggers each validation error so you know what to
expect:

| Misconfiguration | Result at config-load time |
|---|---|
| Missing `description` | `ValidationError: subagent requires description` |
| Both `prompt` and `prompt_file` set | `ValidationError: prompt and prompt_file are mutually exclusive` |
| Neither `prompt` nor `prompt_file` set | `ValidationError: subagent requires either prompt or prompt_file` |
| `prompt_file: ./does-not-exist.md` | `ValidationError: prompt_file not found` |
| `model: claude-3-5-sonnet-20241022` | `ValidationError: model must be one of sonnet, opus, haiku, inherit` |
| `tools: [WebSerach]` (typo) | **Warning** at load time (not an error) — typos in tool names produce a warning so unknown built-ins/MCP tools aren't blocked. |
| `claude.subagents:` block present | `ValidationError: claude.subagents is no longer supported; remove this block. Subagent forwarding is gated solely by the presence of claude.agents. To cap HoloDeck-side test concurrency, set execution.parallel_test_cases instead.` |

---

## 6. Verifying the wiring

A quick sanity check after editing your YAML — both `holodeck chat` and
`holodeck test` load and validate the YAML before they do anything else,
so either one will surface validation errors and warnings:

```bash
holodeck chat path/to/agent.yaml                  # exercise the team interactively
holodeck test path/to/agent.yaml                  # run test cases (honors execution.parallel_test_cases)
```

If your test suite includes a "smoke" case at the top, `holodeck test`
gives you fast YAML validation followed by a single end-to-end exercise.

---

## 7. Common patterns

**Specialized small models for narrow tasks**:

```yaml
claude:
  agents:
    classifier:
      description: Routes questions to one of: technical, billing, general.
      prompt: Classify the user's question. Respond with one word only.
      model: haiku
      tools: []      # no tools needed — pure reasoning
```

**File-based prompts for long instructions**:

```yaml
claude:
  agents:
    legal_reviewer:
      description: Reviews contracts for risk per the firm's playbook.
      prompt_file: ./prompts/legal-reviewer.md     # versioned alongside agent.yaml
      tools: [Read, mcp__contract_db__search]
```

**Inherit-everything subagent for chained reasoning**:

```yaml
claude:
  agents:
    reflector:
      description: Reviews the parent's draft answer for gaps and suggests revisions.
      prompt: You are a critical reviewer. Find gaps in reasoning.
      # tools omitted → inherits parent tools
      # model omitted → inherits parent model
```
