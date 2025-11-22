# CLI Interface Contract

**Feature**: `007-interactive-chat`
**Contract Type**: Command-Line Interface
**Date**: 2025-11-22

## Command Specification

### `holodeck chat`

Start an interactive chat session with an agent.

**Synopsis**:
```bash
holodeck chat <AGENT_CONFIG_PATH> [OPTIONS]
```

**Arguments**:

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `AGENT_CONFIG_PATH` | Path | Yes | Path to agent YAML configuration file |

**Options**:

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--verbose` | `-v` | Flag | False | Show detailed tool execution (parameters, internal state) |
| `--observability` | `-o` | Flag | False | Enable OpenTelemetry tracing and metrics |
| `--max-messages` | `-m` | Integer | 50 | Maximum conversation messages before warning |

**Exit Codes**:

| Code | Meaning |
|------|---------|
| 0 | Success (normal exit via 'exit'/'quit') |
| 1 | Configuration error (invalid YAML, missing file) |
| 2 | Agent initialization error (invalid tools, LLM connection failure) |
| 130 | User interrupt (Ctrl+C) |

---

## Interactive Commands

Once in a chat session, users can send messages or use special commands:

### Message Input

**Format**: Plain text input followed by Enter

**Example**:
```
You: What's the weather in San Francisco?
```

**Validation**:
- Message cannot be empty (whitespace-only rejected)
- Maximum length: 10,000 characters
- Must be valid UTF-8
- Control characters stripped

### Exit Commands

**Commands**: `exit`, `quit`

**Behavior**: Terminates session gracefully with farewell message

**Example**:
```
You: exit
Goodbye!
```

### Keyboard Interrupt

**Trigger**: Ctrl+C or Ctrl+D

**Behavior**: Immediate session termination with cleanup

**Example**:
```
You: ^C
Goodbye!
```

---

## Output Format

### Standard Mode (default)

**User Message Display**:
```
You: <user input>
```

**Agent Response Display**:
```
Agent: <agent response>
```

**Tool Execution Display** (inline):
```
[Tool] <tool_name>(<param_summary>) → <status> (<execution_time>s)
```

**Example Session**:
```
$ holodeck chat examples/weather-agent.yaml

Starting chat with Weather Assistant...
Type 'exit' or 'quit' to end session.

You: What's the weather in San Francisco?

[Tool] get_weather(location="San Francisco, CA") → success (0.3s)

Agent: The current weather in San Francisco is 72°F and sunny.

You: exit
Goodbye!
```

### Verbose Mode (`--verbose`)

**Tool Execution Display** (detailed):
```
[Tool Call] <tool_name>
  Parameters:
    <param_name>: <param_value>
    ...
  Status: <status>
  Execution Time: <time>s
  Result:
    <result_preview>
```

**Example**:
```
You: What's the weather in SF?

[Tool Call] get_weather
  Parameters:
    location: San Francisco, CA
    units: fahrenheit
  Status: success
  Execution Time: 0.3s
  Result:
    {"temperature": 72, "conditions": "Sunny", "humidity": 65}

Agent: The current weather in San Francisco is 72°F and sunny with 65% humidity.
```

---

## Error Handling

### Configuration Errors

**Scenario**: Invalid agent YAML or missing file

**Output**:
```
Error: Failed to load agent configuration
  File: /path/to/agent.yaml
  Reason: File not found

Run 'holodeck init' to create a new agent configuration.
```

**Exit Code**: 1

### Agent Initialization Errors

**Scenario**: Invalid tools or LLM connection failure

**Output**:
```
Error: Failed to initialize agent
  Agent: Customer Support Bot
  Reason: Could not connect to Anthropic API (invalid API key)

Check your .env file and ensure ANTHROPIC_API_KEY is set.
```

**Exit Code**: 2

### Input Validation Errors

**Scenario**: User sends invalid input (empty, too long, etc.)

**Output**:
```
You:

Error: Message cannot be empty. Please enter a message.
```

**Exit Code**: N/A (session continues)

### Tool Execution Errors

**Scenario**: Agent tool fails or times out

**Output**:
```
[Tool] search_database(query="...") → failed (5.0s)
Error: Database connection timeout

Agent: I encountered an error searching the database. Please try again.
```

**Exit Code**: N/A (session continues)

---

## Performance Requirements

| Metric | Target | Measured At |
|--------|--------|-------------|
| Session startup | < 1 second | From command invocation to first prompt |
| Response processing | < 5 seconds | User message to agent response (excluding LLM latency) |
| Tool execution display | < 100ms | Tool start to display update |
| Memory usage | < 200MB | For 50-message conversation |
| Session stability | 30+ minutes | No memory leaks or crashes |

---

## Observability Output (when `--observability` enabled)

OpenTelemetry traces exported to configured backend (console, Jaeger, OTLP).

**Trace Structure**:
```
chat_session (session_id, agent_name, duration)
├── message_turn_1
│   ├── llm_call (model, prompt_tokens, completion_tokens, cost)
│   └── tool_execution_1 (tool_name, duration, status)
├── message_turn_2
│   ├── llm_call
│   ├── tool_execution_1
│   └── tool_execution_2
└── ...
```

**Metrics Exported**:
- `chat.session.duration` (seconds)
- `chat.message.count` (total messages)
- `chat.tool.execution.count` (total tool calls)
- `gen_ai.client.token.usage` (prompt, completion, total tokens)
- `gen_ai.client.operation.duration` (LLM call latency)

---

## Backward Compatibility

N/A - This is a new command with no previous versions.

---

## Future Enhancements (Out of Scope for MVP)

- `--resume <session-id>`: Resume saved session (P3)
- `/save`: In-session command to save conversation (P3)
- `/clear`: Clear history but keep session active
- `/history`: Display conversation summary
- Rich terminal UI with syntax highlighting (prompt_toolkit integration)
- Multi-line input support (Ctrl+D to submit)
