/**
 * Turn an AG-UI message list into the rows the conversation draws: user
 * messages, numbered tool calls (cues), groups of cues, and answers bracketed
 * to the cues that came before them in the same turn.
 *
 * Pure module (type imports only), so Node runs the tests with type stripping.
 */
import type { AssistantMessage, Message, ToolCall, UserMessage } from "@ag-ui/client";

export type ToolCallStatus = "inProgress" | "executing" | "complete";

/** One tool call, numbered in thread order: Q1, Q2, … */
export interface Cue {
  cue: number;
  toolCallId: string;
  name: string;
  /** Parsed arguments, once the streamed JSON is complete. */
  args: Record<string, unknown> | undefined;
  status: ToolCallStatus;
  /** The tool message content, when the result has arrived. */
  result: unknown;
  /** The tool message `error` field. */
  error: string | null;
  /** No result, and the run is over: the call was stopped or lost. */
  stopped: boolean;
}

export type Row =
  | { kind: "user"; key: string; message: UserMessage }
  | { kind: "tool"; key: string; call: Cue }
  | { kind: "group"; key: string; calls: Cue[] }
  | { kind: "answer"; key: string; messageId: string; content: string; cues: number[] };

export interface Conversation {
  rows: Row[];
  /** Number of tool calls in the thread. */
  cueCount: number;
}

/** A run of more than this many tool calls is wrapped in one group row. */
export const GROUP_THRESHOLD = 3;

function parseArgs(text: string): Record<string, unknown> | undefined {
  if (!text.trim()) return undefined;
  try {
    const value: unknown = JSON.parse(text);
    if (typeof value === "object" && value !== null && !Array.isArray(value)) {
      return value as Record<string, unknown>;
    }
    return undefined;
  } catch {
    return undefined;
  }
}

/** Result payload of a tool message: CopilotKit-style JSON decode, else the raw content. */
function toolResult(content: Message["content"]): unknown {
  if (typeof content !== "string") return content;
  try {
    return JSON.parse(content) as unknown;
  } catch {
    return content;
  }
}

function toCue(
  call: ToolCall,
  cue: number,
  tool: Extract<Message, { role: "tool" }> | undefined,
  isRunning: boolean,
): Cue {
  const args = parseArgs(call.function.arguments);
  const hasResult = tool !== undefined;
  let status: ToolCallStatus = "inProgress";
  if (hasResult) status = "complete";
  else if (args !== undefined) status = "executing";
  return {
    cue,
    toolCallId: call.id,
    name: call.function.name,
    args: args ?? (hasResult || !isRunning ? {} : undefined),
    status: !hasResult && !isRunning ? "complete" : status,
    result: tool ? toolResult(tool.content) : undefined,
    error: tool?.error ?? null,
    stopped: !hasResult && !isRunning,
  };
}

export function buildConversation(
  messages: readonly Message[],
  isRunning: boolean,
): Conversation {
  const results = new Map<string, Extract<Message, { role: "tool" }>>();
  for (const message of messages) {
    if (message.role === "tool") results.set(message.toolCallId, message);
  }

  const rows: Row[] = [];
  let cueCount = 0;
  let run: Cue[] = [];
  let unanswered: number[] = [];

  const flush = () => {
    if (run.length > GROUP_THRESHOLD) {
      rows.push({ kind: "group", key: `group:${run[0].toolCallId}`, calls: run });
    } else {
      for (const call of run) rows.push({ kind: "tool", key: `tool:${call.toolCallId}`, call });
    }
    run = [];
  };

  for (const message of messages) {
    if (message.role === "user") {
      flush();
      unanswered = [];
      rows.push({ kind: "user", key: `user:${message.id}`, message });
      continue;
    }
    if (message.role !== "assistant") continue;
    const assistant: AssistantMessage = message;
    const content = assistant.content ?? "";
    if (content.trim()) {
      flush();
      rows.push({
        kind: "answer",
        key: `answer:${assistant.id}`,
        messageId: assistant.id,
        content,
        cues: unanswered,
      });
      unanswered = [];
    }
    for (const call of assistant.toolCalls ?? []) {
      cueCount += 1;
      run.push(toCue(call, cueCount, results.get(call.id), isRunning));
      unanswered = [...unanswered, cueCount];
    }
  }
  flush();
  return { rows, cueCount };
}

/** "Q3" or "Q1–Q6". */
export function cueRange(cues: readonly number[]): string {
  if (cues.length === 0) return "";
  const first = cues[0];
  const last = cues[cues.length - 1];
  return first === last ? `Q${first}` : `Q${first}–Q${last}`;
}

/** Plain text of a user message (its text parts). */
export function userText(message: UserMessage): string {
  if (typeof message.content === "string") return message.content;
  return message.content
    .flatMap((part) => (part.type === "text" ? [part.text] : []))
    .join("\n");
}

/**
 * Tool names of a run of calls, counted in first-seen order:
 * "memory_timeline, memory_search ×2".
 */
export function toolNameSummary(names: readonly string[]): string {
  const counts = new Map<string, number>();
  for (const name of names) counts.set(name, (counts.get(name) ?? 0) + 1);
  return [...counts]
    .map(([name, count]) => (count > 1 ? `${name} ×${count}` : name))
    .join(", ");
}
