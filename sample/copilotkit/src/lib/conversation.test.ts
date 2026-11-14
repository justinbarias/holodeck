import { test } from "node:test";
import assert from "node:assert/strict";
import type { Message } from "@ag-ui/client";
import {
  buildConversation,
  cueRange,
  cueState,
  groupSummary,
  toolNameSummary,
  type Cue,
} from "./conversation.ts";

function call(id: string, name: string, args = "{}") {
  return { id, type: "function" as const, function: { name, arguments: args } };
}

const turn: Message[] = [
  { id: "u1", role: "user", content: "Question one" },
  { id: "a1", role: "assistant", toolCalls: [call("t1", "memory_search"), call("t2", "lookup")] },
  { id: "r1", role: "tool", toolCallId: "t1", content: '{"hits":[]}' },
  { id: "r2", role: "tool", toolCallId: "t2", content: "plain", error: "Timed out." },
  { id: "a2", role: "assistant", content: "The answer." },
];

test("tool calls are numbered and the answer is bracketed to them", () => {
  const { rows, cueCount } = buildConversation(turn, false);
  assert.equal(cueCount, 2);
  assert.deepEqual(rows.map((row) => row.kind), ["user", "tool", "tool", "answer"]);
  const answer = rows[3];
  assert.equal(answer.kind, "answer");
  if (answer.kind === "answer") assert.deepEqual(answer.cues, [1, 2]);
  const second = rows[2];
  if (second.kind === "tool") {
    assert.equal(second.call.cue, 2);
    assert.equal(second.call.error, "Timed out.");
    assert.equal(second.call.status, "complete");
  }
  const first = rows[1];
  if (first.kind === "tool") assert.deepEqual(first.call.result, { hits: [] });
});

test("numbering continues across turns", () => {
  const more: Message[] = [
    ...turn,
    { id: "u2", role: "user", content: "Again" },
    { id: "a3", role: "assistant", toolCalls: [call("t3", "x")] },
  ];
  const { rows, cueCount } = buildConversation(more, true);
  assert.equal(cueCount, 3);
  const last = rows[rows.length - 1];
  assert.equal(last.kind, "tool");
  if (last.kind === "tool") {
    assert.equal(last.call.cue, 3);
    assert.equal(last.call.status, "executing");
  }
});

test("streaming arguments are in progress; a stopped run marks the call stopped", () => {
  const partial: Message[] = [
    { id: "u", role: "user", content: "q" },
    { id: "a", role: "assistant", toolCalls: [call("t", "x", '{"query": "ba')] },
  ];
  const running = buildConversation(partial, true).rows[1];
  if (running.kind === "tool") {
    assert.equal(running.call.status, "inProgress");
    assert.equal(running.call.args, undefined);
  }
  const stopped = buildConversation(partial, false).rows[1];
  if (stopped.kind === "tool") assert.equal(stopped.call.stopped, true);
});

test("more than three calls in a finished run form one group", () => {
  const many: Message[] = [
    { id: "u", role: "user", content: "q" },
    {
      id: "a",
      role: "assistant",
      toolCalls: [call("1", "a"), call("2", "b"), call("3", "c"), call("4", "d")],
    },
  ];
  const { rows } = buildConversation(many, false);
  assert.deepEqual(rows.map((row) => row.kind), ["user", "group"]);
  const group = rows[1];
  if (group.kind === "group") assert.equal(cueRange(group.calls.map((c) => c.cue)), "Q1–Q4");
  assert.equal(cueRange([3]), "Q3");
});

test("calls in the running turn stay ungrouped until the turn ends", () => {
  const four = (prefix: string) =>
    [1, 2, 3, 4].map((n) => call(`${prefix}${n}`, "lookup"));
  const messages: Message[] = [
    { id: "u1", role: "user", content: "first" },
    { id: "a1", role: "assistant", toolCalls: four("old") },
    { id: "r1", role: "assistant", content: "First answer." },
    { id: "u2", role: "user", content: "second" },
    { id: "a2", role: "assistant", toolCalls: four("new") },
  ];
  const kinds = (running: boolean) =>
    buildConversation(messages, running).rows.map((row) => row.kind);
  // While the second turn runs, its calls show one by one; the first turn,
  // already finished, stays collapsed.
  assert.deepEqual(kinds(true), [
    "user", "group", "answer", "user", "tool", "tool", "tool", "tool",
  ]);
  // Once it ends, the second turn collapses too.
  assert.deepEqual(kinds(false), ["user", "group", "answer", "user", "group"]);
});

test("tool names are counted in first-seen order", () => {
  assert.equal(
    toolNameSummary([
      "memory_timeline",
      "memory_search",
      "family_assistance_act_search",
      "memory_search",
      "family_assistance_act_search",
    ]),
    "memory_timeline, memory_search ×2, family_assistance_act_search ×2",
  );
  assert.equal(toolNameSummary(["a"]), "a");
});

test("a call from an earlier turn stays stopped while a later turn runs", () => {
  const messages: Message[] = [
    { id: "u1", role: "user", content: "first" },
    { id: "a1", role: "assistant", toolCalls: [call("old", "lookup")] },
    { id: "u2", role: "user", content: "second" },
    { id: "a2", role: "assistant", toolCalls: [call("new", "lookup")] },
  ];
  const tools = buildConversation(messages, true).rows.flatMap((row) =>
    row.kind === "tool" ? [row.call] : [],
  );
  assert.equal(tools[0].toolCallId, "old");
  assert.equal(tools[0].stopped, true);
  assert.equal(cueState(tools[0]), "stopped");
  assert.equal(tools[1].status, "executing");
  assert.equal(cueState(tools[1]), "running");
});

function cue(toolCallId: string, extra: Partial<Cue>): Cue {
  return {
    cue: 1,
    toolCallId,
    name: "x",
    args: {},
    status: "complete",
    result: "ok",
    error: null,
    stopped: false,
    ...extra,
  };
}

test("a group of stopped calls does not say all done", () => {
  const stopped = [1, 2, 3, 4].map((n) => cue(`s${n}`, { result: undefined, stopped: true }));
  assert.deepEqual(groupSummary(stopped), { state: "stopped", word: "4 stopped" });
  const mixed = [cue("a", {}), cue("b", { result: "Error: boom" }), ...stopped.slice(0, 2)];
  assert.deepEqual(groupSummary(mixed), { state: "failed", word: "1 failed" });
  assert.deepEqual(groupSummary([cue("a", {}), cue("b", {})]), { state: "done", word: "All done" });
});

test("serve's Error: content marks the cue failed", () => {
  assert.equal(cueState(cue("e", { result: "Error: Permission denied" })), "failed");
});
