import { test } from "node:test";
import assert from "node:assert/strict";
import {
  argumentsHeadline,
  base64Bytes,
  firstSentence,
  formatBytes,
  formatDuration,
  measuredElapsed,
  toolFailure,
} from "./tool-view.ts";

test("one argument reads as key and value", () => {
  assert.equal(argumentsHeadline({ order_id: "A-8844" }), "order_id A-8844");
});

test("several arguments read as their first two values", () => {
  assert.equal(
    argumentsHeadline({ amount: 249, reason: "faulty_item", notify: true }),
    "249 · faulty_item",
  );
  assert.equal(argumentsHeadline({}), "No arguments");
  assert.equal(argumentsHeadline(undefined), "");
});

test("first sentence stops at the first full stop", () => {
  assert.equal(firstSentence("Timed out after 30 s. The mail server did not answer."), "Timed out after 30 s.");
  assert.equal(firstSentence("no stop here"), "no stop here");
});

test("failures come from the error field or an error object", () => {
  assert.equal(toolFailure("ok", "boom"), "boom");
  assert.equal(toolFailure({ error: "bad input" }), "bad input");
  assert.equal(toolFailure('{"error":{"code":"x","message":"Nope."}}'), "Nope.");
  assert.equal(toolFailure({ id: 1 }), null);
  assert.equal(toolFailure("plain text"), null);
});

test("sizes and durations", () => {
  assert.equal(formatBytes(144), "144 B");
  assert.equal(formatBytes(4300), "4.2 KB");
  assert.equal(base64Bytes("aGVsbG8="), 5);
  assert.equal(formatDuration(420), "0.42 s");
  assert.equal(formatDuration(30000), "30.0 s");
  assert.equal(formatDuration(64000), "1 min 04 s");
});

test("durations under the measuring floor are not shown", () => {
  assert.deepEqual(measuredElapsed(undefined, 0), { elapsedMs: null, running: false });
  assert.deepEqual(measuredElapsed({ start: 1000, end: 1004 }, 2000), { elapsedMs: null, running: false });
  assert.deepEqual(measuredElapsed({ start: 1000, end: 1420 }, 2000), { elapsedMs: 420, running: false });
  assert.deepEqual(measuredElapsed({ start: 1000, end: null }, 1250), { elapsedMs: 250, running: true });
});
