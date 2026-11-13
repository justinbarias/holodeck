import { test } from "node:test";
import assert from "node:assert/strict";
import {
  dropUnfinished,
  isUserAbort,
  recordResult,
  recordStart,
  timingKey,
  type Timings,
} from "./run-events.ts";
import { measuredElapsed } from "./tool-view.ts";

test("two threads that reuse a tool call id keep separate timings", () => {
  let timings: Timings = new Map();
  timings = recordStart(timings, timingKey("thread-a", "t1"), 1000);
  timings = recordResult(timings, timingKey("thread-a", "t1"), 3000);
  timings = recordStart(timings, timingKey("thread-b", "t1"), 5000);
  timings = recordResult(timings, timingKey("thread-b", "t1"), 5400);
  assert.deepEqual(measuredElapsed(timings.get(timingKey("thread-a", "t1")), 9000), {
    elapsedMs: 2000,
    running: false,
  });
  assert.deepEqual(measuredElapsed(timings.get(timingKey("thread-b", "t1")), 9000), {
    elapsedMs: 400,
    running: false,
  });
});

test("a call with no result gets no duration when the run ends", () => {
  let timings: Timings = new Map();
  timings = recordStart(timings, timingKey("a", "done"), 1000);
  timings = recordResult(timings, timingKey("a", "done"), 1500);
  timings = recordStart(timings, timingKey("a", "lost"), 1000);
  timings = dropUnfinished(timings);
  assert.deepEqual(measuredElapsed(timings.get(timingKey("a", "lost")), 6000), {
    elapsedMs: null,
    running: false,
  });
  assert.equal(measuredElapsed(timings.get(timingKey("a", "done")), 6000).elapsedMs, 500);
});

test("the user's Stop is not a run error; other errors are", () => {
  assert.equal(isUserAbort({ code: "abort" }), true);
  assert.equal(isUserAbort({ name: "AbortError" }), true);
  assert.equal(isUserAbort({ code: "INTERNAL_ERROR" }), false);
  assert.equal(isUserAbort({ name: "Error" }), false);
});
