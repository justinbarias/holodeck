import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  chunkLabel,
  closedStamp,
  invalidDateSentence,
  letterSources,
  searchHeadline,
  shortPeriod,
  splitAsOf,
  timelineHeadline,
} from "./card-view.ts";
import { parseToolResult } from "./parse.ts";

function fixture(name: string): { args: Record<string, unknown>; result: unknown } {
  const url = new URL(`./fixtures/${name}`, import.meta.url);
  return JSON.parse(readFileSync(url, "utf8"));
}

const NOW = Date.UTC(2026, 10, 13);

function timeline() {
  const parsed = parseToolResult(fixture("timeline.family-benefit.json").result);
  if (parsed.kind !== "timeline") throw new Error("fixture");
  return parsed.result.entries;
}

test("letters follow first appearance", () => {
  const letters = letterSources(["b.md", "a.md", "b.md", "c.md"]);
  assert.deepEqual([...letters.entries()], [["b.md", "A"], ["a.md", "B"], ["c.md", "C"]]);
});

test("timeline headline counts current facts and names the latest", () => {
  const head = timelineHeadline(timeline(), NOW);
  assert.equal(head.count, "8 facts · 3 current");
  assert.match(head.latest ?? "", /^From 1 July 2024/);
});

test("as-of split keeps closed facts with a stamp", () => {
  const split = splitAsOf(timeline(), NOW);
  assert.equal(split.holding.length, 3);
  assert.equal(split.closed.length, 5);
  assert.equal(split.future.length, 0);
  assert.equal(closedStamp(split.closed[1]), "CLOSED 1 JUL 2020");
});

test("search headline, degraded and empty", () => {
  const now = parseToolResult(fixture("search.now.json").result);
  if (now.kind !== "search") throw new Error("fixture");
  assert.match(searchHeadline(now.result, "q"), /^\d+ hits? for “q”$/);
  const degraded = parseToolResult(fixture("search.degraded.synthetic.json").result);
  if (degraded.kind !== "search") throw new Error("fixture");
  assert.match(searchHeadline(degraded.result, "x"), /^Keyword passages only · \d+ passages? for “x”$/);
  assert.equal(searchHeadline({ ...now.result, hits: [] }, "q"), "No hits for “q”");
});

test("small wording helpers", () => {
  assert.equal(chunkLabel("docs_rate-2024.md_chunk_2"), "chunk 2");
  assert.equal(chunkLabel("abc"), "abc");
  assert.equal(shortPeriod("2020-07-01T00:00:00Z", null), "1 Jul 2020 –");
  const invalid = fixture("search.invalid.json");
  assert.equal(
    invalidDateSentence("valid_at must be an ISO 8601 date", invalid.args),
    "The date “last July” was not understood.",
  );
});
