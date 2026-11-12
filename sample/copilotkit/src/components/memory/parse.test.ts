import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { isToolNamed, parseToolResult } from "./parse.ts";

interface Fixture {
  name: string;
  args: Record<string, unknown>;
  result: unknown;
}

function fixture(name: string): Fixture {
  const url = new URL(`./fixtures/${name}`, import.meta.url);
  return JSON.parse(readFileSync(url, "utf8")) as Fixture;
}

test("timeline object parses with all eight entries", () => {
  const parsed = parseToolResult(fixture("timeline.family-benefit.json").result);
  assert.equal(parsed.kind, "timeline");
  if (parsed.kind !== "timeline") return;
  assert.equal(parsed.result.entity, "Family Benefit");
  assert.equal(parsed.result.entries.length, 8);
  assert.equal(parsed.result.entries[0].valid_from, null);
  assert.equal(parsed.result.entries[0].sources[0].source_uri, "docs/rate-2024.md");
});

test("timeline given as a JSON string parses the same", () => {
  const raw = fixture("timeline.family-benefit.json").result;
  assert.deepEqual(parseToolResult(JSON.stringify(raw)), parseToolResult(raw));
});

test("timeline wrapped as MCP text content parses the same", () => {
  const raw = fixture("timeline.family-benefit.json").result;
  const wrapped = { content: [{ type: "text", text: JSON.stringify(raw) }] };
  assert.deepEqual(parseToolResult(wrapped), parseToolResult(raw));
});

test("empty timeline parses with no entries", () => {
  assert.deepEqual(parseToolResult(fixture("timeline.unknown.json").result), {
    kind: "timeline",
    result: { entity: "Nobody Payment", entries: [] },
  });
});

test("error shape gives code and message", () => {
  const raw = fixture("search.invalid.json").result;
  const expected = {
    kind: "error",
    code: "invalid_argument",
    message:
      "valid_at must be an ISO 8601 date or datetime, for example 2022-07-01 or 2022-07-01T09:30:00Z",
  };
  assert.deepEqual(parseToolResult(raw), expected);
  assert.deepEqual(parseToolResult(JSON.stringify(raw)), expected);
});

test("garbage input is unknown and keeps the raw value", () => {
  for (const raw of [
    "not json at all",
    "{ broken",
    "42",
    42,
    null,
    undefined,
    [],
    { hello: "world" },
    { error: "flat string" },
    { entity: "X", entries: "nope" },
    { entity: "X", entries: [{ fact_id: "a" }] },
    {
      entity: "X",
      entries: [
        {
          fact_id: "a",
          text: "t",
          valid_from: "last July",
          valid_to: null,
          recorded_at: null,
          expired_at: null,
          sources: [],
        },
      ],
    },
  ]) {
    assert.deepEqual(parseToolResult(raw), { kind: "unknown", raw });
  }
});

test("search result parses with five fact hits", () => {
  const parsed = parseToolResult(fixture("search.now.json").result);
  assert.equal(parsed.kind, "search");
  if (parsed.kind !== "search") return;
  assert.equal(parsed.result.hits.length, 5);
  assert.equal(parsed.result.degraded, false);
  assert.equal(parsed.result.valid_at, "2026-11-11T00:00:00Z");
});

test("search result as a JSON string or MCP text content parses the same", () => {
  const raw = fixture("search.2021.json").result;
  const expected = parseToolResult(raw);
  assert.equal(expected.kind, "search");
  assert.deepEqual(parseToolResult(JSON.stringify(raw)), expected);
  const wrapped = { content: [{ type: "text", text: JSON.stringify(raw) }] };
  assert.deepEqual(parseToolResult(wrapped), expected);
});

test("search with a bad hit, timestamp or flag is unknown", () => {
  const good = fixture("search.now.json").result as {
    hits: Record<string, unknown>[];
  };
  const variants: Record<string, unknown>[] = [
    { ...good, hits: [{ ...good.hits[0], kind: "passage" }] },
    { ...good, hits: [{ ...good.hits[0], score: "high" }] },
    { ...good, hits: [{ ...good.hits[0], valid_from: "last July" }] },
    { ...good, hits: [{ ...good.hits[0], fact_id: undefined }] },
    { ...good, hits: [{ ...good.hits[0], superseded_by: 3 }] },
    { ...good, hits: "none" },
    { ...good, degraded: "no" },
    { ...good, valid_at: "yesterday" },
  ];
  for (const raw of variants) {
    assert.deepEqual(parseToolResult(raw), { kind: "unknown", raw });
  }
});

test("search with zero hits parses", () => {
  const raw = {
    valid_at: "1990-01-01T00:00:00Z",
    known_at: "2026-11-11T12:00:00Z",
    degraded: false,
    hits: [],
  };
  assert.deepEqual(parseToolResult(raw), { kind: "search", result: raw });
});

test("provenance result parses, found and not found", () => {
  const found = parseToolResult(fixture("provenance.rate-150.json").result);
  assert.equal(found.kind, "provenance");
  if (found.kind === "provenance") {
    assert.equal(found.result.found, true);
    assert.equal(found.result.sources.length, 1);
    assert.equal(found.result.fact?.sources.length, 1);
  }
  assert.deepEqual(parseToolResult(fixture("provenance.unknown.json").result), {
    kind: "provenance",
    result: { found: false, fact: null, sources: [] },
  });
});

test("provenance that contradicts itself or is malformed is unknown", () => {
  // Decision: found true with no fact (or found false with a fact) cannot be
  // drawn as either answer, so it falls back to the generic card.
  const good = fixture("provenance.rate-150.json").result as {
    fact: Record<string, unknown>;
    sources: Record<string, unknown>[];
  };
  const variants: unknown[] = [
    { found: true, fact: null, sources: [] },
    { found: true, fact: null, sources: good.sources },
    { found: false, fact: good.fact, sources: [] },
    { found: "yes", fact: good.fact, sources: [] },
    { found: true, fact: good.fact, sources: "none" },
    {
      found: true,
      fact: good.fact,
      sources: [{ ...good.sources[0], reference_time_source: "guess" }],
    },
    {
      found: true,
      fact: good.fact,
      sources: [{ ...good.sources[0], reference_time: "someday" }],
    },
  ];
  for (const raw of variants) {
    assert.deepEqual(parseToolResult(raw), { kind: "unknown", raw });
  }
});

test("tool names match exactly; an MCP tool from another server does not match", () => {
  assert.ok(isToolNamed("memory_timeline", "memory_timeline"));
  assert.ok(isToolNamed("memory_search", "memory_search"));
  assert.ok(!isToolNamed("mcp__unrelated__memory_search", "memory_search"));
  assert.ok(!isToolNamed("mcp__holodeck_tools__memory_timeline", "memory_timeline"));
  assert.ok(!isToolNamed("memory_timeline_v2", "memory_timeline"));
  assert.ok(!isToolNamed("not_memory_timeline", "memory_timeline"));
  assert.ok(!isToolNamed("memory_search", "memory_timeline"));
});

test("impossible dates and naive datetimes are rejected", () => {
  const good = fixture("search.now.json").result as Record<string, unknown>;
  assert.equal(parseToolResult(good).kind, "search");
  for (const bad of [
    "2026-02-30T00:00:00Z", // no 30 February
    "2026-11-11T00:00:00", // no offset: would be read in the browser's zone
    "2026-11-11T24:00:00Z",
    "2026-11-11T00:00:00+25:00",
    "2026-11-11",
  ]) {
    for (const field of ["valid_at", "known_at"]) {
      const raw = { ...good, [field]: bad };
      assert.deepEqual(parseToolResult(raw), { kind: "unknown", raw }, `${field}=${bad}`);
    }
  }
  // An explicit offset is accepted.
  assert.equal(
    parseToolResult({ ...good, valid_at: "2026-11-11T10:00:00+10:00" }).kind,
    "search",
  );
});
