import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { parseToolResult } from "./parse.ts";
import {
  REFERENCE_TIME_SOURCE_TEXT,
  describeDocumentDate,
  describeHitPeriod,
  describeInstants,
  describePassageStart,
  findSuperseding,
  groupHits,
  needsExpander,
  plural,
  shortFactId,
  summariseHits,
  supersedingLinkText,
} from "./search-view.ts";
import type { MemoryProvenanceResult, MemorySearchResult } from "./types.ts";

function result(name: string): unknown {
  const url = new URL(`./fixtures/${name}`, import.meta.url);
  return (JSON.parse(readFileSync(url, "utf8")) as { result: unknown }).result;
}

function search(name: string): MemorySearchResult {
  const parsed = parseToolResult(result(name));
  assert.equal(parsed.kind, "search");
  if (parsed.kind !== "search") throw new Error("not a search result");
  return parsed.result;
}

function provenance(name: string): MemoryProvenanceResult {
  const parsed = parseToolResult(result(name));
  assert.equal(parsed.kind, "provenance");
  if (parsed.kind !== "provenance") throw new Error("not a provenance result");
  return parsed.result;
}

test("search.now: 5 fact hits, 0 passages, not degraded", () => {
  const s = search("search.now.json");
  const groups = groupHits(s.hits);
  assert.equal(groups.facts.length, 5);
  assert.equal(groups.passages.length, 0);
  assert.equal(s.degraded, false);
  assert.equal(summariseHits(groups), "5 facts");
});

test("search.2021: 2 fact hits, one with no start date", () => {
  const s = search("search.2021.json");
  const groups = groupHits(s.hits);
  assert.equal(groups.facts.length, 2);
  assert.equal(groups.passages.length, 0);
  assert.equal(groups.facts.filter((hit) => hit.valid_from === null).length, 1);
  assert.equal(
    describeHitPeriod(groups.facts[0]),
    "1 July 2020 to 1 July 2022",
  );
  assert.equal(
    describeHitPeriod(groups.facts[1]),
    "until 1 July 2024, start date not known",
  );
});

test("synthetic degraded fixture: 0 facts, 3 passages, degraded", () => {
  const s = search("search.degraded.synthetic.json");
  const groups = groupHits(s.hits);
  assert.equal(s.degraded, true);
  assert.equal(groups.facts.length, 0);
  assert.equal(groups.passages.length, 3);
  assert.equal(summariseHits(groups), "3 passages");
  assert.deepEqual(
    groups.passages.map(describePassageStart),
    ["Starts 1 January 2023", "Starts 1 January 2022", null],
  );
});

test("grouping keeps the ranked order inside each group and puts nothing else in", () => {
  const hit = (kind: "fact" | "chunk", id: string) => ({
    kind,
    text: id,
    score: 0,
    source_tool: "t",
    chunk_id: id,
    source_uri: "a.md",
    fact_id: null,
    valid_from: null,
    valid_to: null,
    superseded_by: null,
  });
  const groups = groupHits([
    hit("chunk", "c1"),
    hit("fact", "f1"),
    hit("chunk", "c2"),
    hit("fact", "f2"),
  ]);
  assert.deepEqual(groups.facts.map((h) => h.text), ["f1", "f2"]);
  assert.deepEqual(groups.passages.map((h) => h.text), ["c1", "c2"]);
  assert.equal(summariseHits(groups), "2 facts · 2 passages");
  assert.equal(summariseHits(groupHits([])), "");
});

test("header: same UTC day says so once, with the clock times", () => {
  const s = search("search.now.json");
  assert.deepEqual(describeInstants(s.valid_at, s.known_at), {
    sameDay: true,
    rows: [{ label: "Valid and as known on", value: "11 November 2026" }],
    detail: "Valid at 00:00:00 UTC, known at 12:00:00 UTC",
  });
  assert.deepEqual(
    describeInstants("2026-11-11T00:00:00Z", "2026-11-11T00:00:00Z"),
    {
      sameDay: true,
      rows: [{ label: "Valid and as known on", value: "11 November 2026" }],
      detail: null,
    },
  );
  assert.equal(
    describeInstants("2026-11-11T09:30:00Z", "2026-11-11T09:30:00Z").detail,
    "Both at 09:30:00 UTC",
  );
});

test("header: different days give two labelled rows", () => {
  const s = search("search.2021.json");
  assert.deepEqual(describeInstants(s.valid_at, s.known_at), {
    sameDay: false,
    rows: [
      { label: "Valid at", value: "1 January 2021" },
      { label: "As known at", value: "11 November 2026, 12:00:00 UTC" },
    ],
    detail: null,
  });
});

test("header: the day boundary is UTC", () => {
  const view = describeInstants("2026-11-10T23:59:00Z", "2026-11-11T00:00:00Z");
  assert.equal(view.sameDay, false);
});

test("provenance.rate-150: found, one source with passage text", () => {
  const p = provenance("provenance.rate-150.json");
  assert.equal(p.found, true);
  assert.equal(p.sources.length, 1);
  assert.equal(p.fact?.valid_to, null);
  assert.match(p.sources[0].text ?? "", /^# Family Benefit base rate/);
  assert.equal(
    describeDocumentDate(p.sources[0]),
    "Document dated 1 July 2024, taken from the document's front matter",
  );
});

test("provenance.unknown: not found, no fact, no sources", () => {
  const p = provenance("provenance.unknown.json");
  assert.equal(p.found, false);
  assert.equal(p.fact, null);
  assert.deepEqual(p.sources, []);
});

test("wording for each reference_time_source value", () => {
  assert.deepEqual(REFERENCE_TIME_SOURCE_TEXT, {
    front_matter: "taken from the document's front matter",
    mtime: "taken from the file's last-modified time",
    ingestion: "no date in the document, so the time it was ingested is used",
  });
  const base = { reference_time: "2024-07-01T00:00:00Z" } as const;
  assert.equal(
    describeDocumentDate({ ...base, reference_time_source: "mtime" }),
    "Document dated 1 July 2024, taken from the file's last-modified time",
  );
  assert.equal(
    describeDocumentDate({ ...base, reference_time_source: "ingestion" }),
    "Document dated 1 July 2024, no date in the document, so the time it was ingested is used",
  );
});

test("plural and the expander threshold", () => {
  assert.equal(plural(1, "source", "sources"), "1 source");
  assert.equal(plural(0, "source", "sources"), "0 sources");
  assert.equal(needsExpander("short"), false);
  assert.equal(needsExpander("x".repeat(220)), false);
  assert.equal(needsExpander("x".repeat(221)), true);
  assert.equal(needsExpander("a\nb\nc\nd"), false);
  assert.equal(needsExpander("a\nb\nc\nd\ne"), true);
  const synthetic = search("search.degraded.synthetic.json").hits.map((h) =>
    needsExpander(h.text),
  );
  assert.deepEqual(synthetic, [true, false, false]);
});

test("groupHits leaves out passages with empty or whitespace text", () => {
  const base = search("search.2021.json").hits[0];
  const hit = (kind: "fact" | "chunk", text: string) => ({ ...base, kind, text });
  const groups = groupHits([
    hit("chunk", ""),
    hit("chunk", " \n\t "),
    hit("chunk", "body"),
    hit("fact", "a fact"),
  ]);
  assert.deepEqual(groups.passages.map((h) => h.text), ["body"]);
  assert.equal(groups.facts.length, 1);
  assert.equal(summariseHits(groups), "1 fact · 1 passage");
});

test("header: instants seconds apart on one day render differently, as returned", () => {
  const view = describeInstants(
    "2026-11-11T07:31:34.3277Z",
    "2026-11-11T07:31:53.240257Z",
  );
  assert.equal(view.sameDay, true);
  assert.equal(
    view.detail,
    "Valid at 07:31:34.3277 UTC, known at 07:31:53.240257 UTC",
  );
  // Exact midnight keeps the date-only form, even with a zero fraction.
  assert.equal(
    describeInstants("2026-11-11T00:00:00.000000Z", "2026-11-11T00:00:00Z").detail,
    null,
  );
});

test("superseded fact hit: links to a listed fact, else names the id", () => {
  const base = {
    kind: "fact" as const,
    score: 1,
    source_tool: "t",
    chunk_id: "c",
    source_uri: "docs/a.md",
    valid_from: null,
    valid_to: null,
  };
  const newer = {
    ...base,
    text: "From 1 July 2024, the base rate of Family Benefit is 150 dollars per fortnight for each child.",
    fact_id: "9f87ed20-3484-4d29-910c-38b0b70bed1f",
    superseded_by: null,
  };
  const older = {
    ...base,
    text: "The base rate is 120 dollars.",
    fact_id: "9e9b87f7-fefd-4a3e-a4af-4c013984c9d8",
    superseded_by: newer.fact_id,
  };
  const orphan = { ...older, superseded_by: "ac6ecdee-97f7-45a4-b5c7-a136e14dacba" };
  const facts = [older, newer, orphan];
  assert.deepEqual(findSuperseding(newer, facts), { kind: "none" });
  assert.deepEqual(findSuperseding(older, facts), { kind: "listed", index: 1 });
  assert.deepEqual(findSuperseding(orphan, facts), {
    kind: "unlisted",
    id: "ac6ecdee-97f7-45a4-b5c7-a136e14dacba",
    shortId: "ac6ecdee…",
  });
  assert.equal(shortFactId("f1"), "f1");
  assert.equal(
    supersedingLinkText(newer),
    "From 1 July 2024, the base rate of Family Benefit is 150…",
  );
});
