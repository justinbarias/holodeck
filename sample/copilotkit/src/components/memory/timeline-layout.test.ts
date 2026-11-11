import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  MAX_DRAWN_BARS,
  addUtcMonths,
  buildAxis,
  describePeriod,
  factStatus,
  factsAt,
  fileName,
  formatIsoDate,
  formatUtcDate,
  layoutTimeline,
  shortFactLabel,
  summariseTimeline,
  toTime,
} from "./timeline-layout.ts";
import type { MemoryTimelineResult, TimelineEntry } from "./types";

function fixture(name: string): MemoryTimelineResult {
  const url = new URL(`./fixtures/${name}`, import.meta.url);
  return (JSON.parse(readFileSync(url, "utf8")) as { result: MemoryTimelineResult })
    .result;
}

const timeline = fixture("timeline.family-benefit.json");
const entries = timeline.entries;

// Short ids for the eight facts in timeline.family-benefit.json.
const ADMIN_UNTIL_2024 = "938a53d8-a5ca-427a-884f-4bf7db20373e"; // valid_from null
const RATE_90 = "43d703ca-9e44-4c73-9e8e-11bb01f2f2b1"; // 2019-07-01 .. 2020-07-01
const RATE_100 = "e28a99df-f0c4-4cbf-a457-2010092ce417"; // 2020-07-01 .. 2022-07-01
const SUPPLEMENT = "645906e9-ccc8-46e0-a761-3c3ef8bf580c"; // 2021-07-01 .. open
const RATE_120 = "9e9b87f7-fefd-4a3e-a4af-4c013984c9d8"; // 2022-07-01 .. 2024-07-01
const ADMIN_AGENCY = "d9b08131-ae08-41a0-8361-0f2a54d01587"; // 2022-07-01 .. 2024-07-01
const RATE_150 = "9f87ed20-3484-4d29-910c-38b0b70bed1f"; // 2024-07-01 .. open
const ADMIN_OFFICE = "ac6ecdee-97f7-45a4-b5c7-a136e14dacba"; // 2024-07-01 .. open

// A fixed "today" so the axis is deterministic.
const NOW = Date.parse("2026-11-11T00:00:00Z");

const ids = (list: TimelineEntry[]) => list.map((entry) => entry.fact_id);

test("fixture has eight entries", () => {
  assert.equal(entries.length, 8);
});

test("axis runs from the earliest valid_from to today, with year ticks", () => {
  const axis = buildAxis(entries, NOW);
  assert.ok(axis);
  assert.equal(axis.start, Date.parse("2019-07-01T00:00:00Z"));
  // Latest date in the data is 2024-07-01, so today is the end.
  assert.equal(axis.end, NOW);
  assert.deepEqual(
    axis.ticks.map((tick) => tick.label),
    ["2020", "2021", "2022", "2023", "2024", "2025", "2026"],
  );
  assert.equal(axis.ticks[0].time, Date.parse("2020-01-01T00:00:00Z"));
});

test("axis end is the latest data date when it is after today", () => {
  const axis = buildAxis(entries, Date.parse("2020-01-01T00:00:00Z"));
  assert.ok(axis);
  assert.equal(axis.end, Date.parse("2024-07-01T00:00:00Z"));
});

test("lanes: one per attribute, chained by supersession (same as the earlier rendering)", () => {
  // Worked by hand (chains: valid_to of one fact equals valid_from of the next,
  // paired by word overlap when several end and start at the same instant):
  //  RATE_90 -> RATE_100 (only fact starting 2020-07-01)
  //  RATE_100 -> RATE_120 (rate words beat ADMIN_AGENCY at 2022-07-01)
  //  RATE_120 -> RATE_150 and ADMIN_AGENCY -> ADMIN_OFFICE (2024-07-01)
  //  SUPPLEMENT continues nothing: its own lane.
  // Lanes ordered by first start: rates 2019, supplement 2021, admin 2022.
  // These are the lanes the earlier greedy packing drew for this fixture.
  const layout = layoutTimeline(entries, NOW);
  assert.equal(layout.lanes.length, 3);
  assert.deepEqual(
    layout.lanes.map((lane) => lane.map((bar) => bar.entry.fact_id)),
    [
      [RATE_90, RATE_100, RATE_120, RATE_150],
      [SUPPLEMENT],
      [ADMIN_AGENCY, ADMIN_OFFICE],
    ],
  );
  assert.deepEqual(
    layout.lanes.map((lane) => lane.map((bar) => bar.lane)),
    [[0, 0, 0, 0], [1], [2, 2]],
  );
});

test("open facts extend to the axis end; closed facts end at valid_to", () => {
  const layout = layoutTimeline(entries, NOW);
  const bars = layout.lanes.flat();
  const open = bars.filter((bar) => bar.open).map((bar) => bar.entry.fact_id);
  // lane 0 ends with RATE_150, lane 1 is SUPPLEMENT, lane 2 ends with ADMIN_OFFICE
  assert.deepEqual(open, [RATE_150, SUPPLEMENT, ADMIN_OFFICE]);
  for (const bar of bars) {
    if (bar.open) assert.equal(bar.end, NOW);
    else assert.equal(bar.end, toTime(bar.entry.valid_to ?? ""));
  }
});

test("the entry with valid_from null is undated and not drawn", () => {
  const layout = layoutTimeline(entries, NOW);
  assert.deepEqual(ids(layout.undated), [ADMIN_UNTIL_2024]);
  const drawn = layout.lanes.flat().map((bar) => bar.entry.fact_id);
  assert.equal(drawn.length, 7);
  assert.ok(!drawn.includes(ADMIN_UNTIL_2024));
});

test("facts at 2021-01-01", () => {
  assert.deepEqual(ids(factsAt(entries, Date.parse("2021-01-01T00:00:00Z"))), [
    ADMIN_UNTIL_2024,
    RATE_100,
  ]);
});

test("facts at 2019-12-01", () => {
  assert.deepEqual(ids(factsAt(entries, Date.parse("2019-12-01T00:00:00Z"))), [
    ADMIN_UNTIL_2024,
    RATE_90,
  ]);
});

test("facts at 2018-01-01: only the fact with no known start", () => {
  assert.deepEqual(ids(factsAt(entries, Date.parse("2018-01-01T00:00:00Z"))), [
    ADMIN_UNTIL_2024,
  ]);
});

test("facts today: the three open facts", () => {
  const expected = [SUPPLEMENT, RATE_150, ADMIN_OFFICE];
  assert.deepEqual(ids(factsAt(entries, NOW)), expected);
  assert.deepEqual(ids(factsAt(entries, Date.now())), expected);
});

test("half-open: a fact whose valid_to equals the instant does not hold", () => {
  const t = Date.parse("2022-07-01T00:00:00Z");
  const held = ids(factsAt(entries, t));
  assert.ok(!held.includes(RATE_100), "RATE_100 ends at 2022-07-01");
  assert.deepEqual(held, [ADMIN_UNTIL_2024, SUPPLEMENT, RATE_120, ADMIN_AGENCY]);
  // and the fact whose valid_from equals the instant does hold
  assert.ok(held.includes(RATE_120));
});

test("empty entries: no axis, no lanes, no undated, no facts", () => {
  const empty = fixture("timeline.unknown.json");
  assert.deepEqual(empty.entries, []);
  const layout = layoutTimeline(empty.entries, NOW);
  assert.equal(layout.axis, null);
  assert.deepEqual(layout.lanes, []);
  assert.deepEqual(layout.undated, []);
  assert.deepEqual(factsAt(empty.entries, NOW), []);
});

test("only undated entries: no axis, all undated", () => {
  const only = entries.filter((entry) => entry.fact_id === ADMIN_UNTIL_2024);
  const layout = layoutTimeline(only, NOW);
  assert.equal(layout.axis, null);
  assert.deepEqual(ids(layout.undated), [ADMIN_UNTIL_2024]);
});

test("plain-word periods, in UTC", () => {
  const byId = new Map(entries.map((entry) => [entry.fact_id, entry]));
  const period = (id: string) => describePeriod(byId.get(id) as TimelineEntry);
  assert.equal(period(RATE_100), "1 July 2020 to 1 July 2022");
  assert.equal(period(RATE_150), "from 1 July 2024, still current");
  assert.equal(period(ADMIN_UNTIL_2024), "until 1 July 2024, start date not known");
  assert.equal(
    formatIsoDate("2026-11-11T07:31:34.327700Z"),
    "11 November 2026, 07:31:34.327700 UTC",
  );
  assert.equal(formatIsoDate("2019-12-31T23:30:00Z"), "31 December 2019, 23:30:00 UTC");
});

test("month steps keep the day where it exists", () => {
  assert.equal(
    addUtcMonths(Date.parse("2021-01-31T00:00:00Z"), 1),
    Date.parse("2021-02-28T00:00:00Z"),
  );
  assert.equal(
    addUtcMonths(Date.parse("2021-07-01T00:00:00Z"), -12),
    Date.parse("2020-07-01T00:00:00Z"),
  );
});

test("labels and file names", () => {
  assert.equal(
    shortFactLabel("From 1 July 2024, the base rate of Family Benefit is 150 dollars per fortnight for each child."),
    "The base rate of Family Benefit is 150 dollars per fortnight for each child",
  );
  assert.equal(
    shortFactLabel("Family Benefit is administered by the Benefits Agency.", "Family Benefit"),
    "Family Benefit is administered by the Benefits Agency",
  );
  assert.equal(
    shortFactLabel(
      "The base rate of Family Benefit is 120 dollars per fortnight for each child, starting from 1 July 2022.",
      "Family Benefit",
    ),
    "The base rate is 120 dollars per fortnight for each child",
  );
  assert.equal(
    shortFactLabel(
      "The young child supplement is payable with Family Benefit from 1 July 2021.",
      "Family Benefit",
    ),
    "The young child supplement is payable",
  );
  assert.equal(fileName("docs/rate-2024.md"), "rate-2024.md");
  assert.equal(fileName("rate.md"), "rate.md");
});

const laneIds = (list: readonly TimelineEntry[]) =>
  layoutTimeline(list, NOW).lanes.map((lane) => lane.map((bar) => bar.entry.text));

test("lanes do not depend on fact ids or input order", () => {
  const expected = laneIds(entries);
  // The review's repro: give the Social Services Office fact the lowest id.
  const renamed = entries.map((entry) =>
    entry.fact_id === ADMIN_OFFICE
      ? { ...entry, fact_id: "00000000-0000-4000-8000-000000000000" }
      : entry,
  );
  assert.deepEqual(laneIds(renamed), expected);
  assert.deepEqual(laneIds([...entries].reverse()), expected);
  // Every id replaced, in an order that reverses the original id order.
  const reIded = entries.map((entry, index) => ({
    ...entry,
    fact_id: String(entries.length - index).padStart(4, "0"),
  }));
  assert.deepEqual(laneIds(reIded), expected);
  assert.deepEqual(laneIds([...reIded].reverse()), expected);
  // The rate lane stays the rate lane.
  const rateLane = layoutTimeline(renamed, NOW).lanes[0].map((bar) => bar.entry.fact_id);
  assert.deepEqual(rateLane, [RATE_90, RATE_100, RATE_120, RATE_150]);
});

test("a fact with no overlapping words does not continue a lane", () => {
  const byId = new Map(entries.map((entry) => [entry.fact_id, entry]));
  const only = [byId.get(RATE_100), byId.get(ADMIN_AGENCY)] as TimelineEntry[];
  // RATE_100 ends when ADMIN_AGENCY starts, but they are different attributes.
  assert.equal(layoutTimeline(only, NOW).lanes.length, 2);
});

test("150,000 entries: no RangeError, and the chart draws at most the bar limit", () => {
  const start = Date.parse("1900-01-01T00:00:00Z");
  const big: TimelineEntry[] = [];
  for (let i = 0; i < 150_000; i += 1) {
    const from = new Date(start + i * 60_000).toISOString();
    const to = new Date(start + (i + 1) * 60_000).toISOString();
    big.push({
      fact_id: `f${i}`,
      text: `Fact number ${i}`,
      valid_from: from,
      valid_to: to,
      recorded_at: null,
      expired_at: null,
      sources: [],
    });
  }
  const layout = layoutTimeline(big, NOW);
  assert.ok(layout.axis);
  assert.equal(layout.axis.start, start);
  assert.equal(layout.axis.end, NOW);
  const drawn = layout.lanes.flat();
  assert.equal(drawn.length, MAX_DRAWN_BARS);
  assert.equal(layout.omitted, 150_000 - MAX_DRAWN_BARS);
  // The latest facts are the ones kept.
  assert.equal(drawn[drawn.length - 1].entry.fact_id, "f149999");
});

test("a fact that starts after now is not current, and now stays on the axis", () => {
  const now = Date.parse("2026-11-12T12:00:00Z");
  const future: TimelineEntry = {
    fact_id: "future",
    text: "From 1 January 2027, the base rate is 160 dollars.",
    valid_from: "2027-01-01T00:00:00Z",
    valid_to: null,
    recorded_at: null,
    expired_at: null,
    sources: [],
  };
  assert.equal(factStatus(future, now), "future");
  assert.equal(summariseTimeline([future], now), "1 fact · 0 current");
  assert.equal(describePeriod(future, now), "from 1 January 2027, not yet in force");
  const axis = buildAxis([future], now);
  assert.ok(axis);
  assert.equal(axis.start, now, "the axis extends back to now");
  assert.ok(axis.end >= Date.parse("2027-01-01T00:00:00Z"));
  // The as-of view at now: nothing holds yet.
  assert.deepEqual(factsAt([future], now), []);
  // Once the date passes it is current.
  assert.equal(factStatus(future, Date.parse("2027-01-01T00:00:00Z")), "current");
});

test("fact status on the fixture: current, ended, and expired", () => {
  const byId = new Map(entries.map((entry) => [entry.fact_id, entry]));
  const status = (id: string) => factStatus(byId.get(id) as TimelineEntry, NOW);
  assert.equal(status(RATE_150), "current");
  assert.equal(status(SUPPLEMENT), "current");
  assert.equal(status(RATE_120), "ended");
  assert.equal(status(ADMIN_UNTIL_2024), "ended");
  assert.equal(summariseTimeline(entries, NOW), "8 facts · 3 current");
  const retracted = { ...(byId.get(RATE_150) as TimelineEntry), expired_at: "2026-01-01T00:00:00Z" };
  assert.equal(factStatus(retracted, NOW), "ended");
});

test("times show seconds and fractions as returned; midnight is date only", () => {
  const valid = formatIsoDate("2026-11-11T07:31:34.3277Z");
  const known = formatIsoDate("2026-11-11T07:31:53.240257Z");
  assert.equal(valid, "11 November 2026, 07:31:34.3277 UTC");
  assert.equal(known, "11 November 2026, 07:31:53.240257 UTC");
  assert.notEqual(valid, known);
  assert.equal(formatIsoDate("2026-11-11T00:00:00Z"), "11 November 2026");
  assert.equal(formatUtcDate(Date.parse("2026-11-11T00:00:00Z")), "11 November 2026");
  assert.equal(
    formatUtcDate(Date.parse("2026-11-11T07:31:34.327Z")),
    "11 November 2026, 07:31:34.327 UTC",
  );
});
