/**
 * Pure layout for the fact timeline: time axis, lane packing, the facts that
 * hold at an instant, and plain-word date text. No React and no DOM.
 *
 * Times are milliseconds since the Unix epoch (UTC). Every date is computed
 * and formatted in UTC, exactly as the tool result gives it.
 */
import type { TimelineEntry } from "./types";

const DAY_MS = 24 * 60 * 60 * 1000;

export interface AxisTick {
  time: number;
  label: string;
}

export interface TimeAxis {
  start: number;
  end: number;
  ticks: AxisTick[];
}

export interface TimelineBar {
  entry: TimelineEntry;
  start: number;
  /** `valid_to`, or the axis end for an open fact. */
  end: number;
  /** True when `valid_to` is null: the fact is still current. */
  open: boolean;
  lane: number;
}

export interface TimelineLayout {
  /** null when no entry has a `valid_from`: nothing to draw. */
  axis: TimeAxis | null;
  /** Bars grouped by lane, each lane in time order. */
  lanes: TimelineBar[][];
  /** Entries with `valid_from` null. They are not drawn as bars. */
  undated: TimelineEntry[];
  /** Dated entries left off the chart because of `MAX_DRAWN_BARS`. */
  omitted: number;
}

/**
 * The most bars the chart draws. A larger result keeps the bars with the
 * latest `valid_from` and the view says how many were left out, so a huge
 * result cannot freeze the page.
 */
export const MAX_DRAWN_BARS = 500;

export function toTime(iso: string): number {
  return Date.parse(iso);
}

function utcYearStart(year: number): number {
  return Date.UTC(year, 0, 1);
}

/**
 * The axis runs from the earlier of the earliest `valid_from` and `now` to
 * the later of the latest date in the data and `now`, so "now" is always on
 * the axis (a result with only future facts still shows today). Ticks fall
 * on 1 January of each year inside the domain, thinned to at most `maxTicks`.
 * Loops, not `Math.min(...array)`: a spread of a very long array throws.
 */
export function buildAxis(
  entries: readonly TimelineEntry[],
  now: number,
  maxTicks = 12,
): TimeAxis | null {
  let earliest = Infinity;
  let latest = now;
  for (const entry of entries) {
    if (entry.valid_from !== null) {
      const t = toTime(entry.valid_from);
      if (t < earliest) earliest = t;
      if (t > latest) latest = t;
    }
    if (entry.valid_to !== null) {
      const t = toTime(entry.valid_to);
      if (t > latest) latest = t;
    }
  }
  if (earliest === Infinity) return null;
  const start = Math.min(earliest, now);
  // Keep a visible span when everything starts at `now`.
  const end = Math.max(latest, start + 30 * DAY_MS);

  const firstYear = new Date(start).getUTCFullYear();
  const lastYear = new Date(end).getUTCFullYear();
  const years: number[] = [];
  for (let year = firstYear; year <= lastYear; year += 1) {
    const t = utcYearStart(year);
    if (t >= start && t <= end) years.push(year);
  }
  const step = Math.max(1, Math.ceil(years.length / maxTicks));
  const ticks = years
    .filter((year) => (year - (years[0] ?? 0)) % step === 0)
    .map((year) => ({ time: utcYearStart(year), label: String(year) }));
  return { start, end, ticks };
}

function compareText(a: string, b: string): number {
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

/**
 * Order by content, never by id first: start, end (open last), text. The
 * fact id breaks a tie only between entries that are otherwise identical,
 * where the order cannot change what is drawn.
 */
function compareEntries(a: TimelineEntry, b: TimelineEntry): number {
  const byStart = toTime(a.valid_from ?? "") - toTime(b.valid_from ?? "");
  if (byStart !== 0) return byStart;
  const endA = a.valid_to === null ? Infinity : toTime(a.valid_to);
  const endB = b.valid_to === null ? Infinity : toTime(b.valid_to);
  if (endA !== endB) return endA < endB ? -1 : 1;
  return compareText(a.text, b.text) || compareText(a.fact_id, b.fact_id);
}

/** Lower-case words of a fact without its date clause and without numbers. */
function attributeWords(text: string): Set<string> {
  const stripped = text
    .trim()
    .replace(LEADING_DATE_CLAUSE, "")
    .replace(TRAILING_DATE_CLAUSE, "")
    .toLowerCase();
  return new Set(stripped.match(/[a-z]+/g) ?? []);
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 && b.size === 0) return 0;
  let shared = 0;
  for (const word of a) if (b.has(word)) shared += 1;
  return shared / (a.size + b.size - shared);
}

/** Least word overlap for one fact to continue another's lane. */
const CHAIN_MIN_SIMILARITY = 0.3;

/**
 * Lane rule: one lane per attribute. The wire `TimelineEntry` names no
 * attribute, so lanes come from supersession: a fact whose `valid_to` equals
 * another fact's `valid_from` is replaced by it, and the two share a lane
 * (a chain). When several facts end and start at the same instant, pairs are
 * matched by word overlap of their text (date clause and numbers removed),
 * most similar first; ties go by text and dates. A pair below
 * `CHAIN_MIN_SIMILARITY` is not chained. A fact that continues no chain
 * starts its own lane. Lanes are ordered by their first fact's start, then
 * text. Nothing depends on fact ids or input order.
 */
export function layoutTimeline(
  entries: readonly TimelineEntry[],
  now: number,
): TimelineLayout {
  const axis = buildAxis(entries, now);
  const undated = entries.filter((entry) => entry.valid_from === null);
  if (!axis) return { axis, lanes: [], undated, omitted: 0 };

  const allDated = entries.filter((entry) => entry.valid_from !== null);
  allDated.sort(compareEntries);
  // Keep the latest bars when there are too many.
  const omitted = Math.max(0, allDated.length - MAX_DRAWN_BARS);
  const dated = omitted > 0 ? allDated.slice(omitted) : allDated;

  const words = new Map<TimelineEntry, Set<string>>();
  for (const entry of dated) words.set(entry, attributeWords(entry.text));

  // Facts ending and starting at each instant.
  const ending = new Map<number, TimelineEntry[]>();
  const starting = new Map<number, TimelineEntry[]>();
  for (const entry of dated) {
    const from = toTime(entry.valid_from ?? "");
    const list = starting.get(from);
    if (list) list.push(entry);
    else starting.set(from, [entry]);
    if (entry.valid_to !== null) {
      const to = toTime(entry.valid_to);
      // A fact must end after it starts to have a successor; this also
      // rules out cycles, because starts strictly increase along a chain.
      if (to > from) {
        const ends = ending.get(to);
        if (ends) ends.push(entry);
        else ending.set(to, [entry]);
      }
    }
  }

  const next = new Map<TimelineEntry, TimelineEntry>();
  const hasPrevious = new Set<TimelineEntry>();
  for (const [instant, preds] of ending) {
    const succs = starting.get(instant);
    if (!succs) continue;
    const pairs: { pred: TimelineEntry; succ: TimelineEntry; score: number }[] = [];
    for (const pred of preds) {
      for (const succ of succs) {
        const score = jaccard(words.get(pred) ?? new Set(), words.get(succ) ?? new Set());
        if (score >= CHAIN_MIN_SIMILARITY) pairs.push({ pred, succ, score });
      }
    }
    pairs.sort(
      (a, b) =>
        b.score - a.score ||
        compareEntries(a.pred, b.pred) ||
        compareEntries(a.succ, b.succ),
    );
    for (const { pred, succ } of pairs) {
      if (next.has(pred) || hasPrevious.has(succ)) continue;
      next.set(pred, succ);
      hasPrevious.add(succ);
    }
  }

  // `dated` is sorted, so heads come out ordered by start, then text.
  const lanes: TimelineBar[][] = [];
  for (const head of dated) {
    if (hasPrevious.has(head)) continue;
    const lane = lanes.length;
    const bars: TimelineBar[] = [];
    for (let entry: TimelineEntry | undefined = head; entry; entry = next.get(entry)) {
      const open = entry.valid_to === null;
      bars.push({
        entry,
        start: toTime(entry.valid_from ?? ""),
        end: open ? axis.end : toTime(entry.valid_to ?? ""),
        open,
        lane,
      });
    }
    lanes.push(bars);
  }
  return { axis, lanes, undated, omitted };
}

/**
 * Half-open validity, the rule the backend uses: a fact holds at `t` when
 * `valid_from <= t` (or `valid_from` is null) and (`valid_to` is null or
 * `valid_to > t`). Returns matching entries in input order.
 */
export function holdsAt(entry: TimelineEntry, t: number): boolean {
  const startsBefore =
    entry.valid_from === null || toTime(entry.valid_from) <= t;
  const endsAfter = entry.valid_to === null || toTime(entry.valid_to) > t;
  return startsBefore && endsAfter;
}

export function factsAt(
  entries: readonly TimelineEntry[],
  t: number,
): TimelineEntry[] {
  return entries.filter((entry) => holdsAt(entry, t));
}

/**
 * Where a fact stands at `now`:
 * - "current": it holds at `now` (`holdsAt`) and is not expired as of `now`;
 * - "future": `valid_from` is after `now`, so it is not yet in force;
 * - "ended": anything else (`valid_to <= now`, or expired by `now`).
 */
export type FactStatus = "current" | "future" | "ended";

export function factStatus(entry: TimelineEntry, now: number): FactStatus {
  if (entry.valid_from !== null && toTime(entry.valid_from) > now) return "future";
  const expired = entry.expired_at !== null && toTime(entry.expired_at) <= now;
  return holdsAt(entry, now) && !expired ? "current" : "ended";
}

export const FACT_STATUS_LABEL: Record<FactStatus, string> = {
  current: "Current",
  future: "Not yet in force",
  ended: "Ended",
};

/** Header summary: "8 facts · 3 current". */
export function summariseTimeline(
  entries: readonly TimelineEntry[],
  now: number,
): string {
  let current = 0;
  for (const entry of entries) if (factStatus(entry, now) === "current") current += 1;
  const n = entries.length;
  return `${n} ${n === 1 ? "fact" : "facts"} · ${current} current`;
}

/** Position of `t` on the axis, 0 to 1, clamped. */
export function fractionOf(axis: TimeAxis, t: number): number {
  const span = axis.end - axis.start;
  if (span <= 0) return 0;
  return Math.min(1, Math.max(0, (t - axis.start) / span));
}

/** The time at axis position `fraction`, snapped to the start of a UTC day. */
export function timeAtFraction(axis: TimeAxis, fraction: number): number {
  const clamped = Math.min(1, Math.max(0, fraction));
  return clampToAxis(
    axis,
    startOfUtcDay(axis.start + clamped * (axis.end - axis.start)),
  );
}

export function clampToAxis(axis: TimeAxis, t: number): number {
  return Math.min(axis.end, Math.max(axis.start, t));
}

export function startOfUtcDay(t: number): number {
  return Math.floor(t / DAY_MS) * DAY_MS;
}

/** Move `t` by whole UTC months; the day of month is kept where it exists. */
export function addUtcMonths(t: number, months: number): number {
  const date = new Date(t);
  const day = date.getUTCDate();
  const target = new Date(
    Date.UTC(
      date.getUTCFullYear(),
      date.getUTCMonth() + months,
      1,
      date.getUTCHours(),
      date.getUTCMinutes(),
      date.getUTCSeconds(),
      date.getUTCMilliseconds(),
    ),
  );
  const lastDay = new Date(
    Date.UTC(target.getUTCFullYear(), target.getUTCMonth() + 1, 0),
  ).getUTCDate();
  target.setUTCDate(Math.min(day, lastDay));
  return target.getTime();
}

const DATE_FORMAT = new Intl.DateTimeFormat("en-GB", {
  day: "numeric",
  month: "long",
  year: "numeric",
  timeZone: "UTC",
});

const pad = (n: number, width = 2) => String(n).padStart(width, "0");

/**
 * Fractional-second digits as the string gives them ("34.327700Z" gives
 * "327700"), or "" when there are none. `Date` keeps only milliseconds, so the
 * digits come from the text.
 */
export function fractionDigits(iso: string): string {
  const match = /T\d{2}:\d{2}:\d{2}\.(\d+)/.exec(iso);
  return match ? match[1] : "";
}

/** "07:31:34 UTC", with ".<fraction>" when `fraction` is not empty. */
function clockText(t: number, fraction: string): string {
  const d = new Date(t);
  const base = `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())}`;
  return `${base}${fraction ? `.${fraction}` : ""} UTC`;
}

function millisFraction(t: number): string {
  const ms = ((t % 1000) + 1000) % 1000;
  return ms === 0 ? "" : pad(ms, 3);
}

/** True when the instant is exactly UTC midnight, fraction included. */
export function isUtcMidnight(t: number, fraction = ""): boolean {
  return t % DAY_MS === 0 && !/[1-9]/.test(fraction);
}

/** "07:31:34.327700 UTC": the UTC time of day with seconds, as returned. */
export function formatIsoClock(iso: string): string {
  const fraction = fractionDigits(iso);
  return clockText(toTime(iso), fraction);
}

/**
 * "1 July 2020". A time that is not UTC midnight is added with seconds as
 * "HH:MM:SS UTC" (plus milliseconds when there are any) so the text never
 * hides part of what the result says.
 */
export function formatUtcDate(t: number): string {
  const date = DATE_FORMAT.format(t);
  const fraction = millisFraction(t);
  if (isUtcMidnight(t, fraction)) return date;
  return `${date}, ${clockText(t, fraction)}`;
}

/** Like `formatUtcDate`, with the fractional seconds exactly as the string has them. */
export function formatIsoDate(iso: string): string {
  const t = toTime(iso);
  const fraction = fractionDigits(iso);
  const date = DATE_FORMAT.format(t);
  if (isUtcMidnight(t, fraction)) return date;
  return `${date}, ${clockText(t, fraction)}`;
}

/**
 * The valid period in plain words. With `now`, an open fact that starts
 * after `now` reads "not yet in force" rather than "still current".
 */
export function describePeriod(
  entry: Pick<TimelineEntry, "valid_from" | "valid_to">,
  now?: number,
): string {
  const { valid_from: from, valid_to: to } = entry;
  if (from !== null && to !== null) {
    return `${formatIsoDate(from)} to ${formatIsoDate(to)}`;
  }
  if (from !== null) {
    const future = now !== undefined && toTime(from) > now;
    return `from ${formatIsoDate(from)}, ${future ? "not yet in force" : "still current"}`;
  }
  if (to !== null) return `until ${formatIsoDate(to)}, start date not known`;
  return "start date not known, still current";
}

/** The file name of a source URI: "docs/rate-2024.md" gives "rate-2024.md". */
export function fileName(uri: string): string {
  const path = uri.split(/[?#]/)[0].replace(/[\\/]+$/, "");
  const parts = path.split(/[\\/]/);
  return parts[parts.length - 1] || uri;
}

const LEADING_DATE_CLAUSE =
  /^(from|since|starting|on|as of)\s+\d{1,2}\s+[A-Za-z]+\s+\d{4},?\s+/i;

const TRAILING_DATE_CLAUSE =
  /,?\s+(starting\s+)?(from|since|on)\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}$/i;

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/**
 * A short bar label. The bar position already says when and the card header
 * names the entity, so this drops a leading or trailing date clause
 * ("From 1 July 2024, ...", "..., starting from 1 July 2022"), a mid-sentence
 * mention of the entity ("the base rate of Family Benefit is" gives "the base
 * rate is") and the final full stop. The view truncates it further with an
 * ellipsis; the full text is always shown elsewhere.
 */
export function shortFactLabel(text: string, entity?: string): string {
  let label = text.trim().replace(/\.$/, "");
  label = label.replace(LEADING_DATE_CLAUSE, "").replace(TRAILING_DATE_CLAUSE, "");
  if (entity && entity.trim()) {
    const mention = new RegExp(
      `\\s+(of|for|with|to)\\s+(the\\s+)?${escapeRegExp(entity.trim())}\\b`,
      "gi",
    );
    label = label.replace(mention, "");
  }
  label = label.trim();
  if (!label) return text.trim();
  return label.charAt(0).toUpperCase() + label.slice(1);
}
