/**
 * Pure view logic for the `memory_search` and `memory_provenance` renderers:
 * grouping, header wording and the plain-word text for each document-date
 * source. No React and no DOM.
 */
import {
  describePeriod,
  formatIsoClock,
  formatIsoDate,
  fractionDigits,
  isUtcMidnight,
  startOfUtcDay,
  toTime,
} from "./timeline-layout.ts";
import type {
  MemorySearchHit,
  MemorySource,
  ReferenceTimeSource,
} from "./types";

export function plural(count: number, one: string, many: string): string {
  return `${count} ${count === 1 ? one : many}`;
}

export interface InstantRow {
  label: string;
  value: string;
}

export interface InstantsView {
  /** True when both instants fall on the same UTC day. */
  sameDay: boolean;
  rows: InstantRow[];
  /** Time-of-day text for the same-day case when a time is not midnight. */
  detail: string | null;
}

function isMidnightIso(iso: string): boolean {
  return isUtcMidnight(toTime(iso), fractionDigits(iso));
}

/**
 * The two instants a search answer is for. Different UTC days give two rows,
 * "Valid at" and "As known at". The same UTC day gives one row, "Valid and as
 * known on <date>", plus a detail line with the clock times (seconds, and the
 * fractional seconds as returned) when either one is not UTC midnight, so no
 * part of the result is hidden.
 */
export function describeInstants(validAt: string, knownAt: string): InstantsView {
  const valid = toTime(validAt);
  const known = toTime(knownAt);
  if (startOfUtcDay(valid) !== startOfUtcDay(known)) {
    return {
      sameDay: false,
      rows: [
        { label: "Valid at", value: formatIsoDate(validAt) },
        { label: "As known at", value: formatIsoDate(knownAt) },
      ],
      detail: null,
    };
  }
  let detail: string | null = null;
  if (!isMidnightIso(validAt) || !isMidnightIso(knownAt)) {
    const validClock = formatIsoClock(validAt);
    const knownClock = formatIsoClock(knownAt);
    detail =
      validClock === knownClock
        ? `Both at ${validClock}`
        : `Valid at ${validClock}, known at ${knownClock}`;
  }
  return {
    sameDay: true,
    rows: [
      { label: "Valid and as known on", value: formatIsoDate(new Date(startOfUtcDay(valid)).toISOString()) },
    ],
    detail,
  };
}

export interface HitGroups {
  facts: MemorySearchHit[];
  passages: MemorySearchHit[];
}

/**
 * Fact hits and passage hits, each in the order the result gives. A passage
 * whose text is empty or whitespace is left out, so it is neither shown nor
 * counted.
 */
export function groupHits(hits: readonly MemorySearchHit[]): HitGroups {
  return {
    facts: hits.filter((hit) => hit.kind === "fact"),
    passages: hits.filter((hit) => hit.kind === "chunk" && hit.text.trim() !== ""),
  };
}

export function summariseHits(groups: HitGroups): string {
  const parts: string[] = [];
  if (groups.facts.length > 0) {
    parts.push(plural(groups.facts.length, "fact", "facts"));
  }
  if (groups.passages.length > 0) {
    parts.push(plural(groups.passages.length, "passage", "passages"));
  }
  return parts.join(" · ");
}

/** The valid period of a fact hit in plain words. */
export function describeHitPeriod(hit: MemorySearchHit): string {
  return describePeriod(hit);
}

/** "Starts 1 July 2024" for a passage with a start date, else null. */
export function describePassageStart(hit: MemorySearchHit): string | null {
  return hit.valid_from === null ? null : `Starts ${formatIsoDate(hit.valid_from)}`;
}

export const SUPERSEDED_NOTE = "Superseded by a later fact";

/** Where the fact that supersedes a hit is. */
export type SupersedingFact =
  | { kind: "none" }
  /** In the same result: `index` into the fact hits. */
  | { kind: "listed"; index: number }
  /** Not in this result: the id, and a short form to display. */
  | { kind: "unlisted"; id: string; shortId: string };

/** "9f87ed20-3484-…" gives "9f87ed20…"; a short id is kept whole. */
export function shortFactId(id: string): string {
  return id.length > 12 ? `${id.slice(0, 8)}…` : id;
}

/** Link text naming the superseding fact: its text, cut at a word near 60 characters. */
export function supersedingLinkText(fact: Pick<MemorySearchHit, "text">): string {
  const text = fact.text.trim().replace(/\s+/g, " ");
  if (text.length <= 60) return text;
  const cut = text.slice(0, 59);
  const space = cut.lastIndexOf(" ");
  return `${(space > 30 ? cut.slice(0, space) : cut).trimEnd()}…`;
}

export function findSuperseding(
  hit: MemorySearchHit,
  facts: readonly MemorySearchHit[],
): SupersedingFact {
  const id = hit.superseded_by;
  if (id === null) return { kind: "none" };
  const index = facts.findIndex((fact) => fact !== hit && fact.fact_id === id);
  if (index !== -1) return { kind: "listed", index };
  return { kind: "unlisted", id, shortId: shortFactId(id) };
}

/** Phrase for where a document's date came from. */
export const REFERENCE_TIME_SOURCE_TEXT: Record<ReferenceTimeSource, string> = {
  front_matter: "taken from the document's front matter",
  mtime: "taken from the file's last-modified time",
  ingestion: "no date in the document, so the time it was ingested is used",
};

export function describeReferenceTimeSource(source: ReferenceTimeSource): string {
  return REFERENCE_TIME_SOURCE_TEXT[source];
}

/** "Document dated 1 July 2024, taken from the document's front matter". */
export function describeDocumentDate(
  source: Pick<MemorySource, "reference_time" | "reference_time_source">,
): string {
  return `Document dated ${formatIsoDate(source.reference_time)}, ${describeReferenceTimeSource(
    source.reference_time_source,
  )}`;
}

const EXPAND_CHARS = 220;
const EXPAND_LINES = 4;

/** True when a passage is long enough that the view clamps it. */
export function needsExpander(text: string): boolean {
  return text.length > EXPAND_CHARS || text.split(/\r?\n/).length > EXPAND_LINES;
}
