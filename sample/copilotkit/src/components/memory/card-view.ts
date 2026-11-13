/**
 * Pure view logic for the collapsed and expanded memory cards: headlines,
 * lettered source chips, short dates, stamps and the as-of split of a
 * timeline. No React and no DOM.
 */
import { holdsAt, toTime } from "./timeline-layout.ts";
import type { MemorySearchResult, TimelineEntry } from "./types";

const SHORT_DATE = new Intl.DateTimeFormat("en-GB", {
  day: "numeric",
  month: "short",
  year: "numeric",
  timeZone: "UTC",
});

/** "1 Jul 2024" (UTC). */
export function shortDate(iso: string): string {
  return SHORT_DATE.format(toTime(iso));
}

/** "1 Jul 2024" from epoch ms (UTC). */
export function shortDateAt(t: number): string {
  return SHORT_DATE.format(t);
}

/** "1 JUL 2024", for stamps. */
export function stampDate(iso: string): string {
  return shortDate(iso).toUpperCase();
}

/** "1 Jul 2020 – 1 Jul 2024", "1 Jul 2024 –", "– 1 Jul 2024", or "No dates". */
export function shortPeriod(from: string | null, to: string | null): string {
  if (from && to) return `${shortDate(from)} – ${shortDate(to)}`;
  if (from) return `${shortDate(from)} –`;
  if (to) return `– ${shortDate(to)}`;
  return "No dates";
}

/**
 * Letters for plain source documents, A, B, C… in order of first appearance.
 * After Z the letters double (AA, AB…).
 */
export function letterSources(uris: readonly string[]): Map<string, string> {
  const letters = new Map<string, string>();
  for (const uri of uris) {
    if (!letters.has(uri)) letters.set(uri, letterAt(letters.size));
  }
  return letters;
}

function letterAt(index: number): string {
  const A = 65;
  if (index < 26) return String.fromCharCode(A + index);
  return letterAt(Math.floor(index / 26) - 1) + String.fromCharCode(A + (index % 26));
}

/** A retracted fact: no end date, but the record itself was expired. */
export function isRetracted(entry: TimelineEntry): boolean {
  return entry.valid_to === null && entry.expired_at !== null;
}

export interface AsOfSplit {
  /** Holds on the date and was not retracted. */
  holding: TimelineEntry[];
  /** Ended on or before the date, or retracted. */
  closed: TimelineEntry[];
  /** Starts after the date. */
  future: TimelineEntry[];
}

/** Split the facts of a timeline at an as-of instant. Input order is kept. */
export function splitAsOf(entries: readonly TimelineEntry[], t: number): AsOfSplit {
  const split: AsOfSplit = { holding: [], closed: [], future: [] };
  for (const entry of entries) {
    if (isRetracted(entry)) split.closed.push(entry);
    else if (holdsAt(entry, t)) split.holding.push(entry);
    else if (entry.valid_from !== null && toTime(entry.valid_from) > t) split.future.push(entry);
    else split.closed.push(entry);
  }
  return split;
}

/** The stamp on a closed fact: "CLOSED 1 JUL 2024" or "RETRACTED 3 MAR 2026". */
export function closedStamp(entry: TimelineEntry): string | null {
  if (isRetracted(entry) && entry.expired_at) return `RETRACTED ${stampDate(entry.expired_at)}`;
  if (entry.valid_to) return `CLOSED ${stampDate(entry.valid_to)}`;
  return null;
}

function latestStart(entries: readonly TimelineEntry[]): TimelineEntry | undefined {
  let best: TimelineEntry | undefined;
  for (const entry of entries) {
    if (!best) best = entry;
    else if (
      (entry.valid_from ? toTime(entry.valid_from) : -Infinity) >
      (best.valid_from ? toTime(best.valid_from) : -Infinity)
    ) {
      best = entry;
    }
  }
  return best;
}

/** "8 facts · 3 current" and the most recent current fact, for the collapsed card. */
export function timelineHeadline(
  entries: readonly TimelineEntry[],
  now: number,
): { count: string; latest: string | null } {
  const { holding } = splitAsOf(entries, now);
  const n = entries.length;
  return {
    count: `${n} ${n === 1 ? "fact" : "facts"} · ${holding.length} current`,
    latest: latestStart(holding)?.text ?? null,
  };
}

/** Collapsed headline of a search card. */
export function searchHeadline(search: MemorySearchResult, query: string): string {
  const facts = search.hits.filter((hit) => hit.kind === "fact").length;
  const passages = search.hits.filter(
    (hit) => hit.kind === "chunk" && hit.text.trim() !== "",
  ).length;
  const total = facts + passages;
  const quoted = query ? ` for “${query}”` : "";
  if (total === 0) return `No hits${quoted}`;
  if (search.degraded) {
    return `Keyword passages only · ${passages} ${passages === 1 ? "passage" : "passages"}${quoted}`;
  }
  return `${total} ${total === 1 ? "hit" : "hits"}${quoted}`;
}

/** "chunk 2" from "docs_rate-2024.md_chunk_2"; other ids come back unchanged. */
export function chunkLabel(chunkId: string): string {
  const match = /_chunk_(\d+)$/.exec(chunkId);
  return match ? `chunk ${match[1]}` : chunkId;
}

/**
 * Plain words for an invalid date argument: "The date “last July” was not
 * understood." when the message names a date argument whose value is known;
 * null otherwise.
 */
export function invalidDateSentence(
  message: string,
  args: Record<string, unknown> | undefined,
): string | null {
  for (const key of ["valid_at", "known_at", "as_of"]) {
    const value = args?.[key];
    if (message.includes(key) && typeof value === "string") {
      return `The date “${value}” was not understood.`;
    }
  }
  return null;
}
