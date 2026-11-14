/**
 * Turn a raw tool result into a typed memory result.
 *
 * CopilotKit (1.54 through 1.87, checked) hands a renderer the tool message content after
 * `parseJson(content, content)` (see @copilotkit/react-core
 * src/hooks/use-render-tool-call.ts), so a result arrives as an object when
 * the content was valid JSON and as the original string when it was not.
 * Both are accepted here. Anything that does not match a known shape comes
 * back as `{ kind: "unknown", raw }` so the caller can show it unchanged.
 *
 * Pure module: no React, no DOM. Only type imports, so Node can run its
 * tests with type stripping.
 */
import type {
  MemoryProvenanceResult,
  MemorySearchHit,
  MemorySearchResult,
  MemorySource,
  MemoryTimelineResult,
  ReferenceTimeSource,
  TimelineEntry,
} from "./types";

export type ParsedToolResult =
  | { kind: "timeline"; result: MemoryTimelineResult }
  | { kind: "search"; result: MemorySearchResult }
  | { kind: "provenance"; result: MemoryProvenanceResult }
  | { kind: "error"; code: string; message: string }
  | { kind: "unknown"; raw: unknown };

type JsonObject = Record<string, unknown>;

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * An ISO 8601 date-time with an explicit offset: `Z` or `+HH:MM`/`-HH:MM`.
 * A naive date-time would be read in the browser's time zone, so it is
 * rejected.
 */
const TIMESTAMP =
  /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?(?:Z|[+-](\d{2}):(\d{2}))$/;

/**
 * True for a well-formed instant whose fields are real: the date components
 * must round-trip through `Date.UTC`, so 2026-02-30 is rejected rather than
 * read as 2 March.
 */
function isTimestamp(value: unknown): value is string {
  if (typeof value !== "string") return false;
  const match = TIMESTAMP.exec(value);
  if (!match) return false;
  const [year, month, day, hour, minute] = match.slice(1, 6).map(Number);
  const second = match[6] === undefined ? 0 : Number(match[6]);
  if (hour > 23 || minute > 59 || second > 59) return false;
  if (match[7] !== undefined && (Number(match[7]) > 23 || Number(match[8]) > 59)) {
    return false;
  }
  const date = new Date(Date.UTC(year, month - 1, day));
  if (
    date.getUTCFullYear() !== year ||
    date.getUTCMonth() !== month - 1 ||
    date.getUTCDate() !== day
  ) {
    return false;
  }
  return !Number.isNaN(Date.parse(value));
}

function isNullableTimestamp(value: unknown): value is string | null {
  return value === null || isTimestamp(value);
}

const REFERENCE_TIME_SOURCES: readonly string[] = [
  "front_matter",
  "mtime",
  "ingestion",
];

function toSource(value: unknown): MemorySource | null {
  if (!isObject(value)) return null;
  const {
    episode_id,
    source_tool,
    chunk_id,
    source_uri,
    reference_time,
    reference_time_source,
    text,
  } = value;
  if (
    typeof episode_id !== "string" ||
    typeof source_tool !== "string" ||
    typeof chunk_id !== "string" ||
    typeof source_uri !== "string" ||
    !isTimestamp(reference_time) ||
    typeof reference_time_source !== "string" ||
    !REFERENCE_TIME_SOURCES.includes(reference_time_source)
  ) {
    return null;
  }
  if (text !== undefined && text !== null && typeof text !== "string") {
    return null;
  }
  return {
    episode_id,
    source_tool,
    chunk_id,
    source_uri,
    reference_time,
    reference_time_source: reference_time_source as ReferenceTimeSource,
    text: text ?? null,
  };
}

function toEntry(value: unknown): TimelineEntry | null {
  if (!isObject(value)) return null;
  const {
    fact_id,
    text,
    valid_from,
    valid_to,
    recorded_at,
    expired_at,
    sources,
  } = value;
  if (
    typeof fact_id !== "string" ||
    typeof text !== "string" ||
    !isNullableTimestamp(valid_from) ||
    !isNullableTimestamp(valid_to) ||
    !isNullableTimestamp(recorded_at) ||
    !isNullableTimestamp(expired_at) ||
    !Array.isArray(sources)
  ) {
    return null;
  }
  const parsedSources: MemorySource[] = [];
  for (const source of sources) {
    const parsed = toSource(source);
    if (!parsed) return null;
    parsedSources.push(parsed);
  }
  return {
    fact_id,
    text,
    valid_from,
    valid_to,
    recorded_at,
    expired_at,
    sources: parsedSources,
  };
}

function toTimeline(value: JsonObject): MemoryTimelineResult | null {
  const { entity, entries } = value;
  if (typeof entity !== "string" || !Array.isArray(entries)) return null;
  const parsed: TimelineEntry[] = [];
  for (const entry of entries) {
    const result = toEntry(entry);
    if (!result) return null;
    parsed.push(result);
  }
  return { entity, entries: parsed };
}

function isNullableString(value: unknown): value is string | null {
  return value === null || typeof value === "string";
}

function toHit(value: unknown): MemorySearchHit | null {
  if (!isObject(value)) return null;
  const {
    kind,
    text,
    score,
    source_tool,
    chunk_id,
    source_uri,
    fact_id,
    valid_from,
    valid_to,
    superseded_by,
  } = value;
  if (
    (kind !== "fact" && kind !== "chunk") ||
    typeof text !== "string" ||
    typeof score !== "number" ||
    typeof source_tool !== "string" ||
    typeof chunk_id !== "string" ||
    typeof source_uri !== "string" ||
    !isNullableString(fact_id) ||
    !isNullableTimestamp(valid_from) ||
    !isNullableTimestamp(valid_to) ||
    !isNullableString(superseded_by)
  ) {
    return null;
  }
  return {
    kind,
    text,
    score,
    source_tool,
    chunk_id,
    source_uri,
    fact_id,
    valid_from,
    valid_to,
    superseded_by,
  };
}

function toSearch(value: JsonObject): MemorySearchResult | null {
  const { valid_at, known_at, degraded, hits } = value;
  if (
    !isTimestamp(valid_at) ||
    !isTimestamp(known_at) ||
    typeof degraded !== "boolean" ||
    !Array.isArray(hits)
  ) {
    return null;
  }
  const parsed: MemorySearchHit[] = [];
  for (const hit of hits) {
    const result = toHit(hit);
    if (!result) return null;
    parsed.push(result);
  }
  return { valid_at, known_at, degraded, hits: parsed };
}

/**
 * `found: true` with `fact: null` is contradictory, so it is not a provenance
 * result: it comes back as unknown and the generic card shows it unchanged.
 * `found: false` carries no fact; a fact next to `found: false` is rejected
 * the same way.
 */
function toProvenance(value: JsonObject): MemoryProvenanceResult | null {
  const { found, fact, sources } = value;
  if (typeof found !== "boolean" || !Array.isArray(sources)) return null;
  let entry: TimelineEntry | null = null;
  if (fact !== null) {
    entry = toEntry(fact);
    if (!entry) return null;
  }
  if (found !== (entry !== null)) return null;
  const parsed: MemorySource[] = [];
  for (const source of sources) {
    const result = toSource(source);
    if (!result) return null;
    parsed.push(result);
  }
  return { found, fact: entry, sources: parsed };
}

/**
 * An MCP tool result can reach the client still wrapped as text content:
 * `{ content: [{ type: "text", text: "<json>" }] }` or the bare array.
 * Return the single text part, or the value unchanged.
 */
function unwrapTextContent(value: unknown): unknown {
  const parts = isObject(value) ? value.content : value;
  if (!Array.isArray(parts) || parts.length !== 1) return value;
  const part: unknown = parts[0];
  if (isObject(part) && part.type === "text" && typeof part.text === "string") {
    return part.text;
  }
  return value;
}

function decode(raw: unknown): unknown {
  if (typeof raw !== "string") return raw;
  try {
    return JSON.parse(raw) as unknown;
  } catch {
    return undefined;
  }
}

export function parseToolResult(raw: unknown): ParsedToolResult {
  let value = decode(raw);
  const unwrapped = unwrapTextContent(value);
  if (unwrapped !== value) value = decode(unwrapped);
  if (!isObject(value)) return { kind: "unknown", raw };

  const error = value.error;
  if (isObject(error)) {
    if (typeof error.code === "string" && typeof error.message === "string") {
      return { kind: "error", code: error.code, message: error.message };
    }
    return { kind: "unknown", raw };
  }

  if ("entity" in value && "entries" in value) {
    const timeline = toTimeline(value);
    if (timeline) return { kind: "timeline", result: timeline };
  }

  if ("hits" in value && "valid_at" in value) {
    const search = toSearch(value);
    if (search) return { kind: "search", result: search };
  }

  if ("found" in value && "sources" in value) {
    const provenance = toProvenance(value);
    if (provenance) return { kind: "provenance", result: provenance };
  }

  return { kind: "unknown", raw };
}

/**
 * True when `name` is exactly `tool`. HoloDeck's backends send the bare
 * names (`memory_timeline`) on the AG-UI wire: the Claude backend strips the
 * `mcp__<server>__` prefix and OpenAI Agents uses the fixed names. A suffix
 * match would also catch an unrelated MCP tool such as
 * `mcp__unrelated__memory_search`.
 */
export function isToolNamed(name: string, tool: string): boolean {
  return name === tool;
}
