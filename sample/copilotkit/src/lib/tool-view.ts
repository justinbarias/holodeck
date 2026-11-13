/**
 * Pure wording helpers shared by every tool card: the one-line headline of a
 * collapsed card, failure detection, sizes and durations. No React, no DOM,
 * so Node runs the tests with type stripping.
 */

type JsonObject = Record<string, unknown>;

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

const VALUE_CHARS = 60;

function cut(text: string, max: number): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > max ? `${flat.slice(0, max - 1).trimEnd()}…` : flat;
}

/** One argument value as short text: strings bare, everything else as JSON. */
export function argumentText(value: unknown): string {
  if (typeof value === "string") return cut(value, VALUE_CHARS);
  if (value === undefined) return "";
  return cut(JSON.stringify(value) ?? String(value), VALUE_CHARS);
}

/**
 * Headline of a collapsed generic card from its arguments. One argument reads
 * "order_id A-8844"; several read as their first two values, "249 · faulty_item".
 */
export function argumentsHeadline(args: Record<string, unknown> | undefined): string {
  if (!args) return "";
  const entries = Object.entries(args).filter(([, value]) => value !== undefined);
  if (entries.length === 0) return "No arguments";
  if (entries.length === 1) {
    const [key, value] = entries[0];
    return `${key} ${argumentText(value)}`;
  }
  return entries
    .slice(0, 2)
    .map(([, value]) => argumentText(value))
    .join(" · ");
}

/** The first sentence of a message, for a one-line headline. */
export function firstSentence(text: string): string {
  const flat = text.replace(/\s+/g, " ").trim();
  const match = /^(.+?[.!?])(\s|$)/.exec(flat);
  return match ? match[1] : flat;
}

/** Decode a JSON string result; anything else comes back unchanged. */
export function decodeResult(result: unknown): unknown {
  if (typeof result !== "string") return result;
  const trimmed = result.trim();
  if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) return result;
  try {
    return JSON.parse(trimmed) as unknown;
  } catch {
    return result;
  }
}

/**
 * The error a tool call reported, or null. The AG-UI tool message carries an
 * `error` field; a result object of the shape `{ error: "..." }` or
 * `{ error: { message } }` counts too.
 */
export function toolFailure(result: unknown, error?: string | null): string | null {
  if (error && error.trim()) return error.trim();
  const value = decodeResult(result);
  if (!isObject(value)) return null;
  const reported = value.error;
  if (typeof reported === "string" && reported.trim()) return reported.trim();
  if (isObject(reported) && typeof reported.message === "string") {
    return reported.message.trim() || null;
  }
  return null;
}

/** UTF-8 byte length of a string. */
export function byteLength(text: string): number {
  return new TextEncoder().encode(text).length;
}

/** "144 B", "4.2 KB", "1.3 MB". */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** Decoded size of a base64 string. */
export function base64Bytes(base64: string): number {
  const padding = base64.endsWith("==") ? 2 : base64.endsWith("=") ? 1 : 0;
  return Math.max(0, Math.floor((base64.length * 3) / 4) - padding);
}

/** "0.42 s" under 10 s, "12.4 s" under a minute, then "1 min 04 s". */
export function formatDuration(ms: number): string {
  const seconds = Math.max(0, ms) / 1000;
  if (seconds < 10) return `${seconds.toFixed(2)} s`;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds - minutes * 60);
  return `${minutes} min ${String(rest).padStart(2, "0")} s`;
}

/**
 * Below this gap between TOOL_CALL_START and TOOL_CALL_RESULT the browser did
 * not see the call run (the events arrived together, for example in one
 * burst at the end of a run), so no duration is shown.
 */
export const MIN_MEASURED_MS = 16;

/**
 * Elapsed time of a tool call from browser clock readings. Null when the call
 * was not timed, or when start and end are too close to be a measurement.
 */
export function measuredElapsed(
  timing: { start: number; end: number | null } | undefined,
  now: number,
): { elapsedMs: number | null; running: boolean } {
  if (!timing) return { elapsedMs: null, running: false };
  if (timing.end === null) return { elapsedMs: Math.max(0, now - timing.start), running: true };
  const elapsed = timing.end - timing.start;
  return { elapsedMs: elapsed < MIN_MEASURED_MS ? null : elapsed, running: false };
}
