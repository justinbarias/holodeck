/**
 * Pure bookkeeping for the AG-UI events the message list does not carry:
 * browser clock readings per tool call, keyed by thread, and which run
 * errors are the user's own Stop. No React, so Node runs the tests.
 */

/** Client-side clock readings for one tool call. */
export interface ToolTiming {
  start: number;
  /** Set when TOOL_CALL_RESULT arrives; null while the call is open. */
  end: number | null;
}

export type Timings = ReadonlyMap<string, ToolTiming>;

/** Tool call ids are only unique within a thread, so keys carry both. */
export function timingKey(threadId: string, toolCallId: string): string {
  return `${threadId}::${toolCallId}`;
}

/** TOOL_CALL_START: the first start for a key wins. */
export function recordStart(timings: Timings, key: string, at: number): Timings {
  if (timings.has(key)) return timings;
  const next = new Map(timings);
  next.set(key, { start: at, end: null });
  return next;
}

/** TOOL_CALL_RESULT: closes an open call. */
export function recordResult(timings: Timings, key: string, at: number): Timings {
  const timing = timings.get(key);
  if (!timing || timing.end !== null) return timings;
  const next = new Map(timings);
  next.set(key, { ...timing, end: at });
  return next;
}

/**
 * The run ended: a call that never got its result has no duration (start to
 * the end of the run is not the tool's time), so its reading is dropped.
 */
export function dropUnfinished(timings: Timings): Timings {
  let changed = false;
  const next = new Map<string, ToolTiming>();
  for (const [key, timing] of timings) {
    if (timing.end === null) changed = true;
    else next.set(key, timing);
  }
  return changed ? next : timings;
}

/**
 * True for the error the client raises when the user presses Stop:
 * @ag-ui/client turns an aborted request into RUN_ERROR with code "abort",
 * and a rejected run carries an AbortError.
 */
export function isUserAbort(error: {
  code?: string | null;
  name?: string;
}): boolean {
  return error.code === "abort" || error.name === "AbortError";
}
