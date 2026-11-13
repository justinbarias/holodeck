"use client";

import { useCallback, useEffect, useState } from "react";
import type { AbstractAgent } from "@ag-ui/client";

/** Client-side clock readings for one tool call. */
export interface ToolTiming {
  start: number;
  /** Set when the result arrives, or when the run ends without one. */
  end: number | null;
}

export interface RunError {
  message: string;
  code: string | null;
}

/**
 * Listen to the agent's AG-UI events for two things the message list does
 * not carry: when each tool call started and finished (TOOL_CALL_START to
 * TOOL_CALL_RESULT, measured in the browser), and the last RUN_ERROR.
 */
export function useRunEvents(agent: AbstractAgent | undefined) {
  const [timings, setTimings] = useState<ReadonlyMap<string, ToolTiming>>(() => new Map());
  const [runError, setRunError] = useState<RunError | null>(null);

  useEffect(() => {
    if (!agent) return;
    const update = (change: (next: Map<string, ToolTiming>) => boolean) =>
      setTimings((prev) => {
        const next = new Map(prev);
        return change(next) ? next : prev;
      });
    const subscription = agent.subscribe({
      onRunStartedEvent: () => {
        setRunError(null);
      },
      onToolCallStartEvent: ({ event }) => {
        const start = Date.now();
        update((next) => {
          if (next.has(event.toolCallId)) return false;
          next.set(event.toolCallId, { start, end: null });
          return true;
        });
      },
      onToolCallResultEvent: ({ event }) => {
        const end = Date.now();
        update((next) => {
          const timing = next.get(event.toolCallId);
          if (!timing || timing.end !== null) return false;
          next.set(event.toolCallId, { ...timing, end });
          return true;
        });
      },
      onRunErrorEvent: ({ event }) => {
        setRunError({ message: event.message, code: event.code ?? null });
      },
      onRunFailed: ({ error }) => {
        setRunError((current) => current ?? { message: error.message, code: null });
      },
      onRunFinalized: () => {
        const end = Date.now();
        update((next) => {
          let changed = false;
          for (const [id, timing] of next) {
            if (timing.end === null) {
              next.set(id, { ...timing, end });
              changed = true;
            }
          }
          return changed;
        });
      },
    });
    return () => subscription.unsubscribe();
  }, [agent]);

  const clearRunError = useCallback(() => setRunError(null), []);

  return { timings, runError, clearRunError };
}
