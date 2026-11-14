"use client";

import { useCallback, useEffect, useState } from "react";
import type { AbstractAgent } from "@ag-ui/client";
import {
  dropUnfinished,
  isUserAbort,
  recordResult,
  recordStart,
  timingKey,
  type Timings,
} from "./run-events";

export type { ToolTiming, Timings } from "./run-events";

export interface RunError {
  message: string;
  code: string | null;
}

/**
 * Listen to the agent's AG-UI events for two things the message list does
 * not carry: when each tool call started and got its result (TOOL_CALL_START
 * to TOOL_CALL_RESULT, measured in the browser, keyed by thread), and the
 * last RUN_ERROR that was not the user's own Stop.
 */
export function useRunEvents(agent: AbstractAgent | undefined) {
  const [timings, setTimings] = useState<Timings>(() => new Map());
  const [runError, setRunError] = useState<RunError | null>(null);

  useEffect(() => {
    if (!agent) return;
    const subscription = agent.subscribe({
      onRunStartedEvent: () => {
        setRunError(null);
      },
      onToolCallStartEvent: ({ event, input }) => {
        const at = Date.now();
        const key = timingKey(input.threadId, event.toolCallId);
        setTimings((prev) => recordStart(prev, key, at));
      },
      onToolCallResultEvent: ({ event, input }) => {
        const at = Date.now();
        const key = timingKey(input.threadId, event.toolCallId);
        setTimings((prev) => recordResult(prev, key, at));
      },
      onRunErrorEvent: ({ event }) => {
        if (isUserAbort({ code: event.code })) return;
        setRunError({ message: event.message, code: event.code ?? null });
      },
      onRunFailed: ({ error }) => {
        if (isUserAbort(error)) return;
        setRunError((current) => current ?? { message: error.message, code: null });
      },
      onRunFinalized: () => {
        setTimings((prev) => dropUnfinished(prev));
      },
    });
    return () => subscription.unsubscribe();
  }, [agent]);

  const clearRunError = useCallback(() => setRunError(null), []);

  return { timings, runError, clearRunError };
}
