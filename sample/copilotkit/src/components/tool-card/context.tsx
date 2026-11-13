"use client";

import { createContext, useCallback, useContext, useMemo, useState, type ReactNode } from "react";

/**
 * Per-session expand/collapse choices, keyed by tool call id (or a group or
 * demo key). A choice survives re-renders, streaming and thread switches for
 * as long as the page is open.
 */
interface DisclosureStore {
  isOpen: (key: string) => boolean;
  setOpen: (key: string, open: boolean) => void;
}

const DisclosureContext = createContext<DisclosureStore | null>(null);

export function DisclosureProvider({
  children,
  initial,
}: {
  children: ReactNode;
  initial?: Record<string, boolean>;
}) {
  const [open, setOpenState] = useState<Record<string, boolean>>(initial ?? {});
  const setOpen = useCallback((key: string, value: boolean) => {
    setOpenState((prev) => (prev[key] === value ? prev : { ...prev, [key]: value }));
  }, []);
  const store = useMemo<DisclosureStore>(
    () => ({ isOpen: (key) => open[key] === true, setOpen }),
    [open, setOpen],
  );
  return <DisclosureContext.Provider value={store}>{children}</DisclosureContext.Provider>;
}

/** Open state for one key; falls back to local state outside a provider. */
export function useDisclosure(key: string | undefined): [boolean, (open: boolean) => void] {
  const store = useContext(DisclosureContext);
  const [local, setLocal] = useState(false);
  if (store && key) return [store.isOpen(key), (value) => store.setOpen(key, value)];
  return [local, setLocal];
}

/** Elapsed time of one tool call on the thread's shared scale. */
export interface CallTiming {
  /** Elapsed ms, or null when the call was not timed (for example, history). */
  elapsedMs: number | null;
  /** Longest elapsed time in the thread, the bar's full width. */
  scaleMs: number;
  running: boolean;
}

export interface ToolCallInfo {
  /** Key for the disclosure store. */
  key?: string;
  timing?: CallTiming;
}

const ToolCallInfoContext = createContext<ToolCallInfo>({});

export const ToolCallInfoProvider = ToolCallInfoContext.Provider;

export function useToolCallInfo(): ToolCallInfo {
  return useContext(ToolCallInfoContext);
}
