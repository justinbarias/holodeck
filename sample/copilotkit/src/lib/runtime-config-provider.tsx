"use client";

import { createContext, useContext, type ReactNode } from "react";
import type { ClientRuntimeConfig } from "./runtime-config";

const RuntimeConfigContext = createContext<ClientRuntimeConfig | null>(null);

export function RuntimeConfigProvider({
  value,
  children,
}: {
  value: ClientRuntimeConfig;
  children: ReactNode;
}) {
  return (
    <RuntimeConfigContext.Provider value={value}>
      {children}
    </RuntimeConfigContext.Provider>
  );
}

export function useRuntimeConfig(): ClientRuntimeConfig {
  const ctx = useContext(RuntimeConfigContext);
  if (!ctx) {
    throw new Error(
      "useRuntimeConfig must be used within <RuntimeConfigProvider>",
    );
  }
  return ctx;
}
