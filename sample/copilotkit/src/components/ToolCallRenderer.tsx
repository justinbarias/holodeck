"use client";

import { ToolCallCard, type ToolCallStatus } from "@/components/ToolCallCard";
import { AsOfResults } from "@/components/memory/AsOfResults";
import { FactTimeline } from "@/components/memory/FactTimeline";
import { isToolNamed } from "@/components/memory/parse";
import { ProvenanceCard } from "@/components/memory/ProvenanceCard";

export interface ToolCallViewProps {
  name: string;
  args?: Record<string, unknown>;
  status: ToolCallStatus;
  result?: unknown;
  error?: string | null;
  stopped?: boolean;
}

/**
 * Picks the view for one tool call. The conversation draws tool calls itself
 * (so it can number them in the cue margin), and calls this for each one.
 *
 * Memory tools get a dedicated view, matched by exact name: HoloDeck's
 * backends send the bare names (`memory_timeline`, `memory_search`,
 * `memory_provenance`) on the AG-UI wire. An MCP tool from another server
 * that happens to end in the same name keeps the generic card, as does every
 * other tool. A tool message with an `error` field always uses the generic
 * card, which shows the error.
 */
export function ToolCallView({ name, args, status, result, error, stopped }: ToolCallViewProps) {
  if (!error && !stopped) {
    if (isToolNamed(name, "memory_timeline")) {
      return <FactTimeline name={name} args={args} status={status} result={result} />;
    }
    if (isToolNamed(name, "memory_search")) {
      return <AsOfResults name={name} args={args} status={status} result={result} />;
    }
    if (isToolNamed(name, "memory_provenance")) {
      return <ProvenanceCard name={name} args={args} status={status} result={result} />;
    }
  }
  return (
    <ToolCallCard
      name={name}
      args={args}
      status={status}
      result={result}
      error={error}
      stopped={stopped}
    />
  );
}
