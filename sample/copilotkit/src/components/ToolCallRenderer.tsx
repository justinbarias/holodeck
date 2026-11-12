"use client";

import { useRenderToolCall } from "@copilotkit/react-core";
import { ToolCallCard, type ToolCallStatus } from "@/components/ToolCallCard";
import { AsOfResults } from "@/components/memory/AsOfResults";
import { FactTimeline } from "@/components/memory/FactTimeline";
import { isToolNamed } from "@/components/memory/parse";
import { ProvenanceCard } from "@/components/memory/ProvenanceCard";

interface WildcardRenderProps {
  name: string;
  args: Record<string, unknown>;
  status: ToolCallStatus;
  result?: unknown;
}

/**
 * ToolCallRenderer - Renders tool calls from the AG-UI backend (v2 API)
 *
 * Uses useRenderToolCall with wildcard "*" to catch all backend tool calls
 * and render them inline in the chat message stream.
 *
 * Memory tools get a dedicated view, matched by exact name: HoloDeck's
 * backends send the bare names (`memory_timeline`, `memory_search`,
 * `memory_provenance`) on the AG-UI wire. An MCP tool from another server
 * that happens to end in the same name keeps the generic card, as does every
 * other tool.
 */
export function ToolCallRenderer() {
  useRenderToolCall({
    name: "*",
    render: (props) => {
      const { name, status, result } = props as unknown as WildcardRenderProps;
      const args = (props as unknown as WildcardRenderProps).args as
        | Record<string, unknown>
        | undefined;

      if (isToolNamed(name, "memory_timeline")) {
        return (
          <FactTimeline name={name} args={args} status={status} result={result} />
        );
      }

      if (isToolNamed(name, "memory_search")) {
        return <AsOfResults name={name} args={args} status={status} result={result} />;
      }

      if (isToolNamed(name, "memory_provenance")) {
        return (
          <ProvenanceCard name={name} args={args} status={status} result={result} />
        );
      }

      return <ToolCallCard name={name} args={args} status={status} result={result} />;
    },
  });

  return null;
}
