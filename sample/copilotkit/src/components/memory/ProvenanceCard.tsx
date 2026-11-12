"use client";

import { useState } from "react";
import { ToolCallCard, type ToolCallStatus } from "@/components/ToolCallCard";
import { ClampedText } from "./ClampedText";
import { MemoryCardHeader } from "./MemoryCardHeader";
import { parseToolResult } from "./parse";
import { describeDocumentDate, plural } from "./search-view";
import {
  FACT_STATUS_LABEL,
  describePeriod,
  factStatus,
  fileName,
} from "./timeline-layout";
import type { MemoryProvenanceResult, MemorySource } from "./types";

export interface ProvenanceCardProps {
  /** Tool name as it arrived. */
  name?: string;
  args?: Record<string, unknown>;
  status: ToolCallStatus;
  /** The tool result: object, JSON string, or anything else. */
  result?: unknown;
  /** "Now" in epoch ms, to tell a current fact from a future one. Defaults to the time of first render. */
  now?: number;
}

const TOOL_NAME = "memory_provenance";
const EYEBROW = "Fact provenance";
const TITLE = "Where this fact comes from";

/**
 * Generative-UI view of a `memory_provenance` result: the fact, then each
 * source document with its date and the passage it came from.
 */
export function ProvenanceCard({ name, args, status, result, now }: ProvenanceCardProps) {
  if (status !== "complete") {
    return (
      <section className="holodeck-tool-call holodeck-timeline" aria-busy="true">
        <MemoryCardHeader eyebrow={EYEBROW} title={TITLE} status={status} />
        <p className="holodeck-timeline__status-line">Looking up the sources…</p>
      </section>
    );
  }

  const parsed = parseToolResult(result);

  if (parsed.kind === "error") {
    return (
      <section className="holodeck-tool-call holodeck-timeline">
        <MemoryCardHeader eyebrow={EYEBROW} title={TITLE} status={status} failed />
        <div className="holodeck-timeline__error">
          <p>{parsed.message}</p>
          <span className="holodeck-timeline__code">{parsed.code}</span>
        </div>
      </section>
    );
  }

  if (parsed.kind !== "provenance") {
    return (
      <ToolCallCard name={name ?? TOOL_NAME} args={args} status={status} result={result} />
    );
  }

  return <ProvenanceView provenance={parsed.result} now={now} />;
}

function ProvenanceView({
  provenance,
  now: nowProp,
}: {
  provenance: MemoryProvenanceResult;
  now?: number;
}) {
  const { fact, sources } = provenance;
  const [now] = useState(() => nowProp ?? Date.now());

  if (!provenance.found || !fact) {
    return (
      <section className="holodeck-tool-call holodeck-timeline holodeck-asof">
        <MemoryCardHeader eyebrow={EYEBROW} title={TITLE} status="complete" />
        <p className="holodeck-timeline__status-line">This fact was not found.</p>
      </section>
    );
  }

  return (
    <section
      className="holodeck-tool-call holodeck-timeline holodeck-asof"
      aria-label="Sources of a fact"
    >
      <MemoryCardHeader
        eyebrow={EYEBROW}
        title={TITLE}
        status="complete"
        summary={plural(sources.length, "source", "sources")}
      />

      <div className="holodeck-asof__fact">
        <p className="holodeck-asof__text">{fact.text}</p>
        <p className="holodeck-timeline__fact-meta">
          <span
            className={
              factStatus(fact, now) === "current"
                ? "holodeck-timeline__tag holodeck-timeline__tag--open"
                : "holodeck-timeline__tag"
            }
          >
            {FACT_STATUS_LABEL[factStatus(fact, now)]}
          </span>
          <span>{capitalise(describePeriod(fact, now))}</span>
        </p>
      </div>

      {sources.length === 0 ? (
        <p className="holodeck-timeline__status-line">No source passage is recorded.</p>
      ) : (
        <div className="holodeck-timeline__facts">
          <h4 className="holodeck-timeline__facts-title">Sources</h4>
          <ol className="holodeck-asof__hits">
            {sources.map((source) => (
              <li key={source.episode_id + source.chunk_id} className="holodeck-asof__hit">
                <SourceBlock source={source} />
              </li>
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}

function SourceBlock({ source }: { source: MemorySource }) {
  return (
    <>
      <p className="holodeck-asof__source-file">
        <span className="holodeck-timeline__file">{fileName(source.source_uri)}</span>
      </p>
      <p className="holodeck-timeline__fact-meta">{describeDocumentDate(source)}</p>
      {source.text ? (
        <blockquote className="holodeck-asof__quote">
          <ClampedText text={source.text} className="holodeck-asof__quote-text" />
        </blockquote>
      ) : (
        <p className="holodeck-timeline__quiet">No passage text is recorded for this source.</p>
      )}
    </>
  );
}

function capitalise(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}
