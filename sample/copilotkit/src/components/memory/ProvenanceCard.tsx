"use client";

import { Fragment, useState } from "react";
import { ToolCallCard, type ToolCallStatus } from "@/components/ToolCallCard";
import { ToolCardShell } from "@/components/tool-card/ToolCardShell";
import { ClampedText } from "./ClampedText";
import { chunkLabel, letterSources, shortPeriod } from "./card-view";
import { parseToolResult } from "./parse";
import { describeReferenceTimeSource, plural, shortFactId } from "./search-view";
import { MemoryErrorBody, SourceChip, memoryErrorHeadline } from "./shared";
import { FACT_STATUS_LABEL, factStatus, formatIsoDate } from "./timeline-layout";
import type { MemoryProvenanceResult } from "./types";

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

/**
 * Generative-UI view of a `memory_provenance` result. Collapsed: the fact
 * text. Expanded: when it holds, when it was recorded, and each source
 * document with its date and the passage it came from.
 */
export function ProvenanceCard({ name, args, status, result, now }: ProvenanceCardProps) {
  const toolName = name ?? TOOL_NAME;
  const factArg = typeof args?.fact_id === "string" ? args.fact_id : "";

  if (status !== "complete") {
    return (
      <ToolCardShell
        name={toolName}
        headline={factArg ? `Looking up fact ${shortFactId(factArg)}` : "Looking up the sources…"}
        tone="running"
        status="Running"
        expandable={false}
      />
    );
  }

  const parsed = parseToolResult(result);

  if (parsed.kind === "error") {
    return (
      <ToolCardShell
        name={toolName}
        headline={memoryErrorHeadline(parsed.message, args)}
        tone="failed"
        status="Failed"
        expandable
      >
        <MemoryErrorBody code={parsed.code} message={parsed.message} args={args} />
      </ToolCardShell>
    );
  }

  if (parsed.kind !== "provenance") {
    return <ToolCallCard name={toolName} args={args} status={status} result={result} />;
  }

  const provenance = parsed.result;
  if (!provenance.found || !provenance.fact) {
    return (
      <ToolCardShell
        name={toolName}
        headline={factArg ? `No fact with id ${shortFactId(factArg)}` : "This fact was not found"}
        tone="quiet"
        status="Not found"
        expandable
      >
        <p className="pb-card__message pb-card__pad">
          No fact with this id is in memory. It may have been purged or belong to another
          namespace.
          {factArg ? (
            <>
              {" "}
              <span className="pb-data">{factArg}</span>
            </>
          ) : null}
        </p>
      </ToolCardShell>
    );
  }

  return (
    <ToolCardShell
      name={toolName}
      headline={provenance.fact.text}
      tone="done"
      status={plural(provenance.sources.length, "source", "sources")}
      expandable
      label="Sources of a fact"
    >
      <ProvenanceBody provenance={provenance} now={now} />
    </ToolCardShell>
  );
}

function ProvenanceBody({
  provenance,
  now: nowProp,
}: {
  provenance: MemoryProvenanceResult;
  now?: number;
}) {
  const [now] = useState(() => nowProp ?? Date.now());
  const fact = provenance.fact;
  if (!fact) return null;
  const sources = provenance.sources;
  const letters = letterSources(sources.map((source) => source.source_uri));
  const holds =
    fact.valid_to === null && factStatus(fact, now) === "current"
      ? `${shortPeriod(fact.valid_from, null)} now`
      : shortPeriod(fact.valid_from, fact.valid_to);

  return (
    <>
      <p className="pb-prov__fact">{fact.text}</p>
      <dl className="pb-kv pb-kv--prose">
        <div className="pb-kv__row">
          <dt>Holds</dt>
          <dd>
            {holds}{" "}
            <span className="pb-quiet">· {FACT_STATUS_LABEL[factStatus(fact, now)]}</span>
          </dd>
        </div>
        {fact.recorded_at ? (
          <div className="pb-kv__row">
            <dt>Recorded</dt>
            <dd>{formatIsoDate(fact.recorded_at)}</dd>
          </div>
        ) : null}
        <div className="pb-kv__row">
          <dt>Fact id</dt>
          <dd className="pb-data">{fact.fact_id}</dd>
        </div>
        {sources.map((source) => (
          <Fragment key={source.episode_id + source.chunk_id}>
            <div className="pb-kv__row">
              <dt>Document</dt>
              <dd>
                <SourceChip letter={letters.get(source.source_uri) ?? "?"}>
                  <span className="pb-data">{source.source_uri}</span>
                </SourceChip>{" "}
                <span className="pb-quiet">{chunkLabel(source.chunk_id)}</span>
              </dd>
            </div>
            <div className="pb-kv__row">
              <dt>Dated</dt>
              <dd>
                {formatIsoDate(source.reference_time)},{" "}
                {describeReferenceTimeSource(source.reference_time_source)}
              </dd>
            </div>
            {source.text ? (
              <div className="pb-kv__quote">
                <blockquote className="pb-quote">
                  <ClampedText text={source.text} className="pb-quote__text" />
                </blockquote>
              </div>
            ) : null}
          </Fragment>
        ))}
      </dl>
      {sources.length === 0 ? (
        <p className="pb-card__message pb-card__pad">No source passage is recorded.</p>
      ) : null}
    </>
  );
}
