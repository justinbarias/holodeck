"use client";

import { useId, type MouseEvent } from "react";
import { ToolCallCard, type ToolCallStatus } from "@/components/ToolCallCard";
import { ClampedText } from "./ClampedText";
import { MemoryCardHeader } from "./MemoryCardHeader";
import { parseToolResult } from "./parse";
import {
  SUPERSEDED_NOTE,
  describeHitPeriod,
  describeInstants,
  describePassageStart,
  findSuperseding,
  groupHits,
  supersedingLinkText,
  summariseHits,
} from "./search-view";
import { fileName } from "./timeline-layout";
import type { MemorySearchHit, MemorySearchResult } from "./types";

export interface AsOfResultsProps {
  /** Tool name as it arrived. */
  name?: string;
  args?: Record<string, unknown>;
  status: ToolCallStatus;
  /** The tool result: object, JSON string, or anything else. */
  result?: unknown;
}

const TOOL_NAME = "memory_search";
const EYEBROW = "Memory search";

/**
 * Generative-UI view of a `memory_search` result. The header says which two
 * instants the answer is for; fact hits come first, then passage hits, each
 * group in the ranked order the tool returned.
 */
export function AsOfResults({ name, args, status, result }: AsOfResultsProps) {
  const query = typeof args?.query === "string" && args.query ? args.query : "Search";

  if (status !== "complete") {
    return (
      <section className="holodeck-tool-call holodeck-timeline" aria-busy="true">
        <MemoryCardHeader eyebrow={EYEBROW} title={query} status={status} />
        <p className="holodeck-timeline__status-line">Searching memory…</p>
      </section>
    );
  }

  const parsed = parseToolResult(result);

  if (parsed.kind === "error") {
    return (
      <section className="holodeck-tool-call holodeck-timeline">
        <MemoryCardHeader eyebrow={EYEBROW} title={query} status={status} failed />
        <div className="holodeck-timeline__error">
          <p>{parsed.message}</p>
          <span className="holodeck-timeline__code">{parsed.code}</span>
        </div>
      </section>
    );
  }

  if (parsed.kind !== "search") {
    return (
      <ToolCallCard name={name ?? TOOL_NAME} args={args} status={status} result={result} />
    );
  }

  return <SearchView query={query} search={parsed.result} />;
}

function SearchView({ query, search }: { query: string; search: MemorySearchResult }) {
  const groups = groupHits(search.hits);
  const instants = describeInstants(search.valid_at, search.known_at);

  return (
    <section
      className="holodeck-tool-call holodeck-timeline holodeck-asof"
      aria-label={`Memory search results for ${query}`}
    >
      <MemoryCardHeader
        eyebrow={EYEBROW}
        title={query}
        status="complete"
        summary={summariseHits(groups) || undefined}
      />

      <div className="holodeck-asof__instants">
        {instants.rows.map((row) => (
          <p key={row.label} className="holodeck-asof__instant">
            <span className="holodeck-asof__instant-label">{row.label}</span>{" "}
            <span className="holodeck-asof__instant-date">{row.value}</span>
          </p>
        ))}
        {instants.detail ? (
          <p className="holodeck-timeline__quiet holodeck-asof__instant-detail">
            {instants.detail}
          </p>
        ) : null}
      </div>

      {search.degraded ? (
        <div className="holodeck-asof__degraded" role="note">
          <strong className="holodeck-asof__degraded-label">Passages only</strong>
          <p>
            The fact graph was unavailable, so these are passages from documents,
            not facts. They carry no end dates, so any of them may no longer be true.
          </p>
        </div>
      ) : null}

      {groups.facts.length + groups.passages.length === 0 ? (
        <p className="holodeck-timeline__status-line">
          No facts or passages found for this date.
        </p>
      ) : null}

      {groups.facts.length > 0 ? (
        <HitList title="Facts" hits={groups.facts} kind="fact" />
      ) : null}
      {groups.passages.length > 0 ? (
        <HitList title="Passages from documents" hits={groups.passages} kind="chunk" />
      ) : null}
    </section>
  );
}

function HitList({
  title,
  hits,
  kind,
}: {
  title: string;
  hits: MemorySearchHit[];
  kind: "fact" | "chunk";
}) {
  const prefix = useId();
  const hitId = (index: number) => `${prefix}-hit-${index}`;
  return (
    <div className="holodeck-timeline__facts">
      <h4 className="holodeck-timeline__facts-title">{title}</h4>
      {kind === "chunk" ? (
        <p className="holodeck-timeline__quiet holodeck-timeline__note">
          A passage is document text, not a checked fact.
        </p>
      ) : null}
      <ol className="holodeck-asof__hits">
        {hits.map((hit, index) => (
          <li
            key={`${hit.chunk_id}:${hit.fact_id ?? ""}:${index}`}
            id={hitId(index)}
            tabIndex={kind === "fact" ? -1 : undefined}
            className="holodeck-asof__hit"
            data-kind={kind}
          >
            {kind === "fact" ? (
              <FactHit hit={hit} facts={hits} hitId={hitId} />
            ) : (
              <PassageHit hit={hit} />
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}

/** Move focus to the listed fact a link points at. */
function focusTarget(event: MouseEvent<HTMLAnchorElement>, id: string) {
  const target = document.getElementById(id);
  if (!target) return;
  event.preventDefault();
  target.focus({ preventScroll: true });
  target.scrollIntoView({ block: "nearest" });
}

function FactHit({
  hit,
  facts,
  hitId,
}: {
  hit: MemorySearchHit;
  facts: MemorySearchHit[];
  hitId: (index: number) => string;
}) {
  const superseding = findSuperseding(hit, facts);
  return (
    <>
      <p className="holodeck-asof__text">{hit.text}</p>
      <p className="holodeck-timeline__fact-meta">
        <span className="holodeck-timeline__tag holodeck-timeline__tag--open">Fact</span>
        <span>{capitalise(describeHitPeriod(hit))}</span>
        {superseding.kind !== "none" ? (
          <span className="holodeck-timeline__tag">{SUPERSEDED_NOTE}</span>
        ) : null}
        {superseding.kind === "listed" ? (
          <a
            className="holodeck-asof__link"
            href={`#${hitId(superseding.index)}`}
            onClick={(event) => focusTarget(event, hitId(superseding.index))}
          >
            Replaced by: {supersedingLinkText(facts[superseding.index])}
          </a>
        ) : null}
        {superseding.kind === "unlisted" ? (
          <span>
            Replaced by fact{" "}
            <code className="holodeck-timeline__file" title={superseding.id}>
              {superseding.shortId}
            </code>
            <span className="sr-only"> (full id {superseding.id})</span>; ask for
            its provenance to see it
          </span>
        ) : null}
      </p>
      <p className="holodeck-timeline__source">
        Source: <span className="holodeck-timeline__file">{fileName(hit.source_uri)}</span>
      </p>
    </>
  );
}

function PassageHit({ hit }: { hit: MemorySearchHit }) {
  const start = describePassageStart(hit);
  return (
    <>
      <ClampedText text={hit.text} className="holodeck-asof__text holodeck-asof__passage" />
      <p className="holodeck-timeline__fact-meta">
        <span className="holodeck-timeline__tag">Passage</span>
        <span>
          <span className="holodeck-timeline__file">{fileName(hit.source_uri)}</span>
        </span>
        {start ? <span>{start}</span> : null}
      </p>
    </>
  );
}

function capitalise(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}
