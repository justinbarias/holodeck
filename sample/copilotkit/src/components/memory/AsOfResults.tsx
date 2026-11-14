"use client";

import { useId, type MouseEvent } from "react";
import { ToolCallCard, type ToolCallStatus } from "@/components/ToolCallCard";
import { WarnIcon } from "@/components/Icons";
import { ToolCardShell } from "@/components/tool-card/ToolCardShell";
import { ClampedText } from "./ClampedText";
import { chunkLabel, letterSources, searchHeadline, shortDate, shortPeriod } from "./card-view";
import { parseToolResult } from "./parse";
import {
  describePassageStart,
  findSuperseding,
  groupHits,
  supersedingLinkText,
} from "./search-view";
import { MemoryErrorBody, SourceChip, memoryErrorHeadline } from "./shared";
import { formatIsoDate, isUtcMidnight, toTime, fractionDigits } from "./timeline-layout";
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

/**
 * Generative-UI view of a `memory_search` result. Collapsed: "N hits for
 * <query>". Expanded: which two instants the answer is for, then fact hits
 * and passage hits in the ranked order the tool returned.
 */
export function AsOfResults({ name, args, status, result }: AsOfResultsProps) {
  const toolName = name ?? TOOL_NAME;
  const query = typeof args?.query === "string" ? args.query : "";

  if (status !== "complete") {
    return (
      <ToolCardShell
        name={toolName}
        headline={query ? `Searching for “${query}”` : "Searching memory…"}
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

  if (parsed.kind !== "search") {
    return <ToolCallCard name={toolName} args={args} status={status} result={result} />;
  }

  const search = parsed.result;
  const groups = groupHits(search.hits);
  const empty = groups.facts.length + groups.passages.length === 0;
  return (
    <ToolCardShell
      name={toolName}
      headline={searchHeadline(search, query)}
      tone={empty ? "empty" : search.degraded ? "warn" : "done"}
      status={empty ? "No hits" : search.degraded ? "Degraded" : "Done"}
      expandable
      label={`Memory search results${query ? ` for ${query}` : ""}`}
    >
      <SearchBody search={search} />
    </ToolCardShell>
  );
}

/** "11 Nov 2026" at midnight, otherwise "11 November 2026, 12:00:00 UTC". */
function instantText(iso: string): string {
  return isUtcMidnight(toTime(iso), fractionDigits(iso)) ? shortDate(iso) : formatIsoDate(iso);
}

function SearchBody({ search }: { search: MemorySearchResult }) {
  const groups = groupHits(search.hits);
  const letters = letterSources([...groups.facts, ...groups.passages].map((hit) => hit.source_uri));
  const prefix = useId();
  const hitId = (index: number) => `${prefix}-hit-${index}`;
  const empty = groups.facts.length + groups.passages.length === 0;

  return (
    <>
      <p className="pb-instants">
        <span>
          Facts that held on <b>{instantText(search.valid_at)}</b>
        </span>{" "}
        <span>
          using what was recorded by <b>{instantText(search.known_at)}</b>
        </span>
      </p>

      {search.degraded ? (
        <div className="pb-banner" data-tone="warn" role="note">
          <WarnIcon size={16} />
          <p>
            <b>Keyword passages only.</b> The fact graph did not answer, so these results
            are document passages without end dates. Treat them as unverified for{" "}
            {shortDate(search.valid_at)}.
          </p>
        </div>
      ) : null}

      {empty ? (
        <p className="pb-card__message pb-card__pad">
          Nothing held on {shortDate(search.valid_at)} for this query.
        </p>
      ) : (
        <ol className="pb-hits">
          {groups.facts.map((hit, index) => (
            <li key={`f:${hit.fact_id ?? ""}:${index}`} id={hitId(index)} tabIndex={-1} className="pb-hit">
              <FactHit
                hit={hit}
                facts={groups.facts}
                letter={letters.get(hit.source_uri) ?? "?"}
                hitId={hitId}
              />
            </li>
          ))}
          {groups.passages.map((hit, index) => (
            <li key={`p:${hit.chunk_id}:${index}`} className="pb-hit">
              <PassageHit hit={hit} letter={letters.get(hit.source_uri) ?? "?"} />
            </li>
          ))}
        </ol>
      )}
    </>
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

function Score({ score }: { score: number }) {
  return (
    <span className="pb-hit__score pb-data" title="Ranking score from the search">
      {score.toFixed(3)}
    </span>
  );
}

function FactHit({
  hit,
  facts,
  letter,
  hitId,
}: {
  hit: MemorySearchHit;
  facts: MemorySearchHit[];
  letter: string;
  hitId: (index: number) => string;
}) {
  const superseding = findSuperseding(hit, facts);
  const when = hit.valid_to ? `from ${shortDate(hit.valid_to)} ` : "";
  return (
    <>
      <span className="pb-hit__meta">
        <span className="pb-hit__kind">Fact</span>
        <SourceChip letter={letter} title={hit.source_uri}>
          {shortPeriod(hit.valid_from, hit.valid_to)}
        </SourceChip>
        <Score score={hit.score} />
      </span>
      <p className="pb-hit__text">{hit.text}</p>
      {superseding.kind === "listed" ? (
        <p className="pb-hit__note">
          Replaced {when}by{" "}
          <a
            className="pb-inline-link"
            href={`#${hitId(superseding.index)}`}
            onClick={(event) => focusTarget(event, hitId(superseding.index))}
          >
            {supersedingLinkText(facts[superseding.index])}
          </a>
        </p>
      ) : null}
      {superseding.kind === "unlisted" ? (
        <p className="pb-hit__note">
          Replaced {when}by fact{" "}
          <code className="pb-data" title={superseding.id}>
            {superseding.shortId}
          </code>
          <span className="sr-only"> (full id {superseding.id})</span>; ask for its provenance to see it.
        </p>
      ) : null}
      <p className="pb-hit__source">
        <span className="pb-data">{hit.source_uri}</span>
      </p>
    </>
  );
}

function PassageHit({ hit, letter }: { hit: MemorySearchHit; letter: string }) {
  return (
    <>
      <span className="pb-hit__meta">
        <span className="pb-hit__kind">Passage</span>
        <SourceChip letter={letter} title={hit.source_uri}>
          <span className="pb-data">
            {hit.source_uri} · {chunkLabel(hit.chunk_id)}
          </span>
        </SourceChip>
        <Score score={hit.score} />
      </span>
      <ClampedText text={hit.text} className="pb-hit__text" />
      {describePassageStart(hit) ? (
        <p className="pb-hit__note">{describePassageStart(hit)}. A passage is document text, not a checked fact.</p>
      ) : (
        <p className="pb-hit__note">A passage is document text, not a checked fact.</p>
      )}
    </>
  );
}
