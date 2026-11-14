"use client";

import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
} from "react";
import { ToolCallCard, type ToolCallStatus } from "@/components/ToolCallCard";
import { ChevronIcon } from "@/components/Icons";
import { useDisclosure, useToolCallInfo } from "@/components/tool-card/context";
import { ToolCardShell } from "@/components/tool-card/ToolCardShell";
import {
  closedStamp,
  isRetracted,
  letterSources,
  shortDate,
  shortDateAt,
  shortPeriod,
  splitAsOf,
  timelineHeadline,
} from "./card-view";
import { parseToolResult } from "./parse";
import { describeReferenceTimeSource, shortFactId } from "./search-view";
import { MemoryErrorBody, SourceChip, Stamp, memoryErrorHeadline } from "./shared";
import {
  MAX_DRAWN_BARS,
  addUtcMonths,
  clampToAxis,
  describePeriod,
  formatIsoDate,
  formatUtcDate,
  fractionOf,
  layoutTimeline,
  shortFactLabel,
  startOfUtcDay,
  timeAtFraction,
  type TimeAxis,
  type TimelineBar,
} from "./timeline-layout";
import type { MemoryTimelineResult, TimelineEntry } from "./types";

export interface FactTimelineProps {
  /** Tool name as it arrived. */
  name?: string;
  args?: Record<string, unknown>;
  status: ToolCallStatus;
  /** The tool result: object, JSON string, or anything else. */
  result?: unknown;
  /** "Today" in epoch ms. Defaults to the start of the current UTC day. */
  now?: number;
}

const TOOL_NAME = "memory_timeline";
/** Narrowest width the plot draws at; below this it scrolls sideways. */
const PLOT_MIN_WIDTH = 520;
/** Most rows any fact list renders; the rest are counted, not drawn. */
const MAX_LISTED = 50;
/** Closed facts listed before "Show n more closed". */
const CLOSED_PREVIEW = 5;

/**
 * Generative-UI view of a `memory_timeline` result. Collapsed: "N facts · M
 * current" and the latest current fact. Expanded: one lane per attribute on a
 * valid-time axis with an as-of handle, the facts that hold on that date, and
 * the closed facts, struck and stamped but never hidden.
 */
export function FactTimeline({ name, args, status, result, now }: FactTimelineProps) {
  const toolName = name ?? TOOL_NAME;
  const entityArg = typeof args?.entity === "string" ? args.entity : "";
  const [today] = useState(() => now ?? startOfUtcDay(Date.now()));

  if (status !== "complete") {
    return (
      <ToolCardShell
        name={toolName}
        headline={entityArg ? `Reading the timeline of ${entityArg}` : "Reading the timeline…"}
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

  if (parsed.kind !== "timeline") {
    return <ToolCallCard name={toolName} args={args} status={status} result={result} />;
  }

  const timeline = parsed.result;
  if (timeline.entries.length === 0) {
    return (
      <ToolCardShell
        name={toolName}
        headline={`No facts recorded for ${timeline.entity}`}
        tone="empty"
        status="No hits"
        expandable
      >
        <p className="pb-card__message pb-card__pad">
          Memory has no facts about {timeline.entity}. Check the spelling of the entity, or
          whether its documents were ingested.
        </p>
      </ToolCardShell>
    );
  }

  const head = timelineHeadline(timeline.entries, today);
  return (
    <ToolCardShell
      name={toolName}
      headline={
        <>
          <span className="pb-card__count">{head.count}</span>
          {head.latest ? <span className="pb-card__latest"> · {head.latest}</span> : null}
        </>
      }
      tone="done"
      status="Done"
      expandable
      label={`Timeline of facts about ${timeline.entity}`}
    >
      <TimelineBody timeline={timeline} today={today} />
    </ToolCardShell>
  );
}

function TimelineBody({ timeline, today }: { timeline: MemoryTimelineResult; today: number }) {
  const { entries, entity } = timeline;
  const layout = useMemo(() => layoutTimeline(entries, today), [entries, today]);
  const { axis, lanes, undated, omitted } = layout;

  const [asOfState, setAsOf] = useState<number | null>(null);
  // The axis always contains `today`, so the initial as-of instant is today.
  const asOf = axis ? clampToAxis(axis, asOfState ?? today) : (asOfState ?? today);

  // Display order: dated facts by start, then undated. Letters follow it.
  const ordered = useMemo(
    () => [
      ...lanes
        .flat()
        .sort((a, b) => a.start - b.start || a.lane - b.lane)
        .map((bar) => bar.entry),
      ...undated,
    ],
    [lanes, undated],
  );
  const letters = useMemo(
    () => letterSources(ordered.flatMap((entry) => entry.sources.map((s) => s.source_uri))),
    [ordered],
  );
  const split = useMemo(() => splitAsOf(ordered, asOf), [ordered, asOf]);
  const holdingIds = useMemo(
    () => new Set(split.holding.map((entry) => entry.fact_id)),
    [split],
  );
  const [showAllClosed, setShowAllClosed] = useState(false);
  const info = useToolCallInfo();
  const [chartOpen, setChartOpen] = useDisclosure(info.key ? `${info.key}:chart` : undefined);
  const chartId = useId();

  return (
    <>
      <h4 className="pb-list-title pb-list-title--first" aria-live="polite">
        As of {formatUtcDate(asOf)} · {split.holding.length}{" "}
        {split.holding.length === 1 ? "fact holds" : "facts hold"}
      </h4>
      <FactList entries={split.holding} letters={letters} today={today} />

      {split.future.length > 0 ? (
        <>
          <h4 className="pb-list-title pb-list-title--quiet">
            Not yet in force on {shortDateAt(asOf)} · {split.future.length}
          </h4>
          <FactList entries={split.future} letters={letters} today={today} />
        </>
      ) : null}

      {split.closed.length > 0 ? (
        <>
          <h4 className="pb-list-title pb-list-title--quiet">Closed · {split.closed.length}</h4>
          <FactList
            entries={showAllClosed ? split.closed : split.closed.slice(0, CLOSED_PREVIEW)}
            letters={letters}
            today={today}
            closed
          />
          {split.closed.length > CLOSED_PREVIEW ? (
            <button
              type="button"
              className="pb-link pb-list-more"
              aria-expanded={showAllClosed}
              onClick={() => setShowAllClosed((value) => !value)}
            >
              {showAllClosed
                ? "Show fewer closed"
                : `Show ${split.closed.length - CLOSED_PREVIEW} more closed`}
            </button>
          ) : null}
        </>
      ) : null}

      {letters.size > 0 ? (
        <ul className="pb-sourcekey" aria-label="Source documents">
          {[...letters.entries()].map(([uri, letter]) => (
            <li key={uri}>
              <SourceChip letter={letter} title={uri}>
                <span className="pb-data">{uri}</span>
              </SourceChip>
            </li>
          ))}
        </ul>
      ) : null}

      {axis ? (
        <div className="pb-chart-toggle">
          <button
            type="button"
            className="pb-link"
            aria-expanded={chartOpen}
            aria-controls={chartId}
            onClick={() => setChartOpen(!chartOpen)}
          >
            {chartOpen ? "Hide timeline chart" : "Show timeline chart"}
            <ChevronIcon size={14} className="pb-chart-toggle__chevron" />
          </button>
        </div>
      ) : null}
      {axis && chartOpen ? (
        <div id={chartId}>
          <TimelineChart
            axis={axis}
            entity={entity}
            lanes={lanes}
            asOf={asOf}
            today={today}
            holdingIds={holdingIds}
            onAsOfChange={setAsOf}
          />
          {omitted > 0 ? (
            <p className="pb-card__detail pb-card__pad">
              {omitted} earlier {omitted === 1 ? "fact is" : "facts are"} not drawn: the chart
              shows the {MAX_DRAWN_BARS} facts with the latest start dates.
            </p>
          ) : null}
        </div>
      ) : null}
    </>
  );
}

const pct = (fraction: number) => `${(fraction * 100).toFixed(3)}%`;

function TimelineChart({
  axis,
  entity,
  lanes,
  asOf,
  today,
  holdingIds,
  onAsOfChange,
}: {
  axis: TimeAxis;
  entity: string;
  lanes: TimelineBar[][];
  asOf: number;
  today: number;
  holdingIds: Set<string>;
  onAsOfChange: (t: number) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const thumbRef = useRef<HTMLDivElement>(null);
  const hintId = useId();

  const asOfFraction = fractionOf(axis, asOf);

  // When the plot is wider than the card, keep the as-of marker in view.
  useEffect(() => {
    const el = scrollRef.current;
    const track = trackRef.current;
    if (!el || !track || el.scrollWidth <= el.clientWidth) return;
    const x = track.offsetLeft + asOfFraction * track.offsetWidth;
    const margin = 24;
    if (x < el.scrollLeft + margin || x > el.scrollLeft + el.clientWidth - margin) {
      el.scrollLeft = x - el.clientWidth / 2;
    }
  }, [asOfFraction]);

  const setFromClientX = useCallback(
    (clientX: number) => {
      const track = trackRef.current;
      if (!track) return;
      const rect = track.getBoundingClientRect();
      if (rect.width <= 0) return;
      onAsOfChange(timeAtFraction(axis, (clientX - rect.left) / rect.width));
    },
    [axis, onAsOfChange],
  );

  const dragging = useRef(false);
  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    dragging.current = true;
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
      // Capture is an aid only; the dragging flag drives the moves.
    }
    setFromClientX(event.clientX);
    thumbRef.current?.focus({ preventScroll: true });
    event.preventDefault();
  };
  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (dragging.current && (event.buttons & 1) === 1) setFromClientX(event.clientX);
    else dragging.current = false;
  };
  const endDrag = () => {
    dragging.current = false;
  };

  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const months = event.shiftKey ? 12 : 1;
    let next: number | null = null;
    switch (event.key) {
      case "ArrowRight":
      case "ArrowUp":
        next = addUtcMonths(asOf, months);
        break;
      case "ArrowLeft":
      case "ArrowDown":
        next = addUtcMonths(asOf, -months);
        break;
      case "PageUp":
        next = addUtcMonths(asOf, 12);
        break;
      case "PageDown":
        next = addUtcMonths(asOf, -12);
        break;
      case "Home":
        next = axis.start;
        break;
      case "End":
        next = axis.end;
        break;
      default:
        return;
    }
    event.preventDefault();
    onAsOfChange(clampToAxis(axis, next));
  };

  const todayFraction = fractionOf(axis, today);
  const showToday = today >= axis.start && today <= axis.end;

  return (
    <div className="pb-tl" ref={scrollRef}>
      <div className="pb-tl__plot" style={{ minWidth: PLOT_MIN_WIDTH }}>
        <div className="pb-tl__axis" aria-hidden="true">
          <span />
          <div className="pb-tl__years">
            {axis.ticks.map((tick) => (
              <span key={tick.time} style={{ left: pct(fractionOf(axis, tick.time)) }}>
                {tick.label}
              </span>
            ))}
          </div>
        </div>
        <ol className="pb-tl__lanes" aria-label="Facts on the valid-time axis">
          {lanes.map((lane, index) => {
            const latest = lane[lane.length - 1];
            return (
              <li key={index} className="pb-tl__lane">
                <span className="pb-tl__lane-label" title={latest.entry.text}>
                  {shortFactLabel(latest.entry.text, entity)}
                </span>
                <span className="pb-tl__lane-track">
                  {axis.ticks.map((tick) => (
                    <span
                      key={tick.time}
                      className="pb-tl__grid"
                      style={{ left: pct(fractionOf(axis, tick.time)) }}
                      aria-hidden="true"
                    />
                  ))}
                  {lane.map((bar) => {
                    const left = fractionOf(axis, bar.start);
                    const width = fractionOf(axis, bar.end) - left;
                    const holds = holdingIds.has(bar.entry.fact_id);
                    const closed = !bar.open || isRetracted(bar.entry);
                    return (
                      <button
                        key={bar.entry.fact_id}
                        type="button"
                        className="pb-tl__bar"
                        data-closed={closed ? "true" : "false"}
                        data-holds={holds ? "true" : "false"}
                        style={{ left: pct(left), width: pct(width) }}
                        aria-label={`${bar.entry.text} Valid ${describePeriod(bar.entry, today)}.${
                          holds ? ` Holds on ${formatUtcDate(asOf)}.` : ""
                        }`}
                        aria-describedby={hintId}
                        title={bar.entry.text}
                        onClick={() => onAsOfChange(bar.start)}
                      >
                        {!bar.open ? <span className="pb-tl__end" aria-hidden="true" /> : null}
                      </button>
                    );
                  })}
                  <span
                    className="pb-tl__marker"
                    style={{ left: pct(asOfFraction) }}
                    aria-hidden="true"
                  />
                </span>
              </li>
            );
          })}
        </ol>

        <div className="pb-tl__axis">
          <span className="pb-tl__asof-label">As of</span>
          <div
            className="pb-tl__track"
            ref={trackRef}
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={endDrag}
            onPointerCancel={endDrag}
            onLostPointerCapture={endDrag}
          >
            <span className="pb-tl__baseline" aria-hidden="true" />
            {showToday ? (
              <span
                className="pb-tl__today"
                style={{ left: pct(todayFraction) }}
                aria-hidden="true"
                title="Today"
              />
            ) : null}
            <div
              ref={thumbRef}
              className="pb-tl__thumb"
              style={{ left: pct(asOfFraction) }}
              role="slider"
              tabIndex={0}
              aria-label="As-of date"
              aria-orientation="horizontal"
              aria-valuemin={axis.start}
              aria-valuemax={axis.end}
              aria-valuenow={asOf}
              aria-valuetext={formatUtcDate(asOf)}
              onKeyDown={onKeyDown}
            />
          </div>
        </div>
        <p className="pb-tl__hint">
          Drag the as-of handle, or focus it and use the arrow keys (Shift for a year). Select a
          bar to move the date to the start of that fact.
        </p>
        <span id={hintId} hidden>
          Select to move the as-of date to the start of this fact.
        </span>
      </div>
    </div>
  );
}

function FactList({
  entries,
  letters,
  today,
  closed = false,
}: {
  entries: TimelineEntry[];
  letters: Map<string, string>;
  today: number;
  closed?: boolean;
}) {
  if (entries.length === 0) {
    return <p className="pb-card__message pb-card__pad pb-quiet">None.</p>;
  }
  const more = entries.length - MAX_LISTED;
  return (
    <>
      <ul className="pb-facts">
        {entries.slice(0, MAX_LISTED).map((entry) => (
          <FactRow
            key={entry.fact_id}
            entry={entry}
            letters={letters}
            today={today}
            closed={closed}
          />
        ))}
      </ul>
      {more > 0 ? (
        <p className="pb-card__detail pb-card__pad">
          {more} more {more === 1 ? "fact is" : "facts are"} not listed.
        </p>
      ) : null}
    </>
  );
}

function FactRow({
  entry,
  letters,
  today,
  closed,
}: {
  entry: TimelineEntry;
  letters: Map<string, string>;
  today: number;
  closed: boolean;
}) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const letter = entry.sources[0] ? letters.get(entry.sources[0].source_uri) : undefined;
  const stamp = closed ? closedStamp(entry) : null;
  const period = closed
    ? entry.valid_from
      ? shortDate(entry.valid_from)
      : "No start"
    : shortPeriod(entry.valid_from, entry.valid_to);

  return (
    <li className="pb-fact" data-closed={closed ? "true" : "false"}>
      <span className="pb-fact__when">
        {letter ? (
          <SourceChip letter={letter}>{period}</SourceChip>
        ) : (
          <span className="pb-quiet">{period}</span>
        )}
      </span>
      <span className="pb-fact__text">
        {closed ? <s>{entry.text}</s> : entry.text}
        {stamp ? (
          <>
            {" "}
            <Stamp>{stamp}</Stamp>
          </>
        ) : null}
        {closed && isRetracted(entry) && entry.expired_at ? (
          <span className="pb-fact__note">
            Known until {shortDate(entry.expired_at)}; memory later recorded that it no longer
            holds.
          </span>
        ) : null}
        {!closed && entry.valid_from === null ? (
          <span className="pb-fact__note">Start date not known.</span>
        ) : null}
      </span>
      <button
        type="button"
        className="pb-link pb-link--tight pb-fact__action"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen((value) => !value)}
      >
        Provenance
      </button>
      {open ? (
        <dl id={panelId} className="pb-fact__prov">
          <div>
            <dt>Holds</dt>
            <dd>{describePeriod(entry, today)}</dd>
          </div>
          {entry.recorded_at ? (
            <div>
              <dt>Recorded</dt>
              <dd>{formatIsoDate(entry.recorded_at)}</dd>
            </div>
          ) : null}
          {entry.sources.map((source) => (
            <div key={source.episode_id + source.chunk_id}>
              <dt>Document</dt>
              <dd>
                <span className="pb-data">{source.source_uri}</span>, dated{" "}
                {formatIsoDate(source.reference_time)} (
                {describeReferenceTimeSource(source.reference_time_source)})
              </dd>
            </div>
          ))}
          <div>
            <dt>Fact id</dt>
            <dd className="pb-data" title={entry.fact_id}>
              {shortFactId(entry.fact_id)}
            </dd>
          </div>
        </dl>
      ) : (
        <div id={panelId} hidden />
      )}
    </li>
  );
}
