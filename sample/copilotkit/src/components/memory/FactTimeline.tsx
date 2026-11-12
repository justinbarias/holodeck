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
import { parseToolResult } from "./parse";
import {
  FACT_STATUS_LABEL,
  MAX_DRAWN_BARS,
  addUtcMonths,
  clampToAxis,
  describePeriod,
  factStatus,
  factsAt,
  fileName,
  formatIsoDate,
  formatUtcDate,
  fractionOf,
  layoutTimeline,
  shortFactLabel,
  startOfUtcDay,
  summariseTimeline,
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
const PLOT_MIN_WIDTH = 560;
/** Most rows any fact list renders; the rest are counted, not drawn. */
const MAX_LISTED = MAX_DRAWN_BARS;

/**
 * Generative-UI view of a `memory_timeline` result: one lane per attribute
 * (a chain of facts that replace each other, see `layoutTimeline`), one bar
 * per fact, and an "as of" slider that re-filters the facts in the browser.
 */
export function FactTimeline({ name, args, status, result, now }: FactTimelineProps) {
  const entityArg = typeof args?.entity === "string" ? args.entity : undefined;

  if (status !== "complete") {
    return (
      <section className="holodeck-tool-call holodeck-timeline" aria-busy="true">
        <TimelineHeader entity={entityArg} status={status} />
        <p className="holodeck-timeline__status-line">Reading the timeline…</p>
      </section>
    );
  }

  const parsed = parseToolResult(result);

  if (parsed.kind === "error") {
    return (
      <section className="holodeck-tool-call holodeck-timeline">
        <TimelineHeader entity={entityArg} status={status} failed />
        <div className="holodeck-timeline__error">
          <p>{parsed.message}</p>
          <span className="holodeck-timeline__code">{parsed.code}</span>
        </div>
      </section>
    );
  }

  if (parsed.kind !== "timeline") {
    return (
      <ToolCallCard name={name ?? TOOL_NAME} args={args} status={status} result={result} />
    );
  }

  return <TimelineView timeline={parsed.result} now={now} />;
}

function TimelineHeader({
  entity,
  status,
  failed = false,
  summary,
}: {
  entity?: string;
  status: ToolCallStatus;
  failed?: boolean;
  summary?: string;
}) {
  const complete = status === "complete";
  return (
    <header className="holodeck-timeline__header">
      <div className="holodeck-timeline__heading">
        <span className="holodeck-timeline__eyebrow">Memory timeline</span>
        <span className="holodeck-timeline__entity">{entity ?? "Entity"}</span>
      </div>
      {summary ? <span className="holodeck-timeline__summary">{summary}</span> : null}
      <span
        className={
          failed
            ? "holodeck-tool-badge holodeck-timeline__badge--failed"
            : complete
              ? "holodeck-tool-badge holodeck-tool-badge--complete"
              : "holodeck-tool-badge holodeck-tool-badge--progress"
        }
      >
        {failed ? "Failed" : complete ? "Complete" : "Running…"}
      </span>
    </header>
  );
}

function plural(count: number, one: string, many: string): string {
  return `${count} ${count === 1 ? one : many}`;
}

function TimelineView({
  timeline,
  now: nowProp,
}: {
  timeline: MemoryTimelineResult;
  now?: number;
}) {
  const [today] = useState(() => nowProp ?? startOfUtcDay(Date.now()));
  const { entries, entity } = timeline;
  const layout = useMemo(() => layoutTimeline(entries, today), [entries, today]);
  const { axis, lanes, undated, omitted } = layout;

  const [asOfState, setAsOf] = useState<number | null>(null);
  // The axis always contains `today`, so the initial as-of instant is today
  // even when every fact starts later.
  const asOf = axis ? clampToAxis(axis, asOfState ?? today) : (asOfState ?? today);
  const holding = useMemo(() => factsAt(entries, asOf), [entries, asOf]);
  const holdingIds = useMemo(
    () => new Set(holding.map((entry) => entry.fact_id)),
    [holding],
  );
  const [inspected, setInspected] = useState<string | null>(null);

  const dated = useMemo(
    () =>
      lanes
        .flat()
        .sort((a, b) => a.start - b.start || a.lane - b.lane)
        .map((bar) => bar.entry),
    [lanes],
  );
  const summary = useMemo(() => summariseTimeline(entries, today), [entries, today]);

  if (entries.length === 0) {
    return (
      <section className="holodeck-tool-call holodeck-timeline">
        <TimelineHeader entity={entity} status="complete" />
        <p className="holodeck-timeline__status-line">No facts found for {entity}.</p>
      </section>
    );
  }

  const inspectedEntry = entries.find((entry) => entry.fact_id === inspected);

  return (
    <section
      className="holodeck-tool-call holodeck-timeline"
      aria-label={`Timeline of facts about ${entity}`}
    >
      <TimelineHeader
        entity={entity}
        status="complete"
        summary={summary}
      />

      <div className="holodeck-timeline__asof">
        <p className="holodeck-timeline__asof-line" aria-live="polite">
          <span className="holodeck-timeline__asof-date">As of {formatUtcDate(asOf)}:</span>{" "}
          {holding.length === 0
            ? "no facts hold"
            : `${plural(holding.length, "fact holds", "facts hold")}`}
        </p>
        {holding.length > 0 ? (
          <ul className="holodeck-timeline__asof-list">
            {holding.slice(0, MAX_LISTED).map((entry) => (
              <li key={entry.fact_id}>
                {entry.text}
                {entry.valid_from === null ? (
                  <span className="holodeck-timeline__quiet"> (start date not known)</span>
                ) : null}
              </li>
            ))}
          </ul>
        ) : null}
        <MoreNote count={holding.length - MAX_LISTED} />
      </div>

      {axis ? (
        <>
          <TimelineChart
            axis={axis}
            entity={entity}
            lanes={lanes}
            asOf={asOf}
            today={today}
            holdingIds={holdingIds}
            onAsOfChange={setAsOf}
            onInspect={setInspected}
          />
          <p className="holodeck-timeline__inspect" aria-hidden="true">
            {inspectedEntry ? (
              <>
                <span className="holodeck-timeline__inspect-text">{inspectedEntry.text}</span>{" "}
                <span className="holodeck-timeline__quiet">
                  {describePeriod(inspectedEntry, today)}
                </span>
              </>
            ) : (
              <span className="holodeck-timeline__quiet">
                Drag the date handle, or focus it and use the arrow keys (Shift for a
                year). Select a bar to move the date to the start of that fact.
              </span>
            )}
          </p>
          <Legend />
          {omitted > 0 ? (
            <p className="holodeck-timeline__quiet holodeck-timeline__note">
              {plural(omitted, "earlier fact is", "earlier facts are")} not drawn: the
              chart shows the {MAX_DRAWN_BARS} facts with the latest start dates.
            </p>
          ) : null}
        </>
      ) : null}

      <FactList title="All facts" entries={dated} holdingIds={holdingIds} now={today} />
      {undated.length > 0 ? (
        <FactList
          title="Start date not known"
          note="These facts have no known start date, so they are not drawn on the chart."
          entries={undated}
          holdingIds={holdingIds}
          now={today}
        />
      ) : null}
    </section>
  );
}

function Legend() {
  return (
    <ul className="holodeck-timeline__legend" aria-label="Key">
      <li>
        <span className="holodeck-timeline__swatch holodeck-timeline__swatch--open" aria-hidden="true" />
        No end date (solid edge)
      </li>
      <li>
        <span className="holodeck-timeline__swatch holodeck-timeline__swatch--closed" aria-hidden="true" />
        Ended (dashed edge)
      </li>
      <li>
        <span className="holodeck-timeline__swatch holodeck-timeline__swatch--marker" aria-hidden="true" />
        As-of date
      </li>
    </ul>
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
  onInspect,
}: {
  axis: TimeAxis;
  entity: string;
  lanes: TimelineBar[][];
  asOf: number;
  today: number;
  holdingIds: Set<string>;
  onAsOfChange: (t: number) => void;
  onInspect: (factId: string | null) => void;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const thumbRef = useRef<HTMLDivElement>(null);
  const hintId = useId();

  const asOfFraction = fractionOf(axis, asOf);

  // When the plot is wider than the card, keep the as-of marker in view.
  // On first render this scrolls to "today", where the open facts are.
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
    // `buttons` guards against a release that happened outside the track.
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
  const laneHeight = 34;

  return (
    <div className="holodeck-timeline__scroll" ref={scrollRef}>
      <div className="holodeck-timeline__plot" style={{ minWidth: PLOT_MIN_WIDTH }}>
        <div
          className="holodeck-timeline__lanes"
          style={{ height: lanes.length * laneHeight }}
        >
          {axis.ticks.map((tick) => (
            <span
              key={tick.time}
              className="holodeck-timeline__grid"
              style={{ left: pct(fractionOf(axis, tick.time)) }}
              aria-hidden="true"
            />
          ))}
          {showToday ? (
            <span
              className="holodeck-timeline__today-line"
              style={{ left: pct(todayFraction) }}
              aria-hidden="true"
            />
          ) : null}
          <ol className="holodeck-timeline__lane-list" aria-label="Facts on the time axis">
            {lanes.map((lane, index) => (
              <li
                key={index}
                className="holodeck-timeline__lane"
                style={{ top: index * laneHeight, height: laneHeight }}
              >
                <ul aria-label={`Lane ${index + 1}`}>
                  {lane.map((bar) => {
                    const left = fractionOf(axis, bar.start);
                    const width = fractionOf(axis, bar.end) - left;
                    const holds = holdingIds.has(bar.entry.fact_id);
                    return (
                      <li key={bar.entry.fact_id}>
                        <button
                          type="button"
                          className="holodeck-timeline__bar"
                          data-open={bar.open ? "true" : "false"}
                          data-holds={holds ? "true" : "false"}
                          style={{ left: pct(left), width: pct(width) }}
                          aria-label={`${bar.entry.text} Valid ${describePeriod(bar.entry, today)}.${
                            holds ? ` Holds on ${formatUtcDate(asOf)}.` : ""
                          }`}
                          aria-describedby={hintId}
                          title={bar.entry.text}
                          onClick={() => onAsOfChange(bar.start)}
                          onMouseEnter={() => onInspect(bar.entry.fact_id)}
                          onMouseLeave={() => onInspect(null)}
                          onFocus={() => onInspect(bar.entry.fact_id)}
                          onBlur={() => onInspect(null)}
                        >
                          <span className="holodeck-timeline__bar-label">
                            {shortFactLabel(bar.entry.text, entity)}
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </li>
            ))}
          </ol>
          <span
            className="holodeck-timeline__marker"
            style={{ left: pct(asOfFraction) }}
            aria-hidden="true"
          />
        </div>

        <div
          className="holodeck-timeline__track"
          ref={trackRef}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
          onLostPointerCapture={endDrag}
        >
          <span className="holodeck-timeline__baseline" aria-hidden="true" />
          {axis.ticks.map((tick) => (
            <span
              key={tick.time}
              className="holodeck-timeline__tick"
              style={{ left: pct(fractionOf(axis, tick.time)) }}
              aria-hidden="true"
            />
          ))}
          <div
            ref={thumbRef}
            className="holodeck-timeline__thumb"
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

        <div className="holodeck-timeline__labels" aria-hidden="true">
          {axis.ticks
            .filter(
              (tick) =>
                !showToday || Math.abs(fractionOf(axis, tick.time) - todayFraction) > 0.07,
            )
            .map((tick) => (
              <span
                key={tick.time}
                className="holodeck-timeline__label"
                style={{ left: pct(fractionOf(axis, tick.time)) }}
              >
                {tick.label}
              </span>
            ))}
          {showToday ? (
            <span
              className="holodeck-timeline__label holodeck-timeline__label--today"
              style={{ left: pct(todayFraction) }}
            >
              Today
            </span>
          ) : null}
        </div>
        <span id={hintId} hidden>
          Select to move the as-of date to the start of this fact.
        </span>
      </div>
    </div>
  );
}

function MoreNote({ count }: { count: number }) {
  if (count <= 0) return null;
  return (
    <p className="holodeck-timeline__quiet holodeck-timeline__note">
      {plural(count, "more fact is", "more facts are")} not listed.
    </p>
  );
}

function FactList({
  title,
  note,
  entries,
  holdingIds,
  now,
}: {
  title: string;
  note?: string;
  entries: TimelineEntry[];
  holdingIds: Set<string>;
  now: number;
}) {
  if (entries.length === 0) return null;
  return (
    <div className="holodeck-timeline__facts">
      <h4 className="holodeck-timeline__facts-title">{title}</h4>
      {note ? <p className="holodeck-timeline__quiet holodeck-timeline__note">{note}</p> : null}
      <ul>
        {entries.slice(0, MAX_LISTED).map((entry) => {
          const status = factStatus(entry, now);
          const open = status === "current";
          const holds = holdingIds.has(entry.fact_id);
          return (
            <li
              key={entry.fact_id}
              className="holodeck-timeline__fact"
              data-open={open ? "true" : "false"}
              data-holds={holds ? "true" : "false"}
            >
              <p className="holodeck-timeline__fact-text">{entry.text}</p>
              <p className="holodeck-timeline__fact-meta">
                <span
                  className={
                    open
                      ? "holodeck-timeline__tag holodeck-timeline__tag--open"
                      : "holodeck-timeline__tag"
                  }
                >
                  {FACT_STATUS_LABEL[status]}
                </span>
                <span>{capitalise(describePeriod(entry, now))}</span>
              </p>
              {entry.sources.map((source) => (
                <p key={source.episode_id + source.chunk_id} className="holodeck-timeline__source">
                  Source: <span className="holodeck-timeline__file">{fileName(source.source_uri)}</span>
                  {", document dated "}
                  {formatIsoDate(source.reference_time)}
                </p>
              ))}
            </li>
          );
        })}
      </ul>
      <MoreNote count={entries.length - MAX_LISTED} />
    </div>
  );
}

function capitalise(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}
