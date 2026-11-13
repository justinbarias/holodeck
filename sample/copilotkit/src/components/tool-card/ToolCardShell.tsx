"use client";

import { useId, type ReactNode } from "react";
import {
  CheckIcon,
  ChevronIcon,
  CrossIcon,
  DashIcon,
  InfoIcon,
  RunningIcon,
  WarnIcon,
} from "@/components/Icons";
import { formatDuration } from "@/lib/tool-view";
import { useDisclosure, useToolCallInfo, type CallTiming } from "./context";

export type CardTone = "running" | "done" | "failed" | "warn" | "quiet" | "empty";

const TONE_ICON: Record<CardTone, (props: { size?: number }) => ReactNode> = {
  running: RunningIcon,
  done: CheckIcon,
  failed: CrossIcon,
  warn: WarnIcon,
  quiet: InfoIcon,
  empty: DashIcon,
};

/** Status light plus word. Never colour alone. */
export function StatusWord({ tone, word }: { tone: CardTone; word: string }) {
  const Icon = TONE_ICON[tone];
  return (
    <span className="pb-status" data-tone={tone}>
      <Icon size={13} />
      {word}
    </span>
  );
}

/** A duration bar on the thread's shared scale, with the number as text. */
export function DurationBar({ timing, failed }: { timing: CallTiming; failed?: boolean }) {
  if (timing.elapsedMs === null) return null;
  const fraction = Math.min(1, timing.elapsedMs / Math.max(1, timing.scaleMs));
  const state = failed ? "failed" : timing.running ? "running" : "done";
  return (
    <span className="pb-duration">
      <span className="pb-duration__track" aria-hidden="true">
        <span
          className="pb-duration__fill"
          data-state={state}
          style={{ width: `${(fraction * 100).toFixed(1)}%` }}
        />
      </span>
      <span className="pb-duration__text">{formatDuration(timing.elapsedMs)}</span>
    </span>
  );
}

export interface ToolCardShellProps {
  /** Tool name as it arrived, shown in Courier. */
  name: string;
  /** One plain-language line; ellipsized when collapsed. */
  headline: ReactNode;
  tone: CardTone;
  /** Status word: Running, Done, Failed, Degraded, Not found, No hits. */
  status: string;
  /** False while arguments are still streaming: nothing to show yet. */
  expandable: boolean;
  /** Overrides the disclosure key from the tool call context. */
  disclosureKey?: string;
  /** Accessible name of the card region. */
  label?: string;
  children?: ReactNode;
}

/**
 * The shared progressive-disclosure card. Collapsed, it is one header row:
 * a real button with the tool name, a headline, the status and the duration
 * bar. Expanding reveals the body with a short height and opacity transition.
 */
export function ToolCardShell({
  name,
  headline,
  tone,
  status,
  expandable,
  disclosureKey,
  label,
  children,
}: ToolCardShellProps) {
  const info = useToolCallInfo();
  const [open, setOpen] = useDisclosure(disclosureKey ?? info.key);
  const bodyId = useId();
  const expanded = expandable && open;

  const row = (
    <>
      <span className="pb-card__name">{name}</span>
      <span className="pb-card__headline">{headline}</span>
      <span className="pb-card__meta">
        <StatusWord tone={tone} word={status} />
        {info.timing ? <DurationBar timing={info.timing} failed={tone === "failed"} /> : null}
      </span>
      <span className="pb-card__chevron" aria-hidden="true">
        {expandable ? <ChevronIcon size={14} /> : null}
      </span>
    </>
  );

  return (
    <section
      className="pb-card"
      data-tone={tone}
      data-open={expanded ? "true" : "false"}
      aria-label={label ?? `${name} tool call`}
      aria-busy={tone === "running" ? true : undefined}
    >
      {expandable ? (
        <button
          type="button"
          className="pb-card__head"
          aria-expanded={expanded}
          aria-controls={bodyId}
          onClick={() => setOpen(!expanded)}
        >
          {row}
        </button>
      ) : (
        <div className="pb-card__head">{row}</div>
      )}
      {expandable ? (
        <div
          id={bodyId}
          className="pb-card__reveal"
          data-open={expanded ? "true" : "false"}
          inert={!expanded}
        >
          <div className="pb-card__body">{children}</div>
        </div>
      ) : null}
    </section>
  );
}
