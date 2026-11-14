"use client";

import { useState } from "react";
import { CopyButton } from "@/components/tool-card/CopyButton";
import { ToolCardShell } from "@/components/tool-card/ToolCardShell";
import {
  argumentsHeadline,
  byteLength,
  decodeResult,
  firstSentence,
  formatBytes,
  toolFailure,
} from "@/lib/tool-view";

export type ToolCallStatus = "inProgress" | "complete" | "executing";

export interface ToolCallCardProps {
  name: string;
  args?: Record<string, unknown>;
  status: ToolCallStatus;
  result?: unknown;
  /** The AG-UI tool message `error` field. */
  error?: string | null;
  /** The run ended before a result arrived. */
  stopped?: boolean;
}

/** Plain-text results longer than this start clamped. */
const TEXT_PREVIEW_CHARS = 280;

/**
 * The generic tool card, for every tool without a dedicated view and as the
 * fallback when a dedicated view cannot read the result. Collapsed it is one
 * row; expanded it shows the arguments, the result and copy actions.
 */
export function ToolCallCard({ name, args, status, result, error, stopped }: ToolCallCardProps) {
  const running = status !== "complete";
  const failure = running ? null : toolFailure(result, error);
  const argsHeadline = argumentsHeadline(args);

  if (running) {
    return (
      <ToolCardShell
        name={name}
        headline={argsHeadline || "Preparing the call…"}
        tone="running"
        status="Running"
        expandable={args !== undefined}
      >
        <ArgumentsTable args={args} />
      </ToolCardShell>
    );
  }

  if (failure) {
    return (
      <ToolCardShell
        name={name}
        headline={firstSentence(failure)}
        tone="failed"
        status="Failed"
        expandable
      >
        <p className="pb-card__message">
          <strong>{firstSentence(failure)}</strong>{" "}
          {failure.slice(firstSentence(failure).length).trim()}
        </p>
        <ArgumentsTable args={args} />
        <CopyRow args={args} result={result} />
      </ToolCardShell>
    );
  }

  if (stopped && result === undefined) {
    return (
      <ToolCardShell
        name={name}
        headline={argsHeadline ? `${argsHeadline} · no result` : "No result"}
        tone="empty"
        status="Stopped"
        expandable
      >
        <p className="pb-card__message">The run ended before this tool returned a result.</p>
        <ArgumentsTable args={args} />
      </ToolCardShell>
    );
  }

  return (
    <ToolCardShell name={name} headline={argsHeadline} tone="done" status="Done" expandable>
      <ArgumentsTable args={args} />
      <ResultView result={result} />
      <CopyRow args={args} result={result} />
    </ToolCardShell>
  );
}

function ArgumentsTable({ args }: { args?: Record<string, unknown> }) {
  if (!args) return null;
  const entries = Object.entries(args);
  return (
    <div className="pb-section">
      <h4 className="pb-section__title">Arguments</h4>
      {entries.length === 0 ? (
        <p className="pb-card__message pb-quiet">No arguments.</p>
      ) : (
        <dl className="pb-kv">
          {entries.map(([key, value]) => (
            <div key={key} className="pb-kv__row">
              <dt>{key}</dt>
              <dd className="pb-data">
                {JSON.stringify(value)}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}

function resultText(result: unknown): { text: string; json: boolean } {
  const value = decodeResult(result);
  if (typeof value === "string") return { text: value, json: false };
  if (value === undefined) return { text: "", json: false };
  return { text: JSON.stringify(value, null, 2), json: true };
}

function ResultView({ result }: { result: unknown }) {
  const [full, setFull] = useState(false);
  const { text, json } = resultText(result);
  if (!text) {
    return (
      <div className="pb-section">
        <h4 className="pb-section__title">Result</h4>
        <p className="pb-card__message pb-quiet">The tool returned an empty result.</p>
      </div>
    );
  }
  const size = formatBytes(byteLength(text));
  if (json) {
    return (
      <div className="pb-section">
        <h4 className="pb-section__title">
          Result <span className="pb-section__aside pb-data">{size} · JSON</span>
        </h4>
        <pre className="pb-pre">{text}</pre>
      </div>
    );
  }
  const long = text.length > TEXT_PREVIEW_CHARS;
  return (
    <div className="pb-section">
      <h4 className="pb-section__title">
        Result <span className="pb-section__aside pb-data">{size} · text</span>
      </h4>
      <p className="pb-card__text">
        {long && !full ? `${text.slice(0, TEXT_PREVIEW_CHARS).trimEnd()}…` : text}
      </p>
      {long ? (
        <button type="button" className="pb-link" onClick={() => setFull((v) => !v)}>
          {full ? "Show less" : `Show full result · ${size}`}
        </button>
      ) : null}
    </div>
  );
}

function CopyRow({ args, result }: { args?: Record<string, unknown>; result: unknown }) {
  const { text } = resultText(result);
  const argsText = args ? JSON.stringify(args, null, 2) : "";
  if (!text && !argsText) return null;
  return (
    <div className="pb-actions">
      {text ? <CopyButton text={text} label="Copy result" /> : null}
      {argsText ? <CopyButton text={argsText} label="Copy arguments" /> : null}
    </div>
  );
}
