"use client";

import { useId } from "react";
import { CopilotChatAssistantMessage } from "@copilotkit/react-core/v2";
import type { ContentPart, UserMessage } from "@ag-ui/client";
import { ToolCallView } from "@/components/ToolCallRenderer";
import { AlertIcon, ChevronIcon, FileIcon } from "@/components/Icons";
import { CopyButton } from "@/components/tool-card/CopyButton";
import { StatusWord, type CardTone } from "@/components/tool-card/ToolCardShell";
import { ToolCallInfoProvider, useDisclosure, type CallTiming } from "@/components/tool-card/context";
import { parseToolResult } from "@/components/memory/parse";
import { cueRange, toolNameSummary, userText, type Cue, type Row } from "@/lib/conversation";
import { formatDuration, measuredElapsed, toolFailure } from "@/lib/tool-view";
import type { RunError, ToolTiming } from "@/lib/use-run-events";

export interface ConversationProps {
  rows: Row[];
  timings: ReadonlyMap<string, ToolTiming>;
  now: number;
  isRunning: boolean;
  runError: RunError | null;
  onEditResend: () => void;
}

/** Tone of a cue light: running, failed, done or stopped. */
function cueTone(call: Cue): CardTone {
  if (call.status !== "complete") return "running";
  if (call.stopped) return "empty";
  if (toolFailure(call.result, call.error)) return "failed";
  if (parseToolResult(call.result).kind === "error") return "failed";
  return "done";
}

function timingOf(
  call: Cue,
  timings: ReadonlyMap<string, ToolTiming>,
  now: number,
): { elapsedMs: number | null; running: boolean } {
  return measuredElapsed(timings.get(call.toolCallId), now);
}

/**
 * The conversation on its cue margin. Each row has a margin cell and a
 * content cell; the margin carries the cue number and light of each tool
 * call, and a bracket beside each answer ties it to the cues before it.
 */
export function Conversation({
  rows,
  timings,
  now,
  isRunning,
  runError,
  onEditResend,
}: ConversationProps) {
  // One time scale per thread: the longest call so far is the full bar.
  let scaleMs = 1000;
  for (const row of rows) {
    const calls = row.kind === "tool" ? [row.call] : row.kind === "group" ? row.calls : [];
    for (const call of calls) {
      const { elapsedMs } = timingOf(call, timings, now);
      if (elapsedMs !== null) scaleMs = Math.max(scaleMs, elapsedMs);
    }
  }
  const timingFor = (call: Cue): CallTiming => ({ ...timingOf(call, timings, now), scaleMs });

  const last = rows[rows.length - 1];
  const waiting =
    isRunning && (!last || last.kind === "user" || (last.kind !== "answer" && !anyRunning(last)));

  return (
    <ol className="pb-convo" aria-label="Conversation">
      {rows.map((row) => {
        switch (row.kind) {
          case "user":
            return (
              <li key={row.key} className="pb-row" data-kind="user">
                <span className="pb-margin" />
                <div className="pb-content">
                  <UserBubble message={row.message} />
                </div>
              </li>
            );
          case "tool":
            return (
              <li key={row.key} className="pb-row" data-kind="tool">
                <span className="pb-margin">
                  <CueMark label={`Q${row.call.cue}`} tone={cueTone(row.call)} />
                </span>
                <div className="pb-content">
                  <ToolRow call={row.call} timing={timingFor(row.call)} />
                </div>
              </li>
            );
          case "group":
            return <GroupRow key={row.key} groupKey={row.key} calls={row.calls} timingFor={timingFor} />;
          case "answer":
            return (
              <li key={row.key} className="pb-row" data-kind="answer">
                <span className="pb-margin">
                  {row.cues.length > 0 ? (
                    <span className="pb-bracket" aria-hidden="true" />
                  ) : null}
                </span>
                <div className="pb-content">
                  <Answer content={row.content} messageId={row.messageId} cues={row.cues} />
                </div>
              </li>
            );
        }
      })}
      {waiting ? (
        <li className="pb-row" data-kind="status">
          <span className="pb-margin" />
          <div className="pb-content">
            <p className="pb-working" role="status">
              <span className="pb-working__caret" aria-hidden="true" />
              {last && last.kind !== "user" ? "Writing the answer…" : "Waiting for the agent…"}
            </p>
          </div>
        </li>
      ) : null}
      {runError ? (
        <li className="pb-row" data-kind="error">
          <span className="pb-margin" />
          <div className="pb-content">
            <RunErrorNotice error={runError} onEditResend={onEditResend} />
          </div>
        </li>
      ) : null}
    </ol>
  );
}

function anyRunning(row: Row): boolean {
  if (row.kind === "tool") return row.call.status !== "complete";
  if (row.kind === "group") return row.calls.some((call) => call.status !== "complete");
  return false;
}

const TONE_WORD: Record<CardTone, string> = {
  running: "running",
  done: "done",
  failed: "failed",
  warn: "degraded",
  quiet: "done",
  empty: "stopped",
};

function CueMark({ label, tone }: { label: string; tone: CardTone }) {
  return (
    <span className="pb-cue" data-tone={tone}>
      <span className="pb-cue__light" aria-hidden="true" />
      <span className="pb-cue__label">{label}</span>
      <span className="sr-only">, {TONE_WORD[tone]}</span>
    </span>
  );
}

function ToolRow({ call, timing }: { call: Cue; timing: CallTiming }) {
  return (
    <ToolCallInfoProvider value={{ key: call.toolCallId, timing }}>
      <ToolCallView
        name={call.name}
        args={call.args}
        status={call.status}
        result={call.result}
        error={call.error}
        stopped={call.stopped}
      />
    </ToolCallInfoProvider>
  );
}

function GroupRow({
  groupKey,
  calls,
  timingFor,
}: {
  groupKey: string;
  calls: Cue[];
  timingFor: (call: Cue) => CallTiming;
}) {
  const [open, setOpen] = useDisclosure(groupKey);
  const bodyId = useId();
  const range = cueRange(calls.map((call) => call.cue));
  const tones = calls.map(cueTone);
  const failed = tones.filter((tone) => tone === "failed").length;
  const running = tones.filter((tone) => tone === "running").length;
  const tone: CardTone = running ? "running" : failed ? "failed" : "done";
  const word = running
    ? `${running} running`
    : failed
      ? `${failed} failed`
      : "All done";

  // Total time spent in the tools, when every call was timed.
  const timed = calls.map(timingFor);
  let span: number | null = null;
  if (timed.every((t) => t.elapsedMs !== null)) {
    span = timed.reduce((sum, t) => sum + (t.elapsedMs ?? 0), 0);
  }

  return (
    <li className="pb-row pb-row--group" data-kind="group" data-open={open ? "true" : "false"}>
      <div className="pb-row__pair">
        <span className="pb-margin">
          {open ? null : <CueMark label={range} tone={tone} />}
        </span>
        <div className="pb-content">
          <div className="pb-card pb-group" data-tone={tone} data-open={open ? "true" : "false"}>
            <button
              type="button"
              className="pb-card__head"
              aria-expanded={open}
              aria-controls={bodyId}
              onClick={() => setOpen(!open)}
            >
              <span className="pb-card__name">{range}</span>
              <span className="pb-card__headline">
                {calls.length} tool calls ·{" "}
                <span className="pb-data">{toolNameSummary(calls.map((call) => call.name))}</span>
                {span !== null ? ` · ${formatDuration(span)} in tools` : ""}
              </span>
              <span className="pb-card__meta">
                <StatusWord tone={tone} word={word} />
              </span>
              <span className="pb-card__chevron" aria-hidden="true">
                <ChevronIcon size={14} />
              </span>
            </button>
          </div>
        </div>
      </div>
      <div id={bodyId} className="pb-card__reveal" data-open={open ? "true" : "false"} inert={!open}>
        <ol className="pb-group__calls">
          {calls.map((call) => (
            <li key={call.toolCallId} className="pb-row" data-kind="tool">
              <span className="pb-margin">
                <CueMark label={`Q${call.cue}`} tone={cueTone(call)} />
              </span>
              <div className="pb-content">
                <ToolRow call={call} timing={timingFor(call)} />
              </div>
            </li>
          ))}
        </ol>
      </div>
    </li>
  );
}

function Answer({
  content,
  messageId,
  cues,
}: {
  content: string;
  messageId: string;
  cues: number[];
}) {
  return (
    <div className="pb-answer">
      <CopilotChatAssistantMessage.MarkdownRenderer
        content={content}
        messageId={messageId}
        className="pb-md"
        controls={{ table: false }}
      />
      <div className="pb-answer__foot">
        <CopyButton text={content} label="Copy answer" className="pb-iconbtn" iconOnly />
        {cues.length > 0 ? (
          <span className="pb-answer__cues">
            <span className="pb-quiet">After</span>{" "}
            <span className="pb-data pb-answer__cue-list">
              {cues.map((cue) => `Q${cue}`).join(" ")}
            </span>
          </span>
        ) : null}
      </div>
    </div>
  );
}

function partName(part: Exclude<ContentPart, { type: "text" }>, index: number): string {
  const meta: unknown = part.metadata;
  if (typeof meta === "object" && meta !== null && "filename" in meta) {
    const filename = (meta as { filename: unknown }).filename;
    if (typeof filename === "string" && filename) return filename;
  }
  return `${part.type} ${index + 1}`;
}

function partLabel(name: string, part: Exclude<ContentPart, { type: "text" }>): string {
  const dot = name.lastIndexOf(".");
  if (dot > 0) return name.slice(dot + 1).toUpperCase();
  return part.type.toUpperCase();
}

function UserBubble({ message }: { message: UserMessage }) {
  const text = userText(message);
  const files =
    typeof message.content === "string"
      ? []
      : message.content.filter(
          (part): part is Exclude<ContentPart, { type: "text" }> => part.type !== "text",
        );
  return (
    <div className="pb-user">
      {files.length > 0 ? (
        <ul className="pb-user__files" aria-label="Attached files">
          {files.map((part, index) => {
            const name = partName(part, index);
            const thumb =
              part.type === "image" && part.source.type === "data"
                ? `data:${part.source.mimeType};base64,${part.source.value}`
                : null;
            return (
              <li key={index} className="pb-filechip">
                {thumb ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={thumb} alt="" className="pb-filechip__thumb" />
                ) : (
                  <FileIcon size={15} />
                )}
                <span className="pb-filechip__name">{name}</span>
                <span className="pb-data pb-quiet">{partLabel(name, part)}</span>
              </li>
            );
          })}
        </ul>
      ) : null}
      {text ? <p className="pb-user__bubble">{text}</p> : null}
    </div>
  );
}

function RunErrorNotice({ error, onEditResend }: { error: RunError; onEditResend: () => void }) {
  return (
    <div className="pb-banner" data-tone="failed" role="alert">
      <AlertIcon size={18} />
      <div>
        <p>
          <b>The agent could not run this turn.</b> {error.message}
          {error.code ? <span className="pb-data"> ({error.code})</span> : null}
        </p>
        <div className="pb-banner__actions">
          <button type="button" className="pb-btn pb-btn--danger" onClick={onEditResend}>
            Edit and resend
          </button>
          <CopyButton
            text={error.code ? `${error.message} (${error.code})` : error.message}
            label="Copy error"
            className="pb-link pb-link--danger"
          />
        </div>
      </div>
    </div>
  );
}
