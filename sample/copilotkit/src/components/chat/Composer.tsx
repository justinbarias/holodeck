"use client";

import { useLayoutEffect, type FormEvent, type KeyboardEvent, type RefObject } from "react";
import { CrossIcon, FileIcon, PaperclipIcon, WarnIcon } from "@/components/Icons";
import { base64Bytes, formatBytes } from "@/lib/tool-view";

export type FileUpload = {
  contentType: string;
  bytes: string;
  filename?: string;
};

export interface ComposerProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onStop: () => void;
  onAttach: () => void;
  isRunning: boolean;
  /** Not connected yet: nothing can be sent. */
  disabled: boolean;
  /** The cue number the next tool call will get. */
  nextCue: number;
  /** Cue currently running, for the placeholder. */
  runningCue: number | null;
  agentTitle: string;
  files: FileUpload[];
  onRemoveFile: (index: number) => void;
  rejected: string[];
  onDismissRejected: () => void;
  textareaRef: RefObject<HTMLTextAreaElement | null>;
  /** Centred on the welcome page instead of docked at the bottom. */
  variant?: "docked" | "welcome";
}

function extension(name: string | undefined, contentType: string): string {
  const dot = name ? name.lastIndexOf(".") : -1;
  if (name && dot > 0) return name.slice(dot + 1).toUpperCase();
  return (contentType.split("/")[1] ?? "file").toUpperCase();
}

/** Grow the textarea with its content, up to the CSS max-height. */
function useAutosize(ref: RefObject<HTMLTextAreaElement | null>, value: string) {
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [ref, value]);
}

export function Composer({
  value,
  onChange,
  onSubmit,
  onStop,
  onAttach,
  isRunning,
  disabled,
  nextCue,
  runningCue,
  agentTitle,
  files,
  onRemoveFile,
  rejected,
  onDismissRejected,
  textareaRef,
  variant = "docked",
}: ComposerProps) {
  useAutosize(textareaRef, value);
  const canSend = !disabled && !isRunning && (value.trim().length > 0 || files.length > 0);

  const placeholder = disabled
    ? "Available once connected"
    : isRunning
      ? runningCue
        ? `Q${runningCue} is running; you can stop it`
        : "The agent is working; you can stop it"
      : files.length > 0
        ? `Ask about ${files.length === 1 ? "this file" : `these ${files.length} files`}…`
        : `Message ${agentTitle}…`;

  const submit = (event: FormEvent) => {
    event.preventDefault();
    if (canSend) onSubmit();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      if (canSend) onSubmit();
    }
  };

  return (
    <form className="pb-composer" data-variant={variant} onSubmit={submit}>
      {rejected.length > 0 ? (
        <div className="pb-banner pb-composer__notice" data-tone="warn" role="alert">
          <WarnIcon size={16} />
          <p>
            <b>Not attached (unsupported file type):</b> {rejected.join(", ")}. Supported: PNG,
            JPEG, GIF, WebP, PDF, DOCX, XLSX, PPTX, TXT, CSV, Markdown.
          </p>
          <button
            type="button"
            className="pb-xbtn"
            onClick={onDismissRejected}
            aria-label="Dismiss notice"
          >
            <CrossIcon size={13} />
          </button>
        </div>
      ) : null}

      {files.length > 0 ? (
        <ul className="pb-composer__files" aria-label="Files to send">
          {files.map((file, index) => (
            <li key={index} className="pb-filechip">
              {file.contentType.startsWith("image/") ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  src={`data:${file.contentType};base64,${file.bytes}`}
                  alt=""
                  className="pb-filechip__thumb"
                />
              ) : (
                <FileIcon size={15} />
              )}
              <span className="pb-filechip__name">{file.filename || `File ${index + 1}`}</span>
              <span className="pb-data pb-quiet">
                {extension(file.filename, file.contentType)} · {formatBytes(base64Bytes(file.bytes))}
              </span>
              <button
                type="button"
                className="pb-xbtn"
                onClick={() => onRemoveFile(index)}
                aria-label={`Remove ${file.filename || `file ${index + 1}`}`}
              >
                <CrossIcon size={13} />
              </button>
            </li>
          ))}
        </ul>
      ) : null}

      <div className="pb-composer__line">
        <span className="pb-composer__cue" aria-hidden="true">
          Q{nextCue}
        </span>
        <div className="pb-field">
          <button
            type="button"
            className="pb-iconbtn pb-iconbtn--bare"
            onClick={onAttach}
            disabled={disabled}
            aria-label="Attach a file"
            title="Attach a file"
          >
            <PaperclipIcon size={18} />
          </button>
          <label className="sr-only" htmlFor="pb-message">
            Message
          </label>
          <textarea
            id="pb-message"
            ref={textareaRef}
            className="pb-field__input"
            rows={1}
            value={value}
            placeholder={placeholder}
            disabled={disabled}
            onChange={(event) => onChange(event.target.value)}
            onKeyDown={onKeyDown}
            aria-describedby="pb-next-cue"
          />
          {isRunning ? (
            <button type="button" className="pb-btn pb-btn--secondary" onClick={onStop}>
              Stop
            </button>
          ) : (
            <button type="submit" className="pb-btn pb-btn--primary" disabled={!canSend}>
              Send
            </button>
          )}
        </div>
        <span id="pb-next-cue" className="sr-only">
          The next tool call will be cue Q{nextCue}.
        </span>
      </div>
    </form>
  );
}
