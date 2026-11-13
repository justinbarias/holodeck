"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { useAgent, useCopilotKit } from "@copilotkit/react-core/v2";
import "@copilotkit/react-core/v2/styles.css";
import type { AbstractAgent, ContentPart, Message } from "@ag-ui/client";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Composer, type FileUpload } from "@/components/chat/Composer";
import { Conversation } from "@/components/chat/Conversation";
import { HowToRead } from "@/components/chat/HowToRead";
import { ConnectionLight, Rail, type Connection, type ThreadSummary } from "@/components/chat/Rail";
import { BrandMark, MenuIcon } from "@/components/Icons";
import { DisclosureProvider } from "@/components/tool-card/context";
import { buildConversation, userText } from "@/lib/conversation";
import { useRuntimeConfig } from "@/lib/runtime-config-provider";
import { partitionSupportedFiles } from "@/lib/supported-mime";
import { useRunEvents } from "@/lib/use-run-events";

/**
 * AG-UI 1.0 file part: the MIME family picks the part type, the bytes travel
 * as a base64 data source, and the filename rides in metadata (1.0 parts have
 * no filename field; holodeck serve reads metadata.filename).
 */
function toFilePart(file: FileUpload): ContentPart {
  const source = {
    type: "data" as const,
    value: file.bytes,
    mimeType: file.contentType,
  };
  const metadata = file.filename ? { filename: file.filename } : undefined;
  const family = file.contentType.split("/")[0];
  const type =
    family === "image" || family === "audio" || family === "video"
      ? family
      : "document";
  return { type, source, metadata };
}

/** Read a file as base64; resolves null when the read fails or is empty. */
function readAsUpload(
  file: File,
  contentType: string,
  filename: string,
): Promise<FileUpload | null> {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const base64 = (e.target?.result as string)?.split(",")[1];
      resolve(base64 ? { contentType, bytes: base64, filename } : null);
    };
    reader.onerror = () => resolve(null);
    reader.readAsDataURL(file);
  });
}

function newId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

/** A thread set aside in this tab: its id, messages and when it started. */
interface SavedThread {
  id: string;
  messages: Message[];
  startedAt: number | null;
}

/**
 * Point the agent at another thread: AG-UI sends `threadId` with every run,
 * and holodeck serve keeps one session per thread id.
 */
function bindThread(target: AbstractAgent, id: string, messages: Message[]) {
  target.setMessages(messages);
  target.threadId = id;
}

const TITLE_CHARS = 48;

function threadTitle(messages: readonly Message[]): string {
  const first = messages.find((message) => message.role === "user");
  const text = first && first.role === "user" ? userText(first).trim() : "";
  if (!text) return first ? "Files only" : "New thread";
  const flat = text.replace(/\s+/g, " ");
  return flat.length > TITLE_CHARS ? `${flat.slice(0, TITLE_CHARS - 1).trimEnd()}…` : flat;
}

function cueCount(messages: readonly Message[]): number {
  let count = 0;
  for (const message of messages) {
    if (message.role === "assistant") count += message.toolCalls?.length ?? 0;
  }
  return count;
}

const TIME = new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit" });

export default function Page() {
  const config = useRuntimeConfig();
  const [selectedFiles, setSelectedFiles] = useState<FileUpload[]>([]);
  const [rejectedFiles, setRejectedFiles] = useState<string[]>([]);
  const [inputValue, setInputValue] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const selectedFilesRef = useRef<FileUpload[]>([]);
  const { copilotkit } = useCopilotKit();
  const { agent, isReady } = useAgent({ agentId: config.agentId });
  const { timings, runError, clearRunError } = useRunEvents(agent);

  // Runtime connection, from CopilotKit core.
  const status = useSyncExternalStore(
    useCallback(
      (onChange: () => void) => {
        const subscription = copilotkit.subscribe({ onRuntimeConnectionStatusChanged: onChange });
        return () => subscription.unsubscribe();
      },
      [copilotkit],
    ),
    () => copilotkit.runtimeConnectionStatus,
    () => "connecting",
  );
  const connection: Connection =
    status === "error" || status === "disconnected"
      ? "error"
      : status === "connected" && isReady
        ? "live"
        : "connecting";

  // Threads kept in this tab. The live thread is the agent's own.
  const [saved, setSaved] = useState<SavedThread[]>([]);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [startedAt, setStartedAt] = useState<Record<string, number>>({});
  const [drawerOpen, setDrawerOpen] = useState(false);
  const currentThreadId = threadId ?? agent.threadId;

  useEffect(() => {
    selectedFilesRef.current = selectedFiles;
  }, [selectedFiles]);

  // Queue the files serve accepts; name the rest in a dismissible notice.
  const queueFiles = useCallback(
    async (files: File[], nameFor: (file: File, mimeType: string) => string) => {
      const { accepted, rejected } = partitionSupportedFiles(files);
      setRejectedFiles(rejected.map((file) => file.name || file.type || "file"));

      const loadedFiles = (
        await Promise.all(
          accepted.map(({ file, mimeType }) =>
            readAsUpload(file, mimeType, nameFor(file, mimeType))
          )
        )
      ).filter((f): f is FileUpload => f !== null);
      setSelectedFiles((prev) => [...prev, ...loadedFiles]);
    },
    []
  );

  const handleFileSelect = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const files = event.target.files;
      if (!files || files.length === 0) return;
      await queueFiles(Array.from(files), (file) => file.name);
      if (event.target) {
        event.target.value = "";
      }
    },
    [queueFiles]
  );

  // Clipboard paste of files (screenshots, copied files).
  useEffect(() => {
    const handlePaste = async (e: ClipboardEvent) => {
      const items = Array.from(e.clipboardData?.items || []);
      const pastedFiles = items
        .filter((item) => item.kind === "file")
        .map((item) => item.getAsFile())
        .filter((file): file is File => file !== null);
      if (pastedFiles.length === 0) return;

      e.preventDefault();

      await queueFiles(
        pastedFiles,
        (file, mimeType) =>
          file.name || `pasted-${Date.now()}.${mimeType.split("/")[1]}`
      );
    };

    document.addEventListener("paste", handlePaste);
    return () => document.removeEventListener("paste", handlePaste);
  }, [queueFiles]);

  const removeFile = useCallback((index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  }, []);

  const handleSubmitMessage = useCallback(async () => {
    const text = inputValue;
    const currentFiles = selectedFilesRef.current;
    const hasText = text.trim().length > 0;
    const hasFiles = currentFiles.length > 0;
    if (!hasText && !hasFiles) return;

    // Content: an array of parts when files are attached, else a string.
    let content: string | ContentPart[];
    if (hasFiles) {
      const contentParts: ContentPart[] = [];
      if (hasText) contentParts.push({ type: "text", text: text.trim() });
      for (const file of currentFiles) contentParts.push(toFilePart(file));
      content = contentParts;
    } else {
      content = text.trim();
    }

    setSelectedFiles([]);
    setInputValue("");
    clearRunError();
    const id = agent.threadId;
    setStartedAt((prev) => (prev[id] ? prev : { ...prev, [id]: Date.now() }));

    agent.addMessage({ id: newId(), role: "user" as const, content });
    try {
      await copilotkit.runAgent({ agent });
    } catch (error) {
      console.error("Error running agent:", error);
    }
  }, [agent, copilotkit, inputValue, clearRunError]);

  const handleStop = useCallback(() => {
    agent.abortRun?.();
  }, [agent]);

  const messages = agent.messages;
  const isRunning = agent.isRunning;
  // Rebuilt on every render: the agent may update its message list in place.
  const conversation = buildConversation(messages, isRunning);

  // Tick the clock while a timed tool call is open, so its bar grows.
  const anyOpenTiming = useMemo(
    () => [...timings.values()].some((timing) => timing.end === null),
    [timings],
  );
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!anyOpenTiming) return;
    const timer = window.setInterval(() => setNow(Date.now()), 100);
    return () => window.clearInterval(timer);
  }, [anyOpenTiming]);

  // Edit and resend: put the last question back in the composer.
  const editAndResend = useCallback(() => {
    const lastUser = [...messages].reverse().find((message) => message.role === "user");
    if (lastUser && lastUser.role === "user") setInputValue(userText(lastUser));
    clearRunError();
    textareaRef.current?.focus();
  }, [messages, clearRunError]);

  // Threads: set the live one aside, then load another or start fresh.
  const setAside = useCallback(() => {
    const id = agent.threadId;
    const snapshot = [...agent.messages];
    if (snapshot.length === 0) return;
    setSaved((prev) => [
      { id, messages: snapshot, startedAt: startedAt[id] ?? null },
      ...prev.filter((thread) => thread.id !== id),
    ]);
  }, [agent, startedAt]);

  const switchTo = useCallback(
    (id: string, messagesFor: Message[]) => {
      bindThread(agent, id, messagesFor);
      setThreadId(id);
      clearRunError();
      setInputValue("");
      setSelectedFiles([]);
      setRejectedFiles([]);
      setDrawerOpen(false);
    },
    [agent, clearRunError],
  );

  const handleNewThread = useCallback(() => {
    if (agent.isRunning) agent.abortRun?.();
    setAside();
    switchTo(newId(), []);
    textareaRef.current?.focus();
  }, [agent, setAside, switchTo]);

  const handleSelectThread = useCallback(
    (id: string) => {
      if (id === agent.threadId) {
        setDrawerOpen(false);
        return;
      }
      const target = saved.find((thread) => thread.id === id);
      if (!target || agent.isRunning) return;
      setAside();
      setSaved((prev) => prev.filter((thread) => thread.id !== id));
      switchTo(id, target.messages);
    },
    [agent, saved, setAside, switchTo],
  );

  const threads: ThreadSummary[] = [
    ...(messages.length > 0
      ? [{ id: currentThreadId, title: threadTitle(messages), cues: conversation.cueCount }]
      : []),
    ...saved
      .filter((thread) => thread.id !== currentThreadId)
      .map((thread) => ({
        id: thread.id,
        title: threadTitle(thread.messages),
        cues: cueCount(thread.messages),
      })),
  ];

  const started = startedAt[currentThreadId];
  let runningCue: number | null = null;
  for (const row of conversation.rows) {
    const calls = row.kind === "tool" ? [row.call] : row.kind === "group" ? row.calls : [];
    for (const call of calls) if (runningCue === null && call.status !== "complete") runningCue = call.cue;
  }

  // Follow new content while the reader is at the bottom of the conversation.
  const stageRef = useRef<HTMLElement>(null);
  const atBottom = useRef(true);
  const lastRow = conversation.rows[conversation.rows.length - 1];
  const followKey = `${conversation.rows.length}:${lastRow?.kind === "answer" ? lastRow.content.length : 0}:${runError ? 1 : 0}:${isRunning ? 1 : 0}`;
  useEffect(() => {
    const stage = stageRef.current;
    if (stage && atBottom.current) stage.scrollTop = stage.scrollHeight;
  }, [followKey]);

  const empty = messages.length === 0;
  const composer = (variant: "docked" | "welcome") => (
    <Composer
      value={inputValue}
      onChange={setInputValue}
      onSubmit={handleSubmitMessage}
      onStop={handleStop}
      onAttach={() => fileInputRef.current?.click()}
      isRunning={isRunning}
      disabled={connection !== "live"}
      nextCue={conversation.cueCount + 1}
      runningCue={runningCue}
      agentTitle={config.agentTitle}
      files={selectedFiles}
      onRemoveFile={removeFile}
      rejected={rejectedFiles}
      onDismissRejected={() => setRejectedFiles([])}
      textareaRef={textareaRef}
      variant={variant}
    />
  );

  return (
    <DisclosureProvider>
      <div className="pb-app">
        <Rail
          agentTitle={config.agentTitle}
          agentDescription={config.agentDescription}
          agentId={config.agentId}
          connection={connection}
          threads={threads}
          currentThreadId={currentThreadId}
          canSwitch={!isRunning}
          onSelectThread={handleSelectThread}
          onNewThread={handleNewThread}
          open={drawerOpen}
          onClose={() => setDrawerOpen(false)}
        />
        {drawerOpen ? (
          <button
            type="button"
            className="pb-scrim"
            aria-label="Close menu"
            onClick={() => setDrawerOpen(false)}
          />
        ) : null}

        <main className="pb-main">
          <div className="pb-topbar">
            <button
              type="button"
              className="pb-iconbtn pb-iconbtn--rail"
              onClick={() => setDrawerOpen(true)}
              aria-label="Open menu"
              aria-controls="pb-rail"
              aria-expanded={drawerOpen}
            >
              <MenuIcon size={18} />
            </button>
            <span className="pb-topbar__brand">
              <BrandMark size={20} />
              HoloDeck
            </span>
            <ConnectionLight connection={connection} />
          </div>

          <header className="pb-header">
            <div className="pb-header__title">
              <h1>{threadTitle(messages)}</h1>
              <p>
                {empty ? (
                  "No cues yet"
                ) : (
                  <>
                    Thread <span className="pb-data">{currentThreadId.slice(0, 4)}</span>
                    {started ? ` · started ${TIME.format(started)}` : ""}
                    {` · ${conversation.cueCount} ${conversation.cueCount === 1 ? "cue" : "cues"}`}
                  </>
                )}
              </p>
            </div>
            <div className="pb-header__tools">
              <HowToRead />
              <ThemeToggle />
            </div>
          </header>

          <input
            type="file"
            multiple
            ref={fileInputRef}
            onChange={handleFileSelect}
            accept={config.fileAccept}
            hidden
          />

          {empty ? (
            <section className="pb-stage pb-stage--center" aria-live="polite">
              {connection === "live" ? (
                <div className="pb-welcome">
                  <h2>{config.agentTitle}</h2>
                  <p className="pb-welcome__lead">
                    {config.agentDescription.replace(/[.\s]+$/, "")}. Ask a question; each tool the agent uses
                    appears as a numbered cue beside the answer.
                  </p>
                  {composer("welcome")}
                  <p className="pb-welcome__note">
                    Answers come from the agent; check the cited source before relying on them.
                  </p>
                </div>
              ) : connection === "connecting" ? (
                <div className="pb-welcome pb-welcome--status" role="status">
                  <span className="pb-spinner" aria-hidden="true" />
                  <h2 className="pb-welcome__status">Connecting to {config.agentTitle}</h2>
                  <p className="pb-welcome__lead">
                    Agent <span className="pb-data">{config.agentId}</span> through{" "}
                    <span className="pb-data">/api/copilotkit</span>
                  </p>
                </div>
              ) : (
                <div className="pb-welcome pb-welcome--status" role="alert">
                  <h2 className="pb-welcome__status">Could not reach the agent</h2>
                  <p className="pb-welcome__lead">
                    Check that <span className="pb-data">holodeck serve</span> is running and
                    that <span className="pb-data">AGENT_ID</span> ({config.agentId}) matches
                    the agent name in agent.yaml, then reload.
                  </p>
                  <button
                    type="button"
                    className="pb-btn pb-btn--outline"
                    onClick={() => window.location.reload()}
                  >
                    Reload
                  </button>
                </div>
              )}
            </section>
          ) : (
            <section
              className="pb-stage"
              aria-label="Conversation"
              ref={stageRef}
              onScroll={(event) => {
                const el = event.currentTarget;
                atBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
              }}
            >
              <Conversation
                rows={conversation.rows}
                timings={timings}
                now={now}
                isRunning={isRunning}
                runError={runError}
                onEditResend={editAndResend}
              />
            </section>
          )}

          {empty && connection === "live" ? null : composer("docked")}
        </main>
      </div>
    </DisclosureProvider>
  );
}
