"use client";

import { BrandMark, CrossIcon, PlusIcon } from "@/components/Icons";

export type Connection = "live" | "connecting" | "error";

export interface ThreadSummary {
  id: string;
  title: string;
  cues: number;
}

const CONNECTION_WORD: Record<Connection, string> = {
  live: "Live",
  connecting: "Connecting",
  error: "Not connected",
};

export function ConnectionLight({ connection }: { connection: Connection }) {
  return (
    <span className="pb-live" data-state={connection}>
      <span className="pb-live__light" aria-hidden="true" />
      {CONNECTION_WORD[connection]}
    </span>
  );
}

export interface RailProps {
  agentTitle: string;
  agentDescription: string;
  agentId: string;
  connection: Connection;
  threads: ThreadSummary[];
  currentThreadId: string;
  /** Switching threads is blocked while a run is in flight. */
  canSwitch: boolean;
  onSelectThread: (id: string) => void;
  onNewThread: () => void;
  /** Narrow screens: the rail is a drawer. */
  open: boolean;
  onClose: () => void;
}

/**
 * The house-blue rail: brand, the agent and its connection, this session's
 * threads with their cue counts, and New thread.
 */
export function Rail({
  agentTitle,
  agentDescription,
  agentId,
  connection,
  threads,
  currentThreadId,
  canSwitch,
  onSelectThread,
  onNewThread,
  open,
  onClose,
}: RailProps) {
  return (
    <aside className="pb-rail" id="pb-rail" data-open={open ? "true" : "false"} aria-label="Agent and threads">
      <div className="pb-rail__brand">
        <BrandMark />
        <span>HoloDeck</span>
        <button type="button" className="pb-rail__close" onClick={onClose} aria-label="Close menu">
          <CrossIcon size={16} />
        </button>
      </div>

      <div className="pb-agent">
        <p className="pb-agent__title">{agentTitle}</p>
        <p className="pb-agent__desc">{agentDescription}</p>
        <p className="pb-agent__state">
          <ConnectionLight connection={connection} />
        </p>
        <p className="pb-agent__id">
          Agent id <span className="pb-data">{agentId}</span>
        </p>
      </div>

      {threads.length > 0 ? (
        <nav className="pb-threads" aria-label="Threads in this session">
          <h2 className="pb-threads__title">Threads</h2>
          <ul>
            {threads.map((thread) => {
              const current = thread.id === currentThreadId;
              return (
                <li key={thread.id}>
                  <button
                    type="button"
                    className="pb-thread"
                    aria-current={current ? "true" : undefined}
                    disabled={!current && !canSwitch}
                    onClick={() => onSelectThread(thread.id)}
                  >
                    <span className="pb-thread__title">{thread.title}</span>
                    <span className="pb-thread__cues">
                      {thread.cues} {thread.cues === 1 ? "cue" : "cues"}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        </nav>
      ) : null}

      <button type="button" className="pb-rail__new" onClick={onNewThread}>
        <PlusIcon size={16} />
        New thread
      </button>
      <p className="pb-rail__foot">
        Threads live in this browser tab only. holodeck serve via /api/copilotkit.
      </p>
    </aside>
  );
}
