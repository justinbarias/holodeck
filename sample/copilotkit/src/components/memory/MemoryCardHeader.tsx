import type { ToolCallStatus } from "@/components/ToolCallCard";

/**
 * Header shared by the search and provenance views: an eyebrow, a title, an
 * optional summary and a status badge. Same classes as the fact timeline.
 */
export function MemoryCardHeader({
  eyebrow,
  title,
  status,
  failed = false,
  summary,
}: {
  eyebrow: string;
  title: string;
  status: ToolCallStatus;
  failed?: boolean;
  summary?: string;
}) {
  const complete = status === "complete";
  return (
    <header className="holodeck-timeline__header">
      <div className="holodeck-timeline__heading">
        <span className="holodeck-timeline__eyebrow">{eyebrow}</span>
        <span className="holodeck-timeline__entity">{title}</span>
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
