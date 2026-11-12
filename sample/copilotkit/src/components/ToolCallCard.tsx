export type ToolCallStatus = "inProgress" | "complete" | "executing";

export interface ToolCallCardProps {
  name: string;
  args?: Record<string, unknown>;
  status: ToolCallStatus;
  result?: unknown;
}

/**
 * The generic tool-call card: name, status, arguments and the raw result.
 * Used for every tool without a dedicated view, and as the fallback when a
 * dedicated view cannot read the result.
 */
export function ToolCallCard({ name, args, status, result }: ToolCallCardProps) {
  const isInProgress = status === "inProgress";
  const isComplete = status === "complete";

  return (
    <div className="holodeck-tool-call">
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.2rem",
          marginBottom: isComplete && result ? "0.2rem" : 0,
        }}
      >
        <span style={{ fontSize: "1rem" }}>
          {isInProgress ? "⏳" : isComplete ? "✅" : "🔧"}
        </span>
        <span style={{ fontWeight: 600 }}>{formatToolName(name)}</span>
        <span
          className={
            isInProgress
              ? "holodeck-tool-badge holodeck-tool-badge--progress"
              : "holodeck-tool-badge holodeck-tool-badge--complete"
          }
        >
          {isInProgress ? "Running..." : isComplete ? "Complete" : String(status)}
        </span>
      </div>

      {/* Show arguments (open while in progress, collapsed when complete) */}
      {args && Object.keys(args).length > 0 ? (
        <details className="holodeck-tool-details" open={isInProgress}>
          <summary>Arguments</summary>
          <pre>{JSON.stringify(args, null, 2)}</pre>
        </details>
      ) : null}

      {/* Show result when complete */}
      {isComplete && result ? (
        <details className="holodeck-tool-details">
          <summary>Result</summary>
          <pre>
            {typeof result === "string"
              ? result
              : JSON.stringify(result, null, 2)}
          </pre>
        </details>
      ) : null}
    </div>
  );
}

/**
 * Format tool name for display.
 * Converts snake_case to Title Case with spaces.
 */
export function formatToolName(name: string): string {
  return name
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
