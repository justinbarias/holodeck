"use client";

import { useEffect, useState } from "react";
import { CopyIcon } from "@/components/Icons";

/** A text button that copies `text` and says "Copied" for a moment. */
export function CopyButton({
  text,
  label,
  className = "pb-link",
  iconOnly = false,
}: {
  text: string;
  label: string;
  className?: string;
  iconOnly?: boolean;
}) {
  const [copied, setCopied] = useState<"idle" | "done" | "failed">("idle");
  useEffect(() => {
    if (copied === "idle") return;
    const timer = window.setTimeout(() => setCopied("idle"), 1600);
    return () => window.clearTimeout(timer);
  }, [copied]);

  const onClick = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied("done");
    } catch {
      setCopied("failed");
    }
  };

  const word = copied === "done" ? "Copied" : copied === "failed" ? "Copy failed" : label;
  return (
    <button
      type="button"
      className={className}
      onClick={onClick}
      aria-label={iconOnly ? word : undefined}
      title={iconOnly ? word : undefined}
    >
      <CopyIcon size={15} />
      {iconOnly ? null : <span>{word}</span>}
      <span className="sr-only" aria-live="polite">
        {copied === "done" ? "Copied" : copied === "failed" ? "Copy failed" : ""}
      </span>
    </button>
  );
}
