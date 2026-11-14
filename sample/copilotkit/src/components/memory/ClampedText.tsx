"use client";

import { useId, useState } from "react";
import { needsExpander } from "./search-view";

/**
 * Passage text clamped to a few lines, with a button to show all of it. A
 * short text has no button. The button is a native <button>, so Enter and
 * Space work and the focus ring shows.
 */
export function ClampedText({ text, className }: { text: string; className?: string }) {
  const [expanded, setExpanded] = useState(false);
  const id = useId();
  const long = needsExpander(text);
  return (
    <>
      <p
        id={id}
        className={className}
        data-clamped={long && !expanded ? "true" : "false"}
      >
        {text}
      </p>
      {long ? (
        <button
          type="button"
          className="pb-link pb-link--tight"
          aria-expanded={expanded}
          aria-controls={id}
          onClick={() => setExpanded((value) => !value)}
        >
          {expanded ? "Show less" : "Show more"}
        </button>
      ) : null}
    </>
  );
}
