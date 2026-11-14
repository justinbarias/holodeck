"use client";

import { useEffect, useRef } from "react";

/** The "How to read this" key: cue numbers, brackets, struck facts, chips, bars. */
export function HowToRead() {
  const ref = useRef<HTMLDetailsElement>(null);

  // Close on Escape or a click outside, like a menu.
  useEffect(() => {
    const details = ref.current;
    if (!details) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && details.open) {
        details.open = false;
        details.querySelector("summary")?.focus();
      }
    };
    const onClick = (event: MouseEvent) => {
      if (details.open && !details.contains(event.target as Node)) details.open = false;
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("click", onClick);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("click", onClick);
    };
  }, []);

  return (
    <details className="pb-key" ref={ref}>
      <summary className="pb-btn pb-btn--outline pb-btn--small">How to read this</summary>
      <div className="pb-key__panel">
        <p>
          <b className="pb-data">Q1, Q2</b> number each tool the agent used, in order. The
          light beside each number says whether it is running, done or failed.
        </p>
        <p>
          <span className="pb-key__bracket" aria-hidden="true" /> A bracket beside an answer
          means it was written after those tools ran; the answer lists them by number.
        </p>
        <p>
          The bar on each tool shows how long it took, on one scale for the whole thread, with
          the time in seconds beside it.
        </p>
        <p>
          <span className="pb-chip">
            <b className="pb-chip__key">A</b>file.md
          </span>{" "}
          Lettered grey chips name the documents a fact came from, in the order they appear.
        </p>
        <p>
          <s>Struck text</s> with a <span className="pb-stamp">CLOSED</span> stamp stopped being
          true on that date. It stays visible as evidence.
        </p>
      </div>
    </details>
  );
}
