import { firstSentence } from "@/lib/tool-view";
import { invalidDateSentence } from "./card-view";

/** A lettered neutral chip for a plain source document. */
export function SourceChip({
  letter,
  children,
  title,
}: {
  letter: string;
  children?: React.ReactNode;
  title?: string;
}) {
  return (
    <span className="pb-chip" title={title}>
      <b className="pb-chip__key">{letter}</b>
      {children}
    </span>
  );
}

/** CLOSED / RETRACTED stamp. */
export function Stamp({ children }: { children: React.ReactNode }) {
  return <span className="pb-stamp">{children}</span>;
}

/** The plain-words headline of a memory tool error. */
export function memoryErrorHeadline(
  message: string,
  args: Record<string, unknown> | undefined,
): string {
  return invalidDateSentence(message, args) ?? firstSentence(message);
}

const CODE_TEXT: Record<string, string> = {
  invalid_argument: "The agent passed memory an argument it could not read.",
  not_supported: "This memory store does not support the request.",
  backend_unavailable: "The memory store did not answer.",
};

/** Body of a failed memory card: plain words first, the raw message after. */
export function MemoryErrorBody({
  code,
  message,
  args,
}: {
  code: string;
  message: string;
  args: Record<string, unknown> | undefined;
}) {
  const dateSentence = invalidDateSentence(message, args);
  return (
    <div className="pb-card__pad">
      <p className="pb-card__message">
        <strong>{dateSentence ?? firstSentence(message)}</strong>{" "}
        {CODE_TEXT[code] ?? ""}
        {dateSentence ? " The agent may retry with an exact date, for example 2025-07-01." : ""}
      </p>
      <p className="pb-card__detail">
        <span className="pb-data">{code}</span> · {message}
      </p>
    </div>
  );
}
