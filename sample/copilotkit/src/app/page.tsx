"use client";

import { useRef, useState, useCallback, useEffect } from "react";
import {
  CopilotChat,
  useAgent,
  useCopilotKit,
} from "@copilotkit/react-core/v2";
import "@copilotkit/react-core/v2/styles.css";
import { ToolCallRenderer } from "@/components/ToolCallRenderer";
import { ThemeToggle } from "@/components/ThemeToggle";
import { useRuntimeConfig } from "@/lib/runtime-config-provider";
import { partitionSupportedFiles } from "@/lib/supported-mime";

import type { ContentPart } from "@ag-ui/client";

type FileUpload = {
  contentType: string;
  bytes: string;
  filename?: string;
};

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

export default function Page() {
  const config = useRuntimeConfig();
  const [selectedFiles, setSelectedFiles] = useState<FileUpload[]>([]);
  const [rejectedFiles, setRejectedFiles] = useState<string[]>([]);
  const [inputValue, setInputValue] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const selectedFilesRef = useRef<FileUpload[]>([]);
  const { copilotkit } = useCopilotKit();

  const { agent } = useAgent({ agentId: config.agentId });

  // Keep ref in sync with state for use in callbacks
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

  // Handle file selection
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

  // Handle clipboard paste of files (screenshots, copied files)
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

  // Remove a file from queue
  const removeFile = useCallback((index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  }, []);

  // Handle message submission - ALWAYS use our handler for full control
  const handleSubmitMessage = useCallback(
    async (text: string) => {
      if (!agent) {
        console.error("No agent available");
        return;
      }

      const currentFiles = selectedFilesRef.current;
      const hasText = text.trim().length > 0;
      const hasFiles = currentFiles.length > 0;

      if (!hasText && !hasFiles) return;

      // Build content - use array format when files present, string otherwise
      let content: string | ContentPart[];

      if (hasFiles) {
        const contentParts: ContentPart[] = [];

        if (hasText) {
          contentParts.push({ type: "text", text: text.trim() });
        }

        for (const file of currentFiles) {
          contentParts.push(toFilePart(file));
        }

        content = contentParts;

        console.log("[Multimodal] Sending message with files:", {
          textParts: contentParts.filter((p) => p.type === "text").length,
          fileParts: contentParts.filter((p) => p.type !== "text").length,
        });
      } else {
        content = text.trim();
      }

      // Clear state
      setSelectedFiles([]);
      setInputValue("");

      // Create and send message
      const message = {
        id: globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`,
        role: "user" as const,
        content,
      };

      agent.addMessage(message);
      try {
        await copilotkit.runAgent({ agent });
      } catch (error) {
        console.error("Error running agent:", error);
      }
    },
    [agent, copilotkit]
  );

  // Stop generation
  const handleStop = useCallback(() => {
    agent?.abortRun?.();
  }, [agent]);

  const messages = agent?.messages ?? [];
  const isRunning = agent?.isRunning ?? false;

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: "var(--background)" }}>
      <ToolCallRenderer />
      <header className="holodeck-header">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <h1>{config.agentTitle}</h1>
            <p>{config.agentDescription}</p>
          </div>
          <ThemeToggle />
        </div>
      </header>

      <main className="holodeck-main" style={{ flex: 1, overflow: "hidden", display: "flex", flexDirection: "column" }}>
        {/* Files serve would skip: not queued, named here */}
        {rejectedFiles.length > 0 && (
          <div className="holodeck-file-notice" role="alert">
            <span>
              Not attached (unsupported file type): {rejectedFiles.join(", ")}.
              Supported: PNG, JPEG, GIF, WebP, PDF, DOCX, XLSX, PPTX, TXT, CSV,
              Markdown.
            </span>
            <button
              type="button"
              onClick={() => setRejectedFiles([])}
              className="holodeck-file-notice__dismiss"
              aria-label="Dismiss notice"
            >
              x
            </button>
          </div>
        )}

        {/* File upload queue */}
        {selectedFiles.length > 0 && (
          <div className="holodeck-file-bar">
            {selectedFiles.map((file, index) => (
              <div
                key={index}
                className="relative inline-block w-16 h-16 rounded-lg overflow-hidden border border-zinc-300 dark:border-zinc-600"
              >
                {file.contentType.startsWith("image/") ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={`data:${file.contentType};base64,${file.bytes}`}
                    alt={file.filename || `File ${index + 1}`}
                    className="w-full h-full object-cover"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center bg-zinc-200 dark:bg-zinc-700">
                    <span className="text-xs text-zinc-600 dark:text-zinc-300 text-center px-1">
                      {file.filename?.split(".").pop()?.toUpperCase() || "FILE"}
                    </span>
                  </div>
                )}
                <button
                  onClick={() => removeFile(index)}
                  className="absolute top-0.5 right-0.5 w-5 h-5 flex items-center justify-center bg-black/60 text-white rounded-full text-xs hover:bg-black/80"
                  aria-label="Remove file"
                >
                  x
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Hidden file input */}
        <input
          type="file"
          multiple
          ref={fileInputRef}
          onChange={handleFileSelect}
          accept={config.fileAccept}
          style={{ display: "none" }}
        />

        {/* CopilotChatView - full control over messages and input */}
        <CopilotChat.View
          className="flex-1"
          messages={messages}
          isRunning={isRunning}
          inputValue={inputValue}
          onInputChange={setInputValue}
          onSubmitMessage={handleSubmitMessage}
          onStop={handleStop}
          onAddFile={() => fileInputRef.current?.click()}
        />
      </main>
    </div>
  );
}
