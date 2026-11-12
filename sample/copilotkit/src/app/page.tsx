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

/**
 * Content part types for AG-UI multimodal messages
 */
type TextContent = {
  type: "text";
  text: string;
};

type BinaryContent = {
  type: "binary";
  mimeType: string;
  data: string;
  filename?: string;
};

type ContentPart = TextContent | BinaryContent;

type FileUpload = {
  contentType: string;
  bytes: string;
  filename?: string;
};

export default function Page() {
  const config = useRuntimeConfig();
  const [selectedFiles, setSelectedFiles] = useState<FileUpload[]>([]);
  const [inputValue, setInputValue] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const selectedFilesRef = useRef<FileUpload[]>([]);
  const { copilotkit } = useCopilotKit();

  const { agent } = useAgent({ agentId: config.agentId });

  // Keep ref in sync with state for use in callbacks
  useEffect(() => {
    selectedFilesRef.current = selectedFiles;
  }, [selectedFiles]);

  // Handle file selection
  const handleFileSelect = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const files = event.target.files;
      if (!files || files.length === 0) return;

      const filePromises = Array.from(files).map((file) => {
        return new Promise<FileUpload | null>((resolve) => {
          const reader = new FileReader();
          reader.onload = (e) => {
            const base64 = (e.target?.result as string)?.split(",")[1];
            if (base64) {
              resolve({
                contentType: file.type,
                bytes: base64,
                filename: file.name,
              });
            } else {
              resolve(null);
            }
          };
          reader.onerror = () => resolve(null);
          reader.readAsDataURL(file);
        });
      });

      const loadedFiles = (await Promise.all(filePromises)).filter(
        (f): f is FileUpload => f !== null
      );
      setSelectedFiles((prev) => [...prev, ...loadedFiles]);

      if (event.target) {
        event.target.value = "";
      }
    },
    []
  );

  // Handle clipboard paste for images
  useEffect(() => {
    const handlePaste = async (e: ClipboardEvent) => {
      const items = Array.from(e.clipboardData?.items || []);
      const imageItems = items.filter((item) => item.type.startsWith("image/"));
      if (imageItems.length === 0) return;

      e.preventDefault();

      const imagePromises = imageItems.map((item) => {
        const file = item.getAsFile();
        if (!file) return Promise.resolve(null);

        return new Promise<FileUpload | null>((resolve) => {
          const reader = new FileReader();
          reader.onload = (e) => {
            const base64 = (e.target?.result as string)?.split(",")[1];
            if (base64) {
              resolve({
                contentType: file.type,
                bytes: base64,
                filename: `pasted-${Date.now()}.${file.type.split("/")[1]}`,
              });
            } else {
              resolve(null);
            }
          };
          reader.onerror = () => resolve(null);
          reader.readAsDataURL(file);
        });
      });

      const loadedImages = (await Promise.all(imagePromises)).filter(
        (img): img is FileUpload => img !== null
      );
      setSelectedFiles((prev) => [...prev, ...loadedImages]);
    };

    document.addEventListener("paste", handlePaste);
    return () => document.removeEventListener("paste", handlePaste);
  }, []);

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
          contentParts.push({
            type: "binary",
            mimeType: file.contentType,
            data: file.bytes,
            filename: file.filename,
          });
        }

        content = contentParts;

        console.log("[Multimodal] Sending message with files:", {
          textParts: contentParts.filter((p) => p.type === "text").length,
          binaryParts: contentParts.filter((p) => p.type === "binary").length,
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
        />
      </main>
    </div>
  );
}
