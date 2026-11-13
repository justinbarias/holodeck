import { test } from "node:test";
import assert from "node:assert/strict";
import {
  DEFAULT_FILE_ACCEPT,
  SUPPORTED_MIME_TYPES,
  partitionSupportedFiles,
  supportedMimeType,
} from "./supported-mime.ts";

test("mirrors serve's SUPPORTED_MIME_TYPES exactly", () => {
  assert.deepEqual([...SUPPORTED_MIME_TYPES].sort(), [
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
    "text/csv",
    "text/markdown",
    "text/plain",
  ]);
});

test("default accept has no wildcard and no legacy .doc", () => {
  const entries = DEFAULT_FILE_ACCEPT.split(",");
  assert.ok(!entries.includes("image/*"));
  assert.ok(!entries.includes(".doc"));
  assert.ok(entries.includes("application/pdf"));
  assert.ok(entries.includes(".docx"));
});

test("supported type is accepted as reported", () => {
  assert.equal(supportedMimeType("a.png", "image/png"), "image/png");
});

test("unsupported reported type is rejected even with a known extension", () => {
  assert.equal(supportedMimeType("a.png", "image/heic"), null);
  assert.equal(supportedMimeType("a.doc", "application/msword"), null);
  assert.equal(supportedMimeType("a.bmp", "image/bmp"), null);
});

test("empty type falls back to the extension", () => {
  assert.equal(supportedMimeType("notes.MD", ""), "text/markdown");
  assert.equal(supportedMimeType("data.csv", ""), "text/csv");
  assert.equal(supportedMimeType("archive.zip", ""), null);
  assert.equal(supportedMimeType("noextension", ""), null);
});

test("partition keeps order and splits accepted from rejected", () => {
  const files = [
    { name: "a.pdf", type: "application/pdf" },
    { name: "b.doc", type: "application/msword" },
    { name: "c.md", type: "" },
    { name: "d.mp4", type: "video/mp4" },
  ];
  const { accepted, rejected } = partitionSupportedFiles(files);
  assert.deepEqual(
    accepted.map((a) => [a.file.name, a.mimeType]),
    [
      ["a.pdf", "application/pdf"],
      ["c.md", "text/markdown"],
    ],
  );
  assert.deepEqual(
    rejected.map((r) => r.name),
    ["b.doc", "d.mp4"],
  );
});
