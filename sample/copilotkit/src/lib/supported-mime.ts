/**
 * File types that `holodeck serve` turns into agent context.
 *
 * Mirrors SUPPORTED_MIME_TYPES in src/holodeck/serve/models.py (repository
 * root). serve skips any other type, so the sample does not queue it.
 * Keep the two lists in sync.
 */

/** Supported MIME type -> file extensions (used for `accept` and fallback). */
const SUPPORTED: Record<string, readonly string[]> = {
  "image/png": [".png"],
  "image/jpeg": [".jpg", ".jpeg"],
  "image/gif": [".gif"],
  "image/webp": [".webp"],
  "application/pdf": [".pdf"],
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [
    ".docx",
  ],
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [
    ".xlsx",
  ],
  "application/vnd.openxmlformats-officedocument.presentationml.presentation": [
    ".pptx",
  ],
  "text/plain": [".txt"],
  "text/csv": [".csv"],
  "text/markdown": [".md"],
};

export const SUPPORTED_MIME_TYPES: readonly string[] = Object.keys(SUPPORTED);

/** Default `accept` value for the file input: every supported type and extension. */
export const DEFAULT_FILE_ACCEPT = [
  ...SUPPORTED_MIME_TYPES,
  ...Object.values(SUPPORTED).flat(),
].join(",");

/**
 * Return the supported MIME type for a file, or null when serve would skip it.
 *
 * Browsers report an empty type for some files (often `.md` or `.csv`); then
 * the extension decides. A non-empty but unsupported type is rejected.
 */
export function supportedMimeType(name: string, type: string): string | null {
  if (type) {
    return SUPPORTED_MIME_TYPES.includes(type) ? type : null;
  }
  const dot = name.lastIndexOf(".");
  if (dot < 0) return null;
  const extension = name.slice(dot).toLowerCase();
  for (const [mime, extensions] of Object.entries(SUPPORTED)) {
    if (extensions.includes(extension)) return mime;
  }
  return null;
}

/** Split files into those serve accepts (with their MIME type) and the rest. */
export function partitionSupportedFiles<T extends { name: string; type: string }>(
  files: readonly T[],
): { accepted: { file: T; mimeType: string }[]; rejected: T[] } {
  const accepted: { file: T; mimeType: string }[] = [];
  const rejected: T[] = [];
  for (const file of files) {
    const mimeType = supportedMimeType(file.name, file.type);
    if (mimeType) {
      accepted.push({ file, mimeType });
    } else {
      rejected.push(file);
    }
  }
  return { accepted, rejected };
}
