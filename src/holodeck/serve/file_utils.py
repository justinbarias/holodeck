"""Shared file utilities for multimodal content processing.

This module provides common utilities for handling file content across
REST and AG-UI protocols, including:
- MIME type to file type and extension mappings
- FileContent to FileInput conversion
- Temporary file cleanup
- Multimodal file processing
"""

from __future__ import annotations

import base64
import binascii
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from holodeck.lib.logging_config import get_logger
from holodeck.models.test_case import FileInput
from holodeck.serve.models import SUPPORTED_MIME_TYPES

if TYPE_CHECKING:
    from holodeck.models.config import ExecutionConfig
    from holodeck.serve.models import FileContent

logger = get_logger(__name__)


# =============================================================================
# File Size Limits (per spec)
# =============================================================================

MAX_FILE_SIZE_MB = 50
MAX_TOTAL_SIZE_MB = 100
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_TOTAL_SIZE_BYTES = MAX_TOTAL_SIZE_MB * 1024 * 1024


# =============================================================================
# MIME Type Mappings (shared across protocols)
# =============================================================================

# Office document MIME type constants
_WORD_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_EXCEL_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
_PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"

# MIME type to HoloDeck file type mapping
MIME_TO_FILE_TYPE: dict[str, str] = {
    "image/png": "image",
    "image/jpeg": "image",
    "image/gif": "image",
    "image/webp": "image",
    "application/pdf": "pdf",
    _WORD_MIME: "word",
    _EXCEL_MIME: "excel",
    _PPTX_MIME: "powerpoint",
    "text/plain": "text",
    "text/csv": "csv",
    "text/markdown": "text",
}

# MIME type to file extension mapping
MIME_TO_EXTENSION: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
    _WORD_MIME: ".docx",
    _EXCEL_MIME: ".xlsx",
    _PPTX_MIME: ".pptx",
    "text/plain": ".txt",
    "text/csv": ".csv",
    "text/markdown": ".md",
}


# =============================================================================
# FileInput Conversion and Cleanup
# =============================================================================


def create_temp_file_from_bytes(
    content_bytes: bytes,
    mime_type: str,
    description: str | None = None,
) -> FileInput:
    """Create a temporary file from binary content and return FileInput.

    Args:
        content_bytes: Binary content to write to temp file.
        mime_type: MIME type of the content.
        description: Optional description (e.g., filename).

    Returns:
        FileInput suitable for FileProcessor.process_file().
    """
    file_type = MIME_TO_FILE_TYPE.get(mime_type, "text")
    extension = MIME_TO_EXTENSION.get(mime_type, ".bin")

    with tempfile.NamedTemporaryFile(suffix=extension, delete=False, mode="wb") as tmp:
        tmp.write(content_bytes)
        tmp_path = tmp.name

    logger.debug(
        "Created temp file: %s (type=%s, size=%d bytes)",
        tmp_path,
        file_type,
        len(content_bytes),
    )

    return FileInput(
        path=tmp_path,
        url=None,
        type=file_type,
        description=description,
        pages=None,
        sheet=None,
        range=None,
        cache=None,
    )


def convert_file_content_to_file_input(file_content: FileContent) -> FileInput:
    """Convert REST FileContent (base64) to FileProcessor-compatible FileInput.

    Creates a temporary file with the decoded content and returns a FileInput
    pointing to it.

    Args:
        file_content: FileContent with base64-encoded data and MIME type.

    Returns:
        FileInput suitable for FileProcessor.process_file().
    """
    content_bytes = base64.b64decode(file_content.content)
    return create_temp_file_from_bytes(
        content_bytes=content_bytes,
        mime_type=file_content.mime_type,
        description=file_content.filename,
    )


def convert_binary_dict_to_file_input(
    binary_content: dict[str, Any],
) -> FileInput | None:
    """Convert AG-UI binary content dict to FileProcessor-compatible FileInput.

    Handles two transport options:
    - data: Inline base64-encoded content (supported)
    - url: Remote URL reference (NOT supported - SSRF security risk)
    - id: File ID reference (NOT supported)

    Note: URL downloads are intentionally disabled to prevent SSRF attacks.
    Clients must provide file content inline as base64-encoded data.

    Args:
        binary_content: Dict with type, mimeType, and data field.

    Returns:
        FileInput suitable for FileProcessor, or None if not processable.

    Raises:
        ValueError: If base64 decoding fails.
    """
    mime_type = binary_content.get("mimeType", "")
    filename = binary_content.get("filename")

    logger.debug(
        "Converting binary content to FileInput: mime=%s, filename=%s",
        mime_type,
        filename,
    )

    # Handle inline base64 data
    if binary_content.get("data"):
        data = binary_content["data"]
        logger.debug("Processing base64 data (length=%d chars)", len(data))
        try:
            content_bytes = base64.b64decode(data)
        except (ValueError, binascii.Error) as e:
            logger.error("Base64 decode failed: %s", e, exc_info=True)
            raise ValueError(f"Invalid base64 data: {e}") from e

        return create_temp_file_from_bytes(
            content_bytes=content_bytes,
            mime_type=mime_type,
            description=filename,
        )

    # Handle URL reference (DISABLED for security - SSRF risk)
    if binary_content.get("url"):
        url = binary_content["url"]
        logger.warning(
            "URL file references are disabled for security (SSRF prevention). "
            "Please provide file content as inline base64 data instead. url=%s",
            url,
        )
        return None

    # Handle file ID reference (not supported)
    if binary_content.get("id"):
        file_id = binary_content["id"]
        logger.warning(
            "File ID references are not supported. "
            "Use inline base64 data instead. id=%s",
            file_id,
        )
        return None

    logger.warning(
        "Binary content has no data, url, or id field - skipping. Content keys: %s",
        list(binary_content.keys()),
    )
    return None


def cleanup_temp_file(file_input: FileInput) -> None:
    """Clean up temporary file created during file conversion.

    Args:
        file_input: FileInput with path to temporary file.
    """
    if file_input.path:
        try:
            Path(file_input.path).unlink(missing_ok=True)
            logger.debug("Cleaned up temp file: %s", file_input.path)
        except Exception as e:
            logger.warning("Failed to cleanup temp file %s: %s", file_input.path, e)


def cleanup_temp_files(file_inputs: list[FileInput]) -> None:
    """Clean up multiple temporary files.

    Args:
        file_inputs: List of FileInput objects to clean up.
    """
    for file_input in file_inputs:
        cleanup_temp_file(file_input)


# =============================================================================
# Multimodal File Processing
# =============================================================================


def process_multimodal_files(
    files: list[FileContent] | list[dict[str, Any]],
    execution_config: ExecutionConfig | None = None,
    is_agui_format: bool = False,
) -> tuple[str, list[FileInput]]:
    """Process multimodal files and return combined content with cleanup list.

    This is the unified file processing function for both REST and AG-UI protocols.

    Args:
        files: List of FileContent objects (REST) or binary content dicts (AG-UI).
        execution_config: Optional execution configuration for FileProcessor.
        is_agui_format: If True, treat files as AG-UI binary content dicts.

    Returns:
        Tuple of (combined_markdown_content, list_of_file_inputs_for_cleanup).
    """
    from holodeck.lib.file_processor import FileProcessor

    if not files:
        return "", []

    # Create FileProcessor
    if execution_config:
        processor = FileProcessor.from_execution_config(execution_config)
    else:
        processor = FileProcessor()

    file_inputs: list[FileInput] = []
    file_contents: list[str] = []

    for idx, file_item in enumerate(files):
        file_input: FileInput | None = None
        filename: str | None = None

        try:
            # Convert to FileInput based on format
            if is_agui_format:
                # AG-UI binary content dict
                binary_dict: dict[str, Any] = file_item  # type: ignore[assignment]
                filename = binary_dict.get("filename")
                file_input = convert_binary_dict_to_file_input(binary_dict)
            else:
                # REST FileContent Pydantic model
                file_content: FileContent = file_item  # type: ignore[assignment]
                filename = file_content.filename
                file_input = convert_file_content_to_file_input(file_content)

            if file_input is None:
                logger.debug(
                    "File %d/%d: conversion returned None, skipping",
                    idx + 1,
                    len(files),
                )
                continue

            # Add to cleanup list immediately after creation
            file_inputs.append(file_input)

            # Process file
            result = processor.process_file(file_input)

            if result.error:
                logger.warning(
                    "File %d/%d processing error: %s",
                    idx + 1,
                    len(files),
                    result.error,
                )
                file_contents.append(f"[File processing error: {result.error}]")
            elif result.markdown_content:
                # Add filename header for AG-UI format
                if is_agui_format and filename:
                    file_contents.append(
                        f"## File: {filename}\n\n{result.markdown_content}"
                    )
                else:
                    file_contents.append(result.markdown_content)
            else:
                logger.debug(
                    "File %d/%d: FileProcessor returned no content",
                    idx + 1,
                    len(files),
                )

        except Exception as e:
            logger.error(
                "File %d/%d: exception during processing: %s",
                idx + 1,
                len(files),
                e,
                exc_info=True,
            )
            file_contents.append(f"[Error processing file: {e}]")

    return "\n\n".join(file_contents), file_inputs


# =============================================================================
# AG-UI Binary Content Extraction
# =============================================================================


def extract_binary_parts_from_content(
    content: list[dict[str, Any] | Any],
) -> list[dict[str, Any]]:
    """Extract binary content parts from AG-UI message content list.

    Filters the content list for binary type parts and validates MIME types.
    Handles both dict format and AG-UI Pydantic objects (BinaryInputContent).

    Args:
        content: List of content parts (text, binary, or strings).

    Returns:
        List of binary content dicts with type, mimeType, and data/url/id fields.
    """
    binary_parts: list[dict[str, Any]] = []

    logger.debug(
        "Scanning content list for binary parts (total items: %d)",
        len(content),
    )

    for idx, part in enumerate(content):
        # Handle dict format
        if isinstance(part, dict):
            if part.get("type") != "binary":
                continue

            mime_type = part.get("mimeType", "")
            if mime_type not in SUPPORTED_MIME_TYPES:
                logger.warning(
                    "Skipping binary content with unsupported MIME type: %s",
                    mime_type,
                )
                continue

            logger.debug(
                "Found binary content (dict): idx=%d, mime=%s, filename=%s",
                idx,
                mime_type,
                part.get("filename"),
            )
            binary_parts.append(part)

        # Handle AG-UI Pydantic object (BinaryInputContent)
        elif hasattr(part, "type") and getattr(part, "type", None) == "binary":
            # AG-UI uses 'mime_type' attribute (snake_case), not 'mimeType'
            mime_type = getattr(part, "mime_type", "")
            if mime_type not in SUPPORTED_MIME_TYPES:
                logger.warning(
                    "Skipping binary content with unsupported MIME type: %s",
                    mime_type,
                )
                continue

            # Convert Pydantic object to dict for consistent handling
            binary_parts.append(
                {
                    "type": "binary",
                    "mimeType": mime_type,
                    "data": getattr(part, "data", None),
                    "url": getattr(part, "url", None),
                    "id": getattr(part, "id", None),
                    "filename": getattr(part, "filename", None),
                }
            )
            logger.debug(
                "Found binary content (Pydantic): idx=%d, mime=%s, filename=%s",
                idx,
                mime_type,
                getattr(part, "filename", None),
            )

    logger.debug("Extracted %d binary parts from content", len(binary_parts))
    return binary_parts
