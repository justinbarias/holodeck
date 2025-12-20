"""StructuredDataLoader for loading and iterating over structured data files.

This module provides functionality to load structured data from CSV, JSON, and JSONL
files and iterate over records with field mapping for vector store ingestion.

Features:
- Automatic file type detection from extension
- CSV delimiter auto-detection
- Schema validation before ingestion
- Memory-efficient streaming via generators
- Batch processing for large files
- Field concatenation for multiple vector fields
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import Any

from holodeck.lib.errors import ConfigError

logger = logging.getLogger(__name__)


class FileType(Enum):
    """Supported structured data file types."""

    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"


def detect_file_type(source_path: str) -> FileType:
    """Detect file type from file extension.

    Args:
        source_path: Path to the source file.

    Returns:
        FileType enum value for the detected type.

    Raises:
        ValueError: If file extension is not supported.

    Example:
        >>> detect_file_type("data/products.csv")
        FileType.CSV
    """
    path = Path(source_path)
    extension = path.suffix.lower()

    type_map = {
        ".csv": FileType.CSV,
        ".json": FileType.JSON,
        ".jsonl": FileType.JSONL,
    }

    if extension not in type_map:
        raise ValueError(
            f"Unsupported file type: '{extension}'. "
            f"Supported types: {list(type_map.keys())}"
        )

    return type_map[extension]


def detect_csv_delimiter(content: str) -> str:
    """Auto-detect CSV delimiter from file content.

    Uses csv.Sniffer to detect the delimiter from a sample of the content.
    Falls back to comma if detection fails.

    Args:
        content: Sample content from the CSV file (first few lines).

    Returns:
        Detected delimiter character.

    Example:
        >>> detect_csv_delimiter("id,name,value\\n1,test,100")
        ','
        >>> detect_csv_delimiter("id;name;value\\n1;test;100")
        ';'
    """
    try:
        # Use csv.Sniffer to detect delimiter
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(content, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        # Fall back to comma if detection fails
        logger.debug("CSV delimiter detection failed, defaulting to comma")
        return ","


def load_csv_records(source_path: str, delimiter: str) -> Iterator[dict[str, str]]:
    """Load records from a CSV file as a generator.

    Streams records from the CSV file to minimize memory usage.

    Args:
        source_path: Path to the CSV file.
        delimiter: Field delimiter character.

    Yields:
        Dictionary for each row with column names as keys.

    Raises:
        FileNotFoundError: If the file doesn't exist.

    Example:
        >>> for record in load_csv_records("products.csv", ","):
        ...     print(record["id"], record["name"])
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {source_path}")

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            yield dict(row)


def load_json_records(source_path: str) -> list[dict[str, Any]]:
    """Load records from a JSON file containing an array.

    Loads the entire JSON array into memory.

    Args:
        source_path: Path to the JSON file.

    Returns:
        List of dictionaries from the JSON array.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the JSON is invalid.

    Example:
        >>> records = load_json_records("faqs.json")
        >>> len(records)
        2
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {source_path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array, got {type(data).__name__}")

    return data


def load_jsonl_records(source_path: str) -> Iterator[dict[str, Any]]:
    """Load records from a JSONL file (one JSON object per line).

    Streams records from the JSONL file to minimize memory usage.
    Empty lines are skipped.

    Args:
        source_path: Path to the JSONL file.

    Yields:
        Dictionary for each line.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If any line contains invalid JSON.

    Example:
        >>> for record in load_jsonl_records("products.jsonl"):
        ...     print(record["product_id"])
    """
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {source_path}")

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                # Skip empty lines
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {line_num}: {e.msg}",
                    e.doc,
                    e.pos,
                ) from e


class StructuredDataLoader:
    """Load and iterate over structured data from CSV, JSON, or JSONL files.

    This class provides a unified interface for loading structured data from
    various file formats and iterating over records with field mapping.

    Attributes:
        source_path: Path to the source data file.
        id_field: Field name to use as unique record identifier.
        vector_fields: List of field names whose values will be embedded.
        metadata_fields: List of field names to include as metadata.
        field_separator: Separator for concatenating multiple vector fields.
        delimiter: CSV delimiter (auto-detected if None).
        batch_size: Number of records per batch.

    Example:
        >>> loader = StructuredDataLoader(
        ...     source_path="products.csv",
        ...     id_field="id",
        ...     vector_fields=["description"],
        ...     metadata_fields=["title", "category"],
        ... )
        >>> for record in loader.iter_records():
        ...     print(record["id"], record["content"])
    """

    def __init__(
        self,
        source_path: str,
        id_field: str,
        vector_fields: list[str],
        metadata_fields: list[str] | None = None,
        field_separator: str = "\n",
        delimiter: str | None = None,
        batch_size: int = 10_000,
    ) -> None:
        """Initialize the StructuredDataLoader.

        Args:
            source_path: Path to the source data file.
            id_field: Field name to use as unique record identifier.
            vector_fields: List of field names whose values will be embedded.
            metadata_fields: List of field names to include as metadata.
                If None, includes all fields except id_field and vector_fields.
            field_separator: Separator for concatenating multiple vector fields.
            delimiter: CSV delimiter. If None, auto-detected from content.
            batch_size: Number of records per batch for iter_batches().
        """
        self.source_path = source_path
        self.id_field = id_field
        self.vector_fields = vector_fields
        self.metadata_fields = metadata_fields
        self.field_separator = field_separator
        self.delimiter = delimiter
        self.batch_size = batch_size

        # Cached file type (lazy detection)
        self._file_type: FileType | None = None
        self._detected_delimiter: str | None = None

    @property
    def file_type(self) -> FileType:
        """Get the detected file type (cached)."""
        if self._file_type is None:
            self._file_type = detect_file_type(self.source_path)
        return self._file_type

    def _get_csv_delimiter(self) -> str:
        """Get the CSV delimiter, auto-detecting if not specified."""
        if self.delimiter is not None:
            return self.delimiter

        if self._detected_delimiter is not None:
            return self._detected_delimiter

        # Read first few lines for detection
        path = Path(self.source_path)
        with open(path, encoding="utf-8") as f:
            sample = f.read(8192)  # Read up to 8KB for detection

        self._detected_delimiter = detect_csv_delimiter(sample)
        logger.debug(f"Auto-detected CSV delimiter: {self._detected_delimiter!r}")
        return self._detected_delimiter

    def _get_sample_record(self) -> dict[str, Any]:
        """Get a sample record for schema validation."""
        if self.file_type == FileType.CSV:
            delimiter = self._get_csv_delimiter()
            for record in load_csv_records(self.source_path, delimiter):
                return record
        elif self.file_type == FileType.JSON:
            records = load_json_records(self.source_path)
            if records:
                return records[0]
        elif self.file_type == FileType.JSONL:
            for record in load_jsonl_records(self.source_path):
                return record

        raise ConfigError(
            field="source",
            message=f"No records found in source file: {self.source_path}",
        )

    def validate_schema(self) -> list[str]:
        """Validate that configured fields exist in the source data.

        Checks that id_field, vector_fields, and metadata_fields all exist
        in the source data schema.

        Returns:
            List of available field names in the source.

        Raises:
            ConfigError: If any configured field doesn't exist in the source.

        Example:
            >>> loader = StructuredDataLoader(...)
            >>> available_fields = loader.validate_schema()
            >>> print(available_fields)
            ['id', 'title', 'description', 'category', 'price']
        """
        sample = self._get_sample_record()
        available_fields = list(sample.keys())

        # Validate id_field
        if self.id_field not in available_fields:
            raise ConfigError(
                field="id_field",
                message=(
                    f"Field '{self.id_field}' not found in source. "
                    f"Available fields: {available_fields}"
                ),
            )

        # Validate vector_fields
        for field in self.vector_fields:
            if field not in available_fields:
                raise ConfigError(
                    field="vector_field",
                    message=(
                        f"Field '{field}' not found in source. "
                        f"Available fields: {available_fields}"
                    ),
                )

        # Validate metadata_fields (if specified)
        if self.metadata_fields:
            for field in self.metadata_fields:
                if field not in available_fields:
                    raise ConfigError(
                        field="metadata_field",
                        message=(
                            f"Field '{field}' not found in source. "
                            f"Available fields: {available_fields}"
                        ),
                    )

        return available_fields

    def _get_raw_records(self) -> Iterator[dict[str, Any]]:
        """Get raw records from the source file."""
        if self.file_type == FileType.CSV:
            delimiter = self._get_csv_delimiter()
            yield from load_csv_records(self.source_path, delimiter)
        elif self.file_type == FileType.JSON:
            yield from load_json_records(self.source_path)
        elif self.file_type == FileType.JSONL:
            yield from load_jsonl_records(self.source_path)

    def _determine_metadata_fields(self, available_fields: list[str]) -> list[str]:
        """Determine which fields to include as metadata.

        If metadata_fields is specified, use those.
        Otherwise, include all fields except id_field and vector_fields.
        """
        if self.metadata_fields is not None:
            return self.metadata_fields

        # Auto-include all non-id, non-vector fields
        excluded = {self.id_field} | set(self.vector_fields)
        return [f for f in available_fields if f not in excluded]

    def _build_content(self, record: dict[str, Any]) -> str:
        """Build content string from vector fields."""
        values = []
        for field in self.vector_fields:
            value = record.get(field)
            if value is not None:
                # Convert non-string values to string
                values.append(str(value))
        return self.field_separator.join(values)

    def _build_metadata(
        self, record: dict[str, Any], metadata_fields: list[str]
    ) -> dict[str, str]:
        """Build metadata dictionary from configured fields."""
        metadata = {}
        for field in metadata_fields:
            value = record.get(field)
            if value is not None:
                # Convert non-string values to string
                metadata[field] = str(value)
            else:
                metadata[field] = ""
        return metadata

    def iter_records(self) -> Iterator[dict[str, Any]]:
        """Iterate over records with field mapping applied.

        Yields dictionaries with keys:
        - id: The record identifier (from id_field)
        - content: Concatenated vector field values
        - metadata: Dictionary of metadata field values

        Records with empty content (all vector fields empty) are skipped.

        Yields:
            Dictionary with id, content, and metadata keys.

        Example:
            >>> for record in loader.iter_records():
            ...     print(record["id"])
            ...     print(record["content"])
            ...     print(record["metadata"])
        """
        # Get first record to determine metadata fields
        sample = self._get_sample_record()
        available_fields = list(sample.keys())
        metadata_fields = self._determine_metadata_fields(available_fields)

        skipped_count = 0
        for record in self._get_raw_records():
            # Build content from vector fields
            content = self._build_content(record)

            # Skip records with empty content
            if not content.strip():
                skipped_count += 1
                continue

            # Get record ID (convert to string)
            record_id = str(record.get(self.id_field, ""))

            # Build metadata
            metadata = self._build_metadata(record, metadata_fields)

            yield {
                "id": record_id,
                "content": content,
                "metadata": metadata,
            }

        if skipped_count > 0:
            logger.warning(
                f"Skipped {skipped_count} records with empty vector field content"
            )

    def iter_batches(self) -> Iterator[list[dict[str, Any]]]:
        """Iterate over batches of records.

        Groups records into batches of batch_size for efficient processing.

        Yields:
            List of record dictionaries (up to batch_size per batch).

        Example:
            >>> for batch in loader.iter_batches():
            ...     print(f"Processing {len(batch)} records")
            ...     for record in batch:
            ...         process(record)
        """
        batch: list[dict[str, Any]] = []

        for record in self.iter_records():
            batch.append(record)

            if len(batch) >= self.batch_size:
                yield batch
                batch = []

        # Yield remaining records
        if batch:
            yield batch
