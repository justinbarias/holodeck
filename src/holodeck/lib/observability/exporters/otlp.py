"""OTLP exporter for OpenTelemetry telemetry.

Provides OTLP (gRPC and HTTP) export to any OTLP-compatible collector
(Jaeger, Honeycomb, Datadog, etc.).

Tasks:
    T071 - Implement create_otlp_span_exporter() for gRPC
    T072 - Implement create_otlp_span_exporter() for HTTP
    T073 - Implement create_otlp_metric_exporter() for both protocols
    T074 - Implement create_otlp_log_exporter() for both protocols
    T075 - Implement create_otlp_exporters() factory function
    T076 - Implement resolve_headers() for env var substitution
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse, urlunparse

import grpc
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
    OTLPLogExporter as OTLPLogExporterGRPC,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter as OTLPMetricExporterGRPC,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as OTLPSpanExporterGRPC,
)
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http._log_exporter import (
    OTLPLogExporter as OTLPLogExporterHTTP,
)
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter as OTLPMetricExporterHTTP,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as OTLPSpanExporterHTTP,
)
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

from holodeck.config.env_loader import substitute_env_vars

if TYPE_CHECKING:
    from holodeck.models.observability import OTLPExporterConfig, OTLPProtocol


# Default OTLP ports
GRPC_DEFAULT_PORT = 4317
HTTP_DEFAULT_PORT = 4318


def resolve_headers(headers: dict[str, str]) -> dict[str, str]:
    """Resolve environment variable references in header values.

    Substitutes ${VAR_NAME} patterns with environment variable values.

    Args:
        headers: Dictionary of header names to values (may contain ${VAR} refs)

    Returns:
        Dictionary with all environment variables resolved

    Raises:
        ConfigError: If a referenced environment variable does not exist

    Example:
        >>> import os
        >>> os.environ["API_KEY"] = "secret123"
        >>> resolve_headers({"Authorization": "Bearer ${API_KEY}"})
        {'Authorization': 'Bearer secret123'}
    """
    return {key: substitute_env_vars(value) for key, value in headers.items()}


def get_compression_grpc(compression: str | None) -> grpc.Compression | None:
    """Convert compression string to gRPC Compression enum.

    Args:
        compression: Compression algorithm name ("gzip", "deflate", None)

    Returns:
        grpc.Compression enum value or None
    """
    if compression is None:
        return None

    compression_map = {
        "gzip": grpc.Compression.Gzip,
        "deflate": grpc.Compression.Deflate,
    }
    return compression_map.get(compression.lower())


def get_compression_http(compression: str | None) -> Compression | None:
    """Convert compression string to HTTP Compression enum.

    Args:
        compression: Compression algorithm name ("gzip", "deflate", None)

    Returns:
        opentelemetry Compression enum value or None
    """
    if compression is None:
        return None

    compression_map = {
        "gzip": Compression.Gzip,
        "deflate": Compression.Deflate,
    }
    return compression_map.get(compression.lower())


def adjust_endpoint_for_protocol(
    endpoint: str,
    protocol: OTLPProtocol,
) -> str:
    """Adjust endpoint port based on protocol if using default localhost.

    OTLP conventions:
    - gRPC default port: 4317
    - HTTP default port: 4318

    Only adjusts ports for localhost/127.0.0.1 when using standard OTLP ports.

    Args:
        endpoint: Original endpoint URL
        protocol: OTLP protocol (grpc or http)

    Returns:
        Endpoint with adjusted port if needed
    """
    from holodeck.models.observability import OTLPProtocol

    parsed = urlparse(endpoint)

    # Only adjust if it's localhost
    if parsed.hostname not in ("localhost", "127.0.0.1"):
        return endpoint

    current_port = parsed.port

    # Adjust port based on protocol if using standard OTLP ports
    if protocol == OTLPProtocol.HTTP and current_port == GRPC_DEFAULT_PORT:
        netloc = f"{parsed.hostname}:{HTTP_DEFAULT_PORT}"
        return urlunparse(parsed._replace(netloc=netloc))
    elif protocol == OTLPProtocol.GRPC and current_port == HTTP_DEFAULT_PORT:
        netloc = f"{parsed.hostname}:{GRPC_DEFAULT_PORT}"
        return urlunparse(parsed._replace(netloc=netloc))

    return endpoint


def _headers_to_grpc_metadata(
    headers: dict[str, str] | None,
) -> tuple[tuple[str, str], ...] | None:
    """Convert headers dict to gRPC metadata format.

    gRPC expects headers as a sequence of (key, value) tuples with
    lowercase keys.

    Args:
        headers: Dictionary of headers or None

    Returns:
        Tuple of (key, value) tuples or None
    """
    if not headers:
        return None
    return tuple((key.lower(), value) for key, value in headers.items())


def create_otlp_span_exporter_grpc(config: OTLPExporterConfig) -> OTLPSpanExporterGRPC:
    """Create OTLP span exporter using gRPC protocol.

    Args:
        config: OTLP exporter configuration

    Returns:
        OTLPSpanExporter (gRPC) instance
    """
    endpoint = adjust_endpoint_for_protocol(config.endpoint, config.protocol)
    headers = resolve_headers(config.headers) if config.headers else None
    grpc_headers = _headers_to_grpc_metadata(headers)
    compression = get_compression_grpc(config.compression)
    timeout_seconds = config.timeout_ms / 1000.0

    return OTLPSpanExporterGRPC(
        endpoint=endpoint,
        insecure=config.insecure,
        headers=grpc_headers,
        timeout=timeout_seconds,
        compression=compression,
    )


def create_otlp_span_exporter_http(config: OTLPExporterConfig) -> OTLPSpanExporterHTTP:
    """Create OTLP span exporter using HTTP protocol.

    Args:
        config: OTLP exporter configuration

    Returns:
        OTLPSpanExporter (HTTP) instance
    """
    endpoint = adjust_endpoint_for_protocol(config.endpoint, config.protocol)

    # HTTP endpoint needs /v1/traces suffix
    if not endpoint.endswith("/v1/traces"):
        endpoint = f"{endpoint.rstrip('/')}/v1/traces"

    headers = resolve_headers(config.headers) if config.headers else None
    compression = get_compression_http(config.compression)
    timeout_seconds = config.timeout_ms / 1000.0

    return OTLPSpanExporterHTTP(
        endpoint=endpoint,
        headers=headers,
        timeout=timeout_seconds,
        compression=compression,
    )


def create_otlp_span_exporter(config: OTLPExporterConfig) -> Any:
    """Create OTLP span exporter based on protocol.

    Dispatches to gRPC or HTTP implementation based on config.protocol.

    Args:
        config: OTLP exporter configuration

    Returns:
        OTLPSpanExporter instance (gRPC or HTTP)
    """
    from holodeck.models.observability import OTLPProtocol

    if config.protocol == OTLPProtocol.GRPC:
        return create_otlp_span_exporter_grpc(config)
    else:
        return create_otlp_span_exporter_http(config)


def create_otlp_metric_exporter_grpc(
    config: OTLPExporterConfig,
) -> OTLPMetricExporterGRPC:
    """Create OTLP metric exporter using gRPC protocol.

    Args:
        config: OTLP exporter configuration

    Returns:
        OTLPMetricExporter (gRPC) instance
    """
    endpoint = adjust_endpoint_for_protocol(config.endpoint, config.protocol)
    headers = resolve_headers(config.headers) if config.headers else None
    grpc_headers = _headers_to_grpc_metadata(headers)
    compression = get_compression_grpc(config.compression)
    timeout_seconds = config.timeout_ms / 1000.0

    return OTLPMetricExporterGRPC(
        endpoint=endpoint,
        insecure=config.insecure,
        headers=grpc_headers,
        timeout=timeout_seconds,
        compression=compression,
    )


def create_otlp_metric_exporter_http(
    config: OTLPExporterConfig,
) -> OTLPMetricExporterHTTP:
    """Create OTLP metric exporter using HTTP protocol.

    Args:
        config: OTLP exporter configuration

    Returns:
        OTLPMetricExporter (HTTP) instance
    """
    endpoint = adjust_endpoint_for_protocol(config.endpoint, config.protocol)

    # HTTP endpoint needs /v1/metrics suffix
    if not endpoint.endswith("/v1/metrics"):
        endpoint = f"{endpoint.rstrip('/')}/v1/metrics"

    headers = resolve_headers(config.headers) if config.headers else None
    compression = get_compression_http(config.compression)
    timeout_seconds = config.timeout_ms / 1000.0

    return OTLPMetricExporterHTTP(
        endpoint=endpoint,
        headers=headers,
        timeout=timeout_seconds,
        compression=compression,
    )


def create_otlp_metric_reader(
    config: OTLPExporterConfig,
) -> PeriodicExportingMetricReader:
    """Create OTLP metric reader (wraps exporter in PeriodicExportingMetricReader).

    Args:
        config: OTLP exporter configuration

    Returns:
        PeriodicExportingMetricReader with OTLP exporter
    """
    from holodeck.models.observability import OTLPProtocol

    exporter: OTLPMetricExporterGRPC | OTLPMetricExporterHTTP
    if config.protocol == OTLPProtocol.GRPC:
        exporter = create_otlp_metric_exporter_grpc(config)
    else:
        exporter = create_otlp_metric_exporter_http(config)

    return PeriodicExportingMetricReader(exporter)


def create_otlp_log_exporter_grpc(config: OTLPExporterConfig) -> OTLPLogExporterGRPC:
    """Create OTLP log exporter using gRPC protocol.

    Args:
        config: OTLP exporter configuration

    Returns:
        OTLPLogExporter (gRPC) instance
    """
    endpoint = adjust_endpoint_for_protocol(config.endpoint, config.protocol)
    headers = resolve_headers(config.headers) if config.headers else None
    grpc_headers = _headers_to_grpc_metadata(headers)
    compression = get_compression_grpc(config.compression)
    timeout_seconds = config.timeout_ms / 1000.0

    return OTLPLogExporterGRPC(
        endpoint=endpoint,
        insecure=config.insecure,
        headers=grpc_headers,
        timeout=timeout_seconds,
        compression=compression,
    )


def create_otlp_log_exporter_http(config: OTLPExporterConfig) -> OTLPLogExporterHTTP:
    """Create OTLP log exporter using HTTP protocol.

    Args:
        config: OTLP exporter configuration

    Returns:
        OTLPLogExporter (HTTP) instance
    """
    endpoint = adjust_endpoint_for_protocol(config.endpoint, config.protocol)

    # HTTP endpoint needs /v1/logs suffix
    if not endpoint.endswith("/v1/logs"):
        endpoint = f"{endpoint.rstrip('/')}/v1/logs"

    headers = resolve_headers(config.headers) if config.headers else None
    compression = get_compression_http(config.compression)
    timeout_seconds = config.timeout_ms / 1000.0

    return OTLPLogExporterHTTP(
        endpoint=endpoint,
        headers=headers,
        timeout=timeout_seconds,
        compression=compression,
    )


def create_otlp_log_exporter(config: OTLPExporterConfig) -> Any:
    """Create OTLP log exporter based on protocol.

    Dispatches to gRPC or HTTP implementation based on config.protocol.

    Args:
        config: OTLP exporter configuration

    Returns:
        OTLPLogExporter instance (gRPC or HTTP)
    """
    from holodeck.models.observability import OTLPProtocol

    if config.protocol == OTLPProtocol.GRPC:
        return create_otlp_log_exporter_grpc(config)
    else:
        return create_otlp_log_exporter_http(config)


def create_otlp_exporters(
    config: OTLPExporterConfig,
) -> tuple[Any, PeriodicExportingMetricReader, Any]:
    """Create all OTLP exporters (spans, metrics, logs).

    Factory function that creates all three exporter types for
    the OTLP exporter configuration.

    Args:
        config: OTLP exporter configuration

    Returns:
        Tuple of (span_exporter, metric_reader, log_exporter)

    Example:
        >>> from holodeck.models.observability import OTLPExporterConfig
        >>> config = OTLPExporterConfig(endpoint="http://localhost:4317")
        >>> span_exp, metric_reader, log_exp = create_otlp_exporters(config)
    """
    span_exporter = create_otlp_span_exporter(config)
    metric_reader = create_otlp_metric_reader(config)
    log_exporter = create_otlp_log_exporter(config)

    return span_exporter, metric_reader, log_exporter


__all__ = [
    "resolve_headers",
    "adjust_endpoint_for_protocol",
    "get_compression_grpc",
    "get_compression_http",
    "create_otlp_span_exporter",
    "create_otlp_span_exporter_grpc",
    "create_otlp_span_exporter_http",
    "create_otlp_metric_reader",
    "create_otlp_metric_exporter_grpc",
    "create_otlp_metric_exporter_http",
    "create_otlp_log_exporter",
    "create_otlp_log_exporter_grpc",
    "create_otlp_log_exporter_http",
    "create_otlp_exporters",
]
