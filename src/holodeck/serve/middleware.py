"""Middleware for the Agent Local Server.

Provides logging, error handling, and other cross-cutting concerns
for the FastAPI application.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Awaitable, Callable
from contextlib import nullcontext
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from holodeck.lib.logging_config import get_logger
from holodeck.serve.models import ProblemDetail

logger = get_logger(__name__)

# Type alias for the middleware call_next function
RequestCallNext = Callable[[Request], Awaitable[Response]]


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging and tracing.

    Captures request metadata including timestamp, endpoint, session ID,
    and latency for each request. When observability is enabled, creates
    per-request trace spans with HTTP attributes.
    """

    def __init__(
        self, app: Callable, debug: bool = False, observability_enabled: bool = False
    ) -> None:
        """Initialize logging middleware.

        Args:
            app: The ASGI application.
            debug: Enable verbose logging of full request/response content.
            observability_enabled: Enable OpenTelemetry per-request tracing.
        """
        super().__init__(app)
        self.debug = debug
        self.observability_enabled = observability_enabled

    async def dispatch(self, request: Request, call_next: RequestCallNext) -> Response:
        """Process request, log metadata, and create trace span if enabled.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware/handler in the chain.

        Returns:
            The HTTP response.
        """
        start_time = time.perf_counter()

        # Extract session ID from various sources
        session_id = self._extract_session_id(request)

        # Create span context for per-request tracing
        if self.observability_enabled:
            from opentelemetry import trace

            from holodeck.lib.observability import get_tracer

            tracer = get_tracer(__name__)
            span_context: Any = tracer.start_as_current_span(
                "holodeck.serve.request",
                kind=trace.SpanKind.SERVER,
            )
        else:
            span_context = nullcontext()

        with span_context as span:
            # Set initial span attributes if span exists
            if span:
                span.set_attribute("http.method", request.method)
                span.set_attribute("http.route", request.url.path)
                if session_id:
                    span.set_attribute("session.id", session_id)

            # Log request
            logger.info(
                "Request started",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "session_id": session_id,
                },
            )

            if self.debug:
                logger.debug(
                    "Request details",
                    extra={
                        "headers": dict(request.headers),
                        "query_params": dict(request.query_params),
                    },
                )

            # Process request
            response = await call_next(request)

            # Set response status on span
            if span:
                span.set_attribute("http.status_code", response.status_code)

            # Calculate latency
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Log response
            logger.info(
                "Request completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": round(latency_ms, 2),
                    "session_id": session_id,
                },
            )

            return response

    def _extract_session_id(self, request: Request) -> str | None:
        """Extract session ID from request.

        Args:
            request: The HTTP request.

        Returns:
            Session ID if found, None otherwise.
        """
        # Check header first
        session_id: str | None = request.headers.get("X-Session-ID")
        if session_id:
            return session_id

        # Check query params
        session_id = request.query_params.get("session_id")
        if session_id:
            return str(session_id)

        return None


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for standardized error handling.

    Catches unhandled exceptions and returns RFC 7807 Problem Details
    formatted error responses.
    """

    def __init__(self, app: Callable, debug: bool = False) -> None:
        """Initialize error handling middleware.

        Args:
            app: The ASGI application.
            debug: Include stack traces in error responses.
        """
        super().__init__(app)
        self.debug = debug

    async def dispatch(self, request: Request, call_next: RequestCallNext) -> Response:
        """Process request and handle errors.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware/handler in the chain.

        Returns:
            The HTTP response, or an error response on exception.
        """
        try:
            return await call_next(request)
        except Exception as e:
            return self._create_error_response(request, e)

    def _create_error_response(
        self,
        request: Request,
        error: Exception,
    ) -> JSONResponse:
        """Create RFC 7807 Problem Details error response.

        Args:
            request: The HTTP request that caused the error.
            error: The exception that was raised.

        Returns:
            JSON response with problem details.
        """
        # Log the error
        logger.error(
            f"Unhandled error: {error}",
            extra={
                "path": request.url.path,
                "method": request.method,
                "error_type": type(error).__name__,
            },
            exc_info=True,
        )

        # Determine status code and type
        status_code = 500
        error_type = "https://holodeck.dev/errors/internal-error"
        title = "Internal Server Error"

        # Build detail message
        detail = str(error) if self.debug else "An unexpected error occurred."

        if self.debug:
            detail += f"\n\nStack trace:\n{traceback.format_exc()}"

        problem = ProblemDetail(
            type=error_type,
            title=title,
            status=status_code,
            detail=detail,
            instance=str(request.url.path),
        )

        return JSONResponse(
            status_code=status_code,
            content=problem.model_dump(exclude_none=True),
            media_type="application/problem+json",
        )
