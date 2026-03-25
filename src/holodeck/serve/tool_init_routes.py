"""Tool initialization HTTP routes.

Provides the POST /tools/{tool_name}/init endpoint for triggering
background tool data ingestion jobs.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from holodeck.lib.logging_config import get_logger
from holodeck.serve.models import InitJobResponse, ProblemDetail
from holodeck.serve.tool_init_manager import (
    InitJob,
    InitJobCapacityError,
    InitJobConflictError,
    InitManagerShuttingDownError,
    ToolNotFoundError,
    ToolNotInitializableError,
)

logger = get_logger(__name__)

router = APIRouter(tags=["Tool Init"])


def _job_to_response(job: InitJob) -> InitJobResponse:
    """Convert an internal InitJob to the API response model."""
    return InitJobResponse(
        tool_name=job.tool_name,
        state=job.state,
        href=f"/tools/{job.tool_name}/init",
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        message=job.message,
        error_detail=job.error_detail,
        progress=job.progress,
        force=job.force,
    )


def _problem_response(
    status: int,
    title: str,
    detail: str,
    instance: str,
) -> JSONResponse:
    """Build a ProblemDetail JSONResponse."""
    problem = ProblemDetail(
        title=title,
        status=status,
        detail=detail,
        instance=instance,
    )
    return JSONResponse(
        status_code=status,
        content=problem.model_dump(exclude_none=True),
        media_type="application/problem+json",
    )


# Exception type → (status_code, title)
_ERROR_MAP: dict[type[Exception], tuple[int, str]] = {
    ToolNotInitializableError: (400, "Bad Request"),
    ToolNotFoundError: (404, "Not Found"),
    InitJobConflictError: (409, "Conflict"),
    InitJobCapacityError: (429, "Too Many Requests"),
    InitManagerShuttingDownError: (503, "Service Unavailable"),
}


@router.post(
    "/tools/{tool_name}/init",
    status_code=201,
    response_model=InitJobResponse,
    responses={
        400: {"model": ProblemDetail},
        404: {"model": ProblemDetail},
        409: {"model": ProblemDetail},
        429: {"model": ProblemDetail},
        503: {"model": ProblemDetail},
    },
)
async def start_tool_init(
    tool_name: str,
    request: Request,
    force: bool = False,
) -> JSONResponse:
    """Trigger initialization of a tool's data store.

    Starts a background job to ingest/index the tool's source data.
    Returns immediately with a 201 Created and a Location header
    pointing to the status resource.

    Args:
        tool_name: Name of the tool to initialize.
        request: The incoming HTTP request.
        force: Force full re-ingestion regardless of prior state.

    Returns:
        201 with InitJobResponse on success, or ProblemDetail on error.
    """
    manager = request.app.state.tool_init_manager
    instance = f"/tools/{tool_name}/init"

    try:
        job = manager.start_init_job(tool_name, force=force)
    except tuple(_ERROR_MAP.keys()) as exc:
        status, title = _ERROR_MAP[type(exc)]
        return _problem_response(
            status=status,
            title=title,
            detail=str(exc),
            instance=instance,
        )

    response = _job_to_response(job)
    return JSONResponse(
        status_code=201,
        content=response.model_dump(mode="json", exclude_none=True),
        headers={"Location": instance},
    )
