"""FastAPI application for Marker document processing service."""

from typing import Any

from celery.result import AsyncResult
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import QUALITY_PRESETS
from app.tasks import celery, convert_document
from services.common.health import check_gpu

app = FastAPI(
    title="Marker Document Processing API",
    description="GPU-accelerated document conversion using Marker + Surya",
    version="1.0.0",
)


# Request/Response Models
class ConvertRequest(BaseModel):
    """Request to convert a document."""

    file_path: str = Field(..., description="Path relative to /data/input")
    quality: str = Field(
        default="balanced",
        description="Quality preset: fast, balanced, or quality",
    )
    markdown_only: bool = Field(
        default=False,
        description="If true, return only markdown (smaller response)",
    )
    langs: list[str] = Field(
        default=["English"],
        description="Languages for OCR",
    )


class BatchConvertRequest(BaseModel):
    """Request to convert multiple documents."""

    files: list[ConvertRequest]


class ConvertResult(BaseModel):
    """Result of a conversion."""

    markdown: str
    json: dict[str, Any] | None = None  # Page structure with bounding boxes
    chunks: list[dict[str, Any]] | None = None  # Flattened blocks for RAG
    metadata: dict[str, Any]


class JobResponse(BaseModel):
    """Response for a job status query."""

    job_id: str
    status: str  # pending, processing, completed, failed
    result: ConvertResult | None = None
    error: str | None = None


class JobSubmitResponse(BaseModel):
    """Response when submitting a job."""

    job_id: str
    status: str = "pending"


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    gpu_available: bool
    gpu_name: str | None
    queue_depth: int
    active_workers: int


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check service health, GPU status, and queue depth."""
    gpu_available, gpu_name = await check_gpu()

    # Check Celery queue
    inspect = celery.control.inspect()
    active = inspect.active() or {}
    reserved = inspect.reserved() or {}

    active_count = sum(len(tasks) for tasks in active.values())
    reserved_count = sum(len(tasks) for tasks in reserved.values())

    return HealthResponse(
        status="healthy" if gpu_available else "degraded",
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        queue_depth=reserved_count,
        active_workers=active_count,
    )


@app.post("/convert", response_model=JobSubmitResponse)
async def submit_convert(request: ConvertRequest) -> JobSubmitResponse:
    """Submit a document for conversion."""
    # Validate quality preset
    if request.quality not in QUALITY_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid quality preset. Must be one of: {list(QUALITY_PRESETS.keys())}",
        )

    # Submit task
    task = convert_document.delay(
        file_path=request.file_path,
        quality=request.quality,
        markdown_only=request.markdown_only,
        langs=request.langs,
    )

    return JobSubmitResponse(job_id=task.id, status="pending")


@app.post("/convert/batch", response_model=list[JobSubmitResponse])
async def submit_batch_convert(request: BatchConvertRequest) -> list[JobSubmitResponse]:
    """Submit multiple documents for conversion."""
    responses = []
    for file_request in request.files:
        # Validate each request
        if file_request.quality not in QUALITY_PRESETS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid quality preset for {file_request.file_path}",
            )

        # Submit task
        task = convert_document.delay(
            file_path=file_request.file_path,
            quality=file_request.quality,
            markdown_only=file_request.markdown_only,
            langs=file_request.langs,
        )
        responses.append(JobSubmitResponse(job_id=task.id, status="pending"))

    return responses


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str) -> JobResponse:
    """Get the status and result of a conversion job."""
    result = AsyncResult(job_id, app=celery)

    if result.state == "PENDING":
        return JobResponse(job_id=job_id, status="pending")
    elif result.state == "STARTED":
        return JobResponse(job_id=job_id, status="processing")
    elif result.state == "SUCCESS":
        task_result = result.result
        if task_result.get("status") == "completed":
            return JobResponse(
                job_id=job_id,
                status="completed",
                result=ConvertResult(**task_result["result"]),
            )
        else:
            return JobResponse(
                job_id=job_id,
                status="failed",
                error=task_result.get("error", "Unknown error"),
            )
    elif result.state == "FAILURE":
        return JobResponse(
            job_id=job_id,
            status="failed",
            error=str(result.result),
        )
    else:
        return JobResponse(job_id=job_id, status=result.state.lower())


@app.get("/jobs")
async def list_jobs(limit: int = 100, offset: int = 0) -> dict[str, Any]:
    """List recent jobs."""
    # Note: This is a basic implementation. For production,
    # consider using Flower's API or storing job metadata in Redis
    inspect = celery.control.inspect()

    active = inspect.active() or {}
    reserved = inspect.reserved() or {}
    scheduled = inspect.scheduled() or {}

    all_tasks = []

    for worker, tasks in active.items():
        for task in tasks:
            all_tasks.append({
                "job_id": task["id"],
                "status": "processing",
                "worker": worker,
            })

    for worker, tasks in reserved.items():
        for task in tasks:
            all_tasks.append({
                "job_id": task["id"],
                "status": "pending",
                "worker": worker,
            })

    return {
        "jobs": all_tasks[offset : offset + limit],
        "total": len(all_tasks),
        "limit": limit,
        "offset": offset,
    }


@app.get("/presets")
async def list_presets() -> dict[str, dict]:
    """List available quality presets."""
    return QUALITY_PRESETS
