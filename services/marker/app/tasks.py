"""Celery tasks for document processing."""

import logging

from celery import Celery

from app.config import get_settings
from app.processor import get_processor
from services.common.system_metrics import get_memory_stats

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize Celery
celery = Celery(
    "marker",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Celery configuration
celery.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    result_extended=True,
    # Task time limits - large PDFs (500+ pages) can take 2+ hours
    task_soft_time_limit=10800,  # 3 hours soft limit
    task_time_limit=14400,  # 4 hours hard limit
    # Worker prefetch (1 task at a time for GPU workloads)
    worker_prefetch_multiplier=1,
    # Only ack task after it completes (prevents requeue on worker crash)
    task_acks_late=True,
    # Explicitly ack on failure/timeout to prevent requeue of failed tasks
    task_acks_on_failure_or_timeout=True,
    # Don't requeue tasks that were started but worker died
    task_reject_on_worker_lost=True,
)

# Redis broker visibility timeout - MUST be set separately after app creation
# to ensure it takes effect. When a task is not acked within this time, Redis
# redelivers it to another worker. Must exceed task_time_limit significantly.
# Default is only 3600s (1 hour) which causes premature redelivery of long tasks.
celery.conf.broker_transport_options = {
    "visibility_timeout": 28800,  # 8 hours (2x task_time_limit for safety margin)
}


def _is_blocklisted(file_path: str) -> bool:
    """Check if file is in the blocklist."""
    from pathlib import Path

    blocklist_path = Path("/data/input/blocklist.txt")
    if not blocklist_path.exists():
        return False

    filename = Path(file_path).name
    blocklist = blocklist_path.read_text().strip().splitlines()
    return filename in blocklist or file_path in blocklist


@celery.task(bind=True, name="convert_document")
def convert_document(
    self,
    file_path: str,
    quality: str = "balanced",
    markdown_only: bool = False,
    langs: list[str] | None = None,
) -> dict:
    """
    Convert a document using Marker.

    Args:
        file_path: Path relative to /data/input
        quality: Quality preset (fast, balanced, quality)
        markdown_only: If True, return only markdown (smaller response)
        langs: Languages for OCR

    Returns:
        Dict with markdown, json, chunks, and metadata
    """
    # Idempotency check: if this task already completed successfully, return cached result.
    # This prevents duplicate processing if the same task ID ends up in the queue twice
    # (e.g., due to HTTP timeout/retry during submission or Redis redelivery edge cases).
    from celery.result import AsyncResult

    existing = AsyncResult(self.request.id, app=celery)
    if existing.state == "SUCCESS" and existing.result:
        cached = existing.result
        if cached.get("status") == "completed":
            logger.warning(
                f"[{file_path}] Skipped - task {self.request.id} already completed (idempotency)"
            )
            return cached

    # Check blocklist first
    if _is_blocklisted(file_path):
        logger.warning(f"[{file_path}] Skipped - file is blocklisted")
        return {
            "status": "skipped",
            "result": None,
            "error": "File is blocklisted",
        }

    # Log memory before processing
    before = get_memory_stats()
    logger.info(
        f"[{file_path}] Starting - RAM: {before['ram_gb']:.1f}GB, "
        f"GPU: {before['gpu_used_gb']:.1f}/{before['gpu_total_gb']:.1f}GB"
    )

    processor = get_processor()

    try:
        result = processor.convert(
            file_path=file_path,
            quality=quality,
            markdown_only=markdown_only,
            langs=langs,
        )

        # Log memory after processing
        after = get_memory_stats()
        logger.info(
            f"[{file_path}] Complete - RAM: {after['ram_gb']:.1f}GB, "
            f"GPU: {after['gpu_used_gb']:.1f}/{after['gpu_total_gb']:.1f}GB"
        )

        # Cleanup intermediate memory (keeps models loaded, schedules idle unload)
        processor.cleanup()

        return {
            "status": "completed",
            "result": result,
            "error": None,
        }
    except FileNotFoundError as e:
        logger.error(f"[{file_path}] File not found: {e}")
        processor.cleanup()
        return {
            "status": "failed",
            "result": None,
            "error": f"File not found: {e}",
        }
    except Exception as e:
        # Log memory on failure too
        after = get_memory_stats()
        logger.error(
            f"[{file_path}] Failed - RAM: {after['ram_gb']:.1f}GB, "
            f"GPU: {after['gpu_used_gb']:.1f}/{after['gpu_total_gb']:.1f}GB - Error: {e}"
        )
        processor.cleanup()
        return {
            "status": "failed",
            "result": None,
            "error": str(e),
        }
