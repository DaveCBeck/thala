"""Celery tasks for document processing."""

from celery import Celery

from app.config import get_settings
from app.processor import get_processor

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
    # Task time limits
    task_soft_time_limit=600,  # 10 minutes soft limit
    task_time_limit=900,  # 15 minutes hard limit
    # Worker prefetch (1 task at a time for GPU workloads)
    worker_prefetch_multiplier=1,
)


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
    processor = get_processor()

    try:
        result = processor.convert(
            file_path=file_path,
            quality=quality,
            markdown_only=markdown_only,
            langs=langs,
        )
        return {
            "status": "completed",
            "result": result,
            "error": None,
        }
    except FileNotFoundError as e:
        return {
            "status": "failed",
            "result": None,
            "error": f"File not found: {e}",
        }
    except Exception as e:
        return {
            "status": "failed",
            "result": None,
            "error": str(e),
        }
