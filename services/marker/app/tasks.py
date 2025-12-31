"""Celery tasks for document processing."""

import logging
import subprocess

import psutil
from celery import Celery

from app.config import get_settings
from app.processor import get_processor

logger = logging.getLogger(__name__)
settings = get_settings()


def get_memory_stats() -> dict:
    """Get current RAM and GPU memory usage for monitoring."""
    # RAM usage for this process
    ram = psutil.Process().memory_info()
    ram_gb = ram.rss / (1024**3)

    # GPU memory via nvidia-smi
    gpu_used_gb = 0.0
    gpu_total_gb = 0.0
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpu_used, gpu_total = map(int, result.stdout.strip().split(", "))
            gpu_used_gb = gpu_used / 1024
            gpu_total_gb = gpu_total / 1024
    except Exception:
        pass

    return {"ram_gb": ram_gb, "gpu_used_gb": gpu_used_gb, "gpu_total_gb": gpu_total_gb}

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

        return {
            "status": "completed",
            "result": result,
            "error": None,
        }
    except FileNotFoundError as e:
        logger.error(f"[{file_path}] File not found: {e}")
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
        return {
            "status": "failed",
            "result": None,
            "error": str(e),
        }
