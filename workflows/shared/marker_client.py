"""Async client for Marker document processing service."""

import asyncio
import os
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field


class MarkerJobResult(BaseModel):
    """Result of a completed Marker conversion job."""

    markdown: str
    json_data: Optional[dict[str, Any]] = Field(None, alias="json")
    chunks: Optional[list[dict[str, Any]]] = None
    metadata: dict[str, Any]

    class Config:
        populate_by_name = True


class MarkerClient:
    """
    Async client for the Marker document processing API.

    Uses httpx for async HTTP with polling support for long-running jobs.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        poll_interval: float = 2.0,
    ):
        """
        Initialize MarkerClient.

        Args:
            base_url: Marker API base URL (default: env MARKER_BASE_URL or http://localhost:8001)
            timeout: Max timeout for HTTP requests in seconds (default: None, no timeout)
            poll_interval: Seconds between status polls (default: env MARKER_POLL_INTERVAL or 2.0)
        """
        self.base_url = base_url or os.getenv("MARKER_BASE_URL", "http://localhost:8001")
        self.timeout = timeout
        self.poll_interval = float(os.getenv("MARKER_POLL_INTERVAL", str(poll_interval)))
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client (lazy init)."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "MarkerClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def submit_job(
        self,
        file_path: str,
        quality: str = "balanced",
        langs: Optional[list[str]] = None,
    ) -> str:
        """
        Submit a document conversion job.

        Args:
            file_path: Path to file relative to /data/input
            quality: Quality preset (fast, balanced, quality)
            langs: Languages for OCR (default: ["English"])

        Returns:
            Job ID for polling status
        """
        client = await self._get_client()

        payload = {
            "file_path": file_path,
            "quality": quality,
            "markdown_only": False,
            "langs": langs or ["English"],
        }

        response = await client.post("/convert", json=payload)
        response.raise_for_status()

        data = response.json()
        return data["job_id"]

    async def poll_until_complete(
        self, job_id: str, max_wait: Optional[float] = None
    ) -> MarkerJobResult:
        """
        Poll job status until complete or timeout.

        Args:
            job_id: Job ID from submit_job
            max_wait: Maximum seconds to wait before timeout (None = no timeout)

        Returns:
            MarkerJobResult with conversion output

        Raises:
            TimeoutError: If job doesn't complete within max_wait
            httpx.HTTPStatusError: On API errors
            RuntimeError: If job fails
        """
        client = await self._get_client()
        start_time = asyncio.get_event_loop().time()

        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if max_wait is not None and elapsed > max_wait:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {max_wait}s"
                )

            response = await client.get(f"/jobs/{job_id}")
            response.raise_for_status()

            data = response.json()
            status = data["status"]

            if status == "completed":
                result_data = data["result"]
                return MarkerJobResult(**result_data)
            elif status == "failed":
                error = data.get("error", "Unknown error")
                raise RuntimeError(f"Job {job_id} failed: {error}")
            elif status in ("pending", "processing"):
                await asyncio.sleep(self.poll_interval)
            else:
                raise RuntimeError(f"Unknown job status: {status}")

    async def convert(
        self,
        file_path: str,
        quality: str = "balanced",
        langs: Optional[list[str]] = None,
    ) -> MarkerJobResult:
        """
        Submit job and wait for completion (convenience method).

        Args:
            file_path: Path to file relative to /data/input
            quality: Quality preset (fast, balanced, quality)
            langs: Languages for OCR (default: ["English"])

        Returns:
            MarkerJobResult with conversion output
        """
        job_id = await self.submit_job(file_path, quality, langs)
        return await self.poll_until_complete(job_id)

    async def submit_jobs(
        self,
        jobs: list[dict[str, Any]],
    ) -> list[str]:
        """
        Submit multiple document conversion jobs.

        Args:
            jobs: List of job configs, each with:
                - file_path: Path to file relative to /data/input
                - quality: Quality preset (fast, balanced, quality)
                - langs: Languages for OCR (default: ["English"])

        Returns:
            List of job IDs for polling status
        """
        job_ids = []
        for job in jobs:
            job_id = await self.submit_job(
                file_path=job["file_path"],
                quality=job.get("quality", "balanced"),
                langs=job.get("langs"),
            )
            job_ids.append(job_id)
        return job_ids

    async def poll_job_status(self, job_id: str) -> dict[str, Any]:
        """
        Get current status of a job without blocking.

        Args:
            job_id: Job ID from submit_job

        Returns:
            Dict with status, and result if completed
        """
        client = await self._get_client()
        response = await client.get(f"/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    async def poll_multiple_until_complete(
        self,
        job_ids: list[str],
        max_wait: float | None = None,
    ):
        """
        Poll multiple jobs, yielding results as they complete.

        This is an async generator that yields (job_id, result) tuples
        as each job completes, allowing post-processing to start
        immediately while other jobs continue.

        Args:
            job_ids: List of job IDs to poll
            max_wait: Maximum seconds to wait for any single job

        Yields:
            Tuple of (job_id, MarkerJobResult) as each completes

        Raises:
            TimeoutError: If any job doesn't complete within max_wait
            RuntimeError: If any job fails
        """
        pending = set(job_ids)
        start_times = {job_id: asyncio.get_event_loop().time() for job_id in job_ids}

        while pending:
            completed_this_round = []

            for job_id in list(pending):
                elapsed = asyncio.get_event_loop().time() - start_times[job_id]
                if max_wait is not None and elapsed > max_wait:
                    raise TimeoutError(
                        f"Job {job_id} did not complete within {max_wait}s"
                    )

                data = await self.poll_job_status(job_id)
                status = data["status"]

                if status == "completed":
                    result = MarkerJobResult(**data["result"])
                    completed_this_round.append((job_id, result))
                    pending.discard(job_id)
                elif status == "failed":
                    error = data.get("error", "Unknown error")
                    raise RuntimeError(f"Job {job_id} failed: {error}")
                # pending/processing: continue polling

            # Yield completed jobs
            for job_id, result in completed_this_round:
                yield job_id, result

            # Sleep before next poll round if still have pending
            if pending:
                await asyncio.sleep(self.poll_interval)

    async def health_check(self) -> dict[str, Any]:
        """
        Check Marker service health and GPU status.

        Returns:
            Health check response with status, GPU info, queue depth
        """
        client = await self._get_client()
        response = await client.get("/health")
        response.raise_for_status()
        return response.json()
