"""
Async client for academic document retrieval service.

This service retrieves full-text academic documents when no open access URL
is available. It accepts DOI + optional metadata and returns the retrieved
document file.

The default implementation uses a pluggable backend that can be swapped
for institutional APIs, browser automation, or other retrieval methods.

Endpoints:
- GET /health: Check service health and VPN status
- POST /retrieve: Submit retrieval request (returns job_id)
- GET /jobs/{job_id}: Check job status and result
- GET /jobs/{job_id}/file: Download retrieved file
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field

from core.utils import BaseAsyncHttpClient

logger = logging.getLogger(__name__)


class RetrieveJobResponse(BaseModel):
    """Response when a retrieval job is created."""

    job_id: str
    status: str


class RetrieveResult(BaseModel):
    """Result of a retrieval job."""

    job_id: str
    status: str  # pending, searching, downloading, completed, failed
    doi: str
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    file_format: Optional[str] = None
    source_id: Optional[str] = None
    error: Optional[str] = None
    error_code: Optional[str] = None


class HealthStatus(BaseModel):
    """Health check response."""

    status: str
    vpn_connected: bool
    vpn_ip: Optional[str] = None


class RetrieveAcademicClient(BaseAsyncHttpClient):
    """
    Async client for academic document retrieval service.

    Example:
        async with RetrieveAcademicClient() as client:
            # Check health
            if not await client.health_check():
                raise RuntimeError("Service unavailable")

            # Submit retrieval request
            job = await client.retrieve(
                doi="10.1234/example",
                title="Example Paper",
                authors=["Author One"]
            )

            # Poll for completion
            result = await client.wait_for_completion(job.job_id, timeout=120)

            if result.status == "completed":
                # Download file to local path
                local_path = await client.download_file(
                    result.job_id, "/tmp/paper.pdf"
                )
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: float = 30.0,
    ):
        super().__init__(
            host=host,
            port=port,
            timeout=timeout,
            host_env_var="THALA_RETRIEVE_ACADEMIC_HOST",
            port_env_var="THALA_RETRIEVE_ACADEMIC_PORT",
            host_default="localhost",
            port_default=8002,
        )

    async def health_check(self) -> bool:
        """Check if service is available and VPN is connected."""
        try:
            client = await self._get_client()
            response = await client.get("/health", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                return data.get("vpn_connected", False)
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
        return False

    async def get_health_status(self) -> Optional[HealthStatus]:
        """Get detailed health status."""
        try:
            client = await self._get_client()
            response = await client.get("/health", timeout=5.0)
            if response.status_code == 200:
                return HealthStatus.model_validate(response.json())
        except Exception as e:
            logger.warning(f"Health status failed: {e}")
        return None

    async def retrieve(
        self,
        doi: str,
        title: Optional[str] = None,
        authors: Optional[list[str]] = None,
        preferred_formats: Optional[list[str]] = None,
        timeout_seconds: int = 120,
    ) -> RetrieveJobResponse:
        """
        Submit a retrieval request.

        Args:
            doi: DOI of the document to retrieve
            title: Optional title to improve search matching
            authors: Optional author names to improve matching
            preferred_formats: Preferred formats (default: ["pdf", "epub"])
            timeout_seconds: Max time for retrieval (default: 120)

        Returns:
            RetrieveJobResponse with job_id and initial status
        """
        client = await self._get_client()

        payload = {
            "doi": doi,
            "timeout_seconds": timeout_seconds,
        }
        if title:
            payload["title"] = title
        if authors:
            payload["authors"] = authors
        if preferred_formats:
            payload["preferred_formats"] = preferred_formats

        response = await client.post("/retrieve", json=payload)
        response.raise_for_status()

        return RetrieveJobResponse.model_validate(response.json())

    async def get_job_status(self, job_id: str) -> RetrieveResult:
        """
        Get the status of a retrieval job.

        Args:
            job_id: Job ID from retrieve() response

        Returns:
            RetrieveResult with current status and result data
        """
        client = await self._get_client()

        response = await client.get(f"/jobs/{job_id}")
        response.raise_for_status()

        return RetrieveResult.model_validate(response.json())

    async def wait_for_completion(
        self,
        job_id: str,
        timeout: float = 120.0,
        poll_interval: float = 2.0,
    ) -> RetrieveResult:
        """
        Wait for a job to complete.

        Args:
            job_id: Job ID to wait for
            timeout: Max time to wait in seconds
            poll_interval: Time between status checks

        Returns:
            RetrieveResult with final status

        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        start_time = asyncio.get_event_loop().time()

        while True:
            result = await self.get_job_status(job_id)

            if result.status in ("completed", "failed"):
                return result

            # Check timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed >= timeout:
                raise asyncio.TimeoutError(
                    f"Job {job_id} did not complete within {timeout}s"
                )

            await asyncio.sleep(poll_interval)

    async def download_file(
        self,
        job_id: str,
        local_path: str,
    ) -> str:
        """
        Download the retrieved file for a completed job.

        Args:
            job_id: Job ID of completed retrieval
            local_path: Local path to save the file

        Returns:
            Path where file was saved

        Raises:
            httpx.HTTPStatusError: If job not completed or file unavailable
        """
        client = await self._get_client()

        # Use streaming for potentially large files
        async with client.stream("GET", f"/jobs/{job_id}/file") as response:
            response.raise_for_status()

            local_file = Path(local_path)
            local_file.parent.mkdir(parents=True, exist_ok=True)

            with open(local_file, "wb") as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)

        return str(local_file)

    async def retrieve_and_download(
        self,
        doi: str,
        local_path: str,
        title: Optional[str] = None,
        authors: Optional[list[str]] = None,
        preferred_formats: Optional[list[str]] = None,
        timeout: float = 120.0,
    ) -> tuple[str, RetrieveResult]:
        """
        Convenience method: retrieve document and download to local path.

        Args:
            doi: DOI of the document
            local_path: Where to save the file
            title: Optional title for better matching
            authors: Optional authors for better matching
            preferred_formats: Preferred formats
            timeout: Max time to wait

        Returns:
            Tuple of (local_path, result)

        Raises:
            Exception: If retrieval fails
        """
        # Submit retrieval
        job = await self.retrieve(
            doi=doi,
            title=title,
            authors=authors,
            preferred_formats=preferred_formats,
            timeout_seconds=int(timeout),
        )

        # Wait for completion
        result = await self.wait_for_completion(job.job_id, timeout=timeout)

        if result.status != "completed":
            raise Exception(
                f"Retrieval failed: {result.error_code} - {result.error}"
            )

        # Download file
        saved_path = await self.download_file(job.job_id, local_path)

        return saved_path, result
