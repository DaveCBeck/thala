"""Health check utilities."""

import subprocess
import time
from typing import Optional

import httpx


async def check_gpu() -> tuple[bool, Optional[str]]:
    """Check GPU availability via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return True, result.stdout.strip()
    except Exception:
        pass
    return False, None


async def check_http_health(
    url: str, timeout: float = 5.0
) -> tuple[bool, Optional[float]]:
    """Check HTTP endpoint health, return (healthy, response_time_ms)."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            start = time.perf_counter()
            response = await client.get(url)
            elapsed = (time.perf_counter() - start) * 1000
            return response.status_code == 200, elapsed
    except Exception:
        return False, None
