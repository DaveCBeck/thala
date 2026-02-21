"""Global rate-limit primitives for parallel workflow execution.

Provides:
- Semaphores for concurrency gating (Imagen, OpenAlex, mmdc)
- ImagenDailyTracker: file-based daily usage counter with atomic try_acquire()
- ImagenRPMLimiter: in-memory token bucket for per-minute rate cap

Uses lazy factory functions to avoid stale state when asyncio.run()
is called multiple times (e.g. in tests). Each primitive is created on
first access within the current event loop.
"""

import asyncio
import fcntl
import json
import logging
import os
from datetime import date
from pathlib import Path

from .paths import STATE_DIR
from .utils import write_json_atomic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Semaphores (existing)
# ---------------------------------------------------------------------------

_imagen_semaphore: asyncio.Semaphore | None = None
_openalex_semaphore: asyncio.Semaphore | None = None
_mmdc_semaphore: asyncio.Semaphore | None = None


def get_imagen_semaphore() -> asyncio.Semaphore:
    """Get or create the global Imagen API semaphore."""
    global _imagen_semaphore
    if _imagen_semaphore is None:
        limit = int(os.environ.get("THALA_IMAGEN_CONCURRENCY", "10"))
        _imagen_semaphore = asyncio.Semaphore(limit)
    return _imagen_semaphore


def get_openalex_semaphore() -> asyncio.Semaphore:
    """Get or create the global OpenAlex API semaphore."""
    global _openalex_semaphore
    if _openalex_semaphore is None:
        limit = int(os.environ.get("THALA_OPENALEX_CONCURRENCY", "20"))
        _openalex_semaphore = asyncio.Semaphore(limit)
    return _openalex_semaphore


def get_mmdc_semaphore() -> asyncio.Semaphore:
    """Get or create the global Mermaid (mmdc/PhantomJS) semaphore.

    Default concurrency=1 because PhantomJS uses shared temp files
    that cause race conditions under parallel execution.
    """
    global _mmdc_semaphore
    if _mmdc_semaphore is None:
        limit = int(os.environ.get("THALA_MMDC_CONCURRENCY", "1"))
        _mmdc_semaphore = asyncio.Semaphore(limit)
    return _mmdc_semaphore


# ---------------------------------------------------------------------------
# Imagen daily tracker
# ---------------------------------------------------------------------------


def _today_str() -> str:
    return date.today().isoformat()


class ImagenDailyTracker:
    """File-based daily usage counter at .thala/state/imagen_daily_usage.json.

    All public methods are async (file I/O via asyncio.to_thread).
    Check-and-decrement is atomic under a single file lock to prevent
    TOCTOU races between concurrent tasks.
    """

    def __init__(self, state_dir: Path | None = None, limit: int | None = None):
        self._state_dir = state_dir or STATE_DIR
        self._limit = limit or int(os.environ.get("THALA_IMAGEN_DAILY_LIMIT", "70"))
        self._state_file = self._state_dir / "imagen_daily_usage.json"
        self._lock_file = self._state_dir / "imagen_daily.lock"

    async def try_acquire(self, count: int = 1) -> bool:
        """Atomically check budget and reserve *count* image slots.

        Args:
            count: Number of images to reserve (e.g. sample_count per API call).
        """
        return await asyncio.to_thread(self._try_acquire_sync, count)

    async def remaining(self) -> int:
        """Non-atomic read of remaining budget (for fast-fail checks)."""
        return await asyncio.to_thread(self._remaining_sync)

    def _try_acquire_sync(self, count: int = 1) -> bool:
        """Single flock acquisition: read, check, increment, write."""
        self._state_dir.mkdir(parents=True, exist_ok=True)
        with self._file_lock():
            data = self._read_state()
            if data["date"] != _today_str():
                data = {"date": _today_str(), "count": 0}
            if data["count"] + count > self._limit:
                return False
            data["count"] += count
            write_json_atomic(self._state_file, data)
            return True

    def _remaining_sync(self) -> int:
        """Read remaining budget without locking (approximate)."""
        self._state_dir.mkdir(parents=True, exist_ok=True)
        try:
            data = self._read_state()
            if data["date"] != _today_str():
                return self._limit
            return max(0, self._limit - data["count"])
        except Exception:
            return self._limit

    def _file_lock(self):
        """Context manager for exclusive file lock."""
        return _FileLock(self._lock_file)

    def _read_state(self) -> dict:
        """Read state file, returning default on missing/corrupt."""
        try:
            with open(self._state_file) as f:
                data = json.load(f)
            if not isinstance(data, dict) or "date" not in data or "count" not in data:
                return {"date": _today_str(), "count": 0}
            return data
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            return {"date": _today_str(), "count": 0}


class _FileLock:
    """Simple context manager wrapping fcntl.flock."""

    def __init__(self, path: Path):
        self._path = path
        self._fd: int | None = None

    def __enter__(self):
        self._path.touch(exist_ok=True)
        self._fd = open(self._path, "w")
        fcntl.flock(self._fd.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        fcntl.flock(self._fd.fileno(), fcntl.LOCK_UN)
        self._fd.close()
        return False


# ---------------------------------------------------------------------------
# Imagen RPM limiter (token bucket)
# ---------------------------------------------------------------------------


class ImagenRPMLimiter:
    """In-memory token bucket for per-minute rate cap.

    Uses asyncio.Lock (not threading.Lock) since all contention
    is within a single event loop.
    """

    def __init__(self, rpm: int):
        self._rpm = rpm
        self._tokens = float(rpm)
        self._last_refill: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self, cost: int = 1) -> None:
        """Wait until *cost* image tokens are available, then consume them.

        Args:
            cost: Number of image tokens to consume (e.g. sample_count).
        """
        while True:
            async with self._lock:
                now = asyncio.get_running_loop().time()
                if self._last_refill == 0.0:
                    self._last_refill = now
                elapsed = now - self._last_refill
                self._tokens = min(
                    float(self._rpm),
                    self._tokens + elapsed * (self._rpm / 60.0),
                )
                self._last_refill = now
                if self._tokens >= cost:
                    self._tokens -= cost
                    return
            await asyncio.sleep(60.0 / self._rpm)


# ---------------------------------------------------------------------------
# Lazy factories
# ---------------------------------------------------------------------------

_imagen_daily_tracker: ImagenDailyTracker | None = None
_imagen_rpm_limiter: ImagenRPMLimiter | None = None


def get_imagen_daily_tracker() -> ImagenDailyTracker:
    """Get or create the global Imagen daily usage tracker."""
    global _imagen_daily_tracker
    if _imagen_daily_tracker is None:
        _imagen_daily_tracker = ImagenDailyTracker()
    return _imagen_daily_tracker


def get_imagen_rpm_limiter() -> ImagenRPMLimiter:
    """Get or create the global Imagen RPM limiter."""
    global _imagen_rpm_limiter
    if _imagen_rpm_limiter is None:
        rpm = int(os.environ.get("THALA_IMAGEN_RPM_LIMIT", "5"))
        _imagen_rpm_limiter = ImagenRPMLimiter(rpm)
    return _imagen_rpm_limiter


# ---------------------------------------------------------------------------
# Reset (lifecycle cleanup)
# ---------------------------------------------------------------------------


def reset_rate_limiters() -> None:
    """Reset all rate limiter globals. Call on supervisor shutdown."""
    global _imagen_semaphore, _openalex_semaphore, _mmdc_semaphore
    global _imagen_rpm_limiter, _imagen_daily_tracker
    _imagen_semaphore = None
    _openalex_semaphore = None
    _mmdc_semaphore = None
    _imagen_rpm_limiter = None
    _imagen_daily_tracker = None
