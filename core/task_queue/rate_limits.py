"""Global rate-limit semaphores for parallel workflow execution.

Uses lazy factory functions to avoid stale semaphores when asyncio.run()
is called multiple times (e.g. in tests). Each semaphore is created on
first access within the current event loop.
"""

import asyncio
import os

_imagen_semaphore: asyncio.Semaphore | None = None
_openalex_semaphore: asyncio.Semaphore | None = None


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
