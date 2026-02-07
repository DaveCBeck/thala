"""Global rate-limit semaphores for parallel workflow execution.

Uses lazy factory functions to avoid stale semaphores when asyncio.run()
is called multiple times (e.g. in tests). Each semaphore is created on
first access within the current event loop.
"""

import asyncio

_imagen_semaphore: asyncio.Semaphore | None = None
_openalex_semaphore: asyncio.Semaphore | None = None


def get_imagen_semaphore() -> asyncio.Semaphore:
    """Get or create the global Imagen API semaphore (limit: 10)."""
    global _imagen_semaphore
    if _imagen_semaphore is None:
        _imagen_semaphore = asyncio.Semaphore(10)
    return _imagen_semaphore


def get_openalex_semaphore() -> asyncio.Semaphore:
    """Get or create the global OpenAlex API semaphore (limit: 20)."""
    global _openalex_semaphore
    if _openalex_semaphore is None:
        _openalex_semaphore = asyncio.Semaphore(20)
    return _openalex_semaphore


def reset_semaphores() -> None:
    """Reset semaphores between asyncio.run() invocations (e.g. in tests)."""
    global _imagen_semaphore, _openalex_semaphore
    _imagen_semaphore = None
    _openalex_semaphore = None
