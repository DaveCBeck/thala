"""Async utilities for concurrent processing."""

import asyncio
from typing import Any, Coroutine, List


async def run_with_concurrency(
    tasks: List[Coroutine],
    max_concurrent: int = 5,
    return_exceptions: bool = True,
) -> List[Any]:
    """Run coroutines with semaphore-controlled concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited(coro):
        async with semaphore:
            return await coro

    return await asyncio.gather(
        *[limited(t) for t in tasks],
        return_exceptions=return_exceptions
    )
