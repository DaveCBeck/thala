"""Async utilities for concurrent processing."""

import asyncio
import logging
from typing import Any, Callable, Coroutine, List, TypeVar

T = TypeVar('T')


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


async def gather_with_error_collection(
    tasks: List[Coroutine],
    logger: logging.Logger,
    error_template: str = "Task failed: {error}",
) -> tuple[List[Any], List[dict]]:
    """Run tasks and separate successes from errors."""
    results = await asyncio.gather(*tasks, return_exceptions=True)
    successes = []
    errors = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(error_template.format(error=result))
            errors.append({"index": i, "error": str(result)})
        else:
            successes.append(result)

    return successes, errors
