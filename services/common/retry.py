"""Retry utilities."""

import asyncio
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    retry_on: tuple = (Exception,),
) -> T:
    """Execute async function with exponential backoff retry."""
    for attempt in range(max_attempts):
        try:
            return await fn()
        except retry_on as e:
            if attempt < max_attempts - 1:
                wait_time = backoff_factor**attempt
                await asyncio.sleep(wait_time)
            else:
                raise
