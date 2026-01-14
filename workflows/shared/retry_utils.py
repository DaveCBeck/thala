"""Retry utilities for async operations."""

import asyncio
from typing import TypeVar, Callable, Awaitable, Type

T = TypeVar("T")


async def with_retry(
    fn: Callable[[], Awaitable[T]],
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    retry_on: Type[Exception] = Exception,
    error_message: str = "Operation failed after {attempts} attempts",
) -> T:
    """Execute async function with exponential backoff retry."""
    last_error = None
    for attempt in range(max_attempts):
        try:
            return await fn()
        except retry_on as e:
            last_error = e
            if attempt < max_attempts - 1:
                wait_time = backoff_factor**attempt
                await asyncio.sleep(wait_time)
    raise RuntimeError(error_message.format(attempts=max_attempts)) from last_error
