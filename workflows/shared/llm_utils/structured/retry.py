"""Retry logic for structured output extraction."""

import asyncio
import logging
from typing import Awaitable, Callable, Optional, Type, TypeVar

from pydantic import BaseModel

from .types import (
    StructuredOutputConfig,
    StructuredOutputResult,
    StructuredOutputStrategy,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


async def with_retries(
    fn: Callable[[], Awaitable[StructuredOutputResult[T]]],
    config: StructuredOutputConfig,
    schema: Type[T],
    strategy: StructuredOutputStrategy,
) -> StructuredOutputResult[T]:
    """Execute with retry logic."""
    last_error: Optional[Exception] = None

    for attempt in range(config.max_retries):
        try:
            result = await fn()
            if result.success:
                return result
            last_error = Exception(result.error)
        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < config.max_retries - 1:
                await asyncio.sleep(config.retry_backoff**attempt)

    return StructuredOutputResult.err(
        error=f"Failed after {config.max_retries} attempts: {last_error}"
    )


__all__ = ["with_retries"]
