"""Concurrent batch execution fallback for structured output.

Used when batch API is not selected but multiple requests need processing.
"""

import asyncio
from typing import Callable, Optional, Type, TypeVar

from pydantic import BaseModel

from .executors import executors
from .types import (
    StructuredOutputConfig,
    StructuredOutputResult,
    StructuredOutputStrategy,
    StructuredRequest,
)

T = TypeVar("T", bound=BaseModel)


async def execute_batch_concurrent(
    output_schema: Type[T],
    requests: list[StructuredRequest],
    system_prompt: Optional[str],
    config: StructuredOutputConfig,
    selected_strategy: StructuredOutputStrategy,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict[str, StructuredOutputResult[T]]:
    """Execute batch via concurrent single requests (fallback)."""
    executor = executors[selected_strategy]
    results: dict[str, StructuredOutputResult[T]] = {}
    completed = 0

    async def process_one(
        req: StructuredRequest,
    ) -> tuple[str, StructuredOutputResult[T]]:
        nonlocal completed
        try:
            result = await executor.execute(
                output_schema=output_schema,
                user_prompt=req.user_prompt,
                system_prompt=req.system_prompt or system_prompt,
                config=config,
            )
        except Exception as e:
            result = StructuredOutputResult.err(error=str(e))

        completed += 1
        if progress_callback:
            progress_callback(completed, len(requests))

        return req.id, result

    # Run concurrently with semaphore to limit parallelism
    semaphore = asyncio.Semaphore(5)

    async def bounded_process(req: StructuredRequest):
        async with semaphore:
            return await process_one(req)

    tasks = [bounded_process(req) for req in requests]
    pairs = await asyncio.gather(*tasks)

    for req_id, result in pairs:
        results[req_id] = result

    return results


__all__ = ["execute_batch_concurrent"]
