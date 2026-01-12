"""Execution logic for structured output requests.

Handles single and batch request execution with strategy-specific routing.
"""

import logging
from typing import Callable, Optional, Type, TypeVar

from pydantic import BaseModel

from .batch_executor import execute_batch_concurrent
from .executors import executors
from .executors.batch import BatchToolCallExecutor
from .retry import with_retries
from .types import (
    BatchResult,
    StructuredOutputConfig,
    StructuredOutputError,
    StructuredOutputResult,
    StructuredOutputStrategy,
    StructuredRequest,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


async def execute_single(
    output_schema: Type[T],
    user_prompt: str,
    system_prompt: Optional[str],
    config: StructuredOutputConfig,
    selected_strategy: StructuredOutputStrategy,
) -> T:
    """Execute single request.

    When selected_strategy is BATCH_TOOL_CALL (e.g., prefer_batch_api=True),
    wraps the single request as a batch for 50% cost savings.
    """
    if selected_strategy == StructuredOutputStrategy.BATCH_TOOL_CALL:
        executor = BatchToolCallExecutor()

        async def _invoke() -> StructuredOutputResult[T]:
            results = await executor.execute_batch(
                output_schema=output_schema,
                requests=[StructuredRequest(id="_single", user_prompt=user_prompt)],
                default_system=system_prompt,
                config=config,
            )
            return results["_single"]

    else:
        executor = executors[selected_strategy]

        async def _invoke() -> StructuredOutputResult[T]:
            return await executor.execute(
                output_schema=output_schema,
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                config=config,
            )

    result = await with_retries(_invoke, config, output_schema, selected_strategy)

    if not result.success:
        raise StructuredOutputError(
            message=result.error or "Unknown error",
            schema=output_schema,
            strategy=selected_strategy,
            attempts=config.max_retries,
        )

    return result.value


async def execute_batch(
    output_schema: Type[T],
    requests: list[StructuredRequest],
    system_prompt: Optional[str],
    config: StructuredOutputConfig,
    selected_strategy: StructuredOutputStrategy,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> BatchResult[T]:
    """Execute batch request."""
    if selected_strategy == StructuredOutputStrategy.BATCH_TOOL_CALL:
        executor = BatchToolCallExecutor()
        results = await executor.execute_batch(
            output_schema=output_schema,
            requests=requests,
            default_system=system_prompt,
            config=config,
            progress_callback=progress_callback,
        )
    else:
        results = await execute_batch_concurrent(
            output_schema=output_schema,
            requests=requests,
            system_prompt=system_prompt,
            config=config,
            selected_strategy=selected_strategy,
            progress_callback=progress_callback,
        )

    successful = sum(1 for r in results.values() if r.success)
    failed = len(results) - successful

    return BatchResult(
        results=results,
        total_items=len(requests),
        successful_items=successful,
        failed_items=failed,
        strategy_used=selected_strategy,
    )


__all__ = ["execute_single", "execute_batch"]
