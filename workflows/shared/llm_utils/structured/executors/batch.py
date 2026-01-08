"""Anthropic Batch API executor for structured output.

Uses the Anthropic Batch API with tool calling for bulk processing,
providing 50% cost savings on batch operations.
"""

import json
from typing import Callable, Optional, Type, TypeVar

from pydantic import BaseModel

from ..types import (
    StructuredOutputConfig,
    StructuredOutputResult,
    StructuredOutputStrategy,
    StructuredRequest,
)
from .base import StrategyExecutor

T = TypeVar("T", bound=BaseModel)


class BatchToolCallExecutor(StrategyExecutor[T]):
    """Uses Anthropic Batch API with tool calling for bulk processing."""

    async def execute_batch(
        self,
        output_schema: Type[T],
        requests: list[StructuredRequest],
        default_system: Optional[str],
        config: StructuredOutputConfig,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict[str, StructuredOutputResult[T]]:
        from ...batch_processor import BatchProcessor

        processor = BatchProcessor(poll_interval=30)

        # Build tool definition from schema
        tool = {
            "name": "extract_output",
            "description": f"Extract {output_schema.__name__} from the content",
            "input_schema": output_schema.model_json_schema(),
        }
        tool_choice = {"type": "tool", "name": "extract_output"}

        # Add all requests to batch
        for req in requests:
            system = req.system_prompt or default_system
            processor.add_request(
                custom_id=req.id,
                prompt=req.user_prompt,
                model=config.tier,
                max_tokens=config.max_tokens,
                system=system,
                thinking_budget=config.thinking_budget,
                tools=[tool],
                tool_choice=tool_choice,
            )

        # Execute batch
        batch_results = await processor.execute_batch()

        # Parse results
        results: dict[str, StructuredOutputResult[T]] = {}
        for req in requests:
            batch_result = batch_results.get(req.id)

            if batch_result and batch_result.success:
                try:
                    parsed = json.loads(batch_result.content)
                    validated = output_schema.model_validate(parsed)
                    results[req.id] = StructuredOutputResult.ok(
                        value=validated,
                        strategy=StructuredOutputStrategy.BATCH_TOOL_CALL,
                        usage=batch_result.usage,
                        thinking=batch_result.thinking,
                    )
                except Exception as e:
                    results[req.id] = StructuredOutputResult.err(
                        error=f"Validation failed: {e}"
                    )
            else:
                error = batch_result.error if batch_result else "No result"
                results[req.id] = StructuredOutputResult.err(error=error)

        return results

    async def execute(
        self,
        output_schema: Type[T],
        user_prompt: str,
        system_prompt: Optional[str],
        config: StructuredOutputConfig,
    ) -> StructuredOutputResult[T]:
        """Execute single request via batch (for consistency)."""
        requests = [StructuredRequest(id="_single", user_prompt=user_prompt)]
        results = await self.execute_batch(
            output_schema=output_schema,
            requests=requests,
            default_system=system_prompt,
            config=config,
        )
        return results["_single"]


__all__ = ["BatchToolCallExecutor"]
