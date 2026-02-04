"""Anthropic Batch API executor for structured output.

Uses the Anthropic Batch API with tool calling for bulk processing,
providing 50% cost savings on batch operations.

When THALA_LLM_BROKER_ENABLED=1 and batch_policy is set, routes through
the central LLM broker instead of direct BatchProcessor calls.
"""

import asyncio
import json
import logging
from typing import Callable, Optional, Type, TypeVar

from langsmith import get_current_run_tree, traceable
from pydantic import BaseModel

from ..types import (
    StructuredOutputConfig,
    StructuredOutputResult,
    StructuredOutputStrategy,
    StructuredRequest,
)
from .base import StrategyExecutor, coerce_to_schema

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class BatchToolCallExecutor(StrategyExecutor[T]):
    """Uses Anthropic Batch API with tool calling for bulk processing.

    When THALA_LLM_BROKER_ENABLED=1 and batch_policy is set in the config,
    routes through the central LLM broker for unified cost/speed management.
    """

    def _should_use_broker(self, output_config: StructuredOutputConfig) -> bool:
        """Check if we should route through the central broker."""
        from core.llm_broker import is_broker_enabled

        return is_broker_enabled() and output_config.batch_policy is not None

    async def _execute_via_broker(
        self,
        output_schema: Type[T],
        requests: list[StructuredRequest],
        default_system: Optional[str],
        output_config: StructuredOutputConfig,
    ) -> dict[str, StructuredOutputResult[T]]:
        """Execute batch via the central LLM broker."""
        from core.llm_broker import get_broker

        broker = get_broker()

        # Build tool definition from schema
        tool = {
            "name": "extract_output",
            "description": f"Extract {output_schema.__name__} from the content",
            "input_schema": output_schema.model_json_schema(),
        }
        tool_choice = {"type": "tool", "name": "extract_output"}

        # Submit all requests to broker
        pending: dict[str, asyncio.Future] = {}
        async with broker.batch_group():
            for req in requests:
                system = req.system_prompt or default_system
                future = await broker.request(
                    prompt=req.user_prompt,
                    model=output_config.tier,
                    policy=output_config.batch_policy,
                    max_tokens=output_config.max_tokens,
                    system=system,
                    thinking_budget=output_config.thinking_budget,
                    tools=[tool],
                    tool_choice=tool_choice,
                    metadata={"request_id": req.id},
                )
                pending[req.id] = future

        # Collect results
        results: dict[str, StructuredOutputResult[T]] = {}
        for req_id, future in pending.items():
            try:
                response = await future
                if response.success:
                    parsed = json.loads(response.content)
                    coerced = coerce_to_schema(parsed, output_schema)
                    validated = output_schema.model_validate(coerced)
                    results[req_id] = StructuredOutputResult.ok(
                        value=validated,
                        strategy=StructuredOutputStrategy.BATCH_TOOL_CALL,
                        usage=response.usage,
                    )
                else:
                    results[req_id] = StructuredOutputResult.err(error=response.error or "Unknown error")
            except Exception as e:
                results[req_id] = StructuredOutputResult.err(error=f"Validation failed: {e}")

        return results

    async def _execute_via_batch_processor(
        self,
        output_schema: Type[T],
        requests: list[StructuredRequest],
        default_system: Optional[str],
        output_config: StructuredOutputConfig,
    ) -> tuple[dict[str, StructuredOutputResult[T]], dict]:
        """Execute batch via the traditional BatchProcessor."""
        from workflows.shared.batch_processor import BatchProcessor

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
                model=output_config.tier,
                max_tokens=output_config.max_tokens,
                system=system,
                thinking_budget=output_config.thinking_budget,
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
                    coerced = coerce_to_schema(parsed, output_schema)
                    validated = output_schema.model_validate(coerced)
                    results[req.id] = StructuredOutputResult.ok(
                        value=validated,
                        strategy=StructuredOutputStrategy.BATCH_TOOL_CALL,
                        usage=batch_result.usage,
                        thinking=batch_result.thinking,
                    )
                except Exception as e:
                    results[req.id] = StructuredOutputResult.err(error=f"Validation failed: {e}")
            else:
                error = batch_result.error if batch_result else "No result"
                results[req.id] = StructuredOutputResult.err(error=error)

        return results, batch_results

    @traceable(run_type="llm", name="batch_structured_output")
    async def execute_batch(
        self,
        output_schema: Type[T],
        requests: list[StructuredRequest],
        default_system: Optional[str],
        output_config: StructuredOutputConfig,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict[str, StructuredOutputResult[T]]:
        # Route through broker if enabled and policy is set
        if self._should_use_broker(output_config):
            logger.debug("Routing batch through central LLM broker")
            return await self._execute_via_broker(
                output_schema=output_schema,
                requests=requests,
                default_system=default_system,
                output_config=output_config,
            )

        # Otherwise use traditional BatchProcessor
        results, batch_results = await self._execute_via_batch_processor(
            output_schema=output_schema,
            requests=requests,
            default_system=default_system,
            output_config=output_config,
        )

        # Aggregate usage from all batch results and attach to LangSmith run
        total_input = sum(r.usage.get("input_tokens", 0) for r in batch_results.values() if r.usage)
        total_output = sum(r.usage.get("output_tokens", 0) for r in batch_results.values() if r.usage)

        try:
            run = get_current_run_tree()
            if run and (total_input or total_output):
                run.add_metadata(
                    {
                        "usage_metadata": {
                            "input_tokens": total_input,
                            "output_tokens": total_output,
                            "total_tokens": total_input + total_output,
                        }
                    }
                )
        except Exception as e:
            # Don't let tracing issues break the workflow
            logger.warning(f"[DIAG] LangSmith run tree error: {e}")

        return results

    async def execute(
        self,
        output_schema: Type[T],
        user_prompt: str,
        system_prompt: Optional[str],
        output_config: StructuredOutputConfig,
    ) -> StructuredOutputResult[T]:
        """Execute single request via batch (for consistency)."""
        requests = [StructuredRequest(id="_single", user_prompt=user_prompt)]
        results = await self.execute_batch(
            output_schema=output_schema,
            requests=requests,
            default_system=system_prompt,
            output_config=output_config,
        )
        return results["_single"]


__all__ = ["BatchToolCallExecutor"]
