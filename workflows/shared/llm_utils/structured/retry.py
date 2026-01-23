"""Retry logic for structured output extraction."""

import asyncio
import logging
from typing import Awaitable, Callable, Optional, Type, TypeVar

from pydantic import BaseModel

from ..models import ModelTier
from .types import (
    StructuredOutputConfig,
    StructuredOutputResult,
    StructuredOutputStrategy,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def is_context_limit_error(error: Exception) -> bool:
    """Check if error is due to prompt exceeding context limit."""
    error_str = str(error).lower()
    return "prompt is too long" in error_str or "exceeds the maximum" in error_str


async def with_retries(
    fn: Callable[[], Awaitable[StructuredOutputResult[T]]],
    config: StructuredOutputConfig,
    schema: Type[T],
    strategy: StructuredOutputStrategy,
    fallback_fn_factory: Optional[
        Callable[[StructuredOutputConfig], Callable[[], Awaitable[StructuredOutputResult[T]]]]
    ] = None,
) -> StructuredOutputResult[T]:
    """Execute with retry logic and optional context fallback.

    Args:
        fn: The function to execute
        config: Configuration including retry settings
        schema: Pydantic schema for the output
        strategy: The execution strategy being used
        fallback_fn_factory: Optional factory that creates a new fn with upgraded config.
            Called with upgraded config (SONNET_1M) when context limit error detected.
    """
    last_error: Optional[Exception] = None
    context_fallback_attempted = False

    for attempt in range(config.max_retries):
        try:
            result = await fn()
            if result.success:
                return result
            last_error = Exception(result.error)
        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1} failed: {e}")

            # Check for context limit error and attempt fallback
            if (
                is_context_limit_error(e)
                and config.enable_context_fallback
                and config.tier != ModelTier.SONNET_1M
                and fallback_fn_factory is not None
                and not context_fallback_attempted
            ):
                context_fallback_attempted = True
                logger.info(
                    f"Context limit exceeded with {config.tier.name}, "
                    "falling back to SONNET_1M"
                )
                # Create upgraded config and new fn
                upgraded_config = StructuredOutputConfig(
                    tier=ModelTier.SONNET_1M,
                    max_tokens=config.max_tokens,
                    thinking_budget=config.thinking_budget,
                    strategy=config.strategy,
                    use_json_schema_method=config.use_json_schema_method,
                    prefer_batch_api=config.prefer_batch_api,
                    max_retries=config.max_retries,
                    retry_backoff=config.retry_backoff,
                    enable_context_fallback=False,  # Prevent infinite fallback loop
                    enable_prompt_cache=config.enable_prompt_cache,
                    cache_ttl=config.cache_ttl,
                    tools=config.tools,
                    max_tool_calls=config.max_tool_calls,
                    max_tool_result_chars=config.max_tool_result_chars,
                )
                fallback_fn = fallback_fn_factory(upgraded_config)
                # Recursively retry with upgraded config
                return await with_retries(
                    fallback_fn, upgraded_config, schema, strategy, fallback_fn_factory
                )

            if attempt < config.max_retries - 1:
                await asyncio.sleep(config.retry_backoff**attempt)

    return StructuredOutputResult.err(
        error=f"Failed after {config.max_retries} attempts: {last_error}"
    )


__all__ = ["with_retries", "is_context_limit_error"]
