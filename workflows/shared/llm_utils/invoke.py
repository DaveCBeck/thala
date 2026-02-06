"""Unified LLM invocation layer.

This module provides the invoke() function that consolidates all LLM invocation
patterns into a single coherent API. All workflow code should call invoke()
rather than using direct llm.ainvoke() or broker.request() calls.

Features:
    - Automatic routing based on model tier and configuration
    - Transparent broker integration for cost optimization
    - DeepSeek support with automatic fallback
    - Prompt caching for Anthropic models
    - Extended thinking support
    - Batch input support (list of user prompts)
    - Structured output via schema= parameter

Example:
    from workflows.shared.llm_utils import invoke, InvokeConfig, ModelTier

    # Simple call
    response = await invoke(
        tier=ModelTier.SONNET,
        system="You are helpful.",
        user="Hello",
    )

    # With batching
    from core.llm_broker import BatchPolicy
    response = await invoke(
        tier=ModelTier.HAIKU,
        system="Score this paper.",
        user="Paper content...",
        config=InvokeConfig(batch_policy=BatchPolicy.PREFER_BALANCE),
    )

    # Batch input
    responses = await invoke(
        tier=ModelTier.HAIKU,
        system="Summarize this.",
        user=["Doc 1...", "Doc 2...", "Doc 3..."],
    )

    # Structured output
    from pydantic import BaseModel

    class Analysis(BaseModel):
        summary: str
        score: float

    result = await invoke(
        tier=ModelTier.SONNET,
        system="Analyze the document.",
        user="Document content...",
        schema=Analysis,
    )
    print(result.summary)  # Typed access
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator, Type, TypeVar, Union, overload

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from .config import InvokeConfig
from .models import ModelTier, get_llm, is_deepseek_tier
from .caching import create_cached_messages

if TYPE_CHECKING:
    from core.llm_broker import LLMResponse

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _broker_response_to_message(response: "LLMResponse") -> AIMessage:
    """Convert broker LLMResponse to proper AIMessage.

    This ensures LangSmith can track token usage and costs correctly
    by populating both response_metadata and usage_metadata.

    Args:
        response: LLMResponse from the broker

    Returns:
        AIMessage with proper metadata for observability
    """
    additional_kwargs: dict[str, Any] = {}
    if response.thinking:
        additional_kwargs["thinking"] = response.thinking

    # Build standardized usage_metadata for LangSmith
    usage_metadata = None
    if response.usage:
        usage_metadata = {
            "input_tokens": response.usage.get("input_tokens", 0),
            "output_tokens": response.usage.get("output_tokens", 0),
            "total_tokens": (response.usage.get("input_tokens", 0) + response.usage.get("output_tokens", 0)),
        }
        # Include cache details if present (Anthropic prompt caching)
        if "cache_creation_input_tokens" in response.usage:
            usage_metadata["input_token_details"] = {
                "cache_creation": response.usage.get("cache_creation_input_tokens", 0),
                "cache_read": response.usage.get("cache_read_input_tokens", 0),
            }

    return AIMessage(
        content=response.content,
        response_metadata={
            "usage": response.usage,  # Keep raw for debugging
            "model": response.model,
            "stop_reason": response.stop_reason,
        },
        usage_metadata=usage_metadata,  # Standard field for LangSmith
        additional_kwargs=additional_kwargs,
    )


async def _invoke_direct(
    tier: ModelTier,
    system: str,
    user_prompts: list[str],
    config: InvokeConfig,
) -> list[AIMessage]:
    """Invoke LLM directly without broker.

    Handles both Anthropic (with caching) and DeepSeek models.
    Uses asyncio.gather() for concurrent processing with rate limiting.

    Args:
        tier: Model tier to use
        system: System prompt
        user_prompts: List of user prompts
        config: Invocation configuration

    Returns:
        List of AIMessage responses (order preserved)
    """
    llm = get_llm(
        tier=tier,
        thinking_budget=config.thinking_budget,
        max_tokens=config.max_tokens,
    )

    # Rate limiting semaphore
    semaphore = asyncio.Semaphore(config.max_concurrent)

    async def invoke_one(user_prompt: str) -> AIMessage:
        async with semaphore:
            # Build messages with caching for Anthropic
            if not is_deepseek_tier(tier) and config.cache:
                messages = create_cached_messages(
                    system_content=system,
                    user_content=user_prompt,
                    cache_system=True,
                    cache_ttl=config.cache_ttl,
                )
            else:
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ]
            return await llm.ainvoke(messages)

    return list(await asyncio.gather(*[invoke_one(p) for p in user_prompts]))


async def _invoke_via_broker(
    tier: ModelTier,
    system: str,
    user_prompts: list[str],
    config: InvokeConfig,
) -> list[AIMessage]:
    """Invoke LLM through the central broker.

    Routes requests through the broker for cost optimization via batching.

    Args:
        tier: Model tier to use
        system: System prompt
        user_prompts: List of user prompts
        config: Invocation configuration

    Returns:
        List of AIMessage responses
    """
    from core.llm_broker import get_broker

    broker = get_broker()

    # Submit all requests within a batch group for proper batching
    futures: list[asyncio.Future] = []

    async with broker.batch_group():
        for user_prompt in user_prompts:
            future = await broker.request(
                prompt=user_prompt,
                model=tier,
                policy=config.batch_policy,
                max_tokens=config.max_tokens,
                system=system,
                thinking_budget=config.thinking_budget,
                tools=config.tools,
                tool_choice=config.tool_choice,
                metadata=config.metadata,
            )
            futures.append(future)

    # Collect results
    results: list[AIMessage] = []
    for future in futures:
        response = await future
        if not response.success:
            raise RuntimeError(f"Broker request failed: {response.error}")
        results.append(_broker_response_to_message(response))

    return results


# Type overloads for invoke()


# Single text input, no schema -> AIMessage
@overload
async def invoke(
    *,
    tier: ModelTier,
    system: str,
    user: str,
    config: InvokeConfig | None = None,
    schema: None = None,
) -> AIMessage: ...


# Batch text input, no schema -> list[AIMessage]
@overload
async def invoke(
    *,
    tier: ModelTier,
    system: str,
    user: list[str],
    config: InvokeConfig | None = None,
    schema: None = None,
) -> list[AIMessage]: ...


# Single input with schema -> T
@overload
async def invoke(
    *,
    tier: ModelTier,
    system: str,
    user: str,
    config: InvokeConfig | None = None,
    schema: Type[T],
) -> T: ...


# Batch input with schema -> list[T]
@overload
async def invoke(
    *,
    tier: ModelTier,
    system: str,
    user: list[str],
    config: InvokeConfig | None = None,
    schema: Type[T],
) -> list[T]: ...


async def invoke(
    *,
    tier: ModelTier,
    system: str,
    user: str | list[str],
    config: InvokeConfig | None = None,
    schema: Type[T] | None = None,
) -> Union[AIMessage, T, list[AIMessage], list[T]]:
    """Unified LLM invocation with automatic routing.

    Routes requests through the optimal path based on model tier and configuration:
    - DeepSeek models: Direct invocation (broker doesn't support DeepSeek)
    - Anthropic with batch_policy: Routes through central broker for cost optimization
    - Otherwise: Direct invocation with prompt caching

    When schema= is provided, returns validated Pydantic model(s) instead of AIMessage.

    Args:
        tier: Model tier to use (HAIKU, SONNET, SONNET_1M, OPUS, DEEPSEEK_V3, DEEPSEEK_R1)
        system: System prompt
        user: User prompt (single string) or list of user prompts (batch)
        config: Optional configuration for caching, batching, thinking, tools, etc.
        schema: Optional Pydantic model class for structured output

    Returns:
        - Without schema: AIMessage for single input, list[AIMessage] for batch
        - With schema: Validated Pydantic model (T) for single, list[T] for batch

    Raises:
        RuntimeError: If broker request fails
        ValueError: If config has invalid constraint combinations
        StructuredOutputError: If structured extraction fails after retries

    Example:
        # Simple text call
        response = await invoke(
            tier=ModelTier.SONNET,
            system="You are helpful.",
            user="Hello",
        )
        print(response.content)

        # With batching
        from core.llm_broker import BatchPolicy
        response = await invoke(
            tier=ModelTier.HAIKU,
            system="Score this paper.",
            user="Paper content...",
            config=InvokeConfig(batch_policy=BatchPolicy.PREFER_BALANCE),
        )

        # Batch input (processes all prompts, returns list)
        responses = await invoke(
            tier=ModelTier.HAIKU,
            system="Summarize this.",
            user=["Doc 1...", "Doc 2...", "Doc 3..."],
        )

        # Structured output
        class Analysis(BaseModel):
            summary: str
            score: float

        result = await invoke(
            tier=ModelTier.SONNET,
            system="Analyze the document.",
            user="Document content...",
            schema=Analysis,
        )
        print(result.summary, result.score)  # Typed access

        # Batch structured output
        results = await invoke(
            tier=ModelTier.HAIKU,
            system="Extract metadata.",
            user=["Doc 1...", "Doc 2..."],
            schema=Metadata,
        )  # Returns list[Metadata]
    """
    config = config or InvokeConfig()

    # Validate tier-specific constraints
    if config.cache and config.thinking_budget and not is_deepseek_tier(tier):
        raise ValueError(
            "Cannot use cache with extended thinking on Anthropic. Set cache=False when using thinking_budget."
        )

    # Route to structured output path if schema provided
    if schema is not None:
        return await _invoke_structured(tier, system, user, config, schema)

    # Text output path
    # Normalize to list for internal processing
    is_batch = isinstance(user, list)
    user_prompts = user if is_batch else [user]

    # Route based on tier and config
    if is_deepseek_tier(tier):
        # DeepSeek: broker can't handle, route directly
        logger.debug(f"Routing to direct invocation (DeepSeek tier: {tier.name})")
        results = await _invoke_direct(tier, system, user_prompts, config)

    elif config.batch_policy is not None:
        # Check if broker is enabled
        from core.llm_broker import is_broker_enabled

        if is_broker_enabled():
            logger.debug(f"Routing through broker (policy: {config.batch_policy.name})")
            results = await _invoke_via_broker(tier, system, user_prompts, config)
        else:
            logger.debug("Broker disabled, falling back to direct invocation")
            results = await _invoke_direct(tier, system, user_prompts, config)
    else:
        # Default: direct invocation with caching
        logger.debug(f"Routing to direct invocation (tier: {tier.name})")
        results = await _invoke_direct(tier, system, user_prompts, config)

    return results if is_batch else results[0]


async def _invoke_structured(
    tier: ModelTier,
    system: str,
    user: str | list[str],
    config: InvokeConfig,
    schema: Type[T],
) -> T | list[T]:
    """Invoke LLM with structured output.

    Selects strategy based on configuration:
    - TOOL_AGENT: When tools are provided (multi-turn agent)
    - LANGCHAIN_STRUCTURED: Default strategy using .with_structured_output()

    Args:
        tier: Model tier to use
        system: System prompt
        user: User prompt (single or list)
        config: Invocation configuration
        schema: Pydantic model class for output

    Returns:
        Validated Pydantic model (single) or list of models (batch)

    Raises:
        StructuredOutputError: If extraction fails after retries
    """
    from .structured.types import (
        StructuredOutputConfig,
        StructuredOutputError,
        StructuredOutputResult,
        StructuredOutputStrategy,
    )
    from .structured.executors import get_executor
    from .structured.retry import with_retries

    # Normalize to list
    is_batch = isinstance(user, list)
    user_prompts = user if is_batch else [user]

    # Select strategy
    if config.tools:
        selected_strategy = StructuredOutputStrategy.TOOL_AGENT
    else:
        selected_strategy = StructuredOutputStrategy.LANGCHAIN_STRUCTURED

    logger.debug(f"Structured output strategy: {selected_strategy.name}")

    # Build StructuredOutputConfig from InvokeConfig
    output_config = StructuredOutputConfig(
        tier=tier,
        max_tokens=config.max_tokens,
        thinking_budget=config.thinking_budget,
        strategy=selected_strategy,
        use_json_schema_method=config.use_json_schema_method,
        batch_policy=config.batch_policy,
        max_retries=config.max_retries,
        retry_backoff=config.retry_backoff,
        enable_context_fallback=config.enable_context_fallback,
        enable_prompt_cache=config.cache,
        cache_ttl=config.cache_ttl,
        tools=config.tools if config.tools else [],
        max_tool_calls=config.max_tool_calls,
        max_tool_result_chars=config.max_tool_result_chars,
    )

    # Single request path with retry logic
    async def execute_single(user_prompt: str) -> T:
        def make_invoke_fn(cfg: StructuredOutputConfig):
            """Factory to create invoke function with given config."""
            executor = get_executor(selected_strategy)

            async def _invoke() -> StructuredOutputResult[T]:
                return await executor.execute(
                    output_schema=schema,
                    user_prompt=user_prompt,
                    system_prompt=system,
                    output_config=cfg,
                )

            return _invoke

        result = await with_retries(
            make_invoke_fn(output_config),
            output_config,
            schema,
            selected_strategy,
            fallback_fn_factory=make_invoke_fn,
        )

        if not result.success:
            raise StructuredOutputError(
                message=result.error or "Unknown error",
                schema=schema,
                strategy=selected_strategy,
                attempts=output_config.max_retries,
            )

        return result.value

    # Execute based on batch size
    if len(user_prompts) == 1:
        result = await execute_single(user_prompts[0])
        return result if not is_batch else [result]
    else:
        # Batch path with concurrent execution
        semaphore = asyncio.Semaphore(10)

        async def execute_one(user_prompt: str) -> T:
            async with semaphore:
                return await execute_single(user_prompt)

        results = await asyncio.gather(*[execute_one(p) for p in user_prompts])
        return list(results)


class InvokeBatch:
    """Batch builder for dynamic batch accumulation.

    Used with the invoke_batch() context manager to accumulate requests
    and signal batch boundaries to the broker.

    Example:
        async with invoke_batch() as batch:
            for paper in papers:
                batch.add(
                    tier=ModelTier.HAIKU,
                    system=SYSTEM,
                    user=format(paper),
                )
            results = await batch.results()
    """

    def __init__(self) -> None:
        self._requests: list[tuple[ModelTier, str, str, InvokeConfig]] = []
        self._futures: list[asyncio.Future] = []
        self._results: list[AIMessage] | None = None

    def add(
        self,
        *,
        tier: ModelTier,
        system: str,
        user: str,
        config: InvokeConfig | None = None,
    ) -> None:
        """Add a request to the batch.

        Args:
            tier: Model tier to use
            system: System prompt
            user: User prompt
            config: Optional configuration (batch_policy will be set automatically)
        """
        self._requests.append((tier, system, user, config or InvokeConfig()))

    async def _submit_to_broker(self) -> None:
        """Submit all accumulated requests to the broker."""
        from core.llm_broker import get_broker, BatchPolicy

        broker = get_broker()

        for tier, system, user, config in self._requests:
            # Force batching for batch context
            policy = config.batch_policy or BatchPolicy.PREFER_BALANCE

            future = await broker.request(
                prompt=user,
                model=tier,
                policy=policy,
                max_tokens=config.max_tokens,
                system=system,
                thinking_budget=config.thinking_budget,
                tools=config.tools,
                tool_choice=config.tool_choice,
                metadata=config.metadata,
            )
            self._futures.append(future)

    async def results(self) -> list[AIMessage]:
        """Get batch results.

        Returns:
            List of AIMessage responses in order of add() calls

        Raises:
            RuntimeError: If called before context manager exits or if any request failed
        """
        if self._results is None:
            raise RuntimeError("Results not available. Wait for context manager to exit.")
        return self._results

    async def _collect_results(self) -> None:
        """Collect results from all futures."""
        self._results = []
        for future in self._futures:
            response = await future
            if not response.success:
                raise RuntimeError(f"Batch request failed: {response.error}")
            self._results.append(_broker_response_to_message(response))


@asynccontextmanager
async def invoke_batch() -> AsyncIterator[InvokeBatch]:
    """Context manager for dynamic batch building.

    Wraps broker.batch_group() to provide a simpler interface for
    accumulating requests dynamically and signaling batch boundaries.

    Example:
        async with invoke_batch() as batch:
            for paper in papers:
                batch.add(
                    tier=ModelTier.HAIKU,
                    system=SYSTEM,
                    user=format(paper),
                )

        results = await batch.results()
        for result in results:
            print(result.content)

    Note:
        All requests in the batch should use Anthropic models (not DeepSeek)
        since DeepSeek doesn't support the batch API.
    """
    from core.llm_broker import get_broker

    broker = get_broker()
    batch = InvokeBatch()

    async with broker.batch_group():
        yield batch
        # Submit all accumulated requests before exiting batch_group
        await batch._submit_to_broker()

    # Collect results after batch_group exits (batch has been flushed)
    await batch._collect_results()


__all__ = [
    "invoke",
    "invoke_batch",
    "InvokeBatch",
]
