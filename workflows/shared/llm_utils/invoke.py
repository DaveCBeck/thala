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
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel

from .config import InvokeConfig
from .models import ModelTier, get_llm, is_deepseek_tier
from .caching import create_cached_messages

if TYPE_CHECKING:
    from core.llm_broker import LLMResponse

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Anthropic Batch API pricing (50% of standard) — (input, output) per million tokens
_BATCH_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-6": (2.50, 12.50),
    "claude-sonnet-4-5-20250929": (1.50, 7.50),
    "claude-haiku-4-5-20251001": (0.50, 2.50),
}

# Type alias for multimodal content (list of content blocks)
MultimodalContent = list[dict[str, Any]]


@traceable(run_type="llm", name="anthropic_batch")
async def _trace_batch_llm_result(
    model: str,
    usage: dict[str, int],
) -> dict[str, Any]:
    """Create a LangSmith LLM run for an Anthropic Batch API result.

    wrap_anthropic() only instruments messages.create()/stream(), not the
    batch API. This creates a synthetic LLM run so LangSmith can track
    tokens and costs. Batch pricing is 50% of standard API rates.
    """
    rt = get_current_run_tree()
    if rt:
        rt.extra = rt.extra or {}
        rt.extra.setdefault("metadata", {})
        rt.extra["metadata"]["ls_provider"] = "anthropic"
        rt.extra["metadata"]["ls_model_name"] = model

        input_t = usage.get("input_tokens", 0)
        output_t = usage.get("output_tokens", 0)
        usage_meta: dict[str, Any] = {
            "input_tokens": input_t,
            "output_tokens": output_t,
            "total_tokens": input_t + output_t,
        }

        if "cache_read_input_tokens" in usage:
            usage_meta["input_token_details"] = {
                "cache_read": usage.get("cache_read_input_tokens", 0),
                "cache_creation": usage.get("cache_creation_input_tokens", 0),
            }

        pricing = _BATCH_PRICING.get(model)
        if pricing:
            input_rate, output_rate = pricing
            usage_meta["input_cost"] = input_t * input_rate / 1_000_000
            usage_meta["output_cost"] = output_t * output_rate / 1_000_000
            usage_meta["total_cost"] = usage_meta["input_cost"] + usage_meta["output_cost"]

        rt.set(usage_metadata=usage_meta)

    return {"model": model, "usage": usage}


def _is_multimodal_content(user: Any) -> bool:
    """Check if user input is multimodal content (list of content dicts).

    Multimodal content is a list of dicts with "type" keys, e.g.:
    [{"type": "text", "text": "..."}, {"type": "image", "source": {...}}]

    This is distinct from a batch of string prompts: ["prompt1", "prompt2"]
    """
    if not isinstance(user, list) or len(user) == 0:
        return False
    # Check if first element is a content dict (has "type" key)
    first = user[0]
    return isinstance(first, dict) and "type" in first


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
    user_prompts: list[str | MultimodalContent],
    config: InvokeConfig,
) -> list[AIMessage]:
    """Invoke LLM directly without broker.

    Handles both Anthropic (with caching) and DeepSeek models.
    Uses asyncio.gather() for concurrent processing with rate limiting.

    Args:
        tier: Model tier to use
        system: System prompt
        user_prompts: List of user prompts (text or multimodal content blocks)
        config: Invocation configuration

    Returns:
        List of AIMessage responses (order preserved)
    """
    llm = get_llm(
        tier=tier,
        effort=config.effort,
        max_tokens=config.max_tokens,
    )

    # Rate limiting semaphore
    semaphore = asyncio.Semaphore(config.max_concurrent)

    async def invoke_one(user_prompt: str | MultimodalContent) -> AIMessage:
        async with semaphore:
            if not isinstance(user_prompt, list) and not is_deepseek_tier(tier) and config.cache:
                # Anthropic prompt caching
                messages = create_cached_messages(
                    system_content=system,
                    user_content=user_prompt,
                    cache_system=True,
                    cache_ttl=config.cache_ttl,
                )
            else:
                # Standard messages (text or multimodal)
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ]
            return await llm.ainvoke(messages)

    return list(await asyncio.gather(*[invoke_one(p) for p in user_prompts]))


async def _invoke_via_broker(
    tier: ModelTier,
    system: str,
    user_prompts: list[str | MultimodalContent],
    config: InvokeConfig,
) -> list[AIMessage]:
    """Invoke LLM through the central broker.

    Routes requests through the broker for cost optimization via batching.

    Args:
        tier: Model tier to use
        system: System prompt
        user_prompts: List of user prompts (text or multimodal content blocks)
        config: Invocation configuration

    Returns:
        List of AIMessage responses
    """
    from core.llm_broker import get_broker

    broker = get_broker()

    # Submit all requests within a batch group for proper batching
    futures: list[asyncio.Future] = []

    # Guard: multimodal + tools not supported via broker
    has_multimodal = any(isinstance(p, list) for p in user_prompts)
    if has_multimodal and config.tools:
        raise ValueError(
            "Multimodal content with tools is not supported via the broker path. "
            "Use invoke() without tools for multimodal content."
        )

    async with broker.batch_group():
        for user_prompt in user_prompts:
            kwargs = dict(
                model=tier,
                policy=config.batch_policy,
                max_tokens=config.max_tokens,
                system=system,
                effort=config.effort,
                metadata=config.metadata,
            )
            if isinstance(user_prompt, list):
                # Multimodal: pass pre-built messages
                kwargs["prompt"] = ""  # unused when messages= is set
                kwargs["messages"] = [{"role": "user", "content": user_prompt}]
            else:
                kwargs["prompt"] = user_prompt
                kwargs["tools"] = config.tools
                kwargs["tool_choice"] = config.tool_choice

            future = await broker.request(**kwargs)
            futures.append(future)

    # Collect results, creating LangSmith LLM traces for batch responses
    results: list[AIMessage] = []
    for future in futures:
        response = await future
        if not response.success:
            raise RuntimeError(f"Broker request failed: {response.error}")
        if response.batched and response.usage:
            await _trace_batch_llm_result(
                model=response.model or tier.value,
                usage=response.usage,
            )
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


# Multimodal input, no schema -> AIMessage
@overload
async def invoke(
    *,
    tier: ModelTier,
    system: str,
    user: MultimodalContent,
    config: InvokeConfig | None = None,
    schema: None = None,
) -> AIMessage: ...


# Multimodal input with schema -> T
@overload
async def invoke(
    *,
    tier: ModelTier,
    system: str,
    user: MultimodalContent,
    config: InvokeConfig | None = None,
    schema: Type[T],
) -> T: ...


async def invoke(
    *,
    tier: ModelTier,
    system: str,
    user: str | list[str] | MultimodalContent,
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

    # Route to structured output path if schema provided
    if schema is not None:
        return await _invoke_structured(tier, system, user, config, schema)

    # Text output path
    # Normalize to list for internal processing
    is_multimodal = _is_multimodal_content(user)
    is_batch = isinstance(user, list) and not is_multimodal
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


async def _invoke_structured_via_broker(
    tier: "ModelTier",
    system: str,
    user_prompt: str,
    config: "InvokeConfig",
    schema: Type[T],
) -> T:
    """Invoke structured output through broker using tool_use pattern.

    Converts the Pydantic schema to an Anthropic tool definition,
    forces tool_choice, and parses the broker response back to a
    validated Pydantic model.

    When thinking (effort) is enabled, tool_choice must be "auto" instead
    of forced.  If the model returns text instead of a tool call, we retry
    once with forced tool_choice and no thinking as a fallback.
    """
    import json as json_mod

    from pydantic import ValidationError

    from core.llm_broker import get_broker
    from .structured.executors.base import coerce_to_schema

    broker = get_broker()

    schema_name = schema.__name__
    tool_def = {
        "name": schema_name,
        "description": f"Return the {schema_name} result.",
        "input_schema": schema.model_json_schema(),
    }

    forced_tool_choice = {"type": "tool", "name": schema_name}

    # When thinking (effort) is enabled, we cannot force tool_choice — the
    # Anthropic API rejects that combination.  Use tool_choice "auto" instead;
    # the model reliably picks the only available tool.
    if config.effort:
        tool_choice = {"type": "auto"}
    else:
        tool_choice = forced_tool_choice

    async with broker.batch_group():
        future = await broker.request(
            prompt=user_prompt,
            model=tier,
            policy=config.batch_policy,
            max_tokens=config.max_tokens,
            system=system,
            effort=config.effort,
            tools=[tool_def],
            tool_choice=tool_choice,
            metadata=config.metadata,
        )

    response = await future
    if not response.success:
        raise RuntimeError(f"Broker structured request failed: {response.error}")

    if response.batched and response.usage:
        await _trace_batch_llm_result(
            model=response.model or tier.value,
            usage=response.usage,
        )

    try:
        raw_data = json_mod.loads(response.content)
        coerced = coerce_to_schema(raw_data, schema)
        return schema.model_validate(coerced)
    except (json_mod.JSONDecodeError, ValidationError, TypeError) as exc:
        if not config.effort:
            raise  # forced tool_choice was used; nothing else to try

        logger.warning(
            "Structured broker response did not match schema %s "
            "(tool_choice was 'auto' due to thinking). Retrying with "
            "forced tool_choice and no thinking. Parse error: %s",
            schema_name,
            exc,
        )

        # Retry: drop thinking so we can force the tool call.
        async with broker.batch_group():
            future = await broker.request(
                prompt=user_prompt,
                model=tier,
                policy=config.batch_policy,
                max_tokens=config.max_tokens,
                system=system,
                tools=[tool_def],
                tool_choice=forced_tool_choice,
                metadata=config.metadata,
            )

        response = await future
        if not response.success:
            raise RuntimeError(
                f"Broker structured request failed on retry: {response.error}"
            )

        if response.batched and response.usage:
            await _trace_batch_llm_result(
                model=response.model or tier.value,
                usage=response.usage,
            )

        raw_data = json_mod.loads(response.content)
        coerced = coerce_to_schema(raw_data, schema)
        return schema.model_validate(coerced)


async def _invoke_structured(
    tier: ModelTier,
    system: str,
    user: str | list[str] | MultimodalContent,
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
        user: User prompt (single, list, or multimodal content blocks)
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

    # Normalize to list - multimodal content is NOT a batch, it's a single prompt
    is_multimodal = _is_multimodal_content(user)
    is_batch = isinstance(user, list) and not is_multimodal
    user_prompts = user if is_batch else [user]

    # Select strategy
    if config.tools:
        selected_strategy = StructuredOutputStrategy.TOOL_AGENT
    else:
        selected_strategy = StructuredOutputStrategy.LANGCHAIN_STRUCTURED

    logger.info(
        f"_invoke_structured: strategy={selected_strategy.name}, "
        f"tier={tier}, has_tools={bool(config.tools)}, "
        f"batch_policy={config.batch_policy}"
    )

    # Broker-routed structured output (no tools, batch_policy set)
    if (
        config.batch_policy is not None
        and selected_strategy == StructuredOutputStrategy.LANGCHAIN_STRUCTURED
        and not is_multimodal
        and not is_deepseek_tier(tier)
    ):
        from core.llm_broker import is_broker_enabled

        if is_broker_enabled():
            logger.debug(f"Routing structured output through broker (policy: {config.batch_policy.name})")

            if len(user_prompts) == 1:
                result = await _invoke_structured_via_broker(tier, system, user_prompts[0], config, schema)
                return result if not is_batch else [result]
            else:
                semaphore = asyncio.Semaphore(10)

                async def _broker_one(prompt: str) -> T:
                    async with semaphore:
                        return await _invoke_structured_via_broker(tier, system, prompt, config, schema)

                results = await asyncio.gather(*[_broker_one(p) for p in user_prompts])
                return list(results)

    # Build StructuredOutputConfig from InvokeConfig
    output_config = StructuredOutputConfig(
        tier=tier,
        max_tokens=config.max_tokens,
        effort=config.effort,
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
    async def execute_single(user_prompt: str | MultimodalContent) -> T:
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

        async def execute_one(user_prompt: str | MultimodalContent) -> T:
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
                effort=config.effort,
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
