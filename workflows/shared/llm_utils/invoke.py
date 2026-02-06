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
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator

from langchain_core.messages import AIMessage

from .config import InvokeConfig
from .models import ModelTier, get_llm, is_deepseek_tier
from .caching import create_cached_messages

if TYPE_CHECKING:
    from core.llm_broker import LLMResponse

logger = logging.getLogger(__name__)


def _broker_response_to_message(response: "LLMResponse") -> AIMessage:
    """Convert broker LLMResponse to proper AIMessage.

    This ensures LangSmith can track token usage and costs correctly
    by populating response_metadata with usage data.

    Args:
        response: LLMResponse from the broker

    Returns:
        AIMessage with proper metadata for observability
    """
    additional_kwargs: dict[str, Any] = {}
    if response.thinking:
        additional_kwargs["thinking"] = response.thinking

    return AIMessage(
        content=response.content,
        response_metadata={
            "usage": response.usage,  # Required for LangSmith token/cost tracking
            "model": response.model,
            "stop_reason": response.stop_reason,
        },
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

    Args:
        tier: Model tier to use
        system: System prompt
        user_prompts: List of user prompts
        config: Invocation configuration

    Returns:
        List of AIMessage responses
    """
    llm = get_llm(
        tier=tier,
        thinking_budget=config.thinking_budget,
        max_tokens=config.max_tokens,
    )

    results: list[AIMessage] = []

    for user_prompt in user_prompts:
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

        response = await llm.ainvoke(messages)
        results.append(response)

    return results


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


async def invoke(
    *,
    tier: ModelTier,
    system: str,
    user: str | list[str],
    config: InvokeConfig | None = None,
) -> AIMessage | list[AIMessage]:
    """Unified LLM invocation with automatic routing.

    Routes requests through the optimal path based on model tier and configuration:
    - DeepSeek models: Direct invocation (broker doesn't support DeepSeek)
    - Anthropic with batch_policy: Routes through central broker for cost optimization
    - Otherwise: Direct invocation with prompt caching

    Args:
        tier: Model tier to use (HAIKU, SONNET, SONNET_1M, OPUS, DEEPSEEK_V3, DEEPSEEK_R1)
        system: System prompt
        user: User prompt (single string) or list of user prompts (batch)
        config: Optional configuration for caching, batching, thinking, etc.

    Returns:
        AIMessage for single user prompt, list[AIMessage] for batch input

    Raises:
        RuntimeError: If broker request fails
        ValueError: If config has invalid constraint combinations

    Example:
        # Simple call
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
    """
    config = config or InvokeConfig()

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
