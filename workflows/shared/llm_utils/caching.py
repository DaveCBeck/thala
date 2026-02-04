"""Prompt caching utilities for cost optimization.

Supports both:
- Anthropic: Explicit cache_control with ephemeral blocks
- DeepSeek: Automatic prefix-based caching with warmup delay

When THALA_LLM_BROKER_ENABLED=1 and batch_policy is set, routes Anthropic
calls through the central LLM broker for unified cost/speed management.
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Literal, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek

if TYPE_CHECKING:
    from core.llm_broker import BatchPolicy
    from workflows.shared.llm_utils.models import ModelTier

logger = logging.getLogger(__name__)

CacheTTL = Literal["5m", "1h"]


class BrokerResponseWrapper:
    """Wraps broker LLMResponse to be compatible with LangChain response format.

    This allows code using extract_response_content() and similar utilities
    to work seamlessly with responses from either LangChain models or the broker.
    """

    def __init__(self, content: Any, usage: Optional[dict] = None):
        self.content = content
        self.usage_metadata = usage


# DeepSeek cache warmup tracking (prefix hash -> timestamp)
_deepseek_cache_warmed: dict[int, float] = {}
DEEPSEEK_CACHE_WARMUP_DELAY = 10.0  # seconds to wait for cache construction


def create_cached_messages(
    system_content: str | list[dict],
    user_content: str,
    cache_system: bool = True,
    cache_ttl: CacheTTL = "5m",
) -> list[dict]:
    """Create messages with cache_control on system content."""
    cache_control = {"type": "ephemeral"}
    if cache_ttl == "1h":
        cache_control["ttl"] = "1h"

    if isinstance(system_content, str):
        system_blocks = [
            {
                "type": "text",
                "text": system_content,
                **({"cache_control": cache_control} if cache_system else {}),
            }
        ]
    else:
        system_blocks = list(system_content)
        if cache_system and system_blocks:
            last_block = dict(system_blocks[-1])
            last_block["cache_control"] = cache_control
            system_blocks[-1] = last_block

    messages = [
        {"role": "system", "content": system_blocks},
        {"role": "user", "content": user_content},
    ]

    return messages


def _is_deepseek_model(llm: BaseChatModel) -> bool:
    """Check if LLM is a DeepSeek model."""
    return isinstance(llm, ChatDeepSeek)


async def warm_deepseek_cache(system_prompt: str) -> None:
    """Ensure DeepSeek cache is warmed for this prefix.

    DeepSeek's cache is automatic and prefix-based, but requires
    a few seconds to construct after the first request. This function
    tracks which prefixes have been used and waits for cache construction
    on the first call with a new prefix.
    """
    prefix_hash = hash(system_prompt)
    if prefix_hash not in _deepseek_cache_warmed:
        logger.debug(f"DeepSeek cache warmup: waiting {DEEPSEEK_CACHE_WARMUP_DELAY}s for prefix")
        _deepseek_cache_warmed[prefix_hash] = time.time()
        await asyncio.sleep(DEEPSEEK_CACHE_WARMUP_DELAY)


def _get_model_tier_from_llm(llm: ChatAnthropic) -> "ModelTier":
    """Extract ModelTier from ChatAnthropic model name."""
    from workflows.shared.llm_utils.models import ModelTier

    model_name = llm.model_name or llm.model or "claude-sonnet-4-20250514"
    model_lower = model_name.lower()

    if "opus" in model_lower:
        return ModelTier.OPUS
    elif "haiku" in model_lower:
        return ModelTier.HAIKU
    elif "sonnet" in model_lower:
        if "1m" in model_lower or "200k" in model_lower:
            return ModelTier.SONNET_1M
        return ModelTier.SONNET
    return ModelTier.SONNET


async def _invoke_via_broker(
    llm: ChatAnthropic,
    system_prompt: str,
    user_prompt: str,
    batch_policy: "BatchPolicy",
) -> BrokerResponseWrapper:
    """Route an Anthropic request through the central LLM broker.

    Args:
        llm: ChatAnthropic model (used to extract model tier and max_tokens)
        system_prompt: System message
        user_prompt: User message
        batch_policy: Broker batch policy

    Returns:
        BrokerResponseWrapper compatible with LangChain response format
    """
    from core.llm_broker import get_broker

    broker = get_broker()
    model_tier = _get_model_tier_from_llm(llm)
    max_tokens = getattr(llm, "max_tokens", None) or 4096

    # Submit request to broker
    future = await broker.request(
        prompt=user_prompt,
        model=model_tier,
        policy=batch_policy,
        max_tokens=max_tokens,
        system=system_prompt,
    )

    # Wait for response
    response = await future

    if not response.success:
        raise RuntimeError(f"Broker request failed: {response.error}")

    return BrokerResponseWrapper(
        content=response.content,
        usage=response.usage,
    )


async def invoke_with_cache(
    llm: BaseChatModel,
    system_prompt: str,
    user_prompt: str,
    cache_system: bool = True,
    cache_ttl: CacheTTL = "5m",
    batch_policy: Optional["BatchPolicy"] = None,
) -> Any:
    """Invoke LLM with prompt caching (Anthropic or DeepSeek).

    For Anthropic: Uses explicit cache_control with ephemeral blocks.
    For DeepSeek: Uses automatic prefix-based caching with warmup delay.

    Args:
        llm: LangChain language model
        system_prompt: System message to cache
        user_prompt: User message
        cache_system: Whether to apply cache_control to system (default: True)
        cache_ttl: "5m" (default, ephemeral) or "1h" (longer retention)
        batch_policy: When set and THALA_LLM_BROKER_ENABLED=1, routes Anthropic
            calls through the central LLM broker. Ignored for DeepSeek models.

    Returns:
        LangChain response object (or BrokerResponseWrapper when using broker)
    """
    # Check if we should route through broker (Anthropic only)
    if batch_policy is not None and isinstance(llm, ChatAnthropic):
        from core.llm_broker import is_broker_enabled

        if is_broker_enabled():
            return await _invoke_via_broker(
                llm=llm,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                batch_policy=batch_policy,
            )
    # DeepSeek: automatic prefix caching, needs warmup delay
    if _is_deepseek_model(llm):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Check if cache needs warmup (first call with this prefix)
        prefix_hash = hash(system_prompt)
        if cache_system and prefix_hash not in _deepseek_cache_warmed:
            # First call - make request, then wait for cache construction
            response = await llm.ainvoke(messages)
            _deepseek_cache_warmed[prefix_hash] = time.time()
            logger.debug(f"DeepSeek cache warmup: waiting {DEEPSEEK_CACHE_WARMUP_DELAY}s for prefix")
            await asyncio.sleep(DEEPSEEK_CACHE_WARMUP_DELAY)
            return response

        return await llm.ainvoke(messages)

    # Anthropic: explicit cache_control
    if isinstance(llm, ChatAnthropic) and cache_system:
        messages = create_cached_messages(
            system_content=system_prompt,
            user_content=user_prompt,
            cache_system=cache_system,
            cache_ttl=cache_ttl,
        )
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    response = await llm.ainvoke(messages)

    # Log Anthropic cache usage
    if isinstance(llm, ChatAnthropic):
        usage = getattr(response, "usage_metadata", None)
        if usage:
            details = usage.get("input_token_details", {})
            cache_read = details.get("cache_read", 0)
            cache_creation = details.get("cache_creation", 0)
            if cache_read > 0:
                logger.debug(f"Cache hit: {cache_read} tokens read from cache")
            elif cache_creation > 0:
                logger.debug(f"Cache miss: {cache_creation} tokens written to cache")

    return response


def _compute_prefix_hash(system_prompt: str, cache_prefix: str | None = None) -> int:
    """Compute hash for cache tracking, including optional user prefix."""
    if cache_prefix:
        return hash(system_prompt + cache_prefix)
    return hash(system_prompt)


async def _batch_invoke_via_broker(
    llm: ChatAnthropic,
    system_prompt: str,
    user_prompts: list[tuple[str, str]],
    batch_policy: "BatchPolicy",
) -> dict[str, Any]:
    """Route batch requests through the central LLM broker.

    Args:
        llm: ChatAnthropic model (used to extract model tier and max_tokens)
        system_prompt: System message (shared across all requests)
        user_prompts: List of (request_id, user_prompt) tuples
        batch_policy: Broker batch policy

    Returns:
        Dict mapping request_id to BrokerResponseWrapper
    """
    from core.llm_broker import get_broker

    broker = get_broker()
    model_tier = _get_model_tier_from_llm(llm)
    max_tokens = getattr(llm, "max_tokens", None) or 4096

    # Submit all requests within a batch group
    pending: dict[str, Any] = {}
    async with broker.batch_group():
        for req_id, user_prompt in user_prompts:
            future = await broker.request(
                prompt=user_prompt,
                model=model_tier,
                policy=batch_policy,
                max_tokens=max_tokens,
                system=system_prompt,
                metadata={"request_id": req_id},
            )
            pending[req_id] = future

    # Collect results
    results: dict[str, Any] = {}
    for req_id, future in pending.items():
        response = await future
        if response.success:
            results[req_id] = BrokerResponseWrapper(
                content=response.content,
                usage=response.usage,
            )
        else:
            logger.warning(f"Broker request {req_id} failed: {response.error}")
            results[req_id] = BrokerResponseWrapper(
                content=f"Error: {response.error}",
                usage=None,
            )

    return results


async def batch_invoke_with_cache(
    llm: BaseChatModel,
    system_prompt: str,
    user_prompts: list[tuple[str, str]],
    cache_prefix: str | None = None,
    max_concurrent: int = 10,
    batch_policy: Optional["BatchPolicy"] = None,
) -> dict[str, Any]:
    """Invoke LLM for multiple requests with cache warmup coordination.

    For DeepSeek: Sends first request, waits for cache construction, then
    processes remaining requests concurrently to benefit from prefix caching.

    For Anthropic: Processes all requests concurrently (cache is explicit).
    When batch_policy is set and broker is enabled, routes through central broker.

    Args:
        llm: Language model to use
        system_prompt: System prompt (shared across all requests)
        user_prompts: List of (request_id, user_prompt) tuples
        cache_prefix: Optional shared prefix in user prompts for hash tracking
            (e.g., "Research Topic: X\\nResearch Questions: Y\\n\\n")
        max_concurrent: Maximum concurrent requests after warmup
        batch_policy: When set and THALA_LLM_BROKER_ENABLED=1, routes Anthropic
            calls through the central LLM broker. Ignored for DeepSeek models.

    Returns:
        Dict mapping request_id to response
    """
    if not user_prompts:
        return {}

    results: dict[str, Any] = {}

    # Check if we should route through broker (Anthropic only)
    if batch_policy is not None and isinstance(llm, ChatAnthropic):
        from core.llm_broker import is_broker_enabled

        if is_broker_enabled():
            return await _batch_invoke_via_broker(
                llm=llm,
                system_prompt=system_prompt,
                user_prompts=user_prompts,
                batch_policy=batch_policy,
            )

    # Non-DeepSeek: process all concurrently
    if not _is_deepseek_model(llm):
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(req_id: str, user_prompt: str) -> tuple[str, Any]:
            async with semaphore:
                response = await invoke_with_cache(llm, system_prompt, user_prompt)
                return req_id, response

        tasks = [process_one(req_id, prompt) for req_id, prompt in user_prompts]
        for req_id, response in await asyncio.gather(*tasks):
            results[req_id] = response
        return results

    # DeepSeek: coordinate cache warmup
    prefix_hash = _compute_prefix_hash(system_prompt, cache_prefix)

    # Check if cache needs warmup
    if prefix_hash not in _deepseek_cache_warmed:
        # First request triggers cache construction
        first_id, first_prompt = user_prompts[0]
        remaining = user_prompts[1:]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_prompt},
        ]
        response = await llm.ainvoke(messages)
        results[first_id] = response

        # Mark warmed and wait for cache construction
        _deepseek_cache_warmed[prefix_hash] = time.time()
        logger.info(
            f"DeepSeek cache warmup: waiting {DEEPSEEK_CACHE_WARMUP_DELAY}s "
            f"before processing {len(remaining)} remaining requests"
        )
        await asyncio.sleep(DEEPSEEK_CACHE_WARMUP_DELAY)
    else:
        remaining = user_prompts
        logger.debug(f"DeepSeek cache already warm, processing {len(remaining)} requests")

    # Process remaining requests concurrently
    if remaining:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_remaining(req_id: str, user_prompt: str) -> tuple[str, Any]:
            async with semaphore:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                return req_id, await llm.ainvoke(messages)

        tasks = [process_remaining(req_id, prompt) for req_id, prompt in remaining]
        for req_id, response in await asyncio.gather(*tasks):
            results[req_id] = response

    return results
