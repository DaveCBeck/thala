"""Persistent result caching for LLM responses."""

import hashlib
import logging
from typing import Any, Optional

from workflows.shared.persistent_cache import get_cached, set_cached

logger = logging.getLogger(__name__)

CACHE_TYPE = "llm_results"
CACHE_TTL_DAYS = 7


def _hash_prompt(system_prompt: str, user_prompt: str, model: str) -> str:
    """Generate cache key from prompts and model."""
    combined = f"{model}||{system_prompt}||{user_prompt}"
    return hashlib.sha256(combined.encode()).hexdigest()


async def invoke_with_result_cache(
    llm_invoke_fn,
    system_prompt: str,
    user_prompt: str,
    model_name: str = "sonnet",
    ttl_days: int = CACHE_TTL_DAYS,
    enable_cache: bool = True,
) -> Any:
    """
    Invoke LLM with persistent result caching.

    This caches the complete LLM response based on prompt hash, providing
    100% cost savings and instant results for repeated identical queries.

    Note: This is different from Anthropic's prompt caching (cache_control),
    which caches input tokens. This caches the entire response.

    Use cases:
    - Repeated processing of the same documents
    - Development/testing with fixed inputs
    - Batch reprocessing after workflow changes

    Args:
        llm_invoke_fn: Async function that invokes LLM (returns response)
        system_prompt: System instructions
        user_prompt: User message content
        model_name: Model identifier for cache key
        ttl_days: Cache lifetime in days (default: 7)
        enable_cache: Enable caching (default: True)

    Returns:
        LLM response (from cache or fresh API call)

    Example:
        async def invoke():
            llm = get_llm(ModelTier.SONNET)
            messages = create_cached_messages(system, user)
            return await llm.ainvoke(messages)

        response = await invoke_with_result_cache(
            invoke,
            system_prompt=SYSTEM,
            user_prompt=text,
            model_name="sonnet",
        )
    """
    if not enable_cache:
        return await llm_invoke_fn()

    cache_key = _hash_prompt(system_prompt, user_prompt, model_name)

    cached = get_cached(CACHE_TYPE, cache_key, ttl_days=ttl_days)
    if cached:
        logger.debug(f"LLM result cache hit for {model_name}")
        return cached

    response = await llm_invoke_fn()

    try:
        if hasattr(response, "model_dump"):
            cached_response = response.model_dump()
        elif hasattr(response, "dict"):
            cached_response = response.dict()
        else:
            cached_response = {
                "content": response.content,
                "response_metadata": getattr(response, "response_metadata", {}),
                "usage_metadata": getattr(response, "usage_metadata", {}),
            }

        set_cached(CACHE_TYPE, cache_key, cached_response)
        logger.debug(f"LLM result cached for {model_name}")
    except Exception as e:
        logger.debug(f"Failed to cache LLM result: {e}")

    return response
