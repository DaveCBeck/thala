"""LLM utilities for document processing workflows.

This module provides Anthropic Claude model integration with:
- Tiered model selection (Haiku/Sonnet/Opus)
- Extended thinking support for complex reasoning tasks
- Prompt caching for cost optimization (90% savings on cache hits)
- Result caching for 100% savings on identical queries
- Both synchronous and batch processing modes

Prompt Caching:
    Cache reads cost 10% of base input token price. To use caching:
    1. Use invoke_with_cache() for simple cached calls
    2. Use create_cached_messages() to build messages with cache_control
    3. Structure prompts with static content first, dynamic content last

Result Caching:
    Persistent caching of complete LLM responses for identical inputs:
    1. Use invoke_with_result_cache() to cache complete responses
    2. Default TTL: 7 days (configurable)
    3. Perfect for dev/test and re-running workflows
"""

from .models import ModelTier, get_llm
from .text_processors import (
    summarize_text,
    extract_json,
    analyze_with_thinking,
    extract_structured,
)
from .caching import (
    CacheTTL,
    create_cached_messages,
    invoke_with_cache,
    summarize_text_cached,
    extract_json_cached,
)
from .result_cache import invoke_with_result_cache

__all__ = [
    "ModelTier",
    "get_llm",
    "summarize_text",
    "extract_json",
    "analyze_with_thinking",
    "extract_structured",
    "CacheTTL",
    "create_cached_messages",
    "invoke_with_cache",
    "summarize_text_cached",
    "extract_json_cached",
    "invoke_with_result_cache",
]
