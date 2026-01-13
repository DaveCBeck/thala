"""LLM utilities for document processing workflows.

This module provides Anthropic Claude model integration with:
- Tiered model selection (Haiku/Sonnet/Opus)
- Extended thinking support for complex reasoning tasks
- Prompt caching for cost optimization (90% savings on cache hits)
- Both synchronous and batch processing modes
- Unified structured output interface (recommended)

Structured Output (Recommended):
    The get_structured_output() function provides a unified interface for all
    structured output needs, automatically selecting the best strategy:

    # Single request
    result = await get_structured_output(
        output_schema=MySchema,
        user_prompt="Analyze this...",
        tier=ModelTier.SONNET,
    )

    # Batch request (auto-uses batch API for 5+ items, 50% cost savings)
    results = await get_structured_output(
        output_schema=MySchema,
        requests=[StructuredRequest(id="1", user_prompt="..."), ...],
        tier=ModelTier.HAIKU,
    )

Prompt Caching:
    Cache reads cost 10% of base input token price. To use caching:
    1. Use invoke_with_cache() for simple cached calls
    2. Use create_cached_messages() to build messages with cache_control
    3. Structure prompts with static content first, dynamic content last
"""

from .models import ModelTier, get_llm
from .caching import (
    CacheTTL,
    create_cached_messages,
    invoke_with_cache,
)
from .structured import (
    StructuredOutputStrategy,
    StructuredOutputConfig,
    StructuredRequest,
    StructuredOutputResult,
    BatchResult,
    StructuredOutputError,
    get_structured_output,
    get_structured_output_with_result,
    extract_from_text,
    classify_content,
)

__all__ = [
    # Model utilities
    "ModelTier",
    "get_llm",
    # Structured output (recommended)
    "StructuredOutputStrategy",
    "StructuredOutputConfig",
    "StructuredRequest",
    "StructuredOutputResult",
    "BatchResult",
    "StructuredOutputError",
    "get_structured_output",
    "get_structured_output_with_result",
    "extract_from_text",
    "classify_content",
    # Caching utilities
    "CacheTTL",
    "create_cached_messages",
    "invoke_with_cache",
]
