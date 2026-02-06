"""LLM utilities for document processing workflows.

This module provides Anthropic Claude and DeepSeek model integration with:
- Unified invoke() function for all LLM calls (recommended)
- Tiered model selection (Haiku/Sonnet/Opus/DeepSeek)
- Extended thinking support for complex reasoning tasks
- Prompt caching for cost optimization (90% savings on cache hits)
- Automatic broker routing for batch cost optimization
- Structured output interface

Unified Invocation (Recommended):
    The invoke() function is the recommended way to call LLMs. It handles
    routing, caching, and broker integration automatically:

    from workflows.shared.llm_utils import invoke, InvokeConfig, ModelTier
    from core.llm_broker import BatchPolicy

    # Simple call
    response = await invoke(
        tier=ModelTier.SONNET,
        system="You are helpful.",
        user="Hello",
    )

    # With batching (routes through broker for cost savings)
    response = await invoke(
        tier=ModelTier.HAIKU,
        system="Score this.",
        user="Content...",
        config=InvokeConfig(batch_policy=BatchPolicy.PREFER_BALANCE),
    )

    # Batch input (list of prompts)
    responses = await invoke(
        tier=ModelTier.HAIKU,
        system="Summarize.",
        user=["Doc 1...", "Doc 2...", "Doc 3..."],
    )

Dynamic Batch Building:
    Use invoke_batch() for accumulating requests dynamically:

    async with invoke_batch() as batch:
        for paper in papers:
            batch.add(tier=ModelTier.HAIKU, system=SYSTEM, user=format(paper))
    results = await batch.results()

Structured Output:
    The get_structured_output() function provides structured output with
    automatic strategy selection. Will be migrated to invoke(..., schema=)
    in a future release.

Prompt Caching:
    Cache reads cost 10% of base input token price. The invoke() function
    handles caching automatically. For manual control, use create_cached_messages().
"""

from .models import ModelTier, get_llm
from .config import InvokeConfig
from .invoke import invoke, invoke_batch, InvokeBatch
from .caching import (
    CacheTTL,
    create_cached_messages,
    invoke_with_cache,
    batch_invoke_with_cache,
    warm_deepseek_cache,
    BrokerResponseWrapper,
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
from .response_parsing import extract_response_content

__all__ = [
    # Unified invocation (recommended)
    "invoke",
    "invoke_batch",
    "InvokeBatch",
    "InvokeConfig",
    # Model utilities
    "ModelTier",
    "get_llm",
    # Structured output
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
    "batch_invoke_with_cache",
    "warm_deepseek_cache",
    "BrokerResponseWrapper",
    # Response parsing
    "extract_response_content",
]
