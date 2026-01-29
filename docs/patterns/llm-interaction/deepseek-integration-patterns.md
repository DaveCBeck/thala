---
name: deepseek-integration-patterns
title: "DeepSeek Integration Patterns: V3 and R1 Model Selection"
date: 2026-01-28
category: llm-interaction
applicability:
  - "Cost optimization for high-volume LLM tasks"
  - "Reasoning-intensive tasks requiring explicit thinking"
  - "Document processing pipelines needing classification/extraction"
  - "Batch processing with prefix-based caching"
components: [model_selection, prefix_caching, r1_thinking, structured_output, warmup_delay]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [deepseek, cost-optimization, reasoning, caching, model-selection, v3, r1, structured-output]
---

# DeepSeek Integration Patterns: V3 and R1 Model Selection

## Intent

Integrate DeepSeek V3 and R1 models with proper caching, structured output, and model selection to achieve 70-80% cost savings on high-volume tasks while maintaining quality on reasoning-intensive work.

## Motivation

Claude models are expensive for high-volume tasks:

**The Problem:**
```
Monthly LLM Cost Breakdown:
├── Content classification (100K calls)  → Claude Haiku: $500
├── Relevance scoring (50K calls)        → Claude Haiku: $250
├── Metadata extraction (30K calls)      → Claude Haiku: $150
├── Document summarization (20K calls)   → Claude Sonnet: $600
└── Complex synthesis (5K calls)         → Claude Opus: $500
                                          Total: ~$2,000/month

Problem: Classification and extraction are simple tasks
         being processed by expensive models.
```

**The Solution:**
```
Hybrid Model Strategy:
├── Content classification (100K calls)  → DeepSeek V3: $27   (80% savings)
├── Relevance scoring (50K calls)        → DeepSeek V3: $14   (80% savings)
├── Metadata extraction (30K calls)      → DeepSeek V3: $8    (80% savings)
├── Document summarization (20K calls)   → DeepSeek R1: $44   (80% savings)
└── Complex synthesis (5K calls)         → Claude Opus: $500  (keep quality)
                                          Total: ~$600/month (70% savings)
```

## Applicability

Use DeepSeek V3 when:
- Task is classification, filtering, or simple extraction
- High volume (thousands of calls)
- Speed is more important than reasoning depth
- Cost sensitivity is high

Use DeepSeek R1 when:
- Task requires explicit reasoning or analysis
- Complex summarization or synthesis
- Moderate volume
- Need to see thinking process

Keep Claude when:
- Quality is critical (final synthesis, user-facing content)
- Task requires nuanced judgment
- Low volume makes cost savings minimal
- Need guaranteed reliability

## Structure

```
┌────────────────────────────────────────────────────────────────────┐
│  Model Selection Decision Tree                                     │
│                                                                    │
│              ┌──────────────────────────────────────┐              │
│              │  Is quality critical?                │              │
│              └──────────────────┬───────────────────┘              │
│                    ┌────────────┴────────────┐                     │
│                  Yes                        No                     │
│                    ↓                        ↓                     │
│            Use Claude               ┌──────────────────┐          │
│            (OPUS/SONNET)            │ Needs reasoning? │          │
│                                     └────────┬─────────┘          │
│                              ┌───────────────┴────────────┐       │
│                            Yes                           No       │
│                              ↓                            ↓       │
│                    Use DeepSeek R1              Use DeepSeek V3   │
│                    (reasoner)                   (chat)            │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│  DeepSeek Caching Flow                                             │
│                                                                    │
│  First request with new system prompt:                            │
│  ┌─────────────────┐                                              │
│  │ Make LLM call   │ → Response (full token cost)                 │
│  └─────────────────┘                                              │
│           ↓                                                        │
│  ┌─────────────────┐                                              │
│  │ Wait 10 seconds │ ← Cache warmup delay (DeepSeek internal)     │
│  └─────────────────┘                                              │
│           ↓                                                        │
│  Subsequent requests with same system prompt:                     │
│  ┌─────────────────┐                                              │
│  │ Make LLM call   │ → Response (90% cost reduction)              │
│  └─────────────────┘    ↑ Cached prefix applied                   │
└────────────────────────────────────────────────────────────────────┘
```

## Implementation

### Step 1: Configure Model Tiers

```python
# workflows/shared/llm_utils/models.py

from enum import Enum
from langchain_deepseek import ChatDeepSeek


class ModelTier(str, Enum):
    """Available model tiers with cost/capability trade-offs."""

    # Anthropic models
    HAIKU = "claude-3-5-haiku-20241022"
    SONNET = "claude-sonnet-4-5-20250929"
    OPUS = "claude-opus-4-5-20251101"

    # DeepSeek models
    DEEPSEEK_V3 = "deepseek-chat"      # Fast, cheap classification
    DEEPSEEK_R1 = "deepseek-reasoner"  # Reasoning with thinking


def is_deepseek_tier(tier: ModelTier) -> bool:
    """Check if tier is a DeepSeek model."""
    return tier in (ModelTier.DEEPSEEK_V3, ModelTier.DEEPSEEK_R1)


def get_llm(
    tier: ModelTier,
    max_tokens: int = 4096,
    **kwargs,
) -> BaseChatModel:
    """Get LLM instance for the specified tier."""

    if is_deepseek_tier(tier):
        llm_kwargs = {
            "model": tier.value,
            "max_tokens": max_tokens,
            "max_retries": 3,
        }

        if tier == ModelTier.DEEPSEEK_R1:
            # Enable explicit thinking for R1 reasoner
            llm_kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            # R1 needs higher max_tokens for reasoning content
            llm_kwargs["max_tokens"] = max(max_tokens, 16384)
            # R1 supports tool calling but NOT tool_choice parameter
            llm_kwargs["disabled_params"] = {"tool_choice": None}

        return ChatDeepSeek(**llm_kwargs, **kwargs)

    # Claude models (existing implementation)
    return ChatAnthropic(
        model=tier.value,
        max_tokens=max_tokens,
        max_retries=3,
        **kwargs,
    )
```

### Step 2: Implement Prefix-Based Caching with Warmup

```python
# workflows/shared/llm_utils/caching.py

import asyncio
import time
from typing import Any

from langchain_deepseek import ChatDeepSeek

# DeepSeek cache tracking
_deepseek_cache_warmed: dict[int, float] = {}
DEEPSEEK_CACHE_WARMUP_DELAY = 10.0  # seconds


def _is_deepseek_model(llm: BaseChatModel) -> bool:
    """Check if LLM is a DeepSeek model."""
    return isinstance(llm, ChatDeepSeek)


async def invoke_with_cache(
    llm: BaseChatModel,
    system_prompt: str,
    user_prompt: str,
    cache_system: bool = True,
) -> Any:
    """Invoke LLM with caching support.

    For DeepSeek models, uses automatic prefix-based caching with warmup delay.
    For Anthropic models, uses explicit cache_control blocks.
    """
    if _is_deepseek_model(llm):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prefix_hash = hash(system_prompt)

        if cache_system and prefix_hash not in _deepseek_cache_warmed:
            # First call - make request, then wait for cache construction
            response = await llm.ainvoke(messages)
            _deepseek_cache_warmed[prefix_hash] = time.time()
            await asyncio.sleep(DEEPSEEK_CACHE_WARMUP_DELAY)
            return response

        # Subsequent calls - cache is warm
        return await llm.ainvoke(messages)

    # Anthropic caching (existing implementation with cache_control)
    return await _invoke_anthropic_with_cache(llm, system_prompt, user_prompt)
```

### Step 3: Batch Processing with Cache Warmup

```python
# workflows/shared/llm_utils/caching.py

async def batch_invoke_with_cache(
    llm: BaseChatModel,
    system_prompt: str,
    user_prompts: list[tuple[str, str]],  # (id, prompt) pairs
    max_concurrent: int = 10,
) -> dict[str, Any]:
    """Batch invoke with shared system prompt caching.

    First request warms the cache, subsequent requests run concurrently.
    """
    if not user_prompts:
        return {}

    results = {}
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_one(item_id: str, prompt: str) -> tuple[str, Any]:
        async with semaphore:
            response = await invoke_with_cache(
                llm, system_prompt, prompt, cache_system=True
            )
            return item_id, response

    if _is_deepseek_model(llm):
        prefix_hash = hash(system_prompt)

        # Warm cache with first request
        if prefix_hash not in _deepseek_cache_warmed:
            first_id, first_prompt = user_prompts[0]
            response = await llm.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": first_prompt},
            ])
            results[first_id] = response
            _deepseek_cache_warmed[prefix_hash] = time.time()
            await asyncio.sleep(DEEPSEEK_CACHE_WARMUP_DELAY)
            user_prompts = user_prompts[1:]  # Process remaining

    # Process remaining concurrently
    if user_prompts:
        tasks = [process_one(id, prompt) for id, prompt in user_prompts]
        batch_results = await asyncio.gather(*tasks)
        results.update(dict(batch_results))

    return results
```

### Step 4: Configure Structured Output for DeepSeek

```python
# workflows/shared/llm_utils/structured/executors/langchain.py

async def execute_langchain(
    output_schema: Type[T],
    request: StructuredRequest,
    output_config: StructuredOutputConfig,
) -> StructuredOutputResult[T]:
    """Execute structured output with model-appropriate method."""

    llm = get_llm(tier=output_config.tier, max_tokens=output_config.max_tokens)

    if is_deepseek_tier(output_config.tier):
        # DeepSeek doesn't support json_schema response_format
        # Use function calling for structured output
        structured_llm = llm.with_structured_output(
            output_schema,
            method="function_calling",
        )
    else:
        # Claude supports both methods
        structured_llm = llm.with_structured_output(
            output_schema,
            method="json_schema" if output_config.use_json_schema else "function_calling",
        )

    messages = [
        {"role": "system", "content": request.system_prompt},
        {"role": "user", "content": request.user_prompt},
    ]

    result = await structured_llm.ainvoke(messages)
    return StructuredOutputResult(success=True, result=result)
```

## Complete Example

```python
from workflows.shared.llm_utils import get_llm, ModelTier
from workflows.shared.llm_utils.caching import batch_invoke_with_cache
from workflows.shared.llm_utils.structured import get_structured_output

# 1. Classification task → DeepSeek V3 (fast, cheap)
llm_classifier = get_llm(ModelTier.DEEPSEEK_V3)

classification_results = await batch_invoke_with_cache(
    llm=llm_classifier,
    system_prompt=CLASSIFICATION_SYSTEM,  # Cached after first call
    user_prompts=[
        ("doc1", f"Classify: {doc1_content}"),
        ("doc2", f"Classify: {doc2_content}"),
        ("doc3", f"Classify: {doc3_content}"),
    ],
    max_concurrent=10,  # After 10s warmup
)

# 2. Reasoning task → DeepSeek R1 (explicit thinking)
llm_reasoner = get_llm(ModelTier.DEEPSEEK_R1)
# Note: R1 automatically gets:
#   - extra_body={"thinking": {"type": "enabled"}}
#   - max_tokens bumped to 16384
#   - disabled_params={"tool_choice": None}

summary_response = await llm_reasoner.ainvoke([
    {"role": "system", "content": SUMMARY_SYSTEM},
    {"role": "user", "content": f"Summarize: {document_content}"},
])
# Response includes <thinking>...</thinking> reasoning

# 3. Structured output with V3
metadata = await get_structured_output(
    output_schema=DocumentMetadata,
    user_prompt=f"Extract metadata:\n\n{content}",
    system_prompt=METADATA_SYSTEM,
    tier=ModelTier.DEEPSEEK_V3,
    enable_prompt_cache=True,  # 90% savings on cached prompts
)

# 4. Quality-critical task → Keep Claude
llm_synthesis = get_llm(ModelTier.OPUS)
final_synthesis = await llm_synthesis.ainvoke([
    {"role": "system", "content": SYNTHESIS_SYSTEM},
    {"role": "user", "content": synthesis_prompt},
])
```

## Consequences

### Benefits

- **70-80% cost savings**: V3 is ~10x cheaper than Haiku for classification
- **Explicit reasoning**: R1 shows thinking process for debugging
- **Prefix caching**: 90% token cost reduction after warmup
- **Structured output**: Function calling works reliably
- **Simple integration**: ChatDeepSeek handles API differences

### Trade-offs

- **10s warmup delay**: First request per system prompt has latency
- **No tool_choice**: R1 cannot force tool use, must rely on prompting
- **Quality variation**: V3 may underperform Claude on edge cases
- **API stability**: Newer provider than Anthropic

### Alternatives

- **All Claude**: Higher cost but consistent quality
- **OpenAI GPT-4o-mini**: Similar cost tier, different trade-offs
- **Self-hosted**: LLaMA/Mistral for maximum control (ops overhead)

## Related Patterns

- [Batch API Cost Optimization](./batch-api-cost-optimization.md) - Anthropic batch approach
- [Anthropic Prompt Caching](./anthropic-prompt-caching-cost-optimization.md) - Claude caching
- [Model Tier Optimization](../../solutions/llm-issues/model-tier-optimization.md) - Task-based selection

## Known Uses in Thala

- `workflows/document_processing/nodes/metadata_agent.py` - V3 for metadata extraction
- `workflows/document_processing/nodes/summary_agent.py` - R1 for document summarization
- `workflows/document_processing/nodes/chapter_detector.py` - V3 for chapter detection
- `workflows/shared/llm_utils/models.py` - Model tier configuration
- `workflows/shared/llm_utils/caching.py` - DeepSeek caching implementation

## References

- [DeepSeek API Documentation](https://platform.deepseek.com/api-docs/)
- [DeepSeek R1 Paper](https://arxiv.org/abs/2401.14196)
- [langchain-deepseek](https://python.langchain.com/docs/integrations/chat/deepseek/)
