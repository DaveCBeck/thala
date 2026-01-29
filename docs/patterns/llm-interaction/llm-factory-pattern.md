---
name: llm-factory-pattern
title: "LLM Factory Pattern for Centralized Configuration"
date: 2026-01-26
category: llm-interaction
applicability:
  - "LLM-powered applications with multiple modules needing consistent configuration"
  - "Multi-provider setups requiring centralized model version management"
  - "Production systems requiring consistent retry and observability"
components: [llm_call, configuration]
complexity: simple
verified_in_production: true
related_solutions: ["../solutions/llm-issues/centralized-retry-configuration.md"]
tags: [factory-pattern, model-tier, retry, langsmith, cost-tracking, anthropic, deepseek]
---

# LLM Factory Pattern for Centralized Configuration

## Intent

Centralize LLM instantiation through a factory function that automatically provides retry logic, observability metadata, and model version management across the entire codebase.

## Motivation

Direct LLM instantiation scattered across modules creates maintenance burden and inconsistent behavior.

**The Problem:**

```python
# File: workflows/output/evening_reads/nodes/write_deep_dive.py
from langchain_anthropic import ChatAnthropic
from workflows.shared.llm_utils import ModelTier

llm = ChatAnthropic(
    model=ModelTier.OPUS.value,
    max_tokens=MAX_TOKENS,
)

# File: workflows/shared/diagram_utils/selection.py
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=8000)  # Outdated!

# File: workflows/shared/image_utils.py
from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",  # Outdated!
    max_tokens=500,
)
```

Issues with scattered instantiation:
- **Model versions get outdated**: Some files had `claude-sonnet-4-20250514` while others had the current version
- **No retry configuration**: Transient 429/503 errors caused workflow failures
- **Missing LangSmith metadata**: Cost tracking incomplete, hard to attribute spend
- **Inconsistent configuration**: Each module reinventing the wheel

**The Solution:**

```python
# All files now use:
from workflows.shared.llm_utils import ModelTier, get_llm

llm = get_llm(tier=ModelTier.OPUS, max_tokens=MAX_TOKENS)
```

Factory automatically provides:
1. Current model versions via `ModelTier` enum
2. `max_retries=3` for transient error handling
3. LangSmith metadata for cost attribution
4. Extended thinking support when needed
5. Multi-provider support (Claude and DeepSeek)

## Applicability

Use this pattern when:
- Multiple modules instantiate LLMs
- Need consistent retry handling across the codebase
- Using LangSmith for cost tracking and observability
- Supporting multiple model tiers or providers
- Want to change model versions in one place

Do NOT use this pattern when:
- Single-file script with one LLM call
- Need highly specialized per-call configuration
- Testing/prototyping where simplicity matters more

## Structure

```
workflows/shared/
└── llm_utils/
    ├── __init__.py          # Exports get_llm, ModelTier
    └── models.py            # Factory function and enum

workflows/output/evening_reads/nodes/
├── write_deep_dive.py       # Uses get_llm(ModelTier.OPUS)
└── write_overview.py        # Uses get_llm(ModelTier.OPUS)

workflows/shared/
├── diagram_utils/
│   ├── generation.py        # Uses get_llm(ModelTier.SONNET)
│   └── selection.py         # Uses get_llm(ModelTier.SONNET)
└── image_utils.py           # Uses get_llm(ModelTier.SONNET)
```

## Implementation

### Step 1: Define Model Tier Enum

Centralize model versions in an enum:

```python
# workflows/shared/llm_utils/models.py

from enum import Enum


class ModelTier(Enum):
    """Model tiers for different task complexities.

    Claude tiers:
        HAIKU: Quick tasks, simple text generation
        SONNET: Standard tasks, summarization, metadata extraction (200k context)
        SONNET_1M: Same as SONNET but with 1M token context window (Tier 4+)
        OPUS: Complex tasks requiring deep analysis (supports extended thinking)

    DeepSeek tiers (10-15x cheaper than Claude, OpenAI-compatible API):
        DEEPSEEK_V3: High-volume simple tasks (classification, filtering)
        DEEPSEEK_R1: Reasoning tasks (methodology analysis, complex extraction)
    """

    # Claude tiers
    HAIKU = "claude-haiku-4-5-20251001"
    SONNET = "claude-sonnet-4-5-20250929"
    SONNET_1M = "claude-sonnet-4-5-20250929"  # Requires 1M context beta header
    OPUS = "claude-opus-4-5-20251101"

    # DeepSeek tiers
    DEEPSEEK_V3 = "deepseek-chat"       # V3.2, 128K context, $0.27/$1.10 per MTok
    DEEPSEEK_R1 = "deepseek-reasoner"   # Reasoning model, 128K context
```

### Step 2: Create Factory Function

```python
# workflows/shared/llm_utils/models.py

import os
from typing import Any, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek

# Beta header for 1M context window (Sonnet 4/4.5 only, Tier 4+)
CONTEXT_1M_BETA = "context-1m-2025-08-07"

# DeepSeek tiers for easy checking
DEEPSEEK_TIERS = {ModelTier.DEEPSEEK_V3, ModelTier.DEEPSEEK_R1}


def is_deepseek_tier(tier: ModelTier) -> bool:
    """Check if a tier is a DeepSeek model."""
    return tier in DEEPSEEK_TIERS


def get_llm(
    tier: ModelTier = ModelTier.SONNET,
    thinking_budget: Optional[int] = None,
    max_tokens: int = 4096,
) -> BaseChatModel:
    """
    Get a configured LLM instance (Claude or DeepSeek).

    Args:
        tier: Model tier selection (HAIKU, SONNET, SONNET_1M, OPUS, DEEPSEEK_V3, DEEPSEEK_R1)
        thinking_budget: Token budget for extended thinking (enables if set).
                        Recommended: 8000-16000 for complex tasks.
                        Supported on Claude models (Sonnet 4.5, Haiku 4.5, Opus 4.5).
                        For DEEPSEEK_R1, thinking is always enabled (explicit mode).
                        Ignored for DEEPSEEK_V3.
        max_tokens: Maximum output tokens (must be > thinking_budget if set)

    Returns:
        BaseChatModel instance configured for the specified tier

    Example:
        # Standard task with Sonnet
        llm = get_llm(ModelTier.SONNET)

        # Large document processing with 1M context
        llm = get_llm(ModelTier.SONNET_1M)

        # Complex analysis with Opus and extended thinking
        llm = get_llm(ModelTier.OPUS, thinking_budget=8000, max_tokens=16000)

        # Cost-effective task with DeepSeek
        llm = get_llm(ModelTier.DEEPSEEK_V3)
    """
    # DeepSeek models use native ChatDeepSeek integration
    if is_deepseek_tier(tier):
        kwargs: dict[str, Any] = {
            "model": tier.value,
            "max_tokens": max_tokens,
            "max_retries": 3,
        }

        if tier == ModelTier.DEEPSEEK_R1:
            # Enable explicit thinking for R1 reasoner model
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            # R1 needs higher max_tokens for reasoning content
            kwargs["max_tokens"] = max(max_tokens, 16384)
            # R1 supports tool calling but NOT tool_choice parameter
            kwargs["disabled_params"] = {"tool_choice": None}
        elif thinking_budget is not None:
            import logging
            logging.getLogger(__name__).warning(
                f"thinking_budget ignored for DeepSeek tier {tier.name} (only R1 supports reasoning)"
            )

        return ChatDeepSeek(**kwargs)

    # Claude models use Anthropic API
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    kwargs: dict[str, Any] = {
        "model": tier.value,
        "api_key": api_key,
        "max_tokens": max_tokens,
        "max_retries": 3,  # Handles 429, 500, 502, 503, 529 with exponential backoff
        # LangSmith metadata for automatic cost tracking
        "metadata": {
            "ls_provider": "anthropic",
            "ls_model_name": tier.value,
        },
    }

    # Enable 1M context window beta for SONNET_1M (requires Tier 4+ account)
    if tier == ModelTier.SONNET_1M:
        kwargs["betas"] = [CONTEXT_1M_BETA]

    if thinking_budget is not None:
        if thinking_budget >= max_tokens:
            raise ValueError(
                f"thinking_budget ({thinking_budget}) must be less than max_tokens ({max_tokens})"
            )
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }

    return ChatAnthropic(**kwargs)
```

### Step 3: Migrate Direct Instantiation

**Before migration:**

```python
# workflows/output/evening_reads/nodes/write_deep_dive.py
from langchain_anthropic import ChatAnthropic
from workflows.shared.llm_utils import ModelTier

MAX_TOKENS = 8000

llm = ChatAnthropic(
    model=ModelTier.OPUS.value,
    max_tokens=MAX_TOKENS,
)

response = await llm.ainvoke([
    SystemMessage(content=system_prompt),
    HumanMessage(content=user_prompt),
])
```

**After migration:**

```python
# workflows/output/evening_reads/nodes/write_deep_dive.py
from workflows.shared.llm_utils import ModelTier, get_llm

MAX_TOKENS = 8000

llm = get_llm(tier=ModelTier.OPUS, max_tokens=MAX_TOKENS)

response = await llm.ainvoke([
    SystemMessage(content=system_prompt),
    HumanMessage(content=user_prompt),
])
```

Changes:
- Remove direct `from langchain_anthropic import ChatAnthropic`
- Import `get_llm` from `llm_utils`
- Replace `ChatAnthropic(model=ModelTier.OPUS.value, ...)` with `get_llm(tier=ModelTier.OPUS, ...)`

### Step 4: Fix Outdated Model Versions

The migration also fixes hardcoded outdated versions:

```python
# BEFORE: workflows/shared/diagram_utils/selection.py
llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=8000)

# AFTER: Version managed centrally
llm = get_llm(tier=ModelTier.SONNET, max_tokens=8000)
```

When Anthropic releases a new model, update `ModelTier` enum once:

```python
# Single change propagates everywhere
class ModelTier(Enum):
    SONNET = "claude-sonnet-4-5-20250929"  # Update here only
```

## Complete Example

Migrating 5 files in commit `23a4b20`:

```diff
# workflows/output/evening_reads/nodes/write_deep_dive.py
-from langchain_anthropic import ChatAnthropic
-from workflows.shared.llm_utils import ModelTier
+from workflows.shared.llm_utils import ModelTier, get_llm

-        llm = ChatAnthropic(
-            model=ModelTier.OPUS.value,
-            max_tokens=MAX_TOKENS,
-        )
+        llm = get_llm(tier=ModelTier.OPUS, max_tokens=MAX_TOKENS)

# workflows/shared/diagram_utils/generation.py
-from langchain_anthropic import ChatAnthropic
+from ..llm_utils import ModelTier, get_llm

-        llm = ChatAnthropic(
-            model=tier.value,
-            max_tokens=4000,
-        )
+        llm = get_llm(tier=tier, max_tokens=4000)

# workflows/shared/diagram_utils/selection.py
-from langchain_anthropic import ChatAnthropic
+from workflows.shared.llm_utils import ModelTier, get_llm

-        llm = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=8000)
+        llm = get_llm(tier=ModelTier.SONNET, max_tokens=8000)

# workflows/shared/image_utils.py
-from langchain_anthropic import ChatAnthropic
+from workflows.shared.llm_utils import ModelTier, get_llm

-        llm = ChatAnthropic(
-            model="claude-sonnet-4-20250514",
-            max_tokens=500,
-        )
+        llm = get_llm(tier=ModelTier.SONNET, max_tokens=500)
```

Result: 11 insertions, 34 deletions (net reduction of 23 lines).

## Consequences

### Benefits

- **Single point of version control**: Update model versions in one place
- **Automatic retry handling**: All LLM calls get `max_retries=3` for transient errors (429, 500, 502, 503, 529)
- **LangSmith cost attribution**: `ls_provider` and `ls_model_name` metadata enables cost tracking per model
- **Consistent configuration**: No more forgotten settings in individual files
- **Multi-provider support**: Same interface for Claude and DeepSeek
- **Extended thinking ready**: `thinking_budget` parameter when needed
- **1M context support**: `SONNET_1M` tier enables large document processing

### Trade-offs

- **Additional abstraction layer**: One more function to understand
- **Less flexibility per-call**: Some advanced ChatAnthropic options not exposed
- **Import path change**: Requires updating imports across codebase

### Alternatives

- **Direct instantiation with config module**: Import settings from config but instantiate directly
- **Dependency injection**: Pass LLM instances from main() to all functions
- **LangChain LCEL chains**: Use `RunnableConfig` for per-chain configuration

## Related Patterns

- [Anthropic Claude Extended Thinking](./anthropic-claude-extended-thinking.md) - Extended thinking implementation details
- [DeepSeek Integration Patterns](./deepseek-integration-patterns.md) - Multi-provider model selection
- [LLM Caching Warmup Pattern](./llm-caching-warmup-pattern.md) - Caching on top of factory
- [Batch API Cost Optimization](./batch-api-cost-optimization.md) - Batch processing integration

## Known Uses in Thala

- `workflows/shared/llm_utils/models.py` - Factory function implementation
- `workflows/output/evening_reads/nodes/write_deep_dive.py` - Opus for complex writing
- `workflows/output/evening_reads/nodes/write_overview.py` - Opus for overview generation
- `workflows/shared/diagram_utils/generation.py` - Sonnet for diagram generation
- `workflows/shared/diagram_utils/selection.py` - Sonnet for candidate selection
- `workflows/shared/image_utils.py` - Sonnet for image prompt generation

## References

- [LangChain Anthropic Integration](https://python.langchain.com/docs/integrations/chat/anthropic/)
- [Anthropic Rate Limits](https://docs.anthropic.com/en/docs/build-with-claude/rate-limits)
- [LangSmith Cost Tracking](https://docs.smith.langchain.com/how_to_guides/monitoring/track_costs)
- [Anthropic Extended Thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
