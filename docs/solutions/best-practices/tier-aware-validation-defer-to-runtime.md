---
type: solution
title: Tier-Aware Validation - Defer to Runtime Context
category: best-practices
created: 2026-02-06
severity: p2
module: workflows.shared.llm_utils
symptoms:
  - "Valid configurations rejected by dataclass __post_init__ validation"
  - "DeepSeek R1 with cache=True blocked incorrectly"
  - "ValueError raised before tier context is available"
root_cause: "Validation at config construction lacks runtime context (model tier)"
resolution_type: code_fix
tags: [validation, dataclass, runtime-context, deepseek, anthropic, config]
---

# Tier-Aware Validation - Defer to Runtime Context

## Problem

Config validation in `__post_init__` rejected valid configurations because it lacked runtime context about which model tier would be used.

### Symptom

`InvokeConfig.__post_init__` rejected `cache=True` with `thinking_budget` for ALL models:

```python
# BAD: Too strict - doesn't know the model tier
@dataclass
class InvokeConfig:
    cache: bool = True
    thinking_budget: int | None = None

    def __post_init__(self) -> None:
        if self.cache and self.thinking_budget:
            raise ValueError(
                "Cannot use cache with extended thinking. "
                "Extended thinking responses change on every call, "
                "making caching ineffective."
            )
```

**The issue:** This validation is correct for Anthropic models (Sonnet, Opus) but WRONG for DeepSeek R1.

### Why This Is Wrong

DeepSeek R1 has different behavior than Anthropic:
- **Always has thinking enabled** (not optional - it's a reasoning model)
- **Has automatic prefix caching** that works independently of thinking output
- Cache stores the prompt, thinking is generated fresh each time
- Caching is highly effective and recommended for R1

So blocking `cache=True` with `thinking_budget` prevents valid DeepSeek R1 usage.

### Root Cause

**Validation that depends on runtime context (model tier) cannot be done at config construction time.**

At the point `InvokeConfig()` is created, we don't know:
- Which tier will be used (Anthropic vs DeepSeek)
- Whether the constraint applies to that tier
- What the actual runtime behavior will be

## Solution

Defer validation to `invoke()` where model tier is known.

### Implementation

**Step 1: Remove validation from `__post_init__`**

```python
# workflows/shared/llm_utils/config.py

@dataclass
class InvokeConfig:
    """Configuration for invoke() calls.

    Attributes:
        cache: Enable prompt caching (default: True). For Anthropic, uses
            ephemeral cache_control blocks. For DeepSeek, uses automatic
            prefix-based caching.
        thinking_budget: Token budget for extended thinking (Anthropic only).
            Recommended: 8000-16000 for complex tasks. Cannot be used with
            cache=True on Anthropic models.
        ...
    """
    cache: bool = True
    thinking_budget: int | None = None

    def __post_init__(self) -> None:
        """Validate constraint combinations.

        Note: Cache + thinking_budget validation is deferred to invoke()
        where we know the model tier. DeepSeek R1 allows this combination
        since it has automatic prefix caching independent of thinking.
        """
        pass  # Validation deferred to invoke()
```

**Step 2: Add tier-aware validation in `invoke()`**

```python
# workflows/shared/llm_utils/invoke.py

from workflows.shared.llm_utils.models import is_deepseek_tier

async def invoke(
    *,
    tier: ModelTier,
    system: str,
    user: str | list[str],
    config: InvokeConfig | None = None,
) -> str | list[str]:
    """Unified LLM invocation with caching, batching, and thinking support."""
    config = config or InvokeConfig()

    # Validate tier-specific constraints
    if config.cache and config.thinking_budget and not is_deepseek_tier(tier):
        raise ValueError(
            "Cannot use cache with extended thinking on Anthropic. "
            "Set cache=False when using thinking_budget."
        )

    # Continue with invocation...
    if is_deepseek_tier(tier):
        # DeepSeek: automatic prefix caching works with thinking
        results = await _invoke_direct(tier, system, user_prompts, config)
    else:
        # Anthropic: route based on batch_policy
        ...
```

### Why This Works

1. **Context-aware validation**: Validation happens where `tier` is known
2. **Tier-specific rules**: Different rules for Anthropic vs DeepSeek
3. **Fail early**: Still validates before making API calls
4. **Clear error messages**: Explains the constraint and how to fix it

### Tier-Specific Behavior

**Anthropic (Sonnet, Opus):**
```python
# Cannot combine cache + thinking_budget
config = InvokeConfig(cache=True, thinking_budget=8000)
await invoke(tier=ModelTier.OPUS, config=config, ...)
# ❌ Raises: "Cannot use cache with extended thinking on Anthropic..."
```

**DeepSeek R1:**
```python
# Can combine cache + thinking_budget (automatic prefix caching)
config = InvokeConfig(cache=True, thinking_budget=8000)
await invoke(tier=ModelTier.DEEPSEEK_R1, config=config, ...)
# ✅ Works: Caches prompts, generates fresh thinking
```

## Key Insight

**Validation that depends on runtime context should be deferred to where that context is available.**

This pattern applies to any config validation that depends on:
- Model tier (Anthropic vs DeepSeek)
- Model capabilities (supports tools, supports thinking, etc.)
- Runtime state (broker enabled, batch mode, etc.)
- Environment configuration (API keys, feature flags, etc.)

## Files Modified

- `workflows/shared/llm_utils/config.py` - Remove `__post_init__` validation
- `workflows/shared/llm_utils/invoke.py` - Add tier-aware validation at runtime

## Prevention

### When to Validate at Construction

Use `__post_init__` for constraints that are ALWAYS true:
```python
@dataclass
class Config:
    max_tokens: int
    min_tokens: int

    def __post_init__(self) -> None:
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens cannot exceed max_tokens")
```

### When to Defer Validation

Defer to runtime when constraints depend on context:
```python
@dataclass
class Config:
    use_tools: bool
    cache: bool

    def __post_init__(self) -> None:
        # DON'T validate here - depends on model capabilities
        pass

def invoke(tier: ModelTier, config: Config):
    # Validate where context is known
    if config.use_tools and not model_supports_tools(tier):
        raise ValueError(f"{tier} doesn't support tools")
```

## Design Checklist

When adding config validation:

- [ ] Does validation depend on runtime context? → Defer to call site
- [ ] Is the constraint universal or tier-specific? → Universal = `__post_init__`, tier-specific = runtime
- [ ] Can the config be constructed before context is known? → Yes = defer validation
- [ ] Does the error message explain HOW to fix it? → Include fix in message
- [ ] Are there tier-specific behaviors documented? → Document in config docstring

## Related Patterns

- [Model Tier Optimization](../llm-issues/model-tier-optimization.md) - Different model tiers have different capabilities
- [DeepSeek Integration Patterns](../../patterns/llm-interaction/deepseek-integration-patterns.md) - DeepSeek-specific behaviors
- [Tool Call Prevention and Extended Thinking](../llm-issues/tool-call-prevention-extended-thinking.md) - Extended thinking configuration

## References

- Commit: 4f9258c - feat(llm_utils): add unified invoke() function and InvokeConfig
- `workflows/shared/llm_utils/config.py` - InvokeConfig implementation
- `workflows/shared/llm_utils/invoke.py` - Tier-aware validation
- `workflows/shared/llm_utils/models.py` - is_deepseek_tier() helper
