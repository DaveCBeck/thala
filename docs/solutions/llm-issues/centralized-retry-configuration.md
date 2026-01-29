---
module: workflows/shared/llm_utils
date: 2026-01-28
problem_type: configuration_issue
component: models
symptoms:
  - "429 Too Many Requests errors causing workflow failures"
  - "500/502/503 transient API errors not retried"
  - "Workflows failing on temporary Anthropic service issues"
root_cause: missing_retry_config
resolution_type: code_fix
severity: medium
tags: [retry, max_retries, rate-limiting, transient-errors, exponential-backoff, llm]
---

# Centralized Retry Configuration

## Problem

LLM calls failed on transient API errors (429, 500, 502, 503, 529) instead of automatically retrying with exponential backoff.

### Environment

- **Module**: `workflows/shared/llm_utils`
- **Python**: 3.12
- **LangChain Anthropic**: 0.3.x
- **Affected files**: `workflows/shared/llm_utils/models.py`

### Symptoms

```
# Workflow failure on rate limit
anthropic.RateLimitError: Error code: 429 - {'type': 'error', 'error': {
  'type': 'rate_limit_error', 'message': 'Too many requests'}}

# Workflow failure on transient server error
anthropic.APIStatusError: Error code: 503 - Service temporarily unavailable
```

Workflows would fail completely on temporary issues that should have been retried.

## Root Cause

The `get_llm()` function did not configure `max_retries`, leaving the default (0 or 1 depending on client version), which meant transient errors caused immediate failures.

```python
# BEFORE: No retry configuration
def get_llm(tier: ModelTier, max_tokens: int = 4096, **kwargs) -> ChatAnthropic:
    return ChatAnthropic(
        model=tier.value,
        api_key=api_key,
        max_tokens=max_tokens,
        # Missing: max_retries
        **kwargs,
    )
```

## Solution

Add `max_retries=3` to the centralized `get_llm()` function.

### Code Changes

```python
# workflows/shared/llm_utils/models.py

def get_llm(
    tier: ModelTier,
    max_tokens: int = 4096,
    **kwargs,
) -> ChatAnthropic:
    """Get configured LLM instance with automatic retry handling."""
    return ChatAnthropic(
        model=tier.value,
        api_key=api_key,
        max_tokens=max_tokens,
        max_retries=3,  # Handles 429, 500, 502, 503, 529 with exponential backoff
        metadata={
            "ls_provider": "anthropic",
            "ls_model_name": tier.value,
        },
        **kwargs,
    )
```

### What max_retries Handles

The Anthropic SDK automatically retries these errors with exponential backoff:

| Error Code | Meaning | Retry Behavior |
|------------|---------|----------------|
| 429 | Rate limit exceeded | Wait and retry |
| 500 | Internal server error | Retry immediately |
| 502 | Bad gateway | Retry immediately |
| 503 | Service unavailable | Retry immediately |
| 529 | Overloaded | Wait and retry |

### Files Modified

- `workflows/shared/llm_utils/models.py`: Add `max_retries=3` to ChatAnthropic instantiation

## Prevention

### How to Avoid This

1. **Always configure max_retries in centralized LLM factory**
   - Don't leave retry handling to callers
   - Use a reasonable default (2-5 retries)
   - All LLM calls through `get_llm()` automatically get retry handling

2. **Consider configurable retries for different use cases**
   ```python
   def get_llm(
       tier: ModelTier,
       max_retries: int = 3,  # Configurable with sensible default
       **kwargs,
   ) -> ChatAnthropic:
       ...
   ```

3. **For batch operations, consider higher retry counts**
   - Batch API has different failure modes
   - May want `max_retries=5` for long-running batches

## Related

- [LangChain Anthropic Retry Documentation](https://python.langchain.com/docs/integrations/chat/anthropic/#retry-logic)
- [Anthropic Rate Limits](https://docs.anthropic.com/en/docs/build-with-claude/rate-limits)
