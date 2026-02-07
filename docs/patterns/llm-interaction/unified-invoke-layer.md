---
name: unified-invoke-layer
title: Unified LLM Invoke Layer
date: 2026-02-06
category: llm-interaction
applicability:
  - "Codebases with multiple LLM invocation patterns (direct, batched, cached)"
  - "Systems requiring automatic routing between sync and batch APIs"
  - "Multi-provider LLM architectures (Anthropic + DeepSeek)"
  - "Applications needing consistent caching and rate limiting"
components: [invoke, routing, caching, batching, rate_limiting]
complexity: moderate
verified_in_production: true
tags: [llm, invoke, routing, batching, caching, anthropic, deepseek, rate-limiting]
related_patterns:
  - central-llm-broker-routing
  - anthropic-prompt-caching
  - llm-factory-pattern
  - deepseek-integration-patterns
---

# Unified LLM Invoke Layer

## Intent

Consolidate all LLM invocation patterns into a single coherent API that handles routing, caching, batching, and provider differences automatically, eliminating the need for workflow code to manage these concerns directly.

## Motivation

Modern LLM-powered applications need to balance multiple concerns across different providers:

**The Problem:**

Workflows that directly call `llm.ainvoke()` or `broker.request()` end up with scattered logic:

```python
# File A: Direct invocation with manual caching
from workflows.shared.llm_utils import get_llm, create_cached_messages

llm = get_llm(ModelTier.SONNET)
messages = create_cached_messages(system, user)
response = await llm.ainvoke(messages)

# File B: Broker routing with batch policy
from core.llm_broker import get_broker, BatchPolicy

broker = get_broker()
async with broker.batch_group():
    future = await broker.request(prompt=user, policy=BatchPolicy.PREFER_BALANCE)
response = await future

# File C: Batch input with semaphore
llm = get_llm(ModelTier.HAIKU)
semaphore = asyncio.Semaphore(10)

async def process_one(user_prompt):
    async with semaphore:
        return await llm.ainvoke([...])

responses = await asyncio.gather(*[process_one(p) for p in prompts])
```

**Issues:**
- **Inconsistent patterns**: Each callsite decides how to invoke
- **Scattered routing logic**: DeepSeek checks, batch policy evaluation duplicated
- **Manual rate limiting**: Semaphore patterns repeated everywhere
- **Provider coupling**: Code aware of Anthropic vs DeepSeek differences
- **Complex batch coordination**: Manual batch groups and future management

**The Solution:**

```python
from workflows.shared.llm_utils import invoke, InvokeConfig
from core.llm_broker import BatchPolicy

# Simple call with automatic caching
response = await invoke(
    tier=ModelTier.SONNET,
    system="You are helpful.",
    user="Hello",
)

# Batch processing via broker
response = await invoke(
    tier=ModelTier.HAIKU,
    system="Score this paper.",
    user="Paper content...",
    config=InvokeConfig(batch_policy=BatchPolicy.PREFER_BALANCE),
)

# Batch input with automatic rate limiting
responses = await invoke(
    tier=ModelTier.HAIKU,
    system="Summarize this.",
    user=["Doc 1...", "Doc 2...", "Doc 3..."],
)
```

Single function provides:
1. **Automatic routing**: DeepSeek → direct, Anthropic with batch_policy → broker, otherwise → direct with caching
2. **Transparent caching**: Anthropic prompt caching applied automatically
3. **Built-in rate limiting**: Semaphore for concurrent requests
4. **Provider abstraction**: Works identically for Anthropic and DeepSeek
5. **Batch support**: Both broker batching and list input batching

## Applicability

Use this pattern when:
- Making LLM calls across multiple workflow nodes
- Need consistent caching strategy without manual message building
- Want automatic broker routing based on batch policy
- Processing batch inputs with rate limiting
- Supporting multiple LLM providers (Anthropic + DeepSeek)
- Require tier-aware validation (e.g., cache + thinking constraints)

Do NOT use this pattern when:
- Need specialized LangChain chains or advanced features
- Streaming responses required (not yet supported)
- Using non-standard LLM configurations
- Prototyping where explicitness matters more than convenience

## Structure

```
workflows/shared/llm_utils/
├── invoke.py               # Core invoke() function and routing
│   ├── invoke()           # Main entry point
│   ├── InvokeConfig       # Configuration dataclass
│   ├── _invoke_direct()   # Direct invocation with caching
│   ├── _invoke_via_broker() # Broker routing
│   ├── InvokeBatch        # Batch builder
│   └── invoke_batch()     # Context manager for dynamic batching
├── config.py              # InvokeConfig dataclass
├── models.py              # ModelTier enum, get_llm()
├── caching.py             # create_cached_messages()
└── __init__.py            # Public exports

core/llm_broker/
├── broker.py              # LLMBroker with batch_group()
└── schemas.py             # BatchPolicy enum

Usage sites:
workflows/research/
workflows/output/
workflows/document_processing/
```

## Participants

### invoke()
Main entry point that routes requests based on tier and configuration:
- Validates tier-specific constraints (cache + thinking)
- Normalizes single/batch input
- Routes to direct or broker path
- Returns single response or list based on input

### InvokeConfig
Configuration dataclass controlling invocation behavior:
- `cache`: Enable prompt caching (default: True)
- `cache_ttl`: Cache time-to-live ("5m" or "1h")
- `batch_policy`: When set, routes through broker
- `thinking_budget`: Token budget for extended thinking
- `tools`/`tool_choice`: Tool use configuration
- `max_tokens`: Maximum output tokens
- `max_concurrent`: Rate limit for direct invocation

### _invoke_direct()
Handles direct LLM invocation without broker:
- Applies prompt caching for Anthropic models
- Uses asyncio.gather() with semaphore for batch input
- Preserves response order
- Supports both Anthropic and DeepSeek

### _invoke_via_broker()
Routes requests through central broker:
- Wraps requests in broker.batch_group() context
- Submits all prompts as batch
- Converts broker responses to AIMessage
- Populates usage_metadata for LangSmith

### InvokeBatch / invoke_batch()
Dynamic batch accumulation:
- Context manager for programmatic batch building
- Wraps broker.batch_group() with cleaner interface
- Collects futures and converts responses
- Used when batch size unknown upfront

## Implementation

### Step 1: Define Configuration

```python
# workflows/shared/llm_utils/config.py

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from core.llm_broker import BatchPolicy

@dataclass
class InvokeConfig:
    """Configuration for invoke() calls.

    Attributes:
        cache: Enable prompt caching (default: True)
        cache_ttl: Cache time-to-live ("5m" default, "1h" for long workflows)
        batch_policy: When set, routes requests through broker for cost optimization
        thinking_budget: Token budget for extended thinking (Anthropic only)
        tools: Tool definitions for tool use
        tool_choice: Tool choice configuration
        metadata: Additional metadata for tracking
        max_tokens: Maximum output tokens (default: 4096)
        max_concurrent: Concurrency limit for direct invocation (default: 10)
    """
    cache: bool = True
    cache_ttl: Literal["5m", "1h"] = "5m"
    batch_policy: "BatchPolicy | None" = None
    thinking_budget: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    max_tokens: int = 4096
    max_concurrent: int = 10
```

### Step 2: Core Invoke Function

```python
# workflows/shared/llm_utils/invoke.py

async def invoke(
    *,
    tier: ModelTier,
    system: str,
    user: str | list[str],
    config: InvokeConfig | None = None,
) -> AIMessage | list[AIMessage]:
    """Unified LLM invocation with automatic routing.

    Routes requests through the optimal path:
    - DeepSeek models: Direct invocation (broker doesn't support)
    - Anthropic with batch_policy: Routes through broker for cost optimization
    - Otherwise: Direct invocation with prompt caching

    Args:
        tier: Model tier (HAIKU, SONNET, OPUS, DEEPSEEK_V3, DEEPSEEK_R1)
        system: System prompt
        user: User prompt (single string) or list of user prompts (batch)
        config: Optional configuration for caching, batching, thinking, etc.

    Returns:
        AIMessage for single user prompt, list[AIMessage] for batch input

    Raises:
        RuntimeError: If broker request fails
        ValueError: If config has invalid constraint combinations
    """
    config = config or InvokeConfig()

    # Validate tier-specific constraints
    if config.cache and config.thinking_budget and not is_deepseek_tier(tier):
        raise ValueError(
            "Cannot use cache with extended thinking on Anthropic. "
            "Set cache=False when using thinking_budget."
        )

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
```

### Step 3: Direct Invocation with Caching

```python
# workflows/shared/llm_utils/invoke.py

async def _invoke_direct(
    tier: ModelTier,
    system: str,
    user_prompts: list[str],
    config: InvokeConfig,
) -> list[AIMessage]:
    """Invoke LLM directly without broker.

    Handles both Anthropic (with caching) and DeepSeek models.
    Uses asyncio.gather() for concurrent processing with rate limiting.
    """
    llm = get_llm(
        tier=tier,
        thinking_budget=config.thinking_budget,
        max_tokens=config.max_tokens,
    )

    # Rate limiting semaphore
    semaphore = asyncio.Semaphore(config.max_concurrent)

    async def invoke_one(user_prompt: str) -> AIMessage:
        async with semaphore:
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
            return await llm.ainvoke(messages)

    return list(await asyncio.gather(*[invoke_one(p) for p in user_prompts]))
```

### Step 4: Broker Invocation

```python
# workflows/shared/llm_utils/invoke.py

async def _invoke_via_broker(
    tier: ModelTier,
    system: str,
    user_prompts: list[str],
    config: InvokeConfig,
) -> list[AIMessage]:
    """Invoke LLM through the central broker.

    Routes requests through broker for cost optimization via batching.
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


def _broker_response_to_message(response: "LLMResponse") -> AIMessage:
    """Convert broker LLMResponse to proper AIMessage.

    This ensures LangSmith can track token usage and costs correctly
    by populating both response_metadata and usage_metadata.
    """
    additional_kwargs: dict[str, Any] = {}
    if response.thinking:
        additional_kwargs["thinking"] = response.thinking

    # Build standardized usage_metadata for LangSmith
    usage_metadata = None
    if response.usage:
        usage_metadata = {
            "input_tokens": response.usage.get("input_tokens", 0),
            "output_tokens": response.usage.get("output_tokens", 0),
            "total_tokens": (
                response.usage.get("input_tokens", 0)
                + response.usage.get("output_tokens", 0)
            ),
        }
        # Include cache details if present (Anthropic prompt caching)
        if "cache_creation_input_tokens" in response.usage:
            usage_metadata["input_token_details"] = {
                "cache_creation": response.usage.get("cache_creation_input_tokens", 0),
                "cache_read": response.usage.get("cache_read_input_tokens", 0),
            }

    return AIMessage(
        content=response.content,
        response_metadata={
            "usage": response.usage,
            "model": response.model,
            "stop_reason": response.stop_reason,
        },
        usage_metadata=usage_metadata,
        additional_kwargs=additional_kwargs,
    )
```

### Step 5: Dynamic Batch Building

```python
# workflows/shared/llm_utils/invoke.py

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
```

## Usage Examples

### Example 1: Simple Invocation with Automatic Caching

```python
# workflows/research/web_research/nodes/supervisor/core.py

from workflows.shared.llm_utils import invoke, ModelTier

async def supervisor(state: State) -> dict:
    response = await invoke(
        tier=ModelTier.OPUS,
        system=SUPERVISOR_SYSTEM,
        user=format_user_prompt(state),
    )

    # Automatic prompt caching applied (Anthropic)
    # Response includes usage_metadata for LangSmith
    return process_tools(response)
```

### Example 2: Broker Routing with Batch Policy

```python
# workflows/research/academic_lit_review/utils/relevance_scoring/scorer.py

from workflows.shared.llm_utils import invoke, InvokeConfig, ModelTier
from core.llm_broker import BatchPolicy

async def score_papers(papers: list[Paper]) -> list[RelevanceScore]:
    responses = await invoke(
        tier=ModelTier.HAIKU,
        system=SCORING_SYSTEM,
        user=[format_paper(p) for p in papers],
        config=InvokeConfig(
            batch_policy=BatchPolicy.PREFER_BALANCE,
            max_tokens=1000,
        ),
    )

    # Automatically routes through broker if enabled
    # Falls back to direct invocation if broker disabled
    return [parse_score(r) for r in responses]
```

### Example 3: Batch Input with Rate Limiting

```python
# workflows/document_processing/nodes/chapter_detector.py

from workflows.shared.llm_utils import invoke, InvokeConfig, ModelTier

async def detect_chapters(chunks: list[str]) -> list[ChapterInfo]:
    # Process 100+ chunks with automatic rate limiting
    responses = await invoke(
        tier=ModelTier.SONNET,
        system=CHAPTER_DETECTION_SYSTEM,
        user=chunks,
        config=InvokeConfig(max_concurrent=20),
    )

    # Semaphore automatically limits to 20 concurrent requests
    # Order preserved in responses
    return [parse_chapter(r) for r in responses]
```

### Example 4: Extended Thinking (Cache Disabled)

```python
# workflows/research/methodology_extraction/nodes/extract.py

from workflows.shared.llm_utils import invoke, InvokeConfig, ModelTier

async def extract_methodology(paper: Paper) -> Methodology:
    response = await invoke(
        tier=ModelTier.OPUS,
        system=METHODOLOGY_SYSTEM,
        user=paper.full_text,
        config=InvokeConfig(
            thinking_budget=8000,
            cache=False,  # Required when using thinking_budget
            max_tokens=16000,
        ),
    )

    # Thinking content available in response.additional_kwargs["thinking"]
    return parse_methodology(response)
```

### Example 5: Dynamic Batch Building

```python
# workflows/research/academic_lit_review/nodes/scoring.py

from workflows.shared.llm_utils import invoke_batch, ModelTier

async def score_papers_dynamic(papers: list[Paper]) -> list[Score]:
    async with invoke_batch() as batch:
        for paper in papers:
            if paper.has_abstract:
                batch.add(
                    tier=ModelTier.HAIKU,
                    system=SCORING_SYSTEM,
                    user=format_paper(paper),
                )

    # All requests submitted as batch group
    results = await batch.results()
    return [parse_score(r) for r in results]
```

### Example 6: Multi-Provider Support (DeepSeek)

```python
# workflows/research/relevance_filter.py

from workflows.shared.llm_utils import invoke, InvokeConfig, ModelTier

async def filter_papers(papers: list[Paper]) -> list[Paper]:
    # DeepSeek V3 for cost-effective classification
    responses = await invoke(
        tier=ModelTier.DEEPSEEK_V3,
        system=RELEVANCE_FILTER_SYSTEM,
        user=[format_paper(p) for p in papers],
        config=InvokeConfig(max_concurrent=50),
    )

    # Automatically routes to direct invocation (broker doesn't support DeepSeek)
    # DeepSeek uses automatic prefix caching, no manual configuration needed
    return [p for p, r in zip(papers, responses) if is_relevant(r)]
```

## Routing Decision Matrix

| Tier | batch_policy | broker_enabled | Route |
|------|-------------|----------------|-------|
| DEEPSEEK_V3 | Any | Any | Direct (DeepSeek unsupported by broker) |
| DEEPSEEK_R1 | Any | Any | Direct (DeepSeek unsupported by broker) |
| HAIKU | None | Any | Direct with caching |
| HAIKU | PREFER_BALANCE | True | Broker (user mode determines sync/batch) |
| HAIKU | PREFER_BALANCE | False | Direct with caching (fallback) |
| SONNET | FORCE_BATCH | True | Broker (always batched) |
| OPUS | REQUIRE_SYNC | True | Direct (sync required) |

**Key routing rules:**
1. DeepSeek tiers always route direct (no broker support)
2. `batch_policy=None` always routes direct with caching
3. `batch_policy` set + broker enabled → broker path
4. `batch_policy` set + broker disabled → direct fallback
5. Extended thinking (`thinking_budget`) forces direct path

## Consequences

### Benefits

- **Single API**: One function for all invocation patterns (direct, batched, cached)
- **Automatic routing**: Provider and policy-aware routing without manual logic
- **Transparent caching**: Anthropic prompt caching applied automatically for Anthropic tiers
- **Rate limiting**: Built-in semaphore for concurrent direct invocations
- **Provider abstraction**: Works identically for Anthropic and DeepSeek
- **Batch flexibility**: Both fixed batch (list input) and dynamic batch (invoke_batch) support
- **LangSmith integration**: Proper usage_metadata for cost tracking
- **Graceful degradation**: Automatic fallback to direct when broker disabled
- **Type safety**: Single/batch return type matches input type

### Trade-offs

- **Abstraction layer**: Hides some direct control over LLM invocation
- **Limited streaming**: No streaming support yet (direct ainvoke only)
- **Config complexity**: InvokeConfig has many options, can be overwhelming
- **Debugging difficulty**: Routing logic adds layer between code and API
- **Migration effort**: Existing code needs to adopt new invoke() pattern

### Performance Characteristics

**Direct path (cache=True):**
- Latency: ~1-3 seconds per request (Anthropic/DeepSeek sync API)
- Throughput: Limited by `max_concurrent` semaphore (default: 10)
- Cost: Full price with 90% cache savings after first call

**Broker path (batch_policy set):**
- Latency: 1-60+ minutes depending on user mode and batch size
- Throughput: Unlimited (broker queues and batches)
- Cost: 50% savings via Batch API

**Batch input (list[str]):**
- Latency: Max of all concurrent requests (parallelized)
- Throughput: `max_concurrent` simultaneous requests
- Cost: Depends on route (direct vs broker)

## Related Patterns

- [Central LLM Broker Routing](./central-llm-broker-routing.md) - Broker architecture and routing matrix
- [Anthropic Prompt Caching](./anthropic-prompt-caching.md) - Caching strategy and cost optimization
- [LLM Factory Pattern](./llm-factory-pattern.md) - Model tier management and instantiation
- [DeepSeek Integration Patterns](./deepseek-integration-patterns.md) - Multi-provider support
- [LangSmith Batch Tracing](../observability/langsmith-batch-tracing.md) - Observability for batched requests

## Known Uses in Thala

- `workflows/shared/llm_utils/invoke.py` - Core implementation
- `workflows/research/academic_lit_review/utils/relevance_scoring/scorer.py` - Batch scoring
- `workflows/document_processing/nodes/chapter_detector.py` - Batch chunk processing
- `workflows/research/web_research/nodes/supervisor/core.py` - Single invocation with caching
- `tests/unit/workflows/llm_utils/test_invoke.py` - Test coverage
- `workflows/shared/llm_utils/__init__.py` - Public exports

## Migration Guide

### From Direct LLM Invocation

**Before:**
```python
from workflows.shared.llm_utils import get_llm, create_cached_messages

llm = get_llm(ModelTier.SONNET)
messages = create_cached_messages(system, user)
response = await llm.ainvoke(messages)
```

**After:**
```python
from workflows.shared.llm_utils import invoke, ModelTier

response = await invoke(
    tier=ModelTier.SONNET,
    system=system,
    user=user,
)
```

### From Broker Request

**Before:**
```python
from core.llm_broker import get_broker, BatchPolicy

broker = get_broker()
async with broker.batch_group():
    future = await broker.request(
        prompt=user,
        policy=BatchPolicy.PREFER_BALANCE,
    )
response = await future
```

**After:**
```python
from workflows.shared.llm_utils import invoke, InvokeConfig, ModelTier
from core.llm_broker import BatchPolicy

response = await invoke(
    tier=ModelTier.SONNET,
    system=system,
    user=user,
    config=InvokeConfig(batch_policy=BatchPolicy.PREFER_BALANCE),
)
```

### From Manual Batch Processing

**Before:**
```python
llm = get_llm(ModelTier.HAIKU)
semaphore = asyncio.Semaphore(10)

async def process_one(user_prompt):
    async with semaphore:
        return await llm.ainvoke([...])

responses = await asyncio.gather(*[process_one(p) for p in prompts])
```

**After:**
```python
from workflows.shared.llm_utils import invoke, ModelTier

responses = await invoke(
    tier=ModelTier.HAIKU,
    system=system,
    user=prompts,  # Pass list directly
)
```

## References

- [Anthropic Messages API](https://docs.anthropic.com/en/api/messages)
- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [Anthropic Batch API](https://docs.anthropic.com/en/api/messages-batch)
- [DeepSeek API Documentation](https://api-docs.deepseek.com/)
- [LangChain ChatAnthropic](https://python.langchain.com/docs/integrations/chat/anthropic/)
- [LangSmith Usage Metadata](https://docs.smith.langchain.com/how_to_guides/monitoring/track_costs)
