---
name: central-llm-broker-routing
title: Central LLM Broker Routing Pattern
date: 2026-02-04
category: llm-interaction
applicability:
  - "Systems making dispersed LLM calls across multiple workflow nodes (10+)"
  - "Need for user-configurable speed/cost optimization tradeoffs"
  - "Requirement for centralized batch threshold decisions"
  - "Cross-workflow coordination of API call routing"
components: [llm_broker, batch_processor, workflow_routing, persistence, metrics]
complexity: high
verified_in_production: false
tags: [anthropic, batch-api, cost-reduction, llm, routing, centralized-architecture, configurable-modes]
shared: true
gist_url: https://gist.github.com/DaveCBeck/797eff970c9e2f913e9228f9e39a949c
article_path: .context/libs/thala-dev/content/orchestration/2026-02-04-central-llm-broker-routing.md
---

# Central LLM Broker Routing Pattern

## Intent

Centralize all LLM API routing decisions using a broker that automatically determines batch vs sync execution based on user-configured mode and call-site batch policy, enabling 50% cost savings while maintaining latency flexibility.

## Problem

Large AI systems make LLM calls scattered across many workflow nodes with inconsistent batching strategies:

- **Hardcoded thresholds**: Each call site decides "if >= 5 requests, use batch" independently
- **No user control**: Users can't configure speed/cost preferences globally
- **Duplicate logic**: Batch collection code repeated across workflows
- **Complex coordination**: Manual semaphores and double-batching patterns
- **No observability**: Difficult to monitor or change batching behavior

## Solution

Implement a **Central LLM Broker** with a 2D routing matrix:

### Routing Matrix

| Policy ↓ / Mode → | FAST | BALANCED | ECONOMICAL |
|-------------------|------|----------|------------|
| `FORCE_BATCH` | Batch | Batch | Batch |
| `PREFER_BALANCE` | Sync | Batch | Batch |
| `PREFER_SPEED` | Sync | Sync | Batch |
| `REQUIRE_SYNC` | Sync | Sync | Sync |

**User Mode** (global preference): Controls speed/cost tradeoff for entire workflow
**Call-Site Policy** (local intent): Declares what the code wants, not how to achieve it

## Structure

```
┌─────────────────────────────────────────────────────────────┐
│                      Workflow Layer                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ Node A  │  │ Node B  │  │ Node C  │  │ Node D  │        │
│  │ PREFER_ │  │ PREFER_ │  │ FORCE_  │  │ REQUIRE │        │
│  │ BALANCE │  │ SPEED   │  │ BATCH   │  │ _SYNC   │        │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
│       │            │            │            │              │
│       └────────────┴─────┬──────┴────────────┘              │
│                          │                                   │
│                          ▼                                   │
│              ┌───────────────────────┐                      │
│              │   LLM Broker          │ ◄─── User Mode       │
│              │   _should_batch()     │      (Fast/Balanced/ │
│              │                       │       Economical)    │
│              └───────────┬───────────┘                      │
│                          │                                   │
│            ┌─────────────┴─────────────┐                    │
│            │                           │                    │
│            ▼                           ▼                    │
│     ┌─────────────┐            ┌─────────────┐             │
│     │  Sync API   │            │  Batch API  │             │
│     │  (Instant)  │            │  (50% off)  │             │
│     └─────────────┘            └─────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

## Implementation

### 1. Define Policies and Modes

```python
# core/llm_broker/schemas.py

class BatchPolicy(Enum):
    """Call-site batch policy declaration."""
    FORCE_BATCH = "force_batch"      # Always batch (bulk operations)
    PREFER_BALANCE = "prefer_balance" # Batch in Balanced/Economical
    PREFER_SPEED = "prefer_speed"    # Batch only in Economical
    REQUIRE_SYNC = "sync"            # Never batch (interactive)

class UserMode(Enum):
    """User-configurable processing mode."""
    FAST = "fast"           # No batching, lowest latency
    BALANCED = "balanced"   # Default, reasonable tradeoff
    ECONOMICAL = "economical"  # Aggressive batching, 50% savings
```

### 2. Implement Routing Decision

```python
# core/llm_broker/broker.py

def _should_batch(
    self,
    policy: BatchPolicy,
    model: ModelTier,
    thinking_budget: int | None,
) -> bool:
    """2D routing matrix: policy + mode → sync or batch."""

    # Hard constraints (never batch)
    if is_deepseek_tier(model):  # DeepSeek has no batch API
        return False
    if thinking_budget:  # Extended thinking incompatible
        return False

    # Policy evaluation
    if policy == BatchPolicy.REQUIRE_SYNC:
        return False
    if policy == BatchPolicy.FORCE_BATCH:
        return True

    # Mode evaluation
    if self._mode == UserMode.FAST:
        return False
    if policy == BatchPolicy.PREFER_BALANCE:
        return self._mode in (UserMode.BALANCED, UserMode.ECONOMICAL)
    if policy == BatchPolicy.PREFER_SPEED:
        return self._mode == UserMode.ECONOMICAL

    return False
```

### 3. Future-Based Request API

```python
# core/llm_broker/broker.py

async def request(
    self,
    prompt: str,
    policy: BatchPolicy = BatchPolicy.PREFER_SPEED,
    model: ModelTier = ModelTier.SONNET,
    **kwargs,
) -> asyncio.Future[LLMResponse]:
    """Submit request, return future for async resolution."""

    request = LLMRequest.create(prompt=prompt, model=model, policy=policy, **kwargs)

    # Create future for caller to await
    future: asyncio.Future[LLMResponse] = asyncio.get_running_loop().create_future()
    self._pending_futures[request.request_id] = future

    if self._should_batch(policy, model, kwargs.get("thinking_budget")):
        await self._queue_for_batch(request)
    else:
        self._spawn_sync_task(request)

    return future
```

### 4. Batch Group Context Manager

```python
# core/llm_broker/broker.py

@asynccontextmanager
async def batch_group(self, mode: UserMode | None = None) -> AsyncIterator[BatchGroup]:
    """Group requests for batch submission on context exit."""
    group = BatchGroup(broker=self, mode=mode or self._mode)
    self._current_group = group

    try:
        yield group
    finally:
        self._current_group = None
        if group.request_ids:
            await self._flush_batch_group(group)
```

### 5. Integration Point

```python
# workflows/shared/llm_utils/structured/interface.py

async def get_structured_output(
    output_schema: type[T],
    user_prompt: str,
    batch_policy: BatchPolicy | None = None,  # Routes through broker
    **kwargs,
) -> T:
    """Unified structured output with optional broker routing."""

    if is_broker_enabled() and batch_policy is not None:
        broker = get_broker()
        future = await broker.request(
            prompt=user_prompt,
            policy=batch_policy,
            **kwargs,
        )
        response = await future
        return parse_response(response, output_schema)

    # Fallback to existing path
    return await _execute_legacy(output_schema, user_prompt, **kwargs)
```

## Participants

- **LLMBroker**: Central routing coordinator with lifecycle management
- **BatchPolicy**: Call-site intent declaration (enum)
- **UserMode**: Global user preference (enum)
- **LLMRequest/LLMResponse**: Request/response data models
- **BrokerPersistence**: Cross-process queue coordination
- **BrokerMetrics**: Observability and monitoring

## Consequences

### Benefits

- **50% cost reduction** on batched requests
- **User control**: Single mode setting affects entire workflow
- **Intent declaration**: Call sites declare what, not how
- **Centralized monitoring**: Single point for metrics and tracing
- **Graceful degradation**: Sync fallback on queue overflow
- **Clean call sites**: Just `await broker.request(...)`

### Trade-offs

- **System complexity**: Broker lifecycle, persistence, monitoring
- **Latency variance**: Batched requests take minutes/hours vs instant
- **Migration effort**: Call sites must add batch_policy parameter
- **Feature flag**: Safe rollout requires opt-in

## Known Uses

- `workflows/document_processing/nodes/` - Chapter detection, summarization
- `workflows/research/academic_lit_review/` - Relevance scoring, extraction
- `workflows/shared/llm_utils/structured/` - get_structured_output()
- `workflows/shared/language/query_translator.py` - Multilingual translation

## Related Patterns

- [Batch API Cost Optimization](./batch-api-cost-optimization.md) - Foundation for batch processing
- [Anthropic Prompt Caching](./anthropic-prompt-caching-cost-optimization.md) - Complementary cost optimization
- [LLM Factory Pattern](./llm-factory-pattern.md) - Centralized LLM instantiation
- [LangSmith Batch Tracing](../observability/langsmith-batch-tracing.md) - Observability for batches

## See Also

- [Central LLM Batch Broker Solution](../../solutions/llm-issues/central-llm-batch-broker.md) - Implementation details
- [Structured Output Guide](../../guides/structured_response.md) - Usage documentation
