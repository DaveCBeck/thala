---
module: core/llm_broker
date: 2026-02-04
problem_type: llm_api_management
component: llm_broker
symptoms:
  - "Hardcoded >= 5 batch thresholds scattered across 6+ files"
  - "No user control over cost vs speed tradeoffs"
  - "use_batch_api parameters passed through every function"
  - "Inconsistent batching behavior across workflows"
  - "Manual batch collection logic duplicated in lit review pipeline"
  - "MAX_LLM_CONCURRENT semaphore causing double-batching patterns"
root_cause: decentralized_batch_decision_logic
resolution_type: architectural_refactor
severity: high
verified_fix: true
tags: [batch-api, cost-optimization, llm-routing, async-patterns, centralized-architecture]
---

# Central LLM Batch Broker

## Problem

LLM calls were scattered across the codebase with inconsistent batching strategies:

```python
# Pattern 1: Hardcoded threshold scattered across 6+ files
if len(requests) >= 5:  # Magic number, no user control
    await batch_process(requests)
else:
    await sync_process(requests)

# Pattern 2: Boolean flag passed through every parameter list
async def process_papers(
    papers: list[Paper],
    use_batch_api: bool = False,  # Pollutes function signatures
) -> list[Result]:
    ...

# Pattern 3: Inconsistent thresholds
# File A: >= 5
# File B: >= 10
# File C: >= 3
```

**Key issues:**
- No single point of control for batch decisions
- Users couldn't configure speed/cost preferences
- Duplicate batch collection logic in multiple workflows
- Complex `MAX_LLM_CONCURRENT` semaphore patterns
- Difficult to monitor or change batching behavior globally

## Root Cause

**Decentralized batch decision logic** - each call site independently implemented batching without coordination.

## Solution

**Central LLM Batch Broker** - a unified routing layer that:
1. Provides user-configurable modes (Fast/Balanced/Economical)
2. Allows call-sites to declare batch policy intent
3. Makes routing decisions via a 2D matrix (mode + policy)
4. Eliminates scattered threshold logic

### Architecture

```
core/llm_broker/
├── __init__.py       # Public API exports
├── broker.py         # Core LLMBroker with routing, batching, retry logic
├── config.py         # BrokerConfig, feature flag, thresholds
├── exceptions.py     # QueueOverflowError, BatchSubmissionError, etc.
├── metrics.py        # BrokerMetrics for observability
├── persistence.py    # Async-safe file queue with fcntl locking
└── schemas.py        # LLMRequest, LLMResponse, BatchPolicy, UserMode
```

### User Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `FAST` | No batching, all calls use sync API | Development, debugging, interactive use |
| `BALANCED` | Batch calls marked PREFER_BALANCE or below | Default; balanced cost/speed |
| `ECONOMICAL` | Aggressive batching for all eligible calls | Bulk processing, overnight jobs |

### Call-Site Policies

| Policy | Description | Example Use |
|--------|-------------|-------------|
| `FORCE_BATCH` | Always batch regardless of mode | Large bulk imports |
| `PREFER_BALANCE` | Batch in Balanced/Economical modes | Standard workflow nodes |
| `PREFER_SPEED` | Batch only in Economical mode | User-facing summarization |
| `REQUIRE_SYNC` | Never batch, always synchronous | Streaming, interactive chat |

### Routing Matrix

| Policy | FAST | BALANCED | ECONOMICAL |
|--------|------|----------|------------|
| FORCE_BATCH | Batch | Batch | Batch |
| PREFER_BALANCE | Sync | Batch | Batch |
| PREFER_SPEED | Sync | Sync | Batch |
| REQUIRE_SYNC | Sync | Sync | Sync |

### Routing Decision Logic

```python
# core/llm_broker/broker.py:353-392
def _should_batch(
    self,
    policy: BatchPolicy,
    model: ModelTier,
    effort: str | None,
) -> bool:
    """Determine if a request should be batched based on policy + mode."""

    # DeepSeek doesn't support batch API
    if is_deepseek_tier(model):
        return False

    # Extended thinking incompatible with batch
    if effort:
        return False

    # Policy + Mode routing matrix
    if policy == BatchPolicy.REQUIRE_SYNC:
        return False
    if policy == BatchPolicy.FORCE_BATCH:
        return True
    if self._mode == UserMode.FAST:
        return False
    if policy == BatchPolicy.PREFER_BALANCE and self._mode in (
        UserMode.BALANCED,
        UserMode.ECONOMICAL,
    ):
        return True
    if policy == BatchPolicy.PREFER_SPEED and self._mode == UserMode.ECONOMICAL:
        return True

    return False
```

### Usage Examples

**Single request:**
```python
from core.llm_broker import get_broker, BatchPolicy

broker = get_broker()
await broker.start()

future = await broker.request(
    prompt="Analyze this document",
    policy=BatchPolicy.PREFER_SPEED,
    model=ModelTier.SONNET,
)
response = await future

await broker.stop()
```

**Batch group (grouped requests):**
```python
async with broker.batch_group() as group:
    future1 = await broker.request(prompt1, policy=BatchPolicy.PREFER_BALANCE)
    future2 = await broker.request(prompt2, policy=BatchPolicy.PREFER_BALANCE)
    # Batch submitted automatically on context exit
    results = await asyncio.gather(future1, future2)
```

**Integration with structured output:**
```python
from core.llm_broker import BatchPolicy

results = await get_structured_output(
    output_schema=PaperSummary,
    requests=[
        StructuredRequest(id="p1", user_prompt="Summarize: ..."),
        StructuredRequest(id="p2", user_prompt="Summarize: ..."),
    ],
    batch_policy=BatchPolicy.PREFER_BALANCE,  # Routes through broker
)
```

## Implementation Phases

### Phase 1: Core Infrastructure
- Implemented schemas, persistence, config, metrics, broker
- 79 unit tests covering routing logic, persistence, batch groups
- Feature flag disabled by default for safe rollout

### Phase 2: Interface Integration
- Added `batch_policy` parameter to `get_structured_output()`
- Added `batch_policy` to `invoke_with_cache()` and `batch_invoke_with_cache()`
- Added `llm_mode` field to workflow state types
- Integration tests for broker routing logic

### Phase 3-4: Call Site Migration
- Removed hardcoded `>= 5` batch thresholds (6+ files)
- Replaced `BatchProcessor` direct usage with `broker.batch_group()`
- Removed `use_batch_api` workflow state fields
- Simplified lit review pipeline (removed manual batch collection)
- Removed `MAX_LLM_CONCURRENT` semaphore patterns

### Code Quality Fixes
13 TODOs resolved:
- Moved `ModelTier` to `core/types/` (dependency inversion)
- Added `RateLimitError` handling with retry (60s delay)
- Added async context manager (`__aenter__`/`__aexit__`)
- Added validation in `LLMRequest.from_dict()` deserialization
- Secure file permissions (0o700/0o600) for queue persistence
- Removed unused exception classes and methods
- Compact JSON serialization for queue files

## Configuration

**Feature flag (disabled by default):**
```bash
THALA_LLM_BROKER_ENABLED=1
```

**Mode configuration:**
```bash
THALA_LLM_BROKER_MODE=balanced  # fast|balanced|economical
```

**Thresholds:**
```python
@dataclass
class BrokerConfig:
    enabled: bool = False
    default_mode: UserMode = UserMode.BALANCED
    batch_threshold: int = 50
    max_queue_size: int = 100
    overflow_behavior: Literal["sync", "reject"] = "sync"
```

## Files Modified

**New modules (7 files, ~2,100 lines):**
- `core/llm_broker/__init__.py`
- `core/llm_broker/broker.py`
- `core/llm_broker/config.py`
- `core/llm_broker/exceptions.py`
- `core/llm_broker/metrics.py`
- `core/llm_broker/persistence.py`
- `core/llm_broker/schemas.py`

**New tests (6 files, ~2,100 lines):**
- `tests/unit/core/llm_broker/test_broker.py`
- `tests/unit/core/llm_broker/test_config.py`
- `tests/unit/core/llm_broker/test_metrics.py`
- `tests/unit/core/llm_broker/test_persistence.py`
- `tests/unit/core/llm_broker/test_schemas.py`
- `tests/integration/llm_broker/test_broker_routing.py`

**Migrated call sites (70+ files):**
- `workflows/document_processing/nodes/*.py`
- `workflows/research/academic_lit_review/**/*.py`
- `workflows/shared/llm_utils/structured/*.py`
- `workflows/shared/llm_utils/caching.py`
- `workflows/shared/language/query_translator.py`

## Verification

```bash
# Unit tests
pytest tests/unit/core/llm_broker/ -v

# Integration tests
pytest tests/integration/llm_broker/ -v

# Manual verification with broker enabled
THALA_LLM_BROKER_ENABLED=1 python -m workflows.document_processing
```

## Benefits

- **50% cost reduction** on batched requests (Anthropic pricing)
- **Centralized control** via single configuration point
- **User-configurable modes** for speed/cost tradeoff
- **Simplified call sites** - just `await broker.request(...)`
- **Cross-workflow coordination** of batch decisions
- **Observability** via metrics and LangSmith tracing
- **Graceful degradation** with sync fallback on overflow

## Trade-offs

- **Higher system complexity** (broker lifecycle, persistence, monitoring)
- **Latency for batched requests** (minutes to hours vs immediate)
- **Feature flag required** for safe rollout
- **Migration effort** to update all call sites (one-time)

## Related Solutions

- [Batch API Custom ID Sanitization](./batch-api-custom-id-sanitization.md) - custom_id validation
- [Batch API JSON Structured Output](./batch-api-json-structured-output.md) - tool-based structured output for batch
- [Model Tier Optimization](./model-tier-optimization.md) - model selection framework
- [Centralized Retry Configuration](./centralized-retry-configuration.md) - retry handling patterns

## Related Patterns

- [Central LLM Broker Routing](../../patterns/llm-interaction/central-llm-broker-routing.md) - architectural pattern for this solution
- [Batch API Cost Optimization](../../patterns/llm-interaction/batch-api-cost-optimization.md) - foundational batch pattern
- [Anthropic Prompt Caching](../../patterns/llm-interaction/anthropic-prompt-caching-cost-optimization.md) - complementary cost optimization
- [LangSmith Batch Tracing](../../patterns/observability/langsmith-batch-tracing.md) - observability for batch operations
