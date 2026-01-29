---
name: langsmith-batch-tracing
title: "LangSmith Batch Tracing: Cost Visibility for Anthropic Message Batches"
date: 2026-01-28
category: observability
applicability:
  - "Batch API usage requiring cost visibility in LangSmith"
  - "High-volume LLM calls processed through Anthropic Message Batches API"
  - "Multi-request operations where aggregate token usage matters"
  - "Production systems needing per-batch success/failure metrics"
components: [wrap_anthropic, traceable_decorator, token_aggregation, metadata_attachment]
complexity: low
verified_in_production: true
related_solutions: []
tags: [langsmith, tracing, batch-api, cost-tracking, anthropic, observability, token-usage]
---

# LangSmith Batch Tracing: Cost Visibility for Anthropic Message Batches

## Intent

Wrap Anthropic batch clients and decorate execution functions to provide complete cost visibility and execution tracking for Message Batches API calls in the LangSmith dashboard.

## Motivation

The Anthropic Message Batches API processes multiple requests in a single batch job, but without proper instrumentation:

**The Problem:**
```
LangSmith UI (without batch tracing):
├── lit_review:Climate Change [workflow:lit_review]
│   ├── ChatAnthropic call (cost: $0.12)
│   ├── ChatAnthropic call (cost: $0.08)
│   └── ... (single-request calls visible)
│
│   ← Batch API calls INVISIBLE
│   ← No cost tracking for 100-request batches
│   ← Cannot see batch success/failure rates
```

When batching 100 relevance scoring requests, you see nothing in LangSmith because:
1. Raw Anthropic clients bypass LangSmith's auto-instrumentation
2. Batch jobs don't emit individual request traces
3. Token usage from batch results isn't captured

**The Solution:**
```
LangSmith UI (with batch tracing):
├── lit_review:Climate Change [workflow:lit_review]
│   ├── batch_structured_output
│   │   Metadata:
│   │     batch_id: batch_01jk...
│   │     request_count: 100
│   │     successful_count: 98
│   │     failed_count: 2
│   │     usage_metadata:
│   │       input_tokens: 125,000
│   │       output_tokens: 8,500
│   │       total_tokens: 133,500
│   │   Total Cost: $2.45
```

## Applicability

Use this pattern when:
- Using Anthropic Message Batches API for high-volume requests
- Need cost visibility for batch operations in LangSmith
- Want success/failure tracking for batch submissions
- Require token aggregation across batch results

Do NOT use this pattern when:
- Using LangChain's ChatAnthropic (already auto-instrumented)
- Single-request operations (standard tracing sufficient)
- No LangSmith integration needed

## Structure

```
┌────────────────────────────────────────────────────────────────────┐
│  Application Code                                                  │
│                                                                    │
│  requests = [StructuredRequest(...) for item in items]            │
│  results = await batch_executor.execute_batch(schema, requests)   │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  BatchToolCallExecutor                                            │
│  @traceable(run_type="llm", name="batch_structured_output")       │
│                                                                    │
│  async def execute_batch(...):                                    │
│      processor = BatchProcessor(...)  ← Wrapped clients           │
│      results = await processor.execute_batch()                    │
│                                                                    │
│      # Aggregate tokens from results                              │
│      total_input = sum(r.usage["input_tokens"] for r in results)  │
│      total_output = sum(r.usage["output_tokens"] for r in results)│
│                                                                    │
│      # Attach to LangSmith trace                                  │
│      run = get_current_run_tree()                                 │
│      run.add_metadata({"usage_metadata": {...}})                  │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  BatchProcessor                                                   │
│  @traceable(name="anthropic_batch_execute", run_type="llm")       │
│                                                                    │
│  def __init__(...):                                               │
│      self.client = wrap_anthropic(Anthropic(...))                 │
│      self.async_client = wrap_anthropic(AsyncAnthropic(...))      │
│                                                                    │
│  async def execute_batch(...):                                    │
│      # Submit batch, poll for completion, parse results           │
│      self._attach_usage_metadata(results, batch_id, count)        │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  LangSmith Dashboard                                              │
│                                                                    │
│  ┌─ batch_structured_output ──────────────────────────────────┐  │
│  │  Run Type: llm                                              │  │
│  │  Metadata:                                                  │  │
│  │    batch_id: batch_01jk8x9z...                             │  │
│  │    request_count: 100                                       │  │
│  │    successful_count: 98                                     │  │
│  │    failed_count: 2                                          │  │
│  │    usage_metadata:                                          │  │
│  │      input_tokens: 125,000                                  │  │
│  │      output_tokens: 8,500                                   │  │
│  │      cache_read_input_tokens: 95,000                        │  │
│  │                                                             │  │
│  │  └─ anthropic_batch_execute (child trace)                   │  │
│  └─────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

## Implementation

### Step 1: Wrap Anthropic Clients

```python
# workflows/shared/batch_processor/processor.py

from anthropic import Anthropic, AsyncAnthropic
from langsmith.wrappers import wrap_anthropic
from langsmith import traceable, get_current_run_tree


class BatchProcessor:
    """Process batch requests with LangSmith tracing."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250929",
        api_key: Optional[str] = None,
    ):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")

        # Wrap both sync and async clients for tracing
        self.client = wrap_anthropic(Anthropic(api_key=api_key))
        self.async_client = wrap_anthropic(AsyncAnthropic(api_key=api_key))

        self.model = model
        self.requests: list[BatchRequest] = []
```

**Why wrap?** `wrap_anthropic()` instruments the client to emit traces for all API calls, but batch API calls still need explicit token aggregation since results arrive asynchronously.

### Step 2: Decorate Execution Methods

```python
# workflows/shared/batch_processor/processor.py

@traceable(name="anthropic_batch_execute", run_type="llm")
async def execute_batch(self) -> dict[str, BatchResult]:
    """Execute all queued requests as a batch.

    The @traceable decorator creates a LangSmith trace with:
    - Automatic timing
    - Parent-child relationship with caller
    - Input/output capture (optional)
    """
    if not self.requests:
        return {}

    # Create batch request
    batch = self.client.messages.batches.create(
        requests=[r.to_api_format() for r in self.requests]
    )

    # Poll for completion
    results = await self._poll_batch(batch.id)

    # Attach usage metadata to trace
    self._attach_usage_metadata(results, batch.id, len(self.requests))

    return results
```

```python
# workflows/shared/llm_utils/structured/executors/batch.py

@traceable(run_type="llm", name="batch_structured_output")
async def execute_batch(
    self,
    output_schema: Type[T],
    requests: list[StructuredRequest],
    default_system: Optional[str],
    output_config: StructuredOutputConfig,  # Note: renamed from 'config'
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> dict[str, StructuredOutputResult[T]]:
    """Execute structured output requests via batch API.

    IMPORTANT: The parameter is named 'output_config' not 'config' because
    @traceable reserves 'config' for LangChain RunnableConfig dict.
    """
    # ... batch execution logic ...
```

### Step 3: Aggregate Token Usage from Results

```python
# workflows/shared/batch_processor/processor.py

def _attach_usage_metadata(
    self,
    results: dict[str, BatchResult],
    batch_id: str,
    request_count: int,
) -> None:
    """Attach aggregated token usage to current LangSmith run."""
    # Aggregate tokens across all results
    total_input = sum(
        r.usage.get("input_tokens", 0) for r in results.values() if r.usage
    )
    total_output = sum(
        r.usage.get("output_tokens", 0) for r in results.values() if r.usage
    )
    total_cache_read = sum(
        r.usage.get("cache_read_input_tokens", 0)
        for r in results.values()
        if r.usage
    )
    total_cache_creation = sum(
        r.usage.get("cache_creation_input_tokens", 0)
        for r in results.values()
        if r.usage
    )

    # Attach to LangSmith trace
    run = get_current_run_tree()
    if run:
        run.add_metadata(
            {
                "batch_id": batch_id,
                "request_count": request_count,
                "successful_count": sum(1 for r in results.values() if r.success),
                "failed_count": sum(1 for r in results.values() if not r.success),
                "usage_metadata": {
                    "input_tokens": total_input,
                    "output_tokens": total_output,
                    "total_tokens": total_input + total_output,
                    "cache_read_input_tokens": total_cache_read,
                    "cache_creation_input_tokens": total_cache_creation,
                },
            }
        )
```

### Step 4: Add Defensive Error Handling

```python
# workflows/shared/llm_utils/structured/executors/batch.py

# After batch execution, attach metadata with error handling
try:
    run = get_current_run_tree()
    if run and (total_input or total_output):
        run.add_metadata(
            {
                "usage_metadata": {
                    "input_tokens": total_input,
                    "output_tokens": total_output,
                    "total_tokens": total_input + total_output,
                }
            }
        )
except Exception as e:
    # Don't let tracing issues break the workflow
    logger.warning(f"LangSmith run tree error: {e}")
```

**Critical:** Tracing failures should never break the batch workflow. Always wrap metadata attachment in try-except.

## Complete Example

```python
from workflows.shared.llm_utils.structured import get_structured_output
from workflows.shared.llm_utils.structured.config import StructuredOutputConfig

# Prepare 100 relevance scoring requests
requests = [
    StructuredRequest(
        custom_id=f"relevance-{i}",
        system_prompt=RELEVANCE_SYSTEM,
        user_prompt=f"Score relevance for paper: {paper['title']}",
    )
    for i, paper in enumerate(papers)
]

# Execute batch with automatic tracing
output_config = StructuredOutputConfig(
    strategy=StructuredOutputStrategy.BATCH_TOOL_CALL,  # Force batch API
    model_name="claude-sonnet-4-5-20250929",
    max_tokens=500,
)

results = await get_structured_output(
    output_schema=RelevanceScore,
    requests=requests,
    default_system=None,
    output_config=output_config,  # Note: use output_config to avoid conflicts with @traceable
)

# In LangSmith dashboard:
# - batch_structured_output trace visible
# - Metadata shows: request_count=100, successful_count=98, failed_count=2
# - usage_metadata shows: input_tokens=125000, output_tokens=8500
# - Cost automatically calculated from token usage
```

### Singleton Client Pattern for Classifiers

```python
# core/scraping/classification/classifier.py

from langsmith.wrappers import wrap_anthropic
from langsmith import traceable
import anthropic

_client: Optional[anthropic.AsyncAnthropic] = None


def _get_client() -> anthropic.AsyncAnthropic:
    """Get or create the wrapped Anthropic async client."""
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        _client = wrap_anthropic(anthropic.AsyncAnthropic(api_key=api_key))
    return _client


@traceable(name="classify_content", run_type="llm")
async def classify_content(
    url: str,
    markdown: str,
    links: list[str],
    doi: Optional[str] = None,
) -> ClassificationResult:
    """Classify content with LangSmith tracing."""
    client = _get_client()
    # ... classification logic ...
```

## Consequences

### Benefits

- **Full cost visibility**: See batch API costs aggregated in LangSmith
- **Success/failure tracking**: Monitor batch reliability with counts
- **Cache hit tracking**: See prompt caching savings (`cache_read_input_tokens`)
- **Hierarchical traces**: Batch traces nest under workflow traces
- **Graceful degradation**: Tracing errors don't break batch execution

### Trade-offs

- **Parameter naming**: Must use `output_config` instead of `config` to avoid conflict with `@traceable`
- **Manual aggregation**: Token usage requires explicit summation from batch results
- **Delayed visibility**: Batch traces only appear after batch completes (not during polling)

### Alternatives

- **No tracing**: Skip instrumentation (lose cost visibility)
- **Custom metrics**: Build separate metrics system (more work, less integration)
- **Individual call tracing**: Don't batch (higher cost, simpler tracing)

## Related Patterns

- [LangSmith Tracing Infrastructure](./langsmith-tracing-infrastructure.md) - Foundation for workflow-level cost visibility
- [Batch API Cost Optimization](../llm-interaction/batch-api-cost-optimization.md) - The batch processing strategy this pattern instruments
- [Conditional Development Tracing](../llm-interaction/conditional-development-tracing.md) - Environment-based tracing toggle
- [LangSmith Trace Identification](../langgraph/langsmith-trace-identification.md) - Semantic naming conventions

## Known Uses in Thala

- `workflows/shared/batch_processor/processor.py` - Core batch processor with client wrapping
- `workflows/shared/llm_utils/structured/executors/batch.py` - Structured output batch executor
- `core/scraping/classification/classifier.py` - Content classifier with wrapped client
- `workflows/shared/llm_utils/structured/executors/base.py` - Base executor interface

## References

- [LangSmith Python SDK](https://docs.smith.langchain.com/how_to_guides/tracing/annotate_code)
- [langsmith.wrappers.wrap_anthropic](https://docs.smith.langchain.com/how_to_guides/tracing/trace_with_langchain#wrap-the-anthropic-client)
- [Anthropic Message Batches API](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing)
