---
name: batch-api-cost-optimization
title: Batch API Cost Optimization Pattern
date: 2026-01-02
category: llm-interaction
applicability:
  - "Making 5+ concurrent LLM calls in a single operation"
  - "Processing large datasets (scoring, extraction, summarization)"
  - "Operations where latency is acceptable (minutes to hours)"
  - "Cost-sensitive batch processing workflows"
components: [batch_processor, relevance_scoring, extraction, summarization]
complexity: medium
verified_in_production: true
tags: [anthropic, batch-api, cost-reduction, llm, async, tracing]
---

# Batch API Cost Optimization Pattern

## Intent

Reduce LLM costs by 50% using Anthropic's Message Batches API for concurrent operations, with automatic fallback to individual calls for small batches.

## Problem

Research workflows make many concurrent LLM calls:
- **Relevance scoring**: 100-1000 papers per review
- **Paper extraction**: 10-100 papers for summaries
- **Chapter summarization**: 10-30 chapters per book
- **Query translation**: 5-20 queries per language
- **Section writing**: 5-15 thematic sections

Using standard API calls:
- Full price per token
- Rate limits constrain parallelism
- No bulk discount

## Solution

Use Anthropic's Message Batches API when making 5+ concurrent calls:
1. **Threshold check**: Route to batch API only when beneficial
2. **Request collection**: Build batch of requests with custom IDs
3. **Batch submission**: Submit and poll for completion
4. **Result mapping**: Map results back using custom IDs
5. **Fallback**: Use `asyncio.gather` with semaphores for small batches

**Cost savings**: 50% reduction on batched requests (per Anthropic pricing).

## Structure

```
workflows/shared/batch_processor/
├── __init__.py          # Public exports
├── processor.py         # BatchProcessor class with execute_batch()
├── models.py            # BatchRequest, BatchResult dataclasses
├── request_builder.py   # Build API-formatted requests
└── result_parser.py     # Parse results from batch completion

Integration points:
├── relevance_scoring.py # batch_score_relevance() - papers
├── extraction.py        # batch_extract_summaries() - paper content
├── chapter_summarization.py # batch_summarize_chapters() - books
├── query_translator.py  # batch_translate_queries() - multilingual
├── writing_nodes.py     # batch_write_sections() - synthesis
└── analysis.py          # batch_analyze_clusters() - clustering
```

## Implementation

### BatchProcessor Core

```python
# workflows/shared/batch_processor/processor.py

from anthropic import Anthropic, AsyncAnthropic
from langsmith import traceable
from langsmith.wrappers import wrap_anthropic

from .models import BatchRequest, BatchResult
from .request_builder import RequestBuilder
from .result_parser import ResultParser


class BatchProcessor:
    """Processor for Anthropic Message Batches API.

    Collects LLM requests and submits them as a batch for 50% cost reduction.
    Results are available when the batch completes (typically within 1 hour).
    """

    CONTEXT_1M_BETA = "context-1m-2025-08-07"

    def __init__(self, poll_interval: int = 60, max_wait_hours: float = 24):
        """
        Args:
            poll_interval: Seconds between status checks (default: 60)
            max_wait_hours: Maximum hours to wait for completion (default: 24)
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        # Wrap clients with LangSmith for tracing
        self.client = wrap_anthropic(Anthropic(api_key=api_key))
        self.async_client = wrap_anthropic(AsyncAnthropic(api_key=api_key))
        self.poll_interval = poll_interval
        self.max_wait_hours = max_wait_hours
        self.pending_requests: list[BatchRequest] = []
        self._request_builder = RequestBuilder()
        self._result_parser = ResultParser(self._request_builder.get_original_id)
        self._needs_1m_context: bool = False

    def add_request(
        self,
        custom_id: str,
        prompt: str,
        model: ModelTier = ModelTier.SONNET,
        max_tokens: int = 4096,
        system: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        tools: Optional[list[dict]] = None,
        tool_choice: Optional[dict] = None,
    ) -> None:
        """Add a request to the pending batch."""
        self.pending_requests.append(
            BatchRequest(
                custom_id=custom_id,
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                system=system,
                thinking_budget=thinking_budget,
                tools=tools,
                tool_choice=tool_choice,
            )
        )
        if model == ModelTier.SONNET_1M:
            self._needs_1m_context = True

    @traceable(name="anthropic_batch_execute", run_type="llm")
    async def execute_batch(self) -> dict[str, BatchResult]:
        """Submit batch and wait for results.

        Returns:
            Dictionary mapping custom_id to BatchResult
        """
        if not self.pending_requests:
            return {}

        batch_requests = self._request_builder.build_batch_requests(
            self.pending_requests
        )
        logger.info(f"Submitting batch with {len(batch_requests)} requests")

        # Create batch - use beta API if 1M context needed
        if self._needs_1m_context:
            batch = self.client.beta.messages.batches.create(
                requests=batch_requests,
                betas=[self.CONTEXT_1M_BETA],
            )
        else:
            batch = self.client.messages.batches.create(requests=batch_requests)

        batch_id = batch.id

        # Poll for completion
        max_polls = int(self.max_wait_hours * 3600 / self.poll_interval)
        for _ in range(max_polls):
            if batch.processing_status == "ended":
                break

            await asyncio.sleep(self.poll_interval)
            if self._needs_1m_context:
                batch = self.client.beta.messages.batches.retrieve(batch_id)
            else:
                batch = self.client.messages.batches.retrieve(batch_id)

            if batch.processing_status == "ended":
                break
        else:
            raise RuntimeError(f"Batch {batch_id} did not complete")

        # Fetch and parse results
        results = await self._result_parser.fetch_results(batch.results_url)
        self.clear_requests()

        return results
```

### Request and Result Models

```python
# workflows/shared/batch_processor/models.py

import re
from dataclasses import dataclass
from typing import Optional


def sanitize_custom_id(identifier: str) -> str:
    """Convert identifier to valid Anthropic batch custom_id.

    API requires pattern ^[a-zA-Z0-9_-]{1,64}$.
    """
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", identifier)
    return sanitized[:64]


@dataclass
class BatchRequest:
    """A single request to be included in a batch."""
    custom_id: str
    prompt: str
    model: any  # ModelTier
    max_tokens: int = 4096
    system: Optional[str] = None
    thinking_budget: Optional[int] = None
    tools: Optional[list[dict]] = None
    tool_choice: Optional[dict] = None


@dataclass
class BatchResult:
    """Result from a single batch request."""
    custom_id: str
    success: bool
    content: Optional[str] = None
    thinking: Optional[str] = None
    error: Optional[str] = None
    usage: Optional[dict] = None
```

### Threshold-Based Routing

```python
# Example from workflows/research/subgraphs/academic_lit_review/utils/relevance_scoring.py

async def batch_score_relevance(
    papers: list[PaperMetadata],
    topic: str,
    research_questions: list[str],
    threshold: float = 0.6,
    language_config: LanguageConfig | None = None,
    tier: ModelTier = ModelTier.HAIKU,
    max_concurrent: int = 10,  # Kept for API compatibility
) -> tuple[list[PaperMetadata], list[PaperMetadata]]:
    """Score papers' relevance with automatic batch routing.

    Uses Anthropic Batch API for 50% cost reduction when scoring 5+ papers.
    Falls back to concurrent individual calls for smaller batches.
    """
    if not papers:
        return [], []

    # Threshold check: batch API only beneficial for 5+ requests
    if len(papers) >= 5:
        return await _batch_score_relevance_batched(
            papers, topic, research_questions, threshold, language_config, tier
        )

    # Fallback: individual calls with semaphore limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    async def score_with_limit(paper):
        async with semaphore:
            return await score_paper_relevance(...)

    tasks = [score_with_limit(paper) for paper in papers]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # ... process results ...


async def _batch_score_relevance_batched(
    papers: list[PaperMetadata],
    topic: str,
    research_questions: list[str],
    threshold: float,
    language_config: LanguageConfig | None,
    tier: ModelTier,
) -> tuple[list[PaperMetadata], list[PaperMetadata]]:
    """Score papers using Anthropic Batch API."""
    processor = BatchProcessor(poll_interval=30)
    paper_index = {}  # Map custom_id back to paper

    for i, paper in enumerate(papers):
        custom_id = f"relevance-{i}"
        paper_index[custom_id] = paper

        user_prompt = RELEVANCE_SCORING_USER_TEMPLATE.format(
            topic=topic,
            title=paper.get("title", "Unknown"),
            abstract=(paper.get("abstract") or "")[:1000],
            # ... other fields ...
        )

        processor.add_request(
            custom_id=custom_id,
            prompt=user_prompt,
            model=tier,
            max_tokens=512,
            system=system_prompt,
        )

    logger.info(f"Submitting batch of {len(papers)} papers for relevance scoring")
    results = await processor.execute_batch()

    relevant, rejected = [], []
    for custom_id, paper in paper_index.items():
        result = results.get(custom_id)
        if result and result.success:
            parsed = json.loads(result.content)
            score = float(parsed.get("relevance_score", 0.5))
            paper["relevance_score"] = score
            if score >= threshold:
                relevant.append(paper)
            else:
                rejected.append(paper)
        else:
            paper["relevance_score"] = 0.5
            rejected.append(paper)

    return relevant, rejected
```

## Usage

### Basic Batch Processing

```python
from workflows.shared.batch_processor import BatchProcessor

processor = BatchProcessor(poll_interval=30, max_wait_hours=2)

# Add requests with unique custom_ids
for i, item in enumerate(items):
    processor.add_request(
        custom_id=f"item-{i}",
        prompt=f"Process this: {item}",
        model=ModelTier.HAIKU,
        max_tokens=1024,
    )

# Execute and get results
results = await processor.execute_batch()

# Access results by custom_id
for i, item in enumerate(items):
    result = results.get(f"item-{i}")
    if result and result.success:
        print(f"Item {i}: {result.content}")
    else:
        print(f"Item {i} failed: {result.error if result else 'No result'}")
```

### With Extended Thinking

```python
processor = BatchProcessor(poll_interval=60)

for i, problem in enumerate(complex_problems):
    processor.add_request(
        custom_id=f"analysis-{i}",
        prompt=problem,
        model=ModelTier.SONNET,
        max_tokens=8000,
        thinking_budget=4000,  # Enable extended thinking
    )

results = await processor.execute_batch()

for custom_id, result in results.items():
    if result.success:
        print(f"Thinking: {result.thinking[:200]}...")
        print(f"Answer: {result.content}")
```

### With Progress Callbacks

```python
async def report_progress(batch_id: str, status: str, counts: dict):
    """Called periodically during batch processing."""
    succeeded = counts.get("succeeded", 0)
    total = counts.get("total", 0)
    print(f"Batch {batch_id}: {status}, {succeeded}/{total} complete")

processor = BatchProcessor(poll_interval=30)
# ... add requests ...

results = await processor.execute_batch_with_callback(
    callback=report_progress,
    callback_interval=300,  # Every 5 minutes
)
```

## Guidelines

### Threshold Selection

| Request Count | Recommendation |
|---------------|----------------|
| 1-4 | Use individual calls (overhead not worth it) |
| 5-50 | Batch API beneficial, typical completion < 30 min |
| 50-500 | Batch API strongly recommended |
| 500+ | Consider splitting into multiple batches |

### Custom ID Best Practices

Custom IDs must match `^[a-zA-Z0-9_-]{1,64}$`:
- Use `sanitize_custom_id()` for arbitrary input
- Include index for array items: `relevance-0`, `relevance-1`
- Include type prefix for clarity: `paper-{doi}`, `chapter-{num}`

### Poll Interval Tuning

| Batch Size | Poll Interval | Rationale |
|------------|---------------|-----------|
| < 100 | 30 seconds | Small batches complete quickly |
| 100-500 | 60 seconds | Balance responsiveness vs API calls |
| 500+ | 120 seconds | Large batches take longer |

### Error Handling

```python
results = await processor.execute_batch()

for custom_id, result in results.items():
    if not result:
        logger.error(f"No result for {custom_id}")
        continue

    if not result.success:
        logger.warning(f"Request {custom_id} failed: {result.error}")
        # Handle failure (retry, default value, etc.)
        continue

    # Process successful result
    content = result.content
```

## Known Uses

- `workflows/research/subgraphs/academic_lit_review/utils/relevance_scoring.py` - Paper relevance scoring
- `workflows/research/subgraphs/academic_lit_review/paper_processor/extraction.py` - Summary extraction
- `workflows/research/subgraphs/chapter_summarization.py` - Book chapter summaries
- `workflows/shared/language/query_translator.py` - Multi-language query translation
- `workflows/research/synthesis/nodes/writing_nodes.py` - Section writing
- `workflows/research/subgraphs/academic_lit_review/clustering/analysis.py` - Cluster analysis

## Consequences

### Benefits
- **50% cost reduction**: Direct savings on LLM token costs
- **Simplified concurrency**: No semaphore management for large batches
- **LangSmith integration**: Batch execution traced with aggregated usage
- **Extended thinking support**: Works with thinking budget
- **1M context support**: Beta API for large context windows

### Trade-offs
- **Latency**: Batches complete in minutes to hours (not real-time)
- **Polling overhead**: Need to poll for completion
- **Failure granularity**: Individual request failures in batch results
- **Threshold complexity**: Routing logic for small vs large batches

## Related Patterns

- [Anthropic Claude Integration with Extended Thinking](./anthropic-claude-extended-thinking.md) - Extended thinking support
- [Prompt Caching Patterns](./prompt-caching-patterns.md) - Cache hits in batches

## Related Solutions

- [Batch API JSON/Structured Output Fixes](../../solutions/llm-issues/batch-api-json-structured-output.md) - Common parsing issues
- [Batch API Custom ID Sanitization](../../solutions/llm-issues/batch-api-custom-id-sanitization.md) - ID validation

## References

- [Anthropic Message Batches API](https://docs.anthropic.com/en/api/creating-message-batches)
- [LangSmith Tracing](https://docs.smith.langchain.com/)
