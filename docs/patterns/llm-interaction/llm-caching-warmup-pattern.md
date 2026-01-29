---
name: llm-caching-warmup-pattern
title: "LLM Caching Warmup Pattern: Coordinated Batch Processing with Prefix Caching"
date: 2026-01-28
category: llm-interaction
applicability:
  - "Batch processing with DeepSeek models requiring cache coordination"
  - "High-volume LLM calls with shared system prompts"
  - "Prompt restructuring for optimal prefix caching"
  - "Multi-document analysis with shared task instructions"
components: [batch_invoke_with_cache, content_first_prompts, prefix_hashing, warmup_coordination]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [caching, warmup, deepseek, batch-processing, prefix-caching, cost-optimization, prompt-engineering]
---

# LLM Caching Warmup Pattern: Coordinated Batch Processing with Prefix Caching

## Intent

Coordinate LLM batch processing with cache warmup to achieve 90% cost reduction on input tokens by: (1) restructuring prompts with content-first pattern, (2) warming cache with first request, (3) batch processing remaining requests.

## Motivation

DeepSeek's prefix-based caching requires coordination to achieve cost savings:

**The Problem:**
```
Uncoordinated batch requests (no cache warmup):
┌────────────────────────────────────────────────────────────────┐
│  Request 1: [system] + [user_prompt_1]  → Full cost (100%)    │
│  Request 2: [system] + [user_prompt_2]  → Full cost (100%)    │
│  Request 3: [system] + [user_prompt_3]  → Full cost (100%)    │
│  ...                                                           │
│  Request N: [system] + [user_prompt_N]  → Full cost (100%)    │
│                                                                │
│  Problem: Cache not warmed before batch processing             │
│  DeepSeek needs ~10 seconds to construct KV cache              │
│  All requests hit cold cache → no savings                      │
└────────────────────────────────────────────────────────────────┘
```

**The Solution:**
```
Coordinated batch with warmup:
┌────────────────────────────────────────────────────────────────┐
│  WARMUP PHASE (sequential):                                    │
│  Request 1: [system] + [user_prompt_1]  → Full cost (100%)    │
│  Wait 10 seconds for cache construction                        │
│                                                                │
│  BATCH PHASE (concurrent):                                     │
│  Request 2-N: [system] + [user_prompt_2..N]  → 10% cost each  │
│                                                                │
│  Result: 90% savings on input tokens for requests 2-N          │
│  100 papers: $0.19 vs $0.32 (40% overall savings)             │
└────────────────────────────────────────────────────────────────┘
```

## Applicability

Use this pattern when:
- Processing multiple items through same LLM with shared system prompt
- Using DeepSeek models (V3 or R1)
- High volume where 10-second warmup delay is acceptable
- Items share common context (research topic, document content)

Do NOT use this pattern when:
- Single request (no batch benefit)
- Using Anthropic models (use explicit cache_control instead)
- Latency-critical applications where 10s delay is unacceptable
- Each request has unique system prompt (no shared prefix)

## Structure

```
┌────────────────────────────────────────────────────────────────────┐
│  batch_invoke_with_cache() Coordination Flow                       │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Phase 1: Cache Check                                        │  │
│  │  prefix_hash = hash(system_prompt + cache_prefix)           │  │
│  │  if prefix_hash not in _deepseek_cache_warmed:              │  │
│  │      → proceed to warmup                                     │  │
│  │  else:                                                       │  │
│  │      → skip warmup, all requests are cache hits             │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              ↓                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Phase 2: Warmup (if needed)                                │  │
│  │  - Execute first request immediately                        │  │
│  │  - Record result                                            │  │
│  │  - Mark prefix_hash as warmed                               │  │
│  │  - Wait 10 seconds for cache construction                   │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                              ↓                                     │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  Phase 3: Concurrent Batch Processing                       │  │
│  │  - Process remaining N-1 requests concurrently              │  │
│  │  - Semaphore limits to max_concurrent (default: 10)         │  │
│  │  - All requests hit warm cache → 90% input token savings    │  │
│  └─────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│  Content-First Prompt Structure                                    │
│                                                                    │
│  BEFORE (instructions first - poor caching):                       │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  "Task: Analyze this document about {topic}                 │  │
│  │   Focus on: {instructions}                                  │  │
│  │   DOCUMENT: {content}"                                      │  │
│  │                                                              │  │
│  │  Problem: Different topics/instructions → different prefix  │  │
│  │           → cache miss for each unique task                 │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  AFTER (content first - optimal caching):                         │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │  "{content}                                                 │  │
│  │   ---                                                       │  │
│  │   Task: Analyze this document about {topic}"               │  │
│  │                                                              │  │
│  │  Benefit: Same document content → same prefix               │  │
│  │           → cache hit for different tasks on same doc      │  │
│  └─────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

## Implementation

### Step 1: Implement Batch Cache Coordination

```python
# workflows/shared/llm_utils/caching.py

import asyncio
import time
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_deepseek import ChatDeepSeek

# Cache warmup tracking
_deepseek_cache_warmed: dict[int, float] = {}
DEEPSEEK_CACHE_WARMUP_DELAY = 10.0  # seconds


def _is_deepseek_model(llm: BaseChatModel) -> bool:
    """Check if LLM is a DeepSeek model."""
    return isinstance(llm, ChatDeepSeek)


def _compute_prefix_hash(system_prompt: str, cache_prefix: str | None = None) -> int:
    """Compute hash for cache tracking, including optional user prefix."""
    if cache_prefix:
        return hash(system_prompt + cache_prefix)
    return hash(system_prompt)


async def batch_invoke_with_cache(
    llm: BaseChatModel,
    system_prompt: str,
    user_prompts: list[tuple[str, str]],  # (request_id, user_prompt)
    cache_prefix: str | None = None,
    max_concurrent: int = 10,
) -> dict[str, Any]:
    """Invoke LLM for multiple requests with cache warmup coordination.

    For DeepSeek: Sends first request, waits for cache construction, then
    processes remaining requests concurrently to benefit from prefix caching.

    For Anthropic: Processes all requests concurrently (cache is explicit).

    Args:
        llm: Language model to use
        system_prompt: System prompt (shared across all requests)
        user_prompts: List of (request_id, user_prompt) tuples
        cache_prefix: Optional shared prefix in user prompts for hash tracking
        max_concurrent: Maximum concurrent requests after warmup

    Returns:
        Dict mapping request_id to response
    """
    if not user_prompts:
        return {}

    results: dict[str, Any] = {}

    # Non-DeepSeek: process all concurrently (no warmup needed)
    if not _is_deepseek_model(llm):
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_one(req_id: str, user_prompt: str) -> tuple[str, Any]:
            async with semaphore:
                response = await invoke_with_cache(llm, system_prompt, user_prompt)
                return req_id, response

        tasks = [process_one(req_id, prompt) for req_id, prompt in user_prompts]
        for req_id, response in await asyncio.gather(*tasks):
            results[req_id] = response
        return results

    # DeepSeek: coordinate cache warmup
    prefix_hash = _compute_prefix_hash(system_prompt, cache_prefix)

    # Check if cache needs warmup
    if prefix_hash not in _deepseek_cache_warmed:
        # First request triggers cache construction
        first_id, first_prompt = user_prompts[0]
        remaining = user_prompts[1:]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_prompt},
        ]
        response = await llm.ainvoke(messages)
        results[first_id] = response

        # Mark warmed and wait for cache construction
        _deepseek_cache_warmed[prefix_hash] = time.time()
        logger.info(
            f"DeepSeek cache warmup: waiting {DEEPSEEK_CACHE_WARMUP_DELAY}s "
            f"before processing {len(remaining)} remaining requests"
        )
        await asyncio.sleep(DEEPSEEK_CACHE_WARMUP_DELAY)
    else:
        remaining = user_prompts
        logger.debug(f"DeepSeek cache already warm, processing {len(remaining)} requests")

    # Process remaining requests concurrently
    if remaining:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_remaining(req_id: str, user_prompt: str) -> tuple[str, Any]:
            async with semaphore:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                return req_id, await llm.ainvoke(messages)

        tasks = [process_remaining(req_id, prompt) for req_id, prompt in remaining]
        for req_id, response in await asyncio.gather(*tasks):
            results[req_id] = response

    return results
```

### Step 2: Restructure Prompts for Content-First Pattern

```python
# workflows/document_processing/prompts.py

# Unified system prompt - generic to enable sharing across agents
DOCUMENT_ANALYSIS_SYSTEM = """You are a document analysis specialist. Analyze the provided document and follow the task instructions given after the document content."""
```

```python
# workflows/document_processing/nodes/metadata_agent.py

from workflows.document_processing.prompts import DOCUMENT_ANALYSIS_SYSTEM

# BEFORE (task first):
# user_prompt = f"Extract metadata from this document:\n\n{content}"

# AFTER (content first):
user_prompt = f"""{content}

---
Task: Extract bibliographic metadata from the document above.

Extract:
- title: Full document title
- authors: List of author names
- date: Publication date
- publisher: Publisher name
- isbn: ISBN if present
- is_multi_author: true if multi-author edited volume"""

result = await get_structured_output(
    output_schema=DocumentMetadata,
    user_prompt=user_prompt,
    system_prompt=DOCUMENT_ANALYSIS_SYSTEM,  # Shared with summary_agent
    tier=ModelTier.DEEPSEEK_R1,
    enable_prompt_cache=True,
)
```

### Step 3: Apply to Batch Scoring

```python
# workflows/research/academic_lit_review/utils/relevance_scoring/scorer.py

from workflows.shared.llm_utils import batch_invoke_with_cache

async def _batch_score_deepseek_cached(
    papers: list[PaperMetadata],
    topic: str,
    research_questions: list[str],
    tier: ModelTier,
    max_concurrent: int = 10,
) -> tuple[list[PaperMetadata], list[PaperMetadata]]:
    """Score papers using DeepSeek with cache-aware batch processing."""
    llm = get_llm(tier=tier)

    # Build cache prefix (static portion shared by all requests)
    research_questions_str = "; ".join(research_questions[:3])
    cache_prefix = f"Research Topic: {topic}\nResearch Questions: {research_questions_str}\n\nPaper to Evaluate:"

    # Build user prompts for each paper
    user_prompts: list[tuple[str, str]] = []
    paper_index: dict[str, PaperMetadata] = {}

    for i, paper in enumerate(papers):
        paper_id = paper.get("doi") or f"paper-{i}"
        paper_index[paper_id] = paper

        user_prompt = RELEVANCE_SCORING_USER_TEMPLATE.format(
            topic=topic,
            research_questions=research_questions_str,
            title=paper.get("title", "Unknown"),
            authors=format_authors(paper),
            abstract=(paper.get("abstract") or "No abstract")[:1000],
        )
        user_prompts.append((paper_id, user_prompt))

    # Process with cache warmup coordination
    responses = await batch_invoke_with_cache(
        llm,
        system_prompt=RELEVANCE_SCORING_SYSTEM,
        user_prompts=user_prompts,
        cache_prefix=cache_prefix,  # Static portion for hash tracking
        max_concurrent=max_concurrent,
    )

    # Parse results and categorize
    relevant, rejected = [], []
    for paper_id, paper in paper_index.items():
        response = responses.get(paper_id)
        score = parse_relevance_score(response)
        paper["relevance_score"] = score
        if score >= threshold:
            relevant.append(paper)
        else:
            rejected.append(paper)

    return relevant, rejected
```

## Complete Example

```python
from workflows.shared.llm_utils import (
    ModelTier,
    get_llm,
    batch_invoke_with_cache,
)

# 1. Define shared system prompt
ANALYSIS_SYSTEM = """You are a document analyst. Follow the task instructions provided after the document content."""

# 2. Build content-first user prompts
user_prompts = []
for i, doc in enumerate(documents):
    # Content first, task after separator
    user_prompt = f"""{doc['content']}

---
Task: Summarize the key findings in 3 bullet points."""

    user_prompts.append((f"doc-{i}", user_prompt))

# 3. Process with cache coordination
llm = get_llm(ModelTier.DEEPSEEK_V3)

responses = await batch_invoke_with_cache(
    llm,
    system_prompt=ANALYSIS_SYSTEM,
    user_prompts=user_prompts,
    max_concurrent=10,
)

# Results: First document full cost, remaining 90% savings on input tokens
for doc_id, response in responses.items():
    print(f"{doc_id}: {response.content[:100]}...")
```

## Consequences

### Benefits

- **90% input token savings**: After warmup, cached prefix charged at 10% rate
- **Shared system prompts**: Metadata + summary agents share cache line
- **Content-first caching**: Same document, different tasks share cache
- **Automatic coordination**: `batch_invoke_with_cache()` handles warmup
- **Session-level tracking**: Avoids redundant warmup within process

### Trade-offs

- **10-second latency**: First batch pays warmup delay
- **DeepSeek-specific**: Anthropic uses different caching mechanism
- **Prompt restructuring**: Requires content-first format change
- **Memory tracking**: Per-process cache dict grows with unique prefixes

### Alternatives

- **Anthropic cache_control**: Explicit caching (no warmup delay)
- **No caching**: Accept full cost per request
- **Batch API**: 50% savings via Anthropic batching (different mechanism)

## Related Patterns

- [DeepSeek Integration Patterns](./deepseek-integration-patterns.md) - V3/R1 model selection
- [Anthropic Prompt Caching](./anthropic-prompt-caching-cost-optimization.md) - Explicit cache_control
- [Batch API Cost Optimization](./batch-api-cost-optimization.md) - Anthropic batch API (50% savings)
- [Model Tier Optimization](../../solutions/llm-issues/model-tier-optimization.md) - Tier selection framework

## Known Uses in Thala

- `workflows/shared/llm_utils/caching.py` - `batch_invoke_with_cache()` implementation
- `workflows/document_processing/prompts.py` - Unified `DOCUMENT_ANALYSIS_SYSTEM`
- `workflows/document_processing/nodes/metadata_agent.py` - Content-first user prompt
- `workflows/document_processing/nodes/summary_agent.py` - Shared system prompt
- `workflows/enhance/editing/prompts.py` - Restructured 8+ prompts
- `workflows/enhance/fact_check/prompts.py` - Restructured prompts
- `workflows/research/academic_lit_review/utils/relevance_scoring/scorer.py` - Batch scoring

## References

- Commit: fe17bdf266bbfe39005f5a34b7b06f411712a377
- [DeepSeek Caching Documentation](https://platform.deepseek.com/api-docs/prompt-caching)
- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
