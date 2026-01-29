---
name: anthropic-prompt-caching
title: "Anthropic Prompt Caching for Cost Optimization"
date: 2025-01-28
category: llm-interaction
applicability:
  - "Multi-turn LLM conversations with shared context"
  - "Batch processing with identical system prompts"
  - "Iterative workflows with repeated instructions"
  - "High-volume LLM operations requiring cost control"
components: [llm_call, prompt_design, cost_optimization]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [anthropic, claude, prompt-caching, cost-optimization, deepseek, langchain]
---

# Anthropic Prompt Caching for Cost Optimization

## Intent

Reduce LLM costs by up to 90% through strategic prompt caching, separating static instructions from dynamic content to maximize cache hits across repeated invocations.

## Motivation

In iterative workflows like research supervision or batch document processing, the same system instructions are sent repeatedly with only the user content changing:

- Research supervisor runs 4-12 iterations with identical instructions (~800 tokens)
- Relevance scoring evaluates 100+ papers with the same scoring criteria
- Report generation refines drafts through multiple passes

Without caching, a 12-iteration research workflow sends ~9,600 system prompt tokens (800 x 12). With caching, only the first call incurs the full cost; subsequent calls read from cache at 10% the price.

**Cost impact**: 90% reduction on cached tokens, transforming expensive iterative workflows into cost-efficient operations.

## Applicability

Use this pattern when:
- System prompts are 1024+ tokens (Anthropic's minimum cacheable size)
- Same instructions are reused across multiple calls
- Processing batch operations with shared context
- Running iterative workflows (research, refinement, multi-turn)

Do NOT use this pattern when:
- Each call has unique system prompts
- System prompts are under 1024 tokens (cannot be cached)
- Using models without caching support
- Real-time latency is critical (first call has cache write overhead)

## Structure

```
workflows/shared/llm_utils/
├── caching.py           # Core caching utilities
│   ├── create_cached_messages()
│   ├── invoke_with_cache()
│   └── batch_invoke_with_cache()
└── __init__.py          # Public exports

workflows/research/web_research/
├── prompts/
│   ├── supervision.py   # SUPERVISOR_SYSTEM_CACHED + USER_TEMPLATE
│   └── compression.py   # Cached compression prompts
└── nodes/
    └── supervisor/
        └── core.py      # Uses invoke_with_cache()
```

## Key Principles: Static vs Dynamic Separation

### What Goes in Static (Cached) System Prompt

- Role definition and persona ("You are the lead researcher...")
- Algorithms and procedures (Diffusion Algorithm steps)
- Tool definitions and usage instructions
- Output format specifications
- Guidelines, rules, and constraints
- Evaluation criteria

### What Goes in Dynamic User Prompt

- Current date/timestamp
- User-specific data (research brief, documents)
- Session state (iteration count, findings so far)
- Changing context (draft content, accumulated results)
- Variable parameters (quality settings, thresholds)

## Implementation

### Step 1: Core Caching Utility

```python
# workflows/shared/llm_utils/caching.py
from typing import Literal

CacheTTL = Literal["5m", "1h"]

def create_cached_messages(
    system_content: str | list[dict],
    user_content: str,
    cache_system: bool = True,
    cache_ttl: CacheTTL = "5m",
) -> list[dict]:
    """Create messages with cache_control on system content.

    Args:
        system_content: System prompt text or content blocks
        user_content: User prompt text
        cache_system: Whether to enable caching on system prompt
        cache_ttl: Cache time-to-live ("5m" default, "1h" for long workflows)

    Returns:
        Messages list formatted for Anthropic API with cache_control
    """
    cache_control = {"type": "ephemeral"}
    if cache_ttl == "1h":
        cache_control["ttl"] = "1h"

    if isinstance(system_content, str):
        system_blocks = [
            {
                "type": "text",
                "text": system_content,
                **({"cache_control": cache_control} if cache_system else {}),
            }
        ]
    else:
        # Multi-block system content: cache the last block
        system_blocks = list(system_content)
        if cache_system and system_blocks:
            last_block = dict(system_blocks[-1])
            last_block["cache_control"] = cache_control
            system_blocks[-1] = last_block

    return [
        {"role": "system", "content": system_blocks},
        {"role": "user", "content": user_content},
    ]
```

### Step 2: Invoke with Cache Helper

```python
async def invoke_with_cache(
    llm: BaseChatModel,
    system_prompt: str,
    user_prompt: str,
    cache_system: bool = True,
    cache_ttl: CacheTTL = "5m",
) -> Any:
    """Invoke LLM with prompt caching (Anthropic or DeepSeek).

    For Anthropic: Uses explicit cache_control with ephemeral blocks.
    For DeepSeek: Uses automatic prefix-based caching with warmup delay.
    """
    # Anthropic: explicit cache_control
    if isinstance(llm, ChatAnthropic) and cache_system:
        messages = create_cached_messages(
            system_content=system_prompt,
            user_content=user_prompt,
            cache_system=cache_system,
            cache_ttl=cache_ttl,
        )
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    response = await llm.ainvoke(messages)

    # Log cache usage for monitoring
    if isinstance(llm, ChatAnthropic):
        usage = getattr(response, "usage_metadata", None)
        if usage:
            details = usage.get("input_token_details", {})
            cache_read = details.get("cache_read", 0)
            cache_creation = details.get("cache_creation", 0)
            if cache_read > 0:
                logger.debug(f"Cache hit: {cache_read} tokens read from cache")
            elif cache_creation > 0:
                logger.debug(f"Cache miss: {cache_creation} tokens written to cache")

    return response
```

### Step 3: Batch Processing with Cache Coordination

```python
async def batch_invoke_with_cache(
    llm: BaseChatModel,
    system_prompt: str,
    user_prompts: list[tuple[str, str]],  # (request_id, user_prompt)
    cache_prefix: str | None = None,
    max_concurrent: int = 10,
) -> dict[str, Any]:
    """Invoke LLM for multiple requests with cache warmup coordination.

    For Anthropic: Processes all requests concurrently (cache is explicit).
    For DeepSeek: Sends first request, waits for cache construction (~10s),
                  then processes remaining requests concurrently.

    Args:
        llm: Language model to use
        system_prompt: System prompt (shared across all requests)
        user_prompts: List of (request_id, user_prompt) tuples
        cache_prefix: Optional shared prefix in user prompts for hash tracking
        max_concurrent: Maximum concurrent requests

    Returns:
        Dict mapping request_id to response
    """
    if not user_prompts:
        return {}

    results: dict[str, Any] = {}
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_one(req_id: str, user_prompt: str) -> tuple[str, Any]:
        async with semaphore:
            response = await invoke_with_cache(llm, system_prompt, user_prompt)
            return req_id, response

    tasks = [process_one(req_id, prompt) for req_id, prompt in user_prompts]
    for req_id, response in await asyncio.gather(*tasks):
        results[req_id] = response

    return results
```

## Before/After: Prompt Separation

### BEFORE: Combined Prompt (No Caching)

```python
# Old approach: Everything in one prompt, rebuilt each iteration
SUPERVISOR_PROMPT = """You are the lead researcher coordinating a deep research project.

Today's date is {date}.

<Diffusion Algorithm>
1. Generate research questions that expand understanding
2. Delegate research via ConductResearch tool
3. Refine the draft report with RefineDraftReport
4. Check completeness - identify remaining gaps
5. Either generate more questions or call ResearchComplete
</Diffusion Algorithm>

<Research Brief>
{research_brief}
</Research Brief>

<Current Draft Report>
{draft_report}
</Current Draft Report>

<Iteration Status>
- Iteration: {iteration}/{max_iterations}
- Completeness: {completeness_score}%
</Iteration Status>

Based on the above, decide your next action."""

# Usage: Full prompt rebuilt and sent every iteration
prompt = SUPERVISOR_PROMPT.format(
    date=today,
    research_brief=brief,
    draft_report=draft,
    iteration=i,
    max_iterations=max_iter,
    completeness_score=score,
)
response = await llm.ainvoke([{"role": "user", "content": prompt}])
```

**Problem**: 800+ token instructions sent fresh every iteration. 12 iterations = 12x the instruction cost.

### AFTER: Separated Prompts (With Caching)

```python
# prompts/supervision.py

# Static system prompt (cached) - ~800 tokens, sent once
SUPERVISOR_SYSTEM_CACHED = """You are the lead researcher coordinating a deep research project using the diffusion algorithm.

<Diffusion Algorithm>
1. Generate research questions that expand understanding (diffusion out)
2. Delegate research via ConductResearch tool to gather concrete findings
3. Refine the draft report with RefineDraftReport to reduce noise/gaps
4. Check completeness - identify remaining gaps
5. Either generate more questions or call ResearchComplete
</Diffusion Algorithm>

<Available Tools>
1. **ConductResearch**: Delegate a specific research question to a sub-agent.
2. **RefineDraftReport**: Update the draft report with new findings.
3. **ResearchComplete**: Signal that research is complete.
</Available Tools>

<Instructions>
Think like a research manager:
1. Read the research brief and customized plan carefully
2. Consider what the user already knows (don't duplicate)
3. Assess current findings - are there gaps?
4. Decide: delegate more research OR refine draft OR complete

Rules:
- Generate diverse questions covering different angles
- Respect the customized plan - focus on GAPS, not what user knows
- Complete when completeness > 85% OR max_iterations reached
- Always cite sources in draft updates
</Instructions>

<Output Format>
First, reason through your decision in <thinking> tags.
Then call the appropriate tool.
</Output Format>"""

# Dynamic user prompt (changes each iteration)
SUPERVISOR_USER_TEMPLATE = """Today's date is {date}.

<Research Brief>
{research_brief}
</Research Brief>

<Current Draft Report>
{draft_report}
</Current Draft Report>

<Research Findings So Far>
{findings_summary}
</Research Findings So Far>

<Operational Metadata>
- Iteration: {iteration}/{max_iterations}
- Completeness: {completeness_score}%
- Gaps remaining: {gaps_remaining}
</Operational Metadata>

Based on the above context, decide your next action."""
```

```python
# nodes/supervisor/core.py

from workflows.shared.llm_utils import invoke_with_cache

async def supervisor(state: DeepResearchState) -> dict[str, Any]:
    # Build dynamic user prompt (changes each iteration)
    user_prompt = SUPERVISOR_USER_TEMPLATE.format(
        date=get_today_str(),
        research_brief=json.dumps(brief, indent=2),
        draft_report=draft_content,
        findings_summary=findings_summary,
        iteration=iteration + 1,
        max_iterations=max_iterations,
        completeness_score=int(diffusion["completeness_score"] * 100),
        gaps_remaining=", ".join(gaps_remaining),
    )

    llm = get_llm(ModelTier.OPUS)

    # Cached invocation: 90% cost reduction on iterations 2-N
    response = await invoke_with_cache(
        llm,
        system_prompt=SUPERVISOR_SYSTEM_CACHED,  # Static, cached
        user_prompt=user_prompt,                  # Dynamic, not cached
    )

    return process_response(response)
```

## Usage Examples at Call Sites

### Example 1: Research Supervisor (Iterative Workflow)

```python
# 12-iteration research workflow
for iteration in range(max_iterations):
    user_prompt = SUPERVISOR_USER_TEMPLATE.format(
        date=get_today_str(),
        research_brief=brief,
        draft_report=draft,
        findings_summary=findings,
        iteration=iteration + 1,
        max_iterations=max_iterations,
        completeness_score=score,
        gaps_remaining=gaps,
    )

    # First call: cache write (~800 tokens)
    # Calls 2-12: cache read at 10% cost
    response = await invoke_with_cache(
        llm,
        system_prompt=SUPERVISOR_SYSTEM_CACHED,
        user_prompt=user_prompt,
    )
```

### Example 2: Relevance Scoring (Batch Processing)

```python
# Score 100 papers against same criteria
system_prompt = RELEVANCE_SCORING_SYSTEM  # Scoring criteria, format spec

user_prompts = [
    (paper["doi"], RELEVANCE_USER_TEMPLATE.format(
        topic=topic,
        research_questions=questions,
        title=paper["title"],
        abstract=paper["abstract"],
    ))
    for paper in papers
]

# All 100 papers benefit from cached system prompt
responses = await batch_invoke_with_cache(
    llm,
    system_prompt=system_prompt,
    user_prompts=user_prompts,
    max_concurrent=10,
)
```

### Example 3: Structured Output with Caching

```python
# workflows/shared/llm_utils/structured/executors/langchain.py

async def execute(
    self,
    output_schema: Type[T],
    user_prompt: str,
    system_prompt: Optional[str],
    output_config: StructuredOutputConfig,
) -> StructuredOutputResult[T]:
    llm = get_llm(tier=output_config.tier)
    structured_llm = llm.with_structured_output(output_schema)

    # Build messages with caching if enabled
    if system_prompt and output_config.enable_prompt_cache:
        messages = create_cached_messages(
            system_content=system_prompt,
            user_content=user_prompt,
            cache_ttl=output_config.cache_ttl,
        )
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    result = await structured_llm.ainvoke(messages)
    return StructuredOutputResult.ok(value=result)
```

### Example 4: Multi-Provider Support (DeepSeek)

```python
# DeepSeek uses automatic prefix-based caching
# Different mechanism, same interface

DEEPSEEK_CACHE_WARMUP_DELAY = 10.0  # seconds

async def invoke_with_cache(llm, system_prompt, user_prompt, ...):
    if _is_deepseek_model(llm):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # First call: make request, then wait for cache construction
        prefix_hash = hash(system_prompt)
        if prefix_hash not in _deepseek_cache_warmed:
            response = await llm.ainvoke(messages)
            _deepseek_cache_warmed[prefix_hash] = time.time()
            await asyncio.sleep(DEEPSEEK_CACHE_WARMUP_DELAY)
            return response

        return await llm.ainvoke(messages)

    # Anthropic: explicit cache_control (as shown above)
    ...
```

## Consequences

### Benefits

- **90% cost reduction** on cached tokens after first call
- **Consistent interface**: Same `invoke_with_cache()` works for Anthropic and DeepSeek
- **Transparent**: Caching happens automatically, no call-site changes needed
- **Monitoring**: Cache hit/miss logging for optimization verification
- **Flexible TTL**: 5-minute default, 1-hour option for long workflows

### Trade-offs

- **1024-token minimum**: System prompts under 1024 tokens cannot be cached
- **First-call overhead**: Initial call writes to cache (slightly slower)
- **Prompt design effort**: Must consciously separate static from dynamic content
- **TTL expiration**: Cache expires, requiring re-creation on long pauses

### Cost Analysis Example

Research supervisor workflow (12 iterations, 800-token system prompt):

| Approach | System Tokens | Cost Factor |
|----------|---------------|-------------|
| No caching | 9,600 (800 x 12) | 1.0x |
| With caching | 800 + 7,200 @ 10% | 0.15x |

**Result**: ~85% cost reduction on system prompt tokens.

## Related Patterns

- [Anthropic Claude Integration with Extended Thinking](./anthropic-claude-extended-thinking.md) - Model tiers and thinking budgets
- [Deep Research Workflow Architecture](../langgraph/deep-research-workflow-architecture.md) - Iterative supervisor pattern

## Known Uses in Thala

- `workflows/shared/llm_utils/caching.py`: Core caching utilities
- `workflows/research/web_research/prompts/supervision.py`: Supervisor cached prompts
- `workflows/research/web_research/prompts/compression.py`: Research compression prompts
- `workflows/research/web_research/nodes/supervisor/core.py`: Supervisor using caching
- `workflows/research/academic_lit_review/utils/relevance_scoring/scorer.py`: Batch scoring with cache
- `workflows/shared/llm_utils/structured/executors/langchain.py`: Structured output with caching

## References

- [Anthropic Prompt Caching Documentation](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [Anthropic Pricing (Input Token Costs)](https://www.anthropic.com/pricing)
- [DeepSeek Context Caching](https://api-docs.deepseek.com/guides/kv_cache)
