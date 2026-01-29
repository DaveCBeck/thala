---
name: anthropic-prompt-caching-cost-optimization
title: "Anthropic Prompt Caching for 90% Cost Reduction"
date: 2025-12-22
category: llm-interaction
applicability:
  - "When iterative workflows repeat the same system prompt across multiple LLM calls"
  - "When batch processing documents with identical extraction instructions"
  - "When system prompts are >= 400 tokens and called frequently"
components: [llm_call, workflow_graph]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [anthropic, claude, prompt-caching, cost-optimization, cache-control, ephemeral, input-tokens, langchain]
---

# Anthropic Prompt Caching for 90% Cost Reduction

## Intent

Reduce LLM API costs by up to 90% on input tokens by leveraging Anthropic's prompt caching feature, which allows marking content blocks as cacheable via `cache_control: {type: "ephemeral"}`.

## Motivation

LLM API costs scale with input tokens. In iterative workflows (research loops, multi-document processing, agentic systems), the same system prompts are sent repeatedly on every call:

```python
# BEFORE: Every iteration pays full price for 800-token system prompt
for iteration in range(12):
    prompt = f"""You are a research supervisor...
    <Diffusion Algorithm>...</Diffusion Algorithm>
    <Tools>...</Tools>
    Today: {date}
    Context: {context}
    """
    response = await llm.ainvoke([{"role": "user", "content": prompt}])
```

With 12 iterations of an 800-token system prompt at Opus pricing ($15/MTok), that's $0.14 in input tokens alone—just for the instructions that never change.

**Prompt caching solves this:**
- Cache reads cost 10% of base input token price
- Cache writes cost 25% premium (amortized over subsequent hits)
- Break-even on 2nd call; 85%+ savings on batch workloads

## Applicability

Use this pattern when:
- Iterative workflows repeat system prompts (research loops, agent iterations)
- Batch processing uses identical instructions (document extraction pipelines)
- System prompts are substantial (>= 400 tokens where savings are meaningful)
- Calls occur within cache TTL window (5 minutes default, refreshes on hit)

Do NOT use this pattern when:
- One-shot calls with no reuse opportunity
- System prompts are highly dynamic (change every call)
- Using non-Anthropic models (pattern is Claude-specific)
- System prompts are very short (< 200 tokens, minimal absolute savings)

## Structure

```
┌──────────────────────────────────────────────────────────────┐
│               Prompt Caching Architecture                     │
└──────────────────────────────────────────────────────────────┘

┌─────────────────────┐     ┌─────────────────────┐
│   BEFORE (No Cache) │     │   AFTER (Cached)    │
└─────────────────────┘     └─────────────────────┘

 Combined Prompt              Separated Prompts
 ┌─────────────────┐         ┌─────────────────┐
 │ System + Data   │         │ System Prompt   │◄── cache_control: ephemeral
 │ (all dynamic)   │         │ (STATIC)        │    ~800 tokens, cached
 │                 │         ├─────────────────┤
 │ - Instructions  │         │ User Content    │◄── Dynamic data
 │ - Today's date  │         │ (DYNAMIC)       │    Changes each call
 │ - Context       │         │ - date, context │
 │ - Findings      │         │ - findings      │
 └─────────────────┘         └─────────────────┘

 Paid: 800 tokens × 12       Paid: 800 (write) + 8,800 × 10%
 = 9,600 tokens full         = 1,680 tokens effective
 = $0.144 (Opus)             = $0.025 (Opus) = 83% savings
```

## Implementation

### Step 1: Create Cached Message Utilities

Build messages with `cache_control` on system content:

```python
# workflows/shared/llm_utils.py

from typing import Literal

CacheTTL = Literal["5m", "1h"]


def create_cached_messages(
    system_content: str | list[dict],
    user_content: str,
    cache_system: bool = True,
    cache_ttl: CacheTTL = "5m",
) -> list[dict]:
    """
    Create messages with cache_control on system content.

    Prompt caching reduces input token costs by 90% on cache hits.
    The cache is automatically refreshed for 5 minutes (or 1 hour with extended TTL).

    Args:
        system_content: Static system prompt (will be cached)
        user_content: Dynamic user message content
        cache_system: Whether to apply cache_control (default: True)
        cache_ttl: Cache lifetime - "5m" (free refresh) or "1h" (2x write cost)

    Returns:
        List of message dicts ready for llm.invoke()
    """
    cache_control = {"type": "ephemeral"}
    if cache_ttl == "1h":
        cache_control["ttl"] = "1h"

    # Build system message content blocks with cache_control
    if isinstance(system_content, str):
        system_blocks = [
            {
                "type": "text",
                "text": system_content,
                **({"cache_control": cache_control} if cache_system else {}),
            }
        ]
    else:
        # Already a list of content blocks - add cache_control to last block
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

### Step 2: Create Convenience Wrapper with Cache Logging

```python
# workflows/shared/llm_utils.py

import logging

logger = logging.getLogger(__name__)


async def invoke_with_cache(
    llm: ChatAnthropic,
    system_prompt: str,
    user_prompt: str,
    cache_system: bool = True,
    cache_ttl: CacheTTL = "5m",
) -> Any:
    """
    Invoke LLM with prompt caching on system message.

    This convenience wrapper:
    1. Creates properly structured messages with cache_control
    2. Invokes the LLM
    3. Logs cache hit/miss statistics

    Args:
        llm: ChatAnthropic instance
        system_prompt: Static system instructions (will be cached)
        user_prompt: Dynamic user content
        cache_system: Whether to cache the system prompt (default: True)
        cache_ttl: Cache lifetime - "5m" or "1h"

    Returns:
        LLM response
    """
    messages = create_cached_messages(
        system_content=system_prompt,
        user_content=user_prompt,
        cache_system=cache_system,
        cache_ttl=cache_ttl,
    )

    response = await llm.ainvoke(messages)

    # Log cache statistics if available
    usage = getattr(response, "usage_metadata", None)
    if usage:
        details = usage.get("input_token_details", {})
        cache_read = details.get("cache_read", 0)
        cache_creation = details.get("cache_creation", 0)
        if cache_read > 0:
            logger.debug(f"Cache HIT: {cache_read} tokens read from cache")
        elif cache_creation > 0:
            logger.debug(f"Cache MISS: {cache_creation} tokens written to cache")

    return response
```

### Step 3: Separate Static and Dynamic Prompt Content

The key pattern is splitting prompts into cacheable static instructions and dynamic data:

```python
# workflows/research/prompts.py

# ============================================================================
# BEFORE: Combined prompt (everything rebuilt every call)
# ============================================================================
SUPERVISOR_OLD = """You are the lead researcher coordinating a deep research project.

Today's date is {date}.

<Diffusion Algorithm>
1. Generate research questions that expand understanding
2. Delegate research via ConductResearch tool
3. Refine the draft report with RefineDraftReport
4. Check completeness - identify remaining gaps
5. Either generate more questions or call ResearchComplete
</Diffusion Algorithm>

<Available Tools>
1. **ConductResearch**: Delegate research question to a sub-agent
2. **RefineDraftReport**: Update the draft report with new findings
3. **ResearchComplete**: Signal that research is complete
</Available Tools>

<Research Brief>
{research_brief}
</Research Brief>

<Current Draft>
{draft_report}
</Current Draft>

<Findings So Far>
{findings_summary}
</Findings So Far>

Iteration: {iteration}/{max_iterations}
Completeness: {completeness_score}%

Decide your next action."""


# ============================================================================
# AFTER: Static system prompt (cached) + Dynamic user template
# ============================================================================

# Static system prompt - ~800 tokens, saves 90% on cache hits
SUPERVISOR_SYSTEM_CACHED = """You are the lead researcher coordinating a deep research project using the diffusion algorithm.

<Diffusion Algorithm>
1. Generate research questions that expand understanding (diffusion out)
2. Delegate research via ConductResearch tool to gather concrete findings
3. Refine the draft report with RefineDraftReport to reduce noise/gaps
4. Check completeness - identify remaining gaps
5. Either generate more questions or call ResearchComplete
</Diffusion Algorithm>

<Available Tools>
1. **ConductResearch**: Delegate a specific research question to a sub-agent. Provide detailed context.
2. **RefineDraftReport**: Update the draft report with new findings. Specify updates and remaining gaps.
3. **ResearchComplete**: Signal that research is complete. Only use when findings are comprehensive.
</Available Tools>

<Instructions>
Think like a research manager:
1. Read the research brief and customized plan carefully
2. Consider what the user already knows (don't duplicate)
3. Assess current findings - are there gaps?
4. Decide: delegate more research OR refine draft OR complete

CRITICAL: Never include operational metadata (iteration counts, percentages,
completeness scores, or internal state) in research questions or topics.
Focus purely on the actual research subject matter.

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


# Dynamic user prompt - changes each iteration
SUPERVISOR_USER_TEMPLATE = """Today's date is {date}.

<Research Brief>
{research_brief}
</Research Brief>

<Customized Plan (based on user's existing knowledge)>
{research_plan}
</Customized Plan>

<Memory Context (what the user already knows)>
{memory_context}
</Memory Context>

<Current Draft Report>
{draft_report}
</Current Draft Report>

<Research Findings So Far>
{findings_summary}
</Research Findings So Far>

<Operational Metadata - Internal tracking only, DO NOT reference in research questions>
<!-- These are internal state values for workflow coordination, not research topics -->
- Iteration: {iteration}/{max_iterations}
- Completeness: {completeness_score}%
- Areas explored: {areas_explored}
- Gaps remaining: {gaps_remaining}
<!-- End internal state -->
</Operational Metadata>

Based on the above context, decide your next action."""
```

### Step 4: Update Node to Use Cached Invocation

```python
# workflows/research/nodes/supervisor.py

from workflows.research.prompts import (
    SUPERVISOR_SYSTEM_CACHED,
    SUPERVISOR_USER_TEMPLATE,
    get_today_str,
)
from workflows.shared.llm_utils import ModelTier, get_llm, invoke_with_cache


async def supervisor(state: DeepResearchState) -> dict[str, Any]:
    """Supervisor node with prompt caching for cost efficiency."""
    llm = get_llm(ModelTier.OPUS, max_tokens=4096)
    llm_with_tools = llm.bind_tools(SUPERVISOR_TOOLS)

    # Build dynamic user content
    user_prompt = SUPERVISOR_USER_TEMPLATE.format(
        date=get_today_str(),
        research_brief=format_brief(state["research_brief"]),
        research_plan=state.get("research_plan", ""),
        memory_context=state.get("memory_context", ""),
        draft_report=state.get("draft", {}).get("content", ""),
        findings_summary=format_findings(state.get("findings", [])),
        iteration=state.get("iteration", 1),
        max_iterations=state.get("max_iterations", 8),
        completeness_score=state.get("completeness_score", 0),
        areas_explored=", ".join(state.get("areas_explored", [])),
        gaps_remaining=", ".join(state.get("gaps_remaining", [])),
    )

    # Use cached invocation - system prompt cached, user content dynamic
    response = await invoke_with_cache(
        llm_with_tools,
        system_prompt=SUPERVISOR_SYSTEM_CACHED,  # ~800 tokens, cached
        user_prompt=user_prompt,  # Dynamic each iteration
    )

    return _process_supervisor_response(response, state)
```

### Step 5: Create Reusable Extraction Utilities

For common patterns like JSON extraction and summarization:

```python
# workflows/shared/llm_utils.py

async def extract_json_cached(
    text: str,
    system_instructions: str,
    schema_hint: Optional[str] = None,
    tier: ModelTier = ModelTier.SONNET,
) -> dict[str, Any]:
    """
    Extract structured JSON from text using Claude with prompt caching.

    The system instructions and schema are cached, so repeated extractions
    (e.g., processing multiple documents) benefit from ~90% cost reduction.

    Args:
        text: Text to extract from
        system_instructions: Instructions for extraction (will be cached)
        schema_hint: Optional JSON schema hint for expected output
        tier: Model tier to use (default: SONNET)

    Returns:
        Extracted data as dict
    """
    llm = get_llm(tier=tier)

    # Build cached system prompt
    system_prompt = system_instructions
    if schema_hint:
        system_prompt += f"\n\nExpected schema:\n{schema_hint}"
    system_prompt += "\n\nRespond with ONLY valid JSON, no other text."

    # Dynamic user content
    user_prompt = f"Extract from this text:\n\n{text}"

    response = await invoke_with_cache(
        llm,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    content = response.content if isinstance(response.content, str) else response.content[0].get("text", "")
    content = content.strip()

    # Handle markdown code blocks
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1])

    return json.loads(content)


async def summarize_text_cached(
    text: str,
    target_words: int = 100,
    context: Optional[str] = None,
    tier: ModelTier = ModelTier.SONNET,
) -> str:
    """
    Summarize text using Claude with prompt caching.

    The system instructions are cached, so repeated calls with different
    texts benefit from ~90% input token cost reduction.
    """
    llm = get_llm(tier=tier)

    # Static system prompt (cached)
    system_prompt = f"""You are a skilled summarizer. Create concise summaries that capture the essential information.

Target length: approximately {target_words} words.
{f"Context: {context}" if context else ""}

Guidelines:
- Focus on the main thesis, key arguments, and conclusions
- Preserve critical details and nuance
- Write in clear, professional prose"""

    # Dynamic user content
    user_prompt = f"Summarize the following text:\n\n{text}"

    response = await invoke_with_cache(
        llm,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    return response.content if isinstance(response.content, str) else response.content[0].get("text", "")
```

## Static vs Dynamic Content Guidelines

### What Goes in Static System Prompt (Cached)

| Content Type | Example | Rationale |
|--------------|---------|-----------|
| Role definition | "You are a research supervisor" | Never changes |
| Algorithm descriptions | `<Diffusion Algorithm>...</>` | Core instructions |
| Tool definitions | Available tools, their usage | Fixed per workflow |
| Output format rules | JSON schema, citation rules | Consistent |
| Guidelines/constraints | "Never include metadata in outputs" | Policy |

### What Goes in Dynamic User Content (Not Cached)

| Content Type | Example | Rationale |
|--------------|---------|-----------|
| Current date | `Today's date is 2025-12-22` | Changes daily |
| User data | Research brief, objectives | Per-session |
| Accumulated state | Findings, draft report | Changes each iteration |
| Iteration metadata | `Iteration: 3/8` | Dynamic |
| Session context | Memory, prior knowledge | Per-user |

## Consequences

### Benefits

- **90% cost reduction** on cached content (10% of base price on hits)
- **Predictable costs** for iterative workflows
- **No code complexity** beyond prompt separation
- **Automatic refresh** within TTL window (5 min default)
- **Observable** via `usage_metadata.input_token_details`

### Trade-offs

- **Anthropic-specific** - DeepSeek has automatic prefix caching (no code changes)
- **Cache write premium** - 25% extra on first write (amortized quickly)
- **TTL expiration** - Cache expires after 5 min of no hits (or 1h with extended TTL)
- **Prompt restructuring** - Requires separating static from dynamic content

### Cost Economics

```
Example: 12-iteration research workflow with 800-token system prompt

Without caching:
  12 × 800 = 9,600 input tokens at $15/MTok (Opus)
  Cost: $0.144

With caching:
  1 × 800 write @ 125% = 1,000 effective tokens
  11 × 800 hits @ 10% = 880 effective tokens
  Total: 1,880 effective tokens
  Cost: $0.028

Savings: 81%
```

## Async Considerations

- **Non-blocking calls**: Always use `ainvoke()`, never `invoke()`
- **Exponential backoff**: Retry with `asyncio.sleep()`, not `time.sleep()`
- **Concurrent batching**: Use semaphores for rate limit compliance

```python
async def summarize_batch(texts: list[str], max_concurrent: int = 5):
    """Summarize multiple texts with bounded concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_summarize(text: str):
        async with semaphore:
            return await summarize_text_cached(text)

    results = await asyncio.gather(
        *[bounded_summarize(t) for t in texts],
        return_exceptions=True
    )
    return [r for r in results if not isinstance(r, Exception)]
```

## Related Patterns

- [Anthropic Claude Integration with Extended Thinking](./anthropic-claude-extended-thinking.md) - Model tier selection and batch API
- [Conditional Development Tracing](./conditional-development-tracing.md) - LangSmith observability for cost monitoring
- [Deep Research Workflow Architecture](../langgraph/deep-research-workflow-architecture.md) - Uses prompt caching in supervisor

## Known Uses in Thala

- `workflows/shared/llm_utils.py`: Core caching utilities
- `workflows/research/prompts.py`: SUPERVISOR_SYSTEM_CACHED, COMPRESS_RESEARCH_SYSTEM_CACHED, FINAL_REPORT_SYSTEM_STATIC
- `workflows/research/nodes/supervisor.py`: Cached supervisor invocation
- `workflows/research/nodes/final_report.py`: Cached final report generation
- `workflows/research/subgraphs/researcher.py`: Cached compress_findings
- `workflows/document_processing/nodes/metadata_agent.py`: extract_json_cached
- `workflows/document_processing/nodes/summary_agent.py`: summarize_text_cached

## References

- [Anthropic Prompt Caching Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [LangChain Anthropic Integration](https://python.langchain.com/docs/integrations/chat/anthropic/)
