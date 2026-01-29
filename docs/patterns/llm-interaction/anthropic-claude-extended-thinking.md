---
name: anthropic-claude-extended-thinking
title: "Anthropic Claude Integration with Extended Thinking"
date: 2025-12-17
category: llm-interaction
applicability:
  - "LLM-powered document processing pipelines"
  - "Complex reasoning tasks requiring deep analysis"
  - "Batch processing with cost optimization needs"
components: [llm_call, structured_output, async_task]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [anthropic, claude, extended-thinking, batch-api, model-tier, langchain]
---

# Anthropic Claude Integration with Extended Thinking

## Intent

Provide a tiered model selection system with extended thinking support for complex reasoning tasks, and batch processing capability for 50% cost reduction on bulk LLM operations.

## Motivation

Document processing workflows require different LLM capabilities at different stages:
- Quick metadata extraction (fast, cheap)
- Standard summarization (balanced quality/cost)
- Complex chapter analysis (deep reasoning needed)
- Bulk processing (cost optimization critical)

This pattern establishes:
1. **ModelTier enum** for consistent model selection across the codebase
2. **Extended thinking** for complex reasoning that benefits from step-by-step analysis
3. **Batch API integration** for 50% cost reduction on asynchronous processing

## Applicability

Use this pattern when:
- Building LLM-powered pipelines with varying task complexity
- Need deep reasoning capability for complex analysis tasks
- Processing many documents where cost optimization matters
- Using Anthropic Claude models (Haiku/Sonnet/Opus)

Do NOT use this pattern when:
- Simple single-shot LLM calls suffice
- Real-time responses required (batch API is asynchronous)
- Using non-Anthropic models (pattern is Claude-specific)

## Structure

```
workflows/shared/
├── llm_utils.py         # ModelTier, get_llm(), analyze_with_thinking()
├── batch_processor.py   # BatchProcessor for Message Batches API
└── ...

workflows/document_processing/
├── nodes/
│   ├── summary_agent.py      # Uses Sonnet
│   ├── metadata_agent.py     # Uses Sonnet
│   └── ...
├── subgraphs/
│   └── chapter_summarization.py  # Uses Opus + thinking
└── batch_mode.py              # Batch API integration
```

## Implementation

### Step 1: Model Tier Enum

Define model tiers for task-appropriate selection:

```python
# workflows/shared/llm_utils.py
from enum import Enum

class ModelTier(Enum):
    """Model tiers for different task complexities.

    HAIKU: Quick tasks, simple text generation
    SONNET: Standard tasks, summarization, metadata extraction
    OPUS: Complex tasks requiring deep analysis (supports extended thinking)
    """
    HAIKU = "claude-haiku-4-5-20251001"
    SONNET = "claude-sonnet-4-5-20250929"
    OPUS = "claude-opus-4-5-20251101"
```

### Step 2: LLM Factory with Extended Thinking

Create a factory function that configures Claude with optional extended thinking:

```python
from langchain_anthropic import ChatAnthropic

def get_llm(
    tier: ModelTier = ModelTier.SONNET,
    thinking_budget: Optional[int] = None,
    max_tokens: int = 4096,
) -> ChatAnthropic:
    """
    Get a configured Anthropic Claude LLM instance.

    Args:
        tier: Model tier selection (HAIKU, SONNET, OPUS)
        thinking_budget: Token budget for extended thinking (enables if set).
                        Recommended: 8000-16000 for complex tasks.
        max_tokens: Maximum output tokens (must be > thinking_budget if set)

    Returns:
        ChatAnthropic instance configured for the specified tier
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    kwargs = {
        "model": tier.value,
        "api_key": api_key,
        "max_tokens": max_tokens,
    }

    if thinking_budget is not None:
        if thinking_budget >= max_tokens:
            raise ValueError(
                f"thinking_budget ({thinking_budget}) must be less than max_tokens"
            )
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }

    return ChatAnthropic(**kwargs)
```

### Step 3: Extended Thinking Helper

Provide a convenience function for complex analysis:

```python
async def analyze_with_thinking(
    text: str,
    prompt: str,
    thinking_budget: int = 8000,
    tier: ModelTier = ModelTier.OPUS,
) -> tuple[str, Optional[str]]:
    """
    Perform complex analysis using extended thinking.

    Extended thinking allows Claude to reason step-by-step before
    providing a final answer, improving quality for complex tasks.

    Args:
        text: Text to analyze
        prompt: Analysis instructions
        thinking_budget: Token budget for reasoning (default: 8000)
        tier: Model tier (default: OPUS for complex analysis)

    Returns:
        Tuple of (final_response, thinking_summary)
    """
    llm = get_llm(tier=tier, thinking_budget=thinking_budget, max_tokens=thinking_budget + 4096)

    full_prompt = f"{prompt}\n\nText:\n{text}"
    response = await llm.ainvoke([HumanMessage(content=full_prompt)])

    # Extract thinking and text content from response
    thinking_content = None
    text_content = ""

    if isinstance(response.content, list):
        for block in response.content:
            if hasattr(block, "type"):
                if block.type == "thinking":
                    thinking_content = getattr(block, "thinking", "")
                elif block.type == "text":
                    text_content = getattr(block, "text", "")
    else:
        text_content = response.content

    return text_content, thinking_content
```

### Step 4: Using Extended Thinking in Workflows

Apply to complex analysis nodes:

```python
# workflows/document_processing/subgraphs/chapter_summarization.py

async def summarize_chapter(state: ChapterSummaryState) -> dict:
    """
    Summarize a chapter using Opus with extended thinking.

    Extended thinking enables deeper analysis of chapter structure,
    arguments, and contributions to the broader work.
    """
    chapter = state["chapter"]
    chapter_content = state["chapter_content"]
    target_words = state["target_words"]

    # Build context
    context = f"Chapter: {chapter['title']}"
    if chapter.get("author"):
        context += f" (by {chapter['author']})"

    prompt = f"""Summarize this chapter in approximately {target_words} words.

Context: {context}

Focus on:
- The main arguments and thesis of the chapter
- Key concepts and findings
- How this chapter contributes to the broader work
- Any significant conclusions or implications"""

    # Use Opus with extended thinking for deep analysis
    summary, thinking = await analyze_with_thinking(
        text=chapter_content,
        prompt=prompt,
        thinking_budget=8000,
        tier=ModelTier.OPUS,
    )

    logger.info(f"Summarized chapter '{chapter['title']}' to {len(summary.split())} words")
    if thinking:
        logger.debug(f"Thinking: {thinking[:200]}...")

    return {"summary": summary}
```

### Step 5: Batch API for Cost Optimization

For bulk processing, use the Message Batches API:

```python
# workflows/shared/batch_processor.py

from anthropic import AsyncAnthropic

class BatchProcessor:
    """Processor for Anthropic Message Batches API.

    Collects LLM requests and submits them as a batch for 50% cost reduction.
    Results typically available within 1 hour.
    """

    def __init__(self, poll_interval: int = 60):
        self.async_client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.poll_interval = poll_interval
        self.pending_requests: list[BatchRequest] = []

    def add_request(
        self,
        custom_id: str,
        prompt: str,
        model: ModelTier = ModelTier.SONNET,
        max_tokens: int = 4096,
        thinking_budget: Optional[int] = None,
    ) -> None:
        """Add a request to the pending batch."""
        self.pending_requests.append(BatchRequest(
            custom_id=custom_id,
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
        ))

    async def execute_batch(self) -> dict[str, BatchResult]:
        """Submit batch and poll until complete."""
        # Build batch requests in Anthropic format
        requests = [self._build_request(r) for r in self.pending_requests]

        # Submit batch
        batch = await self.async_client.messages.batches.create(requests=requests)

        # Poll for completion
        while batch.processing_status != "ended":
            await asyncio.sleep(self.poll_interval)
            batch = await self.async_client.messages.batches.retrieve(batch.id)

        # Collect results
        results = {}
        async for result in self.async_client.messages.batches.results(batch.id):
            results[result.custom_id] = self._parse_result(result)

        self.pending_requests.clear()
        return results
```

## Complete Example

Processing multiple documents with batch API:

```python
from workflows.document_processing.batch_mode import process_documents_with_batch_api

# Process multiple documents with 50% cost savings
results = await process_documents_with_batch_api([
    {"id": "doc1", "content": "Full document 1 text...", "title": "Document 1"},
    {"id": "doc2", "content": "Full document 2 text...", "title": "Document 2"},
], include_metadata=True, include_chapter_summaries=True)

for doc_id, result in results.items():
    print(f"{doc_id}: {result.summary[:100]}...")
    if result.chapter_summaries:
        for chapter in result.chapter_summaries:
            print(f"  - {chapter['title']}: {chapter['summary'][:50]}...")
```

## Consequences

### Benefits

- **Task-appropriate models**: Use cheaper models for simple tasks, powerful models for complex ones
- **Extended thinking**: Improved quality for complex reasoning without prompt engineering
- **50% cost reduction**: Batch API significantly reduces costs for bulk processing
- **Consistent interface**: Same `get_llm()` pattern throughout codebase

### Trade-offs

- **Thinking token overhead**: Extended thinking adds tokens to responses
- **Batch latency**: Batch API is asynchronous (typically 1 hour completion)
- **Claude-specific**: Pattern tied to Anthropic's API features
- **max_tokens constraint**: Must be greater than thinking_budget

### Alternatives

- **OpenAI models**: Different thinking approaches (o1 has internal reasoning)
- **Without thinking**: Standard prompting for simpler tasks
- **Streaming**: For real-time response needs (incompatible with batch)

## Related Patterns

- [Anthropic Prompt Caching](./anthropic-prompt-caching-cost-optimization.md) - 90% cost reduction on repeated system prompts
- [GPU-Accelerated Document Processing](../data-pipeline/gpu-accelerated-document-processing.md) - Document extraction preceding LLM analysis
- [Centralized Environment Configuration](../stores/centralized-env-config.md) - ANTHROPIC_API_KEY configuration

## Known Uses in Thala

- `workflows/shared/llm_utils.py`: ModelTier enum and get_llm() factory
- `workflows/shared/batch_processor.py`: BatchProcessor for Message Batches API
- `workflows/document_processing/nodes/summary_agent.py`: Sonnet for standard summarization
- `workflows/document_processing/subgraphs/chapter_summarization.py`: Opus + thinking for complex chapter analysis
- `workflows/document_processing/batch_mode.py`: Batch document processing

## References

- [Anthropic Extended Thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)
- [Anthropic Message Batches API](https://docs.anthropic.com/en/docs/build-with-claude/message-batches)
- [LangChain Anthropic Integration](https://python.langchain.com/docs/integrations/chat/anthropic/)
