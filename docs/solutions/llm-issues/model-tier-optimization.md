---
module: workflows/shared/llm_utils, workflows/enhance, workflows/research
date: 2026-01-28
problem_type: cost_optimization
component: model_selection
symptoms:
  - "High LLM costs (~$2,000/month) for tasks with varying complexity"
  - "Simple classification tasks using expensive Claude models"
  - "Tool-using tasks using Sonnet instead of Haiku"
  - "No systematic framework for tier selection"
root_cause: "All tasks defaulted to expensive Claude models without analyzing actual complexity requirements"
resolution_type: configuration
severity: medium
verified_fix: true
tags: [cost-optimization, model-selection, deepseek, haiku, tier-assignment, batch-api]
---

# Model Tier Optimization

## Problem

LLM costs were ~$2,000/month with all tasks using expensive Claude models regardless of actual complexity requirements.

**Cost breakdown before optimization:**
```
Monthly LLM Cost:
├── Classification (100K calls)        → Claude Haiku: $500
├── Relevance scoring (50K calls)      → Claude Haiku: $250
├── Query generation (30K calls)       → Claude Haiku: $150
├── Verification/finalization          → Claude Sonnet: $300
├── Tool-using tasks (fact-check)      → Claude Sonnet: $400
├── Quality-critical (synthesis)       → Claude Opus: $400
                                        Total: ~$2,000/month
```

**The problem:** Simple structured output tasks (classification, scoring, query generation) were using Claude models when DeepSeek V3 at 10-15x cheaper could handle them. Tool-using tasks were using Sonnet when Haiku with tools is sufficient.

## Solution

Implemented a systematic model tier optimization framework that matches task complexity to appropriate model tier.

### Tier Selection Decision Framework

```
┌──────────────────────────────────────────────────────────────┐
│  Task Complexity Analysis                                     │
│                                                              │
│  1. Is quality critical? (synthesis, final output)           │
│     → YES: Use OPUS/SONNET (no migration)                    │
│     → NO: Continue to step 2                                 │
│                                                              │
│  2. Does task require tools?                                 │
│     → YES: Use HAIKU (reliable tool calling)                 │
│     → NO: Continue to step 3                                 │
│                                                              │
│  3. Is it simple structured output?                          │
│     → YES: Use DEEPSEEK_V3 (10-15x cheaper)                  │
│     → NO: Use HAIKU (balanced cost/quality)                  │
└──────────────────────────────────────────────────────────────┘
```

### Migration Categories

**1. Migrate to DEEPSEEK_V3 (80% cost reduction):**
- Content classification (full_text/abstract/paywall/non_academic)
- Relevance scoring (0-1 numeric output)
- Query generation and validation
- Polish/fact-check screening
- Structure verification and finalization

**2. Downgrade SONNET → HAIKU (67% cost reduction):**
- Fact-checking with tools
- Reference validation with tools
- Content generation (edits, refinements)
- Quality checks and research targets

**3. Keep OPUS/SONNET (quality-critical):**
- Complex synthesis and integration
- Multi-round supervision loops
- Final document generation
- Extended thinking tasks

### Implementation Changes

**Before (expensive):**
```python
# classifier.py - Using Haiku for simple classification
CLASSIFIER_MODEL = "claude-haiku-4-5-20251001"
response = await client.messages.create(
    model=CLASSIFIER_MODEL,
    tools=[...],  # Complex tool schema
    tool_choice={"type": "tool", "name": "classify_content"},
)
```

**After (optimized):**
```python
# classifier.py - Using DeepSeek V3 with structured output
from workflows.shared.llm_utils import ModelTier, get_structured_output

result: ClassificationResult = await get_structured_output(
    output_schema=ClassificationResult,
    user_prompt=user_prompt,
    system_prompt=CLASSIFICATION_SYSTEM_PROMPT,
    tier=ModelTier.DEEPSEEK_V3,  # 10x cheaper
    max_tokens=1024,
)
```

**Tool-using tasks (SONNET → HAIKU):**
```python
# fact_check.py
result = await get_structured_output(
    output_schema=FactCheckResult,
    user_prompt=user_prompt,
    system_prompt=FACT_CHECK_SYSTEM,
    tier=ModelTier.HAIKU,  # Was SONNET, 67% savings
    tools=tools,
    max_tokens=4000,
)
```

### Batch API Handling for DeepSeek

DeepSeek doesn't support Anthropic's Batch API, so automatic disabling was added:

```python
# scorer.py
from workflows.shared.llm_utils.models import is_deepseek_tier

# Disable batch API for DeepSeek models (no batch API available)
effective_use_batch = use_batch_api and not is_deepseek_tier(tier)

if effective_use_batch and len(papers) >= 5:
    return await _batch_score_relevance_batched(...)  # Claude only

# DeepSeek uses prefix caching instead of batch API
if is_deepseek_tier(tier):
    return await _batch_score_deepseek_cached(...)  # 90% cache savings
```

### Cost Impact

**After optimization:**
```
Monthly LLM Cost:
├── Classification (100K calls)        → DeepSeek V3: $28    (94% savings)
├── Relevance scoring (50K calls)      → DeepSeek V3: $14    (94% savings)
├── Query generation (30K calls)       → DeepSeek V3: $8     (95% savings)
├── Verification/finalization          → DeepSeek V3: $20    (93% savings)
├── Tool-using tasks (fact-check)      → Claude Haiku: $135  (67% savings)
├── Quality-critical (synthesis)       → Claude Opus: $400   (no change)
                                        Total: ~$605/month (70% savings)
```

## Files Modified

- `core/scraping/classification/classifier.py` - Migrate to DeepSeek V3
- `workflows/enhance/editing/nodes/execute_edits.py` - SONNET → HAIKU
- `workflows/enhance/editing/nodes/finalize.py` - SONNET → DEEPSEEK_V3
- `workflows/enhance/editing/nodes/polish.py` - HAIKU → DEEPSEEK_V3
- `workflows/enhance/editing/nodes/verify_structure.py` - SONNET → DEEPSEEK_V3
- `workflows/enhance/fact_check/nodes/fact_check.py` - SONNET → HAIKU
- `workflows/enhance/fact_check/nodes/reference_check.py` - SONNET → HAIKU
- `workflows/enhance/fact_check/nodes/screen_sections.py` - HAIKU → DEEPSEEK_V3
- `workflows/research/academic_lit_review/citation_network/scoring.py` - HAIKU → DEEPSEEK_V3
- `workflows/research/academic_lit_review/diffusion_engine/relevance_filters.py` - HAIKU → DEEPSEEK_V3
- `workflows/research/academic_lit_review/keyword_search/query_builder.py` - HAIKU → DEEPSEEK_V3
- `workflows/research/academic_lit_review/keyword_search/searcher.py` - HAIKU → DEEPSEEK_V3
- `workflows/research/academic_lit_review/utils/relevance_scoring/scorer.py` - Add batch API disabling
- `workflows/research/web_research/nodes/clarify_intent.py` - HAIKU → DEEPSEEK_V3
- `workflows/research/web_research/nodes/create_brief.py` - SONNET → DEEPSEEK_V3
- `workflows/research/web_research/subgraphs/researcher_base/query_generator.py` - HAIKU → DEEPSEEK_V3
- `workflows/research/web_research/subgraphs/researcher_base/query_validator.py` - HAIKU → DEEPSEEK_V3
- `workflows/research/web_research/subgraphs/web_researcher.py` - SONNET → HAIKU
- `workflows/wrappers/multi_lang/nodes/relevance_checker.py` - HAIKU → DEEPSEEK_V3
- `workflows/wrappers/synthesis/nodes/quality_check.py` - SONNET → HAIKU
- `workflows/wrappers/synthesis/nodes/research_targets.py` - SONNET → HAIKU
- `docs/models.md` - Update tier assignments documentation

## Prevention

1. **Use tier selection framework** - Always analyze task complexity before choosing model
2. **Default to cheapest viable tier** - Start with DEEPSEEK_V3, upgrade only if needed
3. **Monitor quality metrics** - Track accuracy/quality to catch regressions
4. **Document tier rationale** - Add comments explaining tier choice in code

## Task Classification Checklist

| Characteristic | DEEPSEEK_V3 | HAIKU | SONNET/OPUS |
|----------------|-------------|-------|-------------|
| Structured output only | ✓ | ✓ | ✓ |
| Requires tools | ✗ | ✓ | ✓ |
| Quality-critical | ✗ | ✗ | ✓ |
| Extended thinking | ✗ | ✗ | OPUS only |
| High volume (1000+) | ✓ | Maybe | ✗ |
| Accuracy tolerance | 85-90% | 90-95% | 95%+ |

## Related Patterns

- [DeepSeek Integration Patterns](../../patterns/llm-interaction/deepseek-integration-patterns.md) - V3/R1 model selection and caching
- [Batch API Cost Optimization](../../patterns/llm-interaction/batch-api-cost-optimization.md) - 50% savings for Claude batching
- [Anthropic Prompt Caching](../../patterns/llm-interaction/anthropic-prompt-caching-cost-optimization.md) - 90% savings on cached prompts
- [Unified Quality Tier System](../../patterns/langgraph/unified-quality-tier-system.md) - Quality tier → model tier mapping

## References

- Commit: af157d0bb7a1cb55ecd0b753eee15ff369150d32
- [DeepSeek API Pricing](https://platform.deepseek.com/api-docs/pricing)
- [Anthropic Pricing](https://docs.anthropic.com/en/docs/about-claude/pricing)
