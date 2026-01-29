---
name: unified-quality-tier-system
title: "Unified Quality Tier System for Workflow Orchestration"
date: 2026-01-13
category: langgraph
shared: true
gist_url: https://gist.github.com/DaveCBeck/eb61d01748fc1b9e6fe33b82534e9812
article_path: .context/libs/thala-dev/content/2026-01-13-unified-quality-tier-system-langgraph.md
applicability:
  - "Multi-workflow systems needing consistent quality control"
  - "Applications offering user-selectable depth/speed tradeoffs"
  - "Workflows where quality affects resource consumption (API costs, time)"
  - "Systems requiring standardized return structures across workflows"
components: [quality_config, quality_presets, workflow_api, state_init]
complexity: simple
verified_in_production: true
related_solutions: []
tags: [quality-tiers, workflow-configuration, api-design, standardization, cost-control]
---

# Unified Quality Tier System for Workflow Orchestration

## Intent

Provide a standardized 5-tier quality system that offers consistent depth/speed tradeoffs across all workflows, with workflow-specific preset configurations and standardized return structures.

## Problem

Multi-workflow systems face quality configuration challenges:

1. **Inconsistent naming**: Different workflows use different tier names (e.g., "fast" vs "quick")
2. **Parameter confusion**: Users don't know what `max_iterations=4` means in terms of results
3. **No semantic meaning**: Numeric parameters don't convey quality expectations
4. **Cross-workflow orchestration**: Wrapper workflows can't easily pass quality settings
5. **Validation scattered**: Each workflow validates quality independently

## Solution

Implement a unified quality system with:
- Single `QualityTier` type shared across all workflows
- Descriptive tier names with clear time/scope expectations
- Workflow-specific preset mappings
- Standardized return structures

### Quality Tiers

```python
# workflows/shared/quality_config.py

from typing import Literal

QualityTier = Literal["test", "quick", "standard", "comprehensive", "high_quality"]

QUALITY_TIER_DESCRIPTIONS = {
    "test": "Minimal processing for testing (~1 min)",
    "quick": "Fast results with limited depth (~5 min)",
    "standard": "Balanced quality and speed (~15 min)",
    "comprehensive": "Thorough processing (~30 min)",
    "high_quality": "Maximum depth and quality (45+ min)",
}
```

| Tier | Use Case | Typical Duration |
|------|----------|------------------|
| `test` | CI/CD, quick validation | ~1 min |
| `quick` | Interactive exploration, demos | ~5 min |
| `standard` | Production use (default) | ~15 min |
| `comprehensive` | Important research tasks | ~30 min |
| `high_quality` | Publication-quality output | 45+ min |

## Implementation

### Step 1: Define Shared Quality Type

```python
# workflows/shared/quality_config.py

from typing import Literal

QualityTier = Literal["test", "quick", "standard", "comprehensive", "high_quality"]
```

### Step 2: Workflow-Specific Presets

Each workflow defines what quality means for its domain:

```python
# workflows/research/academic_lit_review/quality_presets.py

from typing import TypedDict

class QualitySettings(TypedDict):
    max_stages: int              # Diffusion engine stages
    max_papers: int              # Papers to process
    target_word_count: int       # Final review length
    min_citations_filter: int    # Citation threshold
    saturation_threshold: float  # Coverage delta for termination
    use_batch_api: bool          # Batch API preference
    recency_years: int           # Recent paper definition
    recency_quota: float         # Recent papers target %

QUALITY_PRESETS: dict[str, QualitySettings] = {
    "test": {
        "max_stages": 1,
        "max_papers": 5,
        "target_word_count": 500,
        "min_citations_filter": 0,
        "saturation_threshold": 0.5,
        "use_batch_api": True,
        "recency_years": 3,
        "recency_quota": 0.0,
    },
    "quick": {
        "max_stages": 2,
        "max_papers": 50,
        "target_word_count": 3000,
        ...
    },
    # ... other tiers
}
```

### Step 3: Standardized API Signatures

All workflow entry points use consistent parameter order:

```python
async def academic_lit_review(
    topic: str,                                          # Primary input
    research_questions: list[str],                       # Secondary input
    quality: QualityTier = "standard",                   # Quality tier
    language: str = "en",                                # Language
    date_range: Optional[tuple[int, int]] = None,        # Optional filters
) -> dict[str, Any]

async def deep_research(
    query: str,
    quality: QualityTier = "standard",
    max_iterations: int = None,                          # Override if needed
    language: str = None,
) -> dict

async def book_finding(
    theme: str,
    brief: Optional[str] = None,
    quality: QualityTier = "standard",
    language: str = "en",
) -> dict[str, Any]
```

### Step 4: Quality Resolution in Workflow

```python
# workflows/research/academic_lit_review/graph/api.py

async def academic_lit_review(
    topic: str,
    research_questions: list[str],
    quality: QualityTier = "standard",
    ...
) -> dict[str, Any]:
    # 1. Validate quality tier
    if quality not in QUALITY_PRESETS:
        logger.warning(f"Unknown quality '{quality}', using 'standard'")
        quality = "standard"

    # 2. Load preset settings
    quality_settings = QUALITY_PRESETS[quality].copy()

    # 3. Build initial state with settings
    initial_state = build_initial_state(input_data, quality_settings)

    # 4. Pass quality to LangSmith for observability
    result = await graph.ainvoke(
        initial_state,
        config={
            "run_name": f"lit_review:{topic[:30]}",
            "tags": [f"quality:{quality}"],
            "metadata": {"quality_tier": quality},
        },
    )

    return standardize_result(result)
```

### Step 5: Standardized Return Structure

All workflows return consistent structure:

```python
{
    "final_report": str,                                    # Main output
    "status": Literal["success", "partial", "failed"],      # Status enum
    "langsmith_run_id": str,                                # Tracing ID
    "errors": list[dict],                                   # Error log
    "source_count": int,                                    # Resources used
    "started_at": datetime,                                 # Start time
    "completed_at": datetime,                               # End time
}
```

### Step 6: Quality Mapping for Wrapper Workflows

Wrapper workflows pass quality through to sub-workflows:

```python
# workflows/wrappers/multi_lang/graph/api.py

async def multi_lang_research(
    topic: str,
    workflow: Literal["web", "academic", "books"] = "web",
    quality: QualityTier = "standard",
    ...
) -> MultiLangResult:
    # Quality passes through to the underlying workflow
    if workflow == "academic":
        result = await academic_lit_review(
            topic=topic,
            quality=quality,  # Pass through
            ...
        )
    elif workflow == "web":
        result = await deep_research(
            query=topic,
            quality=quality,  # Pass through
            ...
        )
```

## Design Decisions

### Why Not Numeric Tiers (1-5)?

```python
# ❌ BAD: Numeric tiers lack semantic meaning
quality: int = 3  # What does "3" mean?

# ✅ GOOD: Named tiers are self-documenting
quality: QualityTier = "standard"  # Clear expectation
```

### Why Workflow-Specific Presets?

Different workflows have different quality dimensions:
- **Academic**: Papers count, citation threshold, word count
- **Web research**: Iterations, recursion depth
- **Books**: Recommendations per category, model choice

A unified mapping would either be too generic or too complex.

### Why Default to "standard"?

- "standard" balances quality and speed for most use cases
- Users can opt up (`comprehensive`, `high_quality`) for important tasks
- Users can opt down (`quick`, `test`) for exploration or CI/CD

## Consequences

### Benefits

- **Clear semantics**: "high_quality" conveys intent better than `max_papers=300`
- **Consistent API**: All workflows use same quality parameter
- **Easy orchestration**: Wrapper workflows pass quality through transparently
- **Observability**: Quality tier visible in LangSmith tags/metadata
- **Testability**: `test` tier enables fast CI/CD runs

### Trade-offs

- **Preset rigidity**: Users can't customize individual parameters easily
- **Workflow maintenance**: Each workflow must maintain its preset table
- **Time estimates are approximate**: Actual duration depends on content/network

## Configuration Reference

See [Workflow Configuration Guide](/docs/workflow_config.md) for complete preset tables by workflow.

## Related Patterns

- [LangSmith Trace Identification](./langsmith-trace-identification.md) - Quality in tracing metadata
- [Multi-Source Research Orchestration](./multi-source-research-orchestration.md) - Quality mapping across workflows
- [Batch API Cost Optimization](../llm-interaction/batch-api-cost-optimization.md) - Cost implications

## Related Solutions

- [Quality Setting Propagation: max_papers](../../solutions/workflow-issues/quality-setting-propagation-max-papers.md) - Propagation fix

## Known Uses

- `workflows/shared/quality_config.py`: Unified `QualityTier` type
- `workflows/research/academic_lit_review/quality_presets.py`: Academic presets
- `workflows/research/web_research/graph/config.py`: Web research presets
- `workflows/research/book_finding/state.py`: Book finding presets
- `docs/workflow_config.md`: Complete configuration reference

## References

- [Python typing.Literal](https://docs.python.org/3/library/typing.html#typing.Literal)
- [LangSmith Tags and Metadata](https://docs.smith.langchain.com/concepts/tracing)
