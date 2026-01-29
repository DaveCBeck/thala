---
name: workflow-modularization-pattern
title: Workflow Modularization Pattern
date: 2026-01-08
category: langgraph
shared: true
gist_url: https://gist.github.com/DaveCBeck/6507c7282c628a5bc468e118bba92f31
article_path: .context/libs/thala-dev/content/2026-01-08-workflow-modularization-pattern-langgraph.md
applicability:
  - "Workflows that can be used independently or as subgraphs"
  - "Separating core functionality from optional enhancement phases"
  - "Promoting subgraphs to top-level workflows when they gain independence"
  - "Creating wrapper workflows that compose existing workflows"
components: [academic_lit_review, supervised_lit_review, book_finding]
complexity: medium
verified_in_production: true
tags: [modularization, refactoring, composition, wrapper-workflow, standalone]
---

# Workflow Modularization Pattern

## Intent

Restructure workflows to separate core functionality from optional enhancement phases, allowing workflows to be used independently or composed with additional processing layers.

## Problem

Monolithic workflows that combine core logic with optional phases have issues:
- Cannot use core functionality without enhancement phases
- Hard to test core workflow independently
- Tight coupling between phases
- Unclear dependency direction for subgraphs

Example: Literature review workflow that always runs supervision loops, even when quick iterations are needed.

## Solution

Restructure into composable workflows:
1. **Core workflow**: Completes a useful unit of work independently
2. **Wrapper workflow**: Composes core workflow with additional phases
3. **Promote independent subgraphs**: Move to top-level when they don't depend on parent

### Before: Monolithic

```
workflows/research/subgraphs/
├── academic_lit_review/      # Has supervision phase embedded
│   ├── graph/
│   │   └── phases/
│   │       └── supervision.py  # Tightly coupled
│   └── supervision/
├── book_finding/             # Actually independent
└── web_researcher/           # True subgraph
```

### After: Modular

```
workflows/
├── academic_lit_review/      # Core workflow (ends at synthesis)
│   └── graph/
│       └── phases/
│           └── synthesis.py  # No supervision
├── supervised_lit_review/    # Wrapper workflow
│   ├── api.py                # Calls academic_lit_review + supervision
│   └── supervision/          # Supervision loops moved here
├── book_finding/             # Promoted to top-level
└── research/
    └── subgraphs/
        └── web_researcher/   # Remains as true subgraph
```

## Implementation

### Core Workflow (Ends at Useful Output)

```python
# workflows/academic_lit_review/graph/construction.py

def create_lit_review_graph() -> StateGraph:
    """Create literature review workflow.

    Ends at synthesis phase with complete review.
    Supervision is optional and handled by wrapper.
    """
    builder = StateGraph(AcademicLitReviewState)

    # Phases
    builder.add_node("discovery", discovery_phase_node)
    builder.add_node("diffusion", diffusion_phase_node)
    builder.add_node("processing", processing_phase_node)
    builder.add_node("clustering", clustering_phase_node)
    builder.add_node("synthesis", synthesis_phase_node)

    # Flow
    builder.add_edge(START, "discovery")
    builder.add_edge("discovery", "diffusion")
    builder.add_edge("diffusion", "processing")
    builder.add_edge("processing", "clustering")
    builder.add_edge("clustering", "synthesis")
    builder.add_edge("synthesis", END)  # Ends here

    return builder.compile()
```

### Wrapper Workflow (Adds Enhancement)

```python
# workflows/supervised_lit_review/api.py

from workflows.academic_lit_review import academic_lit_review
from workflows.supervised_lit_review.supervision import run_supervision_loops


async def supervised_lit_review(
    topic: str,
    research_questions: list[str],
    quality: str = "standard",
    language: str = "en",
) -> dict:
    """Run literature review with full supervision loops.

    Composes:
    1. Core academic_lit_review workflow
    2. Supervision loops 1-5 for quality enhancement
    """
    # Run core workflow
    lit_review_result = await academic_lit_review(
        topic=topic,
        research_questions=research_questions,
        quality=quality,
        language=language,
    )

    # Skip supervision if quick mode with explicit skip
    quality_settings = lit_review_result.get("quality_settings", {})
    if quality_settings.get("supervision_loops") == "none":
        return lit_review_result

    # Run supervision loops
    supervised_result = await run_supervision_loops(
        final_review=lit_review_result["final_review"],
        paper_corpus=lit_review_result["paper_corpus"],
        paper_summaries=lit_review_result["paper_summaries"],
        clusters=lit_review_result["clusters"],
        quality_settings=quality_settings,
        input_data=lit_review_result["input"],
        zotero_keys=lit_review_result.get("zotero_keys", {}),
    )

    return {
        **lit_review_result,
        "final_review": supervised_result["final_review_v2"],
        "supervision_state": supervised_result,
    }
```

### Promoting Subgraphs

When a subgraph becomes independent:

1. **Move to top-level**: `research/subgraphs/X/` → `workflows/X/`
2. **Update imports across codebase**: All references to the old path
3. **Expose clean public API**: `__init__.py` with explicit exports
4. **Remove parent dependencies**: No imports from former parent

```python
# workflows/book_finding/__init__.py

"""Standalone book finding workflow.

Public API:
    book_finding(theme, quality, language) -> BookFindingResult
"""

from workflows.book_finding.graph.api import book_finding
from workflows.book_finding.state import BookResult, BookFindingState

__all__ = [
    "book_finding",
    "BookResult",
    "BookFindingState",
]
```

### Import Updates

When restructuring, update all imports:

```python
# Before
from workflows.research.subgraphs.academic_lit_review import academic_lit_review
from workflows.research.subgraphs.book_finding import book_finding

# After
from workflows.academic_lit_review import academic_lit_review
from workflows.book_finding import book_finding
```

## When to Restructure

| Signal | Action |
|--------|--------|
| Subgraph used without parent | Promote to top-level |
| Optional phases added to core | Extract to wrapper |
| Import cycles between subgraphs | Identify true dependencies |
| Testing requires full parent | Core workflow too coupled |

## Guidelines

### Core Workflow Design

- Ends at a **useful** output (not intermediate state)
- Has **no dependencies** on optional phases
- Can be **tested independently**
- Exposes **clean public API**

### Wrapper Workflow Design

- **Composes** core workflow (doesn't duplicate)
- Adds **optional** enhancement phases
- Can **skip** enhancement based on config
- Returns **superset** of core result

### Subgraph vs Top-Level

| Aspect | Subgraph | Top-Level Workflow |
|--------|----------|-------------------|
| Used by | Single parent | Multiple callers |
| Dependencies | On parent | Independent |
| Public API | Internal | Exported |
| Testing | With parent | Standalone |

## Files Modified

In this refactoring (~90 files):
- `workflows/academic_lit_review/` - Moved from `research/subgraphs/`
- `workflows/supervised_lit_review/` - New wrapper workflow
- `workflows/book_finding/` - Moved from `research/subgraphs/`
- `workflows/research/subgraphs/` - Retains only `web_researcher`
- All import references updated

## Known Uses

- `workflows/academic_lit_review/` - Core literature review
- `workflows/supervised_lit_review/` - Enhanced with supervision
- `workflows/book_finding/` - Standalone book discovery
- `workflows/wrapped/` - Multi-source research composer

## Consequences

### Benefits
- **Independent testing**: Core workflow testable alone
- **Flexible composition**: Use core or enhanced version
- **Clear dependencies**: No hidden coupling
- **Faster iterations**: Skip enhancement for quick tests

### Trade-offs
- **Import churn**: Many files need updating
- **Initial complexity**: More files to maintain
- **API surface**: More public exports to manage

## Related Patterns

- [Multi-Loop Supervision System](./multi-loop-supervision-system.md) - Enhancement phases
- [Multi-Source Research Orchestration](./multi-source-research-orchestration.md) - Workflow composition

## References

- [LangGraph Subgraphs](https://langchain-ai.github.io/langgraph/concepts/subgraphs/)
- [Composition Over Inheritance](https://en.wikipedia.org/wiki/Composition_over_inheritance)
