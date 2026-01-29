---
name: iterative-theoretical-depth-supervision
title: Iterative Theoretical Depth via Supervision
date: 2026-01-03
category: langgraph
shared: true
gist_url: https://gist.github.com/DaveCBeck/df5714ca24e9fd4d0c56c55345e77951
article_path: .context/libs/thala-dev/content/2026-01-03-iterative-theoretical-depth-supervision-langgraph.md
applicability:
  - "Long-form documents needing iterative improvement"
  - "Literature reviews requiring theoretical grounding verification"
  - "Content quality assurance with targeted expansion"
  - "Opus-powered analysis guiding targeted research"
components: [supervision, analyze_review, expand_topic, integrate_content]
complexity: high
verified_in_production: false
deprecated: true
deprecated_date: 2026-01-29
deprecated_reason: "Supervision architecture restructured; now at workflows/enhance/supervision/ with different loop structure"
superseded_by: "./multi-loop-supervision-system.md"
tags: [supervision, opus, iteration, quality-assurance, literature-review, extended-thinking]
---

> **DEPRECATED**: This documentation describes a superseded architecture.
>
> **Reason:** Supervision was restructured into `workflows/enhance/supervision/` with a different multi-loop approach. The file paths and node structure described here no longer exist.
> **Date:** 2026-01-29
> **See instead:** [Multi-Loop Supervision System](./multi-loop-supervision-system.md) (note: that doc also needs updates for current implementation)

# Iterative Theoretical Depth via Supervision

## Intent

Iteratively improve document quality through an Opus-powered supervisor that identifies theoretical gaps, triggers focused research expansion, and integrates findings until quality thresholds are met or iteration limits reached.

## Problem

Long-form academic documents (literature reviews) often have uneven theoretical depth:
- Some sections well-grounded, others superficial
- Citations may miss foundational works
- Theoretical frameworks may be incomplete
- Quality varies based on initial corpus coverage

Single-pass generation cannot catch these gaps. Manual review is time-consuming and inconsistent.

## Solution

Implement a supervision loop as a post-processing phase:
1. **Analyze**: Opus examines document for theoretical gaps (one issue per iteration)
2. **Decide**: Either identify a specific gap to address, or pass through (approve)
3. **Expand**: If gap found, run targeted research (discovery → diffusion → processing)
4. **Integrate**: Merge new findings into document with full restructuring allowed
5. **Loop**: Continue until pass-through or max iterations

## Structure

```
workflows/research/subgraphs/academic_lit_review/supervision/
├── __init__.py            # Public exports
├── graph.py               # Supervision subgraph construction
├── routing.py             # Flow control functions
├── types.py               # SupervisorDecision, IssueIdentification Pydantic models
├── prompts.py             # SUPERVISOR_SYSTEM, SUPERVISOR_USER prompts
├── focused_expansion.py   # Reuses main workflow phases for targeted research
├── loops/
│   └── __init__.py        # Loop utilities
└── nodes/
    ├── __init__.py
    ├── analyze_review.py  # Opus analysis with extended thinking
    ├── expand_topic.py    # Focused research on identified gap
    └── integrate_content.py # Merge findings into review
```

## Implementation

### Supervision State

```python
# workflows/research/subgraphs/academic_lit_review/supervision/types.py

from pydantic import BaseModel, Field
from typing import Literal, Optional


class IssueIdentification(BaseModel):
    """A theoretical gap identified by the supervisor."""
    topic: str = Field(description="The specific topic/concept lacking depth")
    section: str = Field(description="Section of review where gap appears")
    severity: Literal["minor", "moderate", "significant"] = Field(
        description="How much this affects review quality"
    )
    suggested_queries: list[str] = Field(
        default_factory=list,
        description="Search queries to find relevant papers",
    )
    foundational_concepts: list[str] = Field(
        default_factory=list,
        description="Key concepts/theories that should be present",
    )


class SupervisorDecision(BaseModel):
    """Supervisor's decision after analyzing the review."""
    action: Literal["research_needed", "pass_through"] = Field(
        description="Whether more research is needed or review passes"
    )
    reasoning: str = Field(description="Explanation for the decision")
    issue: Optional[IssueIdentification] = Field(
        default=None,
        description="Identified issue if action is research_needed",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in this decision",
    )
```

### Supervisor Analysis Node

```python
# workflows/research/subgraphs/academic_lit_review/supervision/nodes/analyze_review.py

async def analyze_review_node(state: dict[str, Any]) -> dict[str, Any]:
    """Analyze the literature review for theoretical gaps.

    Uses Opus with extended thinking to carefully assess whether the
    review has adequate theoretical grounding.
    """
    current_review = state.get("current_review", "")
    input_data = state.get("input", {})
    issues_explored = state.get("issues_explored", [])
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    # Use Opus with extended thinking for deep analysis
    llm = get_llm(
        tier=ModelTier.OPUS,
        thinking_budget=8000,
        max_tokens=12096,
    )

    # Get structured output
    structured_llm = llm.with_structured_output(SupervisorDecision)
    messages = [
        {"role": "system", "content": SUPERVISOR_SYSTEM},
        {"role": "user", "content": SUPERVISOR_USER.format(
            final_review=current_review,
            topic=input_data.get("topic", ""),
            issues_explored=_format_explored(issues_explored),
            iteration=iteration + 1,
            max_iterations=max_iterations,
        )},
    ]

    decision: SupervisorDecision = await structured_llm.ainvoke(messages)

    updates = {"decision": decision.model_dump()}

    if decision.action == "pass_through":
        updates["is_complete"] = True
    elif decision.issue:
        # Track explored issues to prevent re-exploration
        updates["issues_explored"] = issues_explored + [decision.issue.topic]

    return updates
```

### Routing Logic

```python
# workflows/research/subgraphs/academic_lit_review/supervision/routing.py

def route_after_analysis(state: dict[str, Any]) -> str:
    """Route based on supervisor decision."""
    decision = state.get("decision")
    if decision is None:
        return "finalize"

    action = decision.get("action", "pass_through")
    if action == "pass_through":
        return "finalize"

    return "expand"


def should_continue_supervision(state: dict[str, Any]) -> str:
    """Check if more supervision iterations are needed."""
    if state.get("is_complete", False):
        return "complete"

    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)

    if iteration >= max_iterations:
        return "complete"

    return "continue"
```

### Supervision Subgraph

```python
# workflows/research/subgraphs/academic_lit_review/supervision/graph.py

def create_supervision_subgraph() -> StateGraph:
    """Create the supervision loop subgraph.

    Flow:
        START -> analyze_review -> route
            -> (research_needed) -> expand_topic -> integrate_content
                -> check_continue -> (continue) -> analyze_review
                               -> (complete) -> finalize -> END
            -> (pass_through) -> finalize -> END
    """
    builder = StateGraph(SupervisionSubgraphState)

    # Add nodes
    builder.add_node("analyze_review", analyze_review_node)
    builder.add_node("expand_topic", expand_topic_node)
    builder.add_node("integrate_content", integrate_content_node)
    builder.add_node("finalize", finalize_supervision_node)

    # Entry point
    builder.add_edge(START, "analyze_review")

    # Route based on supervisor decision
    builder.add_conditional_edges(
        "analyze_review",
        route_after_analysis,
        {"expand": "expand_topic", "finalize": "finalize"},
    )

    # Expansion -> Integration -> Check continue
    builder.add_edge("expand_topic", "integrate_content")

    builder.add_conditional_edges(
        "integrate_content",
        should_continue_supervision,
        {"continue": "analyze_review", "complete": "finalize"},
    )

    builder.add_edge("finalize", END)

    return builder.compile()
```

### Running Supervision

```python
# workflows/research/subgraphs/academic_lit_review/supervision/graph.py

async def run_supervision(
    final_review: str,
    paper_corpus: dict[str, Any],
    paper_summaries: dict[str, Any],
    clusters: list[dict],
    quality_settings: dict[str, Any],
    input_data: dict[str, Any],
    zotero_keys: dict[str, str],
) -> dict[str, Any]:
    """Run the supervision loop on a completed literature review."""

    # Determine max iterations from quality settings
    max_iterations = quality_settings.get("max_stages", 3)

    initial_state = {
        "current_review": final_review,
        "input": input_data,
        "paper_corpus": paper_corpus,
        "paper_summaries": paper_summaries,
        "clusters": clusters,
        "quality_settings": quality_settings,
        "zotero_keys": zotero_keys,
        "iteration": 0,
        "max_iterations": max_iterations,
        "issues_explored": [],
        "is_complete": False,
        "supervision_expansions": [],
    }

    final_state = await supervision_subgraph.ainvoke(initial_state)

    return {
        "final_review_v2": final_state.get("final_review_v2"),
        "iterations": final_state.get("iteration", 0),
        "expansions": final_state.get("supervision_expansions", []),
    }
```

## Usage

```python
from workflows.research.subgraphs.academic_lit_review.supervision import run_supervision

# After main literature review generation
result = await run_supervision(
    final_review=lit_review_output["final_review"],
    paper_corpus=lit_review_output["paper_corpus"],
    paper_summaries=lit_review_output["paper_summaries"],
    clusters=lit_review_output["clusters"],
    quality_settings={"max_stages": 3},  # Max 3 improvement iterations
    input_data={"topic": "AI agents", "research_questions": [...]},
    zotero_keys=lit_review_output["zotero_keys"],
)

improved_review = result["final_review_v2"]
print(f"Improved in {result['iterations']} iterations")
```

## Guidelines

### Quality Settings Integration

| Quality Tier | Max Iterations | Expansion Depth |
|--------------|----------------|-----------------|
| quick | 1 | quick |
| standard | 2 | standard |
| comprehensive | 3 | standard |
| high_quality | 5 | comprehensive |

### Issue Tracking

Track explored issues to prevent re-exploration:
```python
updates["issues_explored"] = issues_explored + [decision.issue.topic]
```

The supervisor prompt includes previously explored issues to avoid loops.

### Termination Conditions

Supervision terminates when:
1. **Pass-through**: Supervisor approves current quality
2. **Max iterations**: Iteration limit reached
3. **Error fallback**: On analysis error, defaults to pass-through

### Integration Strategy

The integrate_content node is allowed to:
- Add new sections
- Restructure existing sections
- Update citations throughout
- Rewrite transitions

This is more flexible than simple appending.

## Known Uses

- `workflows/research/subgraphs/academic_lit_review/graph/phases/supervision.py` - Phase 6 wrapper
- `workflows/research/subgraphs/academic_lit_review/supervision/` - Full implementation

## Consequences

### Benefits
- **Targeted improvement**: Only researches specific gaps
- **Quality assurance**: Opus-level analysis catches subtle issues
- **Bounded iteration**: Quality settings control effort
- **Reuses infrastructure**: Expansion uses existing workflow phases

### Trade-offs
- **Cost**: Multiple Opus calls (analysis + integration)
- **Latency**: Each iteration adds discovery/diffusion/processing time
- **Complexity**: Additional subgraph with routing logic

## Related Patterns

- [Multi-Source Research Orchestration](./multi-source-research-orchestration.md) - Main workflow this supervises
- [Citation Network Academic Review Workflow](./citation-network-academic-review-workflow.md) - Discovery/diffusion phases

## References

- [LangGraph Cycles and Loops](https://langchain-ai.github.io/langgraph/concepts/looping/)
- [Anthropic Claude Extended Thinking](https://docs.anthropic.com/en/docs/extended-thinking)
