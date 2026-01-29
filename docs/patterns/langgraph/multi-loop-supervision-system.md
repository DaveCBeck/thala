---
name: multi-loop-supervision-system
title: Multi-Loop Supervision System Pattern
date: 2026-01-08
category: langgraph
shared: true
gist_url: https://gist.github.com/DaveCBeck/777256058a3e8a1cff3a284a204a4d42
article_path: .context/libs/thala-dev/content/2026-01-08-multi-loop-supervision-system-langgraph.md
applicability:
  - "Long-form document quality assurance with multiple dimensions"
  - "Iterative improvement across literature, structure, editing, and factuality"
  - "Configurable supervision depth based on quality tier"
  - "Academic literature reviews requiring comprehensive refinement"
components: [supervision, orchestration, loop2_literature, loop3_structure, loop4_editing, loop5_factcheck]
complexity: high
verified_in_production: true
tags: [supervision, multi-loop, quality-assurance, editing, factcheck, orchestration, opus, sonnet]
---

# Multi-Loop Supervision System Pattern

## Intent

Systematically improve document quality through multiple specialized supervision loops, each targeting a specific quality dimension (literature coverage, structure, detailed editing, fact-checking), with configurable execution based on quality tier.

## Problem

Single-pass document generation has multiple quality dimensions that require different expertise:
- **Literature coverage**: Missing perspectives or foundational works
- **Structure**: Logical flow, section organization, cohesion
- **Detail**: Section-level writing quality, clarity, depth
- **Factuality**: Citation accuracy, reference validity

A single supervisor cannot effectively address all dimensions. Different loops require different approaches (expansion vs editing vs verification).

## Solution

Implement five specialized supervision loops executed in sequence:

| Loop | Name | Focus | Approach |
|------|------|-------|----------|
| 1 | Theoretical Depth | Gap identification | Opus analysis → expansion → integration |
| 2 | Literature Base | Missing perspectives | Mini-review subgraph for targeted expansion |
| 3 | Structure | Organization, flow | Two-agent structural analysis with edit manifests |
| 4 | Section Editing | Detailed refinement | Parallel section editing + holistic review |
| 4.5 | Cohesion | Inter-section flow | May return to Loop 3 if needed |
| 5 | Fact-Check | Citation verification | Haiku-based reference validation |

## Structure

```
workflows/research/subgraphs/academic_lit_review/supervision/
├── orchestration/
│   ├── graph.py              # Main orchestration graph
│   └── types.py              # OrchestrationState, LoopConfig
├── loops/
│   ├── loop2_literature.py   # Literature base expansion
│   ├── loop3_structure.py    # Structural editing (two-agent)
│   ├── loop4_editing.py      # Section-level editing
│   ├── loop4_5_cohesion.py   # Cohesion check
│   └── loop5_factcheck.py    # Fact and reference checking
├── mini_review/
│   └── graph.py              # Mini-review subgraph for Loop 2
├── utils/
│   ├── section_splitting.py  # Parse document into sections
│   ├── paragraph_numbering.py # Add/remove paragraph numbers
│   ├── edit_application.py   # Apply edit manifests
│   └── revision_history.py   # Track changes across iterations
└── tools/
    ├── paper_search.py       # Search existing corpus
    └── store_query.py        # Query stores for verification
```

## Implementation

### Quality-Based Loop Configuration

```python
# Configurable via quality_settings.supervision_loops
LOOP_CONFIGS = {
    "none": [],                              # Skip all supervision
    "one": [1],                              # Only theoretical depth
    "two": [1, 2],                           # + Literature base
    "three": [1, 2, 3],                      # + Structure
    "four": [1, 2, 3, 4],                    # + Section editing
    "all": [1, 2, 3, 4, 5],                  # Full supervision
}

# Quality preset defaults (all use "all" loops)
QUALITY_PRESETS = {
    "quick": {"supervision_loops": "all", "max_stages": 1},
    "standard": {"supervision_loops": "all", "max_stages": 2},
    "comprehensive": {"supervision_loops": "all", "max_stages": 3},
    "high_quality": {"supervision_loops": "all", "max_stages": 5},
}
```

### Orchestration Graph

```python
# workflows/research/subgraphs/academic_lit_review/supervision/orchestration/graph.py

from langgraph.graph import StateGraph, START, END

def create_orchestration_graph() -> StateGraph:
    """Create the supervision orchestration graph.

    Flow:
        START → loop1_theoretical_depth
              → loop2_literature_base
              → loop3_structure
              → loop4_section_editing
              → loop4_5_cohesion → (return to loop3 OR continue)
              → loop5_factcheck
              → END
    """
    builder = StateGraph(OrchestrationState)

    # Add loop nodes
    builder.add_node("loop1", run_loop1_theoretical_depth)
    builder.add_node("loop2", run_loop2_literature_base)
    builder.add_node("loop3", run_loop3_structure)
    builder.add_node("loop4", run_loop4_section_editing)
    builder.add_node("loop4_5", run_loop4_5_cohesion)
    builder.add_node("loop5", run_loop5_factcheck)
    builder.add_node("finalize", finalize_supervision)

    # Sequential flow with conditional skip based on config
    builder.add_conditional_edges(START, should_run_loop1, {True: "loop1", False: "loop2"})
    builder.add_conditional_edges("loop1", should_run_loop2, {True: "loop2", False: "loop3"})
    builder.add_conditional_edges("loop2", should_run_loop3, {True: "loop3", False: "loop4"})
    builder.add_conditional_edges("loop3", should_run_loop4, {True: "loop4", False: "loop5"})
    builder.add_edge("loop4", "loop4_5")

    # Loop 4.5 can return to Loop 3 if cohesion issues detected
    builder.add_conditional_edges(
        "loop4_5",
        route_after_cohesion_check,
        {"return_to_loop3": "loop3", "continue": "loop5"},
    )

    builder.add_conditional_edges("loop5", should_run_finalize, {True: "finalize", False: END})
    builder.add_edge("finalize", END)

    return builder.compile()
```

### Loop 2: Literature Base Expansion

```python
# workflows/research/subgraphs/academic_lit_review/supervision/loops/loop2_literature.py

async def run_loop2_literature_base(state: OrchestrationState) -> dict:
    """Identify and fill missing literature perspectives.

    Uses Opus to identify missing perspectives, then runs a mini-review
    subgraph to find relevant papers and integrate them.
    """
    current_review = state["current_review"]
    existing_corpus = state["paper_corpus"]

    # Analyze for missing perspectives
    llm = get_llm(ModelTier.OPUS, thinking_budget=4000, max_tokens=8000)
    analysis = await analyze_missing_perspectives(current_review, existing_corpus, llm)

    if not analysis.missing_perspectives:
        return {"loop2_complete": True, "loop2_skipped": True}

    # Run mini-review for each missing perspective
    expanded_corpus = dict(existing_corpus)
    for perspective in analysis.missing_perspectives[:3]:  # Limit to 3
        mini_result = await run_mini_review(
            topic=perspective.topic,
            context=perspective.context,
            quality_settings=state["quality_settings"],
        )
        expanded_corpus.update(mini_result["papers"])

    # Integrate new papers into review
    updated_review = await integrate_literature_expansion(
        current_review, expanded_corpus, analysis.integration_points
    )

    return {
        "current_review": updated_review,
        "paper_corpus": expanded_corpus,
        "loop2_complete": True,
    }
```

### Loop 3: Structural Editing (Two-Agent Pattern)

```python
# workflows/research/subgraphs/academic_lit_review/supervision/loops/loop3_structure.py

async def run_loop3_structure(state: OrchestrationState) -> dict:
    """Two-agent structural analysis and editing.

    Agent 1 (Analyzer): Identifies structural issues
    Agent 2 (Editor): Creates edit manifest to fix issues
    """
    current_review = state["current_review"]

    # Agent 1: Structural analysis
    analyzer_llm = get_llm(ModelTier.OPUS, thinking_budget=8000)
    analysis = await analyze_structure(current_review, analyzer_llm)

    if not analysis.issues:
        return {"loop3_complete": True, "loop3_skipped": True}

    # Agent 2: Generate edit manifest
    editor_llm = get_llm(ModelTier.SONNET, max_tokens=16000)
    edit_manifest = await generate_structural_edits(
        current_review, analysis.issues, editor_llm
    )

    # Apply edits
    updated_review = apply_structural_edits(current_review, edit_manifest)

    return {
        "current_review": updated_review,
        "structural_edits": edit_manifest,
        "loop3_complete": True,
    }
```

### Loop 4: Section-Level Editing

```python
# workflows/research/subgraphs/academic_lit_review/supervision/loops/loop4_editing.py

async def run_loop4_section_editing(state: OrchestrationState) -> dict:
    """Parallel section editing with holistic review.

    1. Split document into sections
    2. Edit each section in parallel (Sonnet)
    3. Holistic review of all edits (Opus)
    4. Reassemble document
    """
    current_review = state["current_review"]

    # Split into sections
    sections = split_into_sections(current_review)

    # Parallel section editing
    async def edit_section(section: Section) -> EditedSection:
        llm = get_llm(ModelTier.SONNET, max_tokens=8000)
        return await improve_section(section, llm)

    edited_sections = await asyncio.gather(*[
        edit_section(s) for s in sections
    ])

    # Holistic review
    holistic_llm = get_llm(ModelTier.OPUS, thinking_budget=4000)
    coherence_edits = await review_coherence(edited_sections, holistic_llm)

    # Apply coherence edits and reassemble
    final_sections = apply_coherence_edits(edited_sections, coherence_edits)
    updated_review = reassemble_sections(final_sections)

    return {
        "current_review": updated_review,
        "section_edits": edited_sections,
        "loop4_complete": True,
    }
```

### Loop 4.5: Cohesion Check

```python
# workflows/research/subgraphs/academic_lit_review/supervision/loops/loop4_5_cohesion.py

async def run_loop4_5_cohesion(state: OrchestrationState) -> dict:
    """Check inter-section cohesion, may return to Loop 3.

    Uses Sonnet to quickly assess whether structural issues remain
    after section editing that require another Loop 3 pass.
    """
    current_review = state["current_review"]
    loop3_repeats = state.get("loop3_repeats", 0)
    max_loop3_repeats = state.get("max_loop3_repeats", 2)

    llm = get_llm(ModelTier.SONNET, max_tokens=2000)
    cohesion_check = await check_cohesion(current_review, llm)

    needs_loop3_repeat = (
        cohesion_check.has_major_issues
        and loop3_repeats < max_loop3_repeats
    )

    return {
        "cohesion_result": cohesion_check,
        "needs_loop3_repeat": needs_loop3_repeat,
        "loop3_repeats": loop3_repeats + 1 if needs_loop3_repeat else loop3_repeats,
    }
```

### Loop 5: Fact-Check

```python
# workflows/research/subgraphs/academic_lit_review/supervision/loops/loop5_factcheck.py

async def run_loop5_factcheck(state: OrchestrationState) -> dict:
    """Verify facts and references using Haiku.

    Fast, cheap verification of:
    - Citation accuracy
    - Reference validity
    - Factual claims against corpus
    """
    current_review = state["current_review"]
    paper_corpus = state["paper_corpus"]

    llm = get_llm(ModelTier.HAIKU, max_tokens=4000)

    # Extract claims and citations
    claims = extract_claims_and_citations(current_review)

    # Verify against corpus
    verification_results = await verify_claims_batch(claims, paper_corpus, llm)

    # Generate correction edits
    corrections = generate_fact_corrections(verification_results)

    # Apply corrections
    updated_review = apply_fact_corrections(current_review, corrections)

    return {
        "current_review": updated_review,
        "fact_check_results": verification_results,
        "loop5_complete": True,
    }
```

## Usage

```python
from workflows.research.subgraphs.academic_lit_review import academic_lit_review

# Standard quality with all loops
result = await academic_lit_review(
    topic="AI agents in creative work",
    research_questions=["How do AI agents...?"],
    quality="standard",  # supervision_loops="all" by default
)

# Quick mode with reduced iterations
result = await academic_lit_review(
    topic="AI agents",
    quality="quick",  # Still runs all loops, but fewer iterations per loop
)

# Custom loop configuration
result = await academic_lit_review(
    topic="AI agents",
    quality_settings={
        "supervision_loops": "three",  # Only loops 1-3
        "max_stages": 2,
    },
)
```

## Guidelines

### Iteration Budget

Iterations are shared across loops, scaled by `max_stages`:

| Quality | max_stages | Typical Iterations per Loop |
|---------|------------|----------------------------|
| quick | 1 | 1-2 |
| standard | 2 | 2-3 |
| comprehensive | 3 | 3-5 |
| high_quality | 5 | 5-8 |

### Loop Dependencies

- Loops 1-2: Can run independently
- Loop 3: Depends on stable content from 1-2
- Loop 4: Depends on stable structure from 3
- Loop 4.5: May return to 3 (max 2 repeats)
- Loop 5: Final pass, no returns

### Model Selection

| Loop | Model | Rationale |
|------|-------|-----------|
| 1, 2 | Opus | Deep analysis, expansion |
| 3 | Opus + Sonnet | Analysis (Opus), edits (Sonnet) |
| 4 | Sonnet + Opus | Section edits (Sonnet), coherence (Opus) |
| 4.5 | Sonnet | Quick cohesion check |
| 5 | Haiku | Fast, cheap verification |

## Known Uses

- `workflows/research/subgraphs/academic_lit_review/supervision/orchestration/graph.py`
- `workflows/research/subgraphs/academic_lit_review/graph/phases/supervision.py`

## Consequences

### Benefits
- **Comprehensive quality**: Multiple dimensions addressed systematically
- **Configurable depth**: Skip loops for quick iterations
- **Specialized approaches**: Each loop uses optimal strategy
- **Conditional refinement**: Loop 4.5 can return for structural fixes

### Trade-offs
- **Complexity**: Multiple loops with inter-dependencies
- **Cost**: Multiple Opus calls across loops
- **Duration**: Full supervision can take hours for large reviews

## Related Patterns

- [Iterative Theoretical Depth Supervision](./iterative-theoretical-depth-supervision.md) - Loop 1 implementation
- [Multi-Source Research Orchestration](./multi-source-research-orchestration.md) - Main workflow

## References

- [LangGraph Conditional Edges](https://langchain-ai.github.io/langgraph/concepts/conditional_edges/)
- [Two-Agent Pattern](https://www.anthropic.com/research/building-effective-agents)
