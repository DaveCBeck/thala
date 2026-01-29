---
name: synthesis-workflow-orchestration
title: "Synthesis Workflow Orchestration: Multi-Phase Research Integration"
date: 2026-01-16
category: langgraph
applicability:
  - "Research projects requiring multiple source types (academic, web, books)"
  - "Long-form content generation from diverse research inputs"
  - "Quality-sensitive synthesis requiring iterative supervision"
  - "Workflows with parallelizable research phases and sequential synthesis"
components: [multi_phase_orchestration, parallel_research_dispatch, two_path_synthesis, quality_checkpoints, model_tiering]
complexity: complex
verified_in_production: true
related_solutions: []
tags: [orchestration, synthesis, parallel-research, multi-source, quality-tiers, send-pattern, section-writing]
---

# Synthesis Workflow Orchestration: Multi-Phase Research Integration

## Intent

Orchestrate a comprehensive synthesis workflow that integrates multiple research sources (academic literature, web research, books) through parallel dispatch, then synthesizes them into structured content with quality validation and editing.

## Motivation

Complex research synthesis requires:
- **Multiple source types**: Academic papers, web articles, books each provide different perspectives
- **Parallel efficiency**: Research phases are independent and can run concurrently
- **Quality control**: Generated content needs validation against thresholds
- **Flexible depth**: Different use cases need different quality/cost tradeoffs

A single-pass approach fails to capture the depth needed for comprehensive synthesis. This pattern structures the workflow into distinct phases with quality-based routing.

## Applicability

Use this pattern when:
- Synthesis requires integrating 3+ distinct source types
- Research phases can run in parallel without dependencies
- Content quality justifies multi-phase processing
- Different quality tiers serve different use cases (quick draft vs publication-ready)

Do NOT use this pattern when:
- Simple single-source summarization is sufficient
- Research phases have tight dependencies
- Cost constraints prevent parallel execution
- Quality validation is unnecessary

## Structure

```
START
  │
  ▼
╔════════════════════════════════════════════════════╗
║  PHASE 1: LITERATURE REVIEW                        ║
║  └─> academic_lit_review workflow                  ║
║      (Papers → DOI corpus → Zotero keys)           ║
╚════════════════════════════════════════════════════╝
  │
  ▼
╔════════════════════════════════════════════════════╗
║  PHASE 2: SUPERVISION (conditional)                ║
║  └─> enhance.supervision (Loop 1 + Loop 2)         ║
║      (Theoretical depth + Literature expansion)    ║
║      [skipped in test mode]                        ║
╚════════════════════════════════════════════════════╝
  │
  ▼
╔════════════════════════════════════════════════════╗
║  PHASE 3: PARALLEL RESEARCH                        ║
║  ┌─────────────────────────────────────────────┐   ║
║  │ generate_research_targets (Haiku)           │   ║
║  │  → N web queries + N book themes            │   ║
║  └─────────────────────────────────────────────┘   ║
║                     │                              ║
║      ┌──────────────┼──────────────┐               ║
║      ▼              ▼              ▼               ║
║  ┌────────┐    ┌────────┐    ┌────────┐           ║
║  │ Web    │    │ Web    │    │ Book   │  ...      ║
║  │ Worker │    │ Worker │    │ Worker │           ║
║  │ [i=0]  │    │ [i=1]  │    │ [i=0]  │           ║
║  └────────┘    └────────┘    └────────┘           ║
║      │              │              │               ║
║      └──────────────┼──────────────┘               ║
║                     ▼                              ║
║           ┌─────────────────┐                      ║
║           │ aggregate_      │ (Reducer merges)    ║
║           │ research        │                      ║
║           └─────────────────┘                      ║
╚════════════════════════════════════════════════════╝
  │
  ▼
╔════════════════════════════════════════════════════╗
║  PHASE 4: SYNTHESIS                                ║
║  ┌─────────────────────────────────────────────┐   ║
║  │ synthesis_router                            │   ║
║  └─────────┬───────────────────────┬───────────┘   ║
║            ▼                       ▼               ║
║     [simple_synthesis]      [suggest_structure]   ║
║     (test mode only)        (Opus planning)       ║
║            │                       │               ║
║            │                 [select_books]       ║
║            │                       │               ║
║            │                 [fetch_summaries]    ║
║            │                       │               ║
║            │              ┌────────┼────────┐      ║
║            │              ▼        ▼        ▼      ║
║            │           [write_section_worker]     ║
║            │           (parallel × N sections)    ║
║            │                       │               ║
║            │              [assemble_sections]     ║
║            │              [check_quality]         ║
║            │                       │               ║
║            └───────────────────────┘               ║
╚════════════════════════════════════════════════════╝
  │
  ▼
╔════════════════════════════════════════════════════╗
║  PHASE 5: EDITING                                  ║
║  └─> enhance.editing workflow                      ║
║      (Structure → Enhancement → Verification)      ║
╚════════════════════════════════════════════════════╝
  │
  ▼
[finalize] → END
```

## Implementation

### Step 1: Define State with Reducer Aggregation

```python
# state.py

from operator import add
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict


class WebResearchResult(TypedDict):
    iteration: int
    query: str
    final_report: str
    source_count: int
    langsmith_run_id: str
    status: str


class BookFindingResult(TypedDict):
    iteration: int
    theme: str
    final_report: str
    processed_books: list[dict]
    zotero_keys: list[str]
    status: str


class SynthesisSection(TypedDict):
    section_id: str
    title: str
    content: str
    citations: list[str]
    quality_score: Optional[float]
    needs_revision: bool


class SynthesisState(TypedDict, total=False):
    # Input
    input: dict
    quality_settings: dict

    # Phase 1: Literature Review
    lit_review_result: Optional[dict]
    paper_corpus: dict[str, Any]      # DOI -> metadata
    zotero_keys: dict[str, str]       # DOI -> Zotero key

    # Phase 2: Supervision
    supervision_result: Optional[dict]

    # Phase 3: Parallel Research (reducer aggregation)
    generated_queries: list[dict]
    generated_themes: list[dict]
    web_research_results: Annotated[list[WebResearchResult], add]
    book_finding_results: Annotated[list[BookFindingResult], add]

    # Phase 4: Synthesis
    synthesis_structure: Optional[dict]
    selected_books: list[dict]
    book_summaries_cache: dict[str, str]  # zotero_key -> summary
    section_drafts: Annotated[list[SynthesisSection], add]

    # Phase 5: Editing
    final_report: Optional[str]

    # Workflow metadata
    current_phase: str
    errors: Annotated[list[dict], add]
```

Key patterns:
- **Reducer aggregation**: `Annotated[list[T], add]` for parallel worker outputs
- **Direct assignment**: Regular fields for sequential phase outputs
- **Error accumulation**: Errors collected from all phases

### Step 2: Quality Tier Presets

```python
# quality_presets.py

from typing import TypedDict


class SynthesisQualitySettings(TypedDict):
    skip_supervision: bool
    web_research_runs: int
    book_finding_runs: int
    simple_synthesis: bool
    max_books_to_select: int
    target_word_count: int
    use_opus_for_structure: bool
    use_opus_for_sections: bool
    section_quality_threshold: float


SYNTHESIS_QUALITY_PRESETS: dict[str, SynthesisQualitySettings] = {
    "test": {
        "skip_supervision": True,
        "web_research_runs": 1,
        "book_finding_runs": 1,
        "simple_synthesis": True,
        "max_books_to_select": 2,
        "target_word_count": 3000,
        "use_opus_for_structure": False,
        "use_opus_for_sections": False,
        "section_quality_threshold": 0.5,
    },
    "quick": {
        "skip_supervision": False,
        "web_research_runs": 2,
        "book_finding_runs": 2,
        "simple_synthesis": False,
        "max_books_to_select": 3,
        "target_word_count": 12000,
        "use_opus_for_structure": True,
        "use_opus_for_sections": False,
        "section_quality_threshold": 0.6,
    },
    "standard": {
        "skip_supervision": False,
        "web_research_runs": 3,
        "book_finding_runs": 3,
        "simple_synthesis": False,
        "max_books_to_select": 4,
        "target_word_count": 18000,
        "use_opus_for_structure": True,
        "use_opus_for_sections": True,
        "section_quality_threshold": 0.7,
    },
    "comprehensive": {
        "skip_supervision": False,
        "web_research_runs": 4,
        "book_finding_runs": 4,
        "simple_synthesis": False,
        "max_books_to_select": 5,
        "target_word_count": 26250,
        "use_opus_for_structure": True,
        "use_opus_for_sections": True,
        "section_quality_threshold": 0.75,
    },
    "high_quality": {
        "skip_supervision": False,
        "web_research_runs": 5,
        "book_finding_runs": 5,
        "simple_synthesis": False,
        "max_books_to_select": 6,
        "target_word_count": 37500,
        "use_opus_for_structure": True,
        "use_opus_for_sections": True,
        "section_quality_threshold": 0.8,
    },
}
```

### Step 3: Parallel Research Dispatch

```python
# nodes/research_workers.py

from langgraph.types import Send
from workflows.research.web_research import deep_research
from workflows.research.book_finding import book_finding


def route_to_parallel_research(state: dict) -> list[Send]:
    """Dispatch parallel web research and book finding workers."""
    generated_queries = state.get("generated_queries", [])
    generated_themes = state.get("generated_themes", [])
    quality = state.get("input", {}).get("quality", "standard")
    multi_lang_config = state.get("input", {}).get("multi_lang_config")

    sends = []

    # Web research workers (one per query)
    for i, query_data in enumerate(generated_queries):
        sends.append(
            Send(
                "web_research_worker",
                {
                    "iteration": i,
                    "query": query_data["query"],
                    "quality": quality,
                    "multi_lang_config": multi_lang_config,
                },
            )
        )

    # Book finding workers (one per theme)
    for i, theme_data in enumerate(generated_themes):
        sends.append(
            Send(
                "book_finding_worker",
                {
                    "iteration": i,
                    "theme": theme_data["theme"],
                    "quality": quality,
                    "multi_lang_config": multi_lang_config,
                },
            )
        )

    return sends


async def web_research_worker(state: dict) -> dict[str, Any]:
    """Execute single web research query."""
    iteration = state.get("iteration", 0)
    query = state.get("query", "")
    quality = state.get("quality", "standard")

    try:
        result = await deep_research(query=query, quality=quality)

        return {
            "web_research_results": [{
                "iteration": iteration,
                "query": query,
                "final_report": result.get("final_report", ""),
                "source_count": result.get("source_count", 0),
                "langsmith_run_id": result.get("langsmith_run_id", ""),
                "status": "success",
            }]
        }

    except Exception as e:
        return {
            "web_research_results": [{
                "iteration": iteration,
                "query": query,
                "final_report": "",
                "source_count": 0,
                "langsmith_run_id": "",
                "status": "failed",
            }],
            "errors": [{"phase": "web_research", "iteration": iteration, "error": str(e)}],
        }


async def book_finding_worker(state: dict) -> dict[str, Any]:
    """Execute single book finding search."""
    iteration = state.get("iteration", 0)
    theme = state.get("theme", "")
    quality = state.get("quality", "standard")

    try:
        result = await book_finding(topic=theme, quality=quality)

        return {
            "book_finding_results": [{
                "iteration": iteration,
                "theme": theme,
                "final_report": result.get("final_report", ""),
                "processed_books": result.get("processed_books", []),
                "zotero_keys": result.get("zotero_keys", []),
                "status": "success",
            }]
        }

    except Exception as e:
        return {
            "book_finding_results": [{
                "iteration": iteration,
                "theme": theme,
                "final_report": "",
                "processed_books": [],
                "zotero_keys": [],
                "status": "failed",
            }],
            "errors": [{"phase": "book_finding", "iteration": iteration, "error": str(e)}],
        }
```

### Step 4: Two-Path Synthesis Router

```python
# nodes/synthesis.py

from workflows.shared.llm_utils import get_llm, ModelTier


def route_synthesis_path(state: dict) -> str:
    """Route to simple or structured synthesis based on quality."""
    quality_settings = state.get("quality_settings", {})

    if quality_settings.get("simple_synthesis", False):
        return "simple_synthesis"
    return "suggest_structure"


async def simple_synthesis(state: dict) -> dict[str, Any]:
    """Single-pass synthesis for test mode."""
    # Aggregate all sources into context
    lit_review = state.get("supervision_result", {}).get("final_report", "")
    web_results = state.get("web_research_results", [])
    book_results = state.get("book_finding_results", [])

    web_summary = "\n\n".join(
        f"### {r['query']}\n{r['final_report'][:3000]}"
        for r in web_results if r.get("status") == "success"
    )[:10000]

    book_summary = "\n\n".join(
        f"### {r['theme']}\n{r['final_report'][:3000]}"
        for r in book_results if r.get("status") == "success"
    )[:10000]

    # Single LLM call
    llm = get_llm(ModelTier.SONNET, max_tokens=16000)
    prompt = f"""Synthesize these research sources into a coherent document:

## Academic Literature
{lit_review[:10000]}

## Web Research
{web_summary}

## Books
{book_summary}

Write a comprehensive synthesis with proper [@CITATION] format."""

    response = await llm.ainvoke([{"role": "user", "content": prompt}])

    return {
        "final_report": response.content,
        "current_phase": "editing",
    }


async def suggest_structure(state: dict) -> dict[str, Any]:
    """Plan document structure using Opus."""
    quality_settings = state.get("quality_settings", {})

    model_tier = (
        ModelTier.OPUS
        if quality_settings.get("use_opus_for_structure", True)
        else ModelTier.SONNET
    )
    llm = get_llm(model_tier, max_tokens=4000)
    llm_structured = llm.with_structured_output(StructureSuggestion)

    # Build context from all sources...
    result = await llm_structured.ainvoke([{"role": "user", "content": prompt}])

    return {
        "synthesis_structure": {
            "title": result.title,
            "sections": [s.model_dump() for s in result.sections],
            "introduction_guidance": result.introduction_guidance,
            "conclusion_guidance": result.conclusion_guidance,
        },
        "current_phase": "select_books",
    }
```

### Step 5: Section-Level Parallel Writing

```python
# nodes/quality_check.py

from langgraph.types import Send
from pydantic import BaseModel


class SectionQuality(BaseModel):
    quality_score: float  # 0.0 to 1.0
    strengths: list[str]
    weaknesses: list[str]
    needs_revision: bool


def route_to_section_workers(state: dict) -> list[Send]:
    """Dispatch parallel workers for each section."""
    synthesis_structure = state.get("synthesis_structure", {})
    sections = synthesis_structure.get("sections", [])

    if not sections:
        return [Send("assemble_sections", state)]

    sends = []
    for section in sections:
        sends.append(
            Send(
                "write_section_worker",
                {
                    "section_id": section.get("section_id"),
                    "section_title": section.get("title"),
                    "section_description": section.get("description"),
                    "key_sources": section.get("key_sources", []),
                    # Pass shared context
                    "lit_review": state.get("supervision_result", {}).get("final_report", ""),
                    "web_research_results": state.get("web_research_results", []),
                    "book_summaries_cache": state.get("book_summaries_cache", {}),
                    "quality_settings": state.get("quality_settings", {}),
                },
            )
        )

    return sends


async def write_section_worker(state: dict) -> dict[str, Any]:
    """Write a single section."""
    section_id = state.get("section_id")
    section_title = state.get("section_title")
    quality_settings = state.get("quality_settings", {})

    model_tier = (
        ModelTier.OPUS
        if quality_settings.get("use_opus_for_sections", True)
        else ModelTier.SONNET
    )
    llm = get_llm(model_tier, max_tokens=8000)

    # Build context from shared state...
    response = await llm.ainvoke([{"role": "user", "content": prompt}])

    # Extract citations
    import re
    citations = re.findall(r'\[@([A-Za-z0-9_-]+)\]', response.content)

    return {
        "section_drafts": [{
            "section_id": section_id,
            "title": section_title,
            "content": response.content,
            "citations": list(set(citations)),
            "quality_score": None,
            "needs_revision": False,
        }],
    }


async def check_section_quality(state: dict) -> dict[str, Any]:
    """Evaluate quality of all sections."""
    section_drafts = state.get("section_drafts", [])
    quality_settings = state.get("quality_settings", {})
    threshold = quality_settings.get("section_quality_threshold", 0.7)

    llm = get_llm(ModelTier.HAIKU, max_tokens=1000)
    llm_structured = llm.with_structured_output(SectionQuality)

    updated_sections = []
    for section in section_drafts:
        result = await llm_structured.ainvoke([
            {"role": "user", "content": f"Evaluate this section:\n{section['content'][:5000]}"}
        ])

        section["quality_score"] = result.quality_score
        section["needs_revision"] = result.quality_score < threshold
        updated_sections.append(section)

    return {
        "section_drafts": updated_sections,
        "current_phase": "assemble",
    }
```

### Step 6: Graph Construction

```python
# graph/construction.py

from langgraph.graph import END, START, StateGraph

from .state import SynthesisState
from .nodes import (
    run_lit_review, run_supervision, generate_research_targets,
    route_to_parallel_research, web_research_worker, book_finding_worker,
    aggregate_research, route_synthesis_path, simple_synthesis,
    suggest_structure, select_books, fetch_book_summaries,
    route_to_section_workers, write_section_worker, assemble_sections,
    check_section_quality, run_editing, finalize,
)


def create_synthesis_graph() -> StateGraph:
    builder = StateGraph(SynthesisState)

    # Phase 1: Literature Review
    builder.add_node("run_lit_review", run_lit_review)

    # Phase 2: Supervision
    builder.add_node("run_supervision", run_supervision)

    # Phase 3: Parallel Research
    builder.add_node("generate_research_targets", generate_research_targets)
    builder.add_node("parallel_router", lambda s: s)
    builder.add_node("web_research_worker", web_research_worker)
    builder.add_node("book_finding_worker", book_finding_worker)
    builder.add_node("aggregate_research", aggregate_research)

    # Phase 4: Synthesis
    builder.add_node("synthesis_router", lambda s: s)
    builder.add_node("simple_synthesis", simple_synthesis)
    builder.add_node("suggest_structure", suggest_structure)
    builder.add_node("select_books", select_books)
    builder.add_node("fetch_book_summaries", fetch_book_summaries)
    builder.add_node("section_router", lambda s: s)
    builder.add_node("write_section_worker", write_section_worker)
    builder.add_node("assemble_sections", assemble_sections)
    builder.add_node("check_section_quality", check_section_quality)

    # Phase 5: Editing
    builder.add_node("run_editing", run_editing)
    builder.add_node("finalize", finalize)

    # Edges: Phase 1-2
    builder.add_edge(START, "run_lit_review")
    builder.add_edge("run_lit_review", "run_supervision")
    builder.add_edge("run_supervision", "generate_research_targets")

    # Edges: Phase 3 (parallel dispatch)
    builder.add_edge("generate_research_targets", "parallel_router")
    builder.add_conditional_edges(
        "parallel_router",
        route_to_parallel_research,
        ["web_research_worker", "book_finding_worker"],
    )
    builder.add_edge("web_research_worker", "aggregate_research")
    builder.add_edge("book_finding_worker", "aggregate_research")

    # Edges: Phase 4 (synthesis routing)
    builder.add_edge("aggregate_research", "synthesis_router")
    builder.add_conditional_edges(
        "synthesis_router",
        route_synthesis_path,
        {
            "simple_synthesis": "simple_synthesis",
            "suggest_structure": "suggest_structure",
        },
    )
    builder.add_edge("simple_synthesis", "run_editing")
    builder.add_edge("suggest_structure", "select_books")
    builder.add_edge("select_books", "fetch_book_summaries")
    builder.add_edge("fetch_book_summaries", "section_router")
    builder.add_conditional_edges(
        "section_router",
        route_to_section_workers,
        ["write_section_worker", "assemble_sections"],
    )
    builder.add_edge("write_section_worker", "assemble_sections")
    builder.add_edge("assemble_sections", "check_section_quality")
    builder.add_edge("check_section_quality", "run_editing")

    # Edges: Phase 5
    builder.add_edge("run_editing", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()
```

## Complete Example

```python
from workflows.wrappers.synthesis import synthesis

# Full synthesis with standard quality
result = await synthesis(
    topic="AI in Healthcare",
    research_questions=[
        "How is AI transforming diagnostic imaging?",
        "What are the regulatory challenges?",
    ],
    quality="standard",  # Uses standard presets
)

print(f"Status: {result['status']}")
print(f"Final length: {len(result['final_report'])} chars")
print(f"Web sources: {sum(r['source_count'] for r in result['web_research_results'])}")
print(f"Books integrated: {len(result.get('selected_books', []))}")

# Quick synthesis for iteration
quick_result = await synthesis(
    topic="AI in Healthcare",
    research_questions=["How is AI used in diagnostics?"],
    quality="quick",  # 2 research runs, no supervision
)

# Test mode for debugging
test_result = await synthesis(
    topic="AI in Healthcare",
    research_questions=["Test question"],
    quality="test",  # 1 research run, simple synthesis
)
```

## Consequences

### Benefits

- **Parallel efficiency**: Research phases run concurrently (3-5 workers)
- **Quality flexibility**: Five tiers from test to publication-ready
- **Multi-source integration**: Academic papers + web + books unified
- **Section-level parallelism**: All sections written concurrently
- **Quality validation**: Configurable thresholds catch low-quality output
- **Model tiering**: Cost-efficient model selection per task

### Trade-offs

- **Complexity**: Five phases with conditional routing adds maintenance burden
- **State size**: Aggregated research results can be large
- **Cost at scale**: High-quality tier uses many Opus calls
- **Sequential phases**: Supervision must complete before research targets

### Alternatives

- **Single-pass synthesis**: Faster but lower quality
- **Manual research**: Human-curated sources instead of automated
- **Iterative refinement**: Single agent with revision loops

## Related Patterns

- [Multi-Source Research Orchestration](./multi-source-research-orchestration.md) - Parallel research with asyncio.gather()
- [Workflow Chaining](./workflow-chaining-pattern.md) - Sequential workflow composition
- [Multi-Loop Supervision System](./multi-loop-supervision-system.md) - Supervision loop architecture
- [Multi-Phase Document Editing](./multi-phase-document-editing.md) - Editing workflow details
- [Unified Quality Tier System](./unified-quality-tier-system.md) - Quality parameterization

## Known Uses in Thala

- `workflows/wrappers/synthesis/graph/construction.py` - Graph construction
- `workflows/wrappers/synthesis/nodes/research_workers.py` - Parallel dispatch
- `workflows/wrappers/synthesis/nodes/synthesis.py` - Two-path synthesis
- `workflows/wrappers/synthesis/nodes/quality_check.py` - Section quality validation
- `workflows/wrappers/synthesis/quality_presets.py` - Quality tier configuration
- `workflows/wrappers/synthesis/state.py` - State with reducers

## References

- [LangGraph Send Pattern](https://langchain-ai.github.io/langgraph/concepts/low_level/#send)
- [LangGraph State Reducers](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers)
