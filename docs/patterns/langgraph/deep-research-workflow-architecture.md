---
name: deep-research-workflow-architecture
title: "Deep Research Workflow with Diffusion Algorithm and Parallel Agents"
date: 2025-12-18
category: langgraph
applicability:
  - "Multi-phase research requiring iterative refinement"
  - "Workflows needing parallel agent execution with result aggregation"
  - "Research tasks benefiting from memory-first approach (existing knowledge before web)"
components: [langgraph_graph, langgraph_node, langgraph_state, llm_call, workflow_graph]
complexity: complex
verified_in_production: false
stale: true
stale_date: 2026-01-29
stale_reason: "Architecture evolved: 3-way researcher dispatch (web/academic/book) replaced with web-only; academic and book are now standalone workflows"
related_solutions: []
tags: [langgraph, research, parallel, diffusion, supervisor, agents, send, reducers]
---

> **STALE**: This documentation describes an older architecture.
>
> **What changed:** The 3-way researcher dispatch (web, academic, book) was removed. Academic and book research are now standalone workflows (`academic_lit_review`, `book_finding`). The web research workflow only dispatches to web researchers.
> **Date:** 2026-01-29
> **Current patterns:**
> - Web research: [Web Research Workflow](../../../workflows/research/web_research/) (web-only)
> - Academic: [Citation Network Academic Review](./citation-network-academic-review-workflow.md)
> - Books: [Standalone Book Finding](./standalone-book-finding-workflow.md)
> - Orchestration: [Synthesis Workflow Orchestration](./synthesis-workflow-orchestration.md)

# Deep Research Workflow with Diffusion Algorithm and Parallel Agents

## Intent

Implement a multi-phase research workflow that uses a "diffusion algorithm" to iteratively expand and consolidate research through parallel agents, with memory-first search and quality-tiered execution.

## Motivation

Deep research tasks require:
1. **Iterative refinement**: Single-pass research often misses important angles
2. **Parallel execution**: Multiple research questions can be pursued simultaneously
3. **Knowledge reuse**: Existing knowledge should inform research direction
4. **Quality control**: Research depth should be configurable without changing architecture
5. **Graceful completion**: The system should know when to stop

The diffusion algorithm solves these by:
- **Diffusion out**: Generating research questions and dispatching to parallel agents
- **Diffusion in**: Aggregating findings and refining the draft report
- **Self-regulation**: Using completeness scores and iteration limits to terminate

## Applicability

Use this pattern when:
- Research requires multiple rounds of investigation
- Questions can be pursued independently in parallel
- You have existing knowledge stores that should be consulted first
- Research depth needs to be configurable (quick to comprehensive)
- Results need aggregation and synthesis

Do NOT use this pattern when:
- Research is simple enough for a single LLM call
- Questions must be answered sequentially (dependent answers)
- No existing knowledge base exists to consult
- Cost is more important than thoroughness

## Structure

```
                                    +----------------+
                                    |     START      |
                                    +-------+--------+
                                            |
                                            v
                                 +----------+----------+
                                 |   clarify_intent    |
                                 +----------+----------+
                                            |
                                            v
                                 +----------+----------+
                                 |    create_brief     |
                                 +----------+----------+
                                            |
                                            v
                                 +----------+----------+
                                 |   search_memory     | ← Memory-first pattern
                                 +----------+----------+
                                            |
                                            v
                                 +----------+----------+
                                 |    iterate_plan     | ← Customizes based on memory
                                 +----------+----------+
                                            |
                                            v
                           +----------------+----------------+
                           |           SUPERVISOR           |◄─────────┐
                           | (diffusion algorithm control)  |          │
                           +-----+--------+--------+--------+          │
                                 │        │        │                   │
            conduct_research     │        │        │  research_complete│
               (Send x 3)        │        │        │                   │
                                 v        v        v                   │
                         +───────+──┐ ┌───+───┐ ┌──+───────┐           │
                         │researcher│ │refine_│ │ final_  │           │
                         │   (1)    │ │ draft │ │ report  │           │
                         +────┬─────┘ └───┬───┘ └────┬────┘           │
                              │           │          │                 │
                         +────┴─────┐     │          v                 │
                         │researcher│     │   ┌──────┴──────┐          │
                         │   (2)    │     │   │ save_findings│         │
                         +────┬─────┘     │   └──────┬──────┘          │
                              │           │          │                 │
                         +────┴─────┐     │          v                 │
                         │researcher│     │       ┌──┴──┐              │
                         │   (3)    │     │       │ END │              │
                         +────┬─────┘     │       └─────┘              │
                              │           │                            │
                              v           │                            │
                    ┌─────────┴────────┐  │                            │
                    │aggregate_findings├──┴────────────────────────────┘
                    └──────────────────┘
```

## Implementation

### Step 1: Define State with Annotated Reducers

State keys that receive parallel writes MUST use reducers.

```python
"""State schemas for deep research workflow."""

from datetime import datetime
from operator import add
from typing import Annotated, Any, Literal, Optional
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class DiffusionState(TypedDict):
    """State for tracking diffusion algorithm progress."""
    iteration: int
    max_iterations: int
    completeness_score: float  # 0-1 estimated completeness
    areas_explored: list[str]
    areas_to_explore: list[str]


class ResearchFinding(TypedDict):
    """Compressed finding from a researcher."""
    question_id: str
    finding: str
    sources: list[dict]
    confidence: float
    gaps: list[str]


class ResearchQuestion(TypedDict):
    """A research question for a researcher agent."""
    question_id: str
    question: str
    context: str
    priority: int


class DeepResearchState(TypedDict):
    """Main workflow state."""
    # Input
    input: dict  # query, depth, max_sources, etc.

    # Research phases
    research_brief: Optional[dict]
    memory_findings: list[dict]
    memory_context: str
    research_plan: Optional[str]

    # Parallel writes - MUST use Annotated reducers
    pending_questions: list[ResearchQuestion]
    research_findings: Annotated[list[ResearchFinding], add]
    errors: Annotated[list[dict], add]
    supervisor_messages: Annotated[list[BaseMessage], add_messages]

    # Diffusion tracking
    diffusion: DiffusionState
    draft_report: Optional[dict]

    # Output
    final_report: Optional[str]
    citations: list[dict]

    # Metadata
    current_status: str
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
```

### Step 2: Implement Parallel Fan-Out with Send()

The supervisor routes to multiple researcher agents using LangGraph's `Send()`.

```python
"""Routing functions for the research workflow."""

from typing import Any
from langgraph.types import Send
from workflows.research.state import DeepResearchState, ResearcherState

MAX_CONCURRENT_RESEARCHERS = 3


def route_supervisor_action(state: DeepResearchState) -> str | list[Send]:
    """Route based on supervisor's chosen action."""
    current_status = state.get("current_status", "")

    if current_status == "conduct_research":
        # Fan out to researcher agents (max 3 parallel)
        pending = state.get("pending_questions", [])[:MAX_CONCURRENT_RESEARCHERS]

        if not pending:
            return "final_report"

        # Create Send() for each researcher - each gets isolated state
        return [
            Send("researcher", ResearcherState(
                question=q,
                search_queries=[],
                search_results=[],
                scraped_content=[],
                finding=None,
            ))
            for q in pending
        ]

    elif current_status == "refine_draft":
        return "refine_draft"

    elif current_status == "research_complete":
        return "final_report"

    else:
        return "supervisor"
```

### Step 3: Implement Supervisor with Diffusion Algorithm

The supervisor uses a high-capability model for strategic reasoning.

```python
"""Supervisor node implementing the diffusion algorithm."""

import logging
from typing import Any
from workflows.research.state import DeepResearchState, DiffusionState, ResearchQuestion
from workflows.shared.llm_utils import ModelTier, get_llm

logger = logging.getLogger(__name__)
MAX_CONCURRENT_RESEARCHERS = 3


async def supervisor(state: DeepResearchState) -> dict[str, Any]:
    """Supervisor agent coordinating research via diffusion algorithm.

    Actions:
    - conduct_research: Generate questions, dispatch to researchers
    - refine_draft: Update draft report with new findings
    - research_complete: Signal completion
    """
    diffusion = state.get("diffusion", {})
    iteration = diffusion.get("iteration", 0)
    max_iterations = diffusion.get("max_iterations", 4)

    # Check termination condition
    if iteration >= max_iterations:
        logger.info(f"Max iterations ({max_iterations}) reached - completing")
        return {
            "current_status": "research_complete",
            "diffusion": {**diffusion, "iteration": iteration},
        }

    # Use OPUS for strategic reasoning
    llm = get_llm(ModelTier.OPUS)

    # Build context from state
    brief = state.get("research_brief", {})
    findings = state.get("research_findings", [])
    draft = state.get("draft_report")

    prompt = _build_supervisor_prompt(
        brief=brief,
        findings=findings,
        draft=draft,
        iteration=iteration,
        max_iterations=max_iterations,
        diffusion=diffusion,
    )

    try:
        response = await llm.ainvoke([{"role": "user", "content": prompt}])
        action, action_data = _parse_supervisor_response(response.content, brief)

        if action == "conduct_research":
            questions = _build_research_questions(action_data, iteration)
            return {
                "pending_questions": questions,
                "diffusion": {
                    **diffusion,
                    "iteration": iteration + 1,
                    "areas_explored": diffusion.get("areas_explored", []) +
                                      [q["question"][:50] for q in questions],
                },
                "current_status": "conduct_research",
            }

        elif action == "refine_draft":
            return {
                "draft_report": action_data.get("draft"),
                "diffusion": {
                    **diffusion,
                    "completeness_score": action_data.get("completeness", 0.5),
                },
                "current_status": "refine_draft",
            }

        elif action == "research_complete":
            return {
                "diffusion": {**diffusion, "completeness_score": 1.0},
                "current_status": "research_complete",
            }

    except Exception as e:
        logger.error(f"Supervisor failed: {e}")
        # Graceful fallback
        if iteration < 1:
            return {
                "pending_questions": [_fallback_question(brief)],
                "diffusion": {**diffusion, "iteration": iteration + 1},
                "errors": [{"node": "supervisor", "error": str(e)}],
                "current_status": "conduct_research",
            }
        else:
            return {
                "diffusion": {**diffusion, "completeness_score": 0.7},
                "errors": [{"node": "supervisor", "error": str(e)}],
                "current_status": "research_complete",
            }


def _fallback_question(brief: dict) -> ResearchQuestion:
    """Create fallback question when supervisor fails."""
    return ResearchQuestion(
        question_id="q_fallback",
        question=brief.get("topic", "Research topic"),
        context="Fallback due to supervisor error",
        priority=1,
    )
```

### Step 4: Implement Researcher Subgraph

Each researcher is an independent subgraph with linear flow.

```python
"""Individual researcher agent subgraph."""

from langgraph.graph import END, START, StateGraph
from workflows.research.state import ResearcherState, ResearchFinding


async def generate_queries(state: ResearcherState) -> dict:
    """Generate search queries from question."""
    # Use HAIKU for cost-effective query generation
    llm = get_llm(ModelTier.HAIKU)
    # ... generate queries
    return {"search_queries": queries}


async def execute_searches(state: ResearcherState) -> dict:
    """Execute web searches."""
    # ... search via Firecrawl
    return {"search_results": results}


async def scrape_pages(state: ResearcherState) -> dict:
    """Scrape top results for full content."""
    # ... scrape pages
    return {"scraped_content": content}


async def compress_findings(state: ResearcherState) -> dict:
    """Compress research into structured finding."""
    # Use SONNET for quality compression
    llm = get_llm(ModelTier.SONNET)
    # ... compress findings

    finding = ResearchFinding(
        question_id=state["question"]["question_id"],
        finding=compressed_text,
        sources=sources,
        confidence=confidence,
        gaps=gaps,
    )

    # Return as list for aggregation via add reducer
    return {"finding": finding, "research_findings": [finding]}


def create_researcher_subgraph() -> StateGraph:
    """Create researcher agent subgraph."""
    builder = StateGraph(ResearcherState)

    builder.add_node("generate_queries", generate_queries)
    builder.add_node("execute_searches", execute_searches)
    builder.add_node("scrape_pages", scrape_pages)
    builder.add_node("compress_findings", compress_findings)

    builder.add_edge(START, "generate_queries")
    builder.add_edge("generate_queries", "execute_searches")
    builder.add_edge("execute_searches", "scrape_pages")
    builder.add_edge("scrape_pages", "compress_findings")
    builder.add_edge("compress_findings", END)

    return builder.compile()


researcher_subgraph = create_researcher_subgraph()
```

### Step 5: Construct Main Graph with Aggregation

```python
"""Main deep research workflow graph."""

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from workflows.research.state import DeepResearchState
from workflows.research.nodes import (
    clarify_intent, create_brief, search_memory_node,
    iterate_plan, supervisor, refine_draft, final_report, save_findings,
)
from workflows.research.subgraphs.researcher import researcher_subgraph


def aggregate_researcher_findings(state: DeepResearchState) -> dict:
    """Sync point after parallel researchers complete."""
    findings = state.get("research_findings", [])
    logger.info(f"Aggregated {len(findings)} research findings")

    return {
        "pending_questions": [],  # Clear queue
        "current_status": "supervising",
    }


def create_deep_research_graph():
    """Create the main deep research workflow graph."""
    builder = StateGraph(DeepResearchState)

    # Add nodes
    builder.add_node("clarify_intent", clarify_intent)
    builder.add_node("create_brief", create_brief)
    builder.add_node("search_memory", search_memory_node)
    builder.add_node("iterate_plan", iterate_plan)
    builder.add_node(
        "supervisor",
        supervisor,
        retry=RetryPolicy(max_attempts=3, backoff_factor=2.0),
    )
    builder.add_node("researcher", researcher_subgraph)
    builder.add_node("aggregate_findings", aggregate_researcher_findings)
    builder.add_node("refine_draft", refine_draft)
    builder.add_node(
        "final_report",
        final_report,
        retry=RetryPolicy(max_attempts=2, backoff_factor=2.0),
    )
    builder.add_node("save_findings", save_findings)

    # Entry flow
    builder.add_edge(START, "clarify_intent")
    builder.add_edge("clarify_intent", "create_brief")
    builder.add_edge("create_brief", "search_memory")
    builder.add_edge("search_memory", "iterate_plan")
    builder.add_edge("iterate_plan", "supervisor")

    # Supervisor routing (diffusion loop)
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor_action,
        ["researcher", "refine_draft", "final_report", "supervisor"],
    )

    # Researchers converge to aggregation
    builder.add_edge("researcher", "aggregate_findings")
    builder.add_edge("aggregate_findings", "supervisor")

    # Refine loops back to supervisor
    builder.add_edge("refine_draft", "supervisor")

    # Final stages
    builder.add_edge("final_report", "save_findings")
    builder.add_edge("save_findings", END)

    return builder.compile()


deep_research_graph = create_deep_research_graph()
```

## Complete Example

```python
"""Run deep research with quality tiers."""

async def deep_research(
    query: str,
    depth: Literal["quick", "standard", "comprehensive"] = "standard",
    max_sources: int = 20,
) -> DeepResearchState:
    """
    Run deep research on a topic.

    Args:
        query: Research question or topic
        depth: Research depth
            - "quick": 2 iterations, ~5 min
            - "standard": 4 iterations, ~15 min
            - "comprehensive": 8 iterations, ~30+ min
        max_sources: Maximum web sources

    Returns:
        DeepResearchState with final_report and citations
    """
    max_iterations = {"quick": 2, "standard": 4, "comprehensive": 8}[depth]

    initial_state: DeepResearchState = {
        "input": {
            "query": query,
            "depth": depth,
            "max_sources": max_sources,
        },
        "research_brief": None,
        "memory_findings": [],
        "memory_context": "",
        "research_plan": None,
        "pending_questions": [],
        "research_findings": [],
        "supervisor_messages": [],
        "diffusion": DiffusionState(
            iteration=0,
            max_iterations=max_iterations,
            completeness_score=0.0,
            areas_explored=[],
            areas_to_explore=[],
        ),
        "draft_report": None,
        "final_report": None,
        "citations": [],
        "errors": [],
        "started_at": datetime.utcnow(),
        "completed_at": None,
        "current_status": "starting",
    }

    result = await deep_research_graph.ainvoke(initial_state)
    return result


# Usage
result = await deep_research(
    "Impact of AI on software engineering jobs in 2025",
    depth="standard",
)
print(result["final_report"])
```

## Consequences

### Benefits

- **Iterative refinement**: Multiple research rounds catch different angles
- **Parallel efficiency**: Up to 3 researchers work simultaneously
- **Knowledge reuse**: Memory-first search avoids redundant research
- **Self-regulation**: Completeness scores and iteration limits ensure termination
- **Graceful degradation**: Every node has fallback behavior
- **Quality tiers**: Same architecture serves quick explorations to deep dives

### Trade-offs

- **Complexity**: Multiple states, subgraphs, and routing logic
- **Cost**: OPUS calls for supervisor are expensive but necessary for quality
- **Latency**: Iterative approach takes longer than single-pass
- **State accumulation**: Findings grow across iterations (may need compression)

### Alternatives

- **Single-pass research**: Simpler but less thorough
- **Sequential researchers**: Avoids parallel complexity but slower
- **Fixed question set**: Simpler routing but less adaptive
- **No memory integration**: Simpler but may repeat known information

## Related Patterns

- [Compression-Level Index Routing](../stores/compression-level-index-routing.md) - Store integration for findings
- [Conditional Development Tracing](../llm-interaction/conditional-development-tracing.md) - LangSmith tracing for debugging
- [Parallel AI Search Integration](../data-pipeline/parallel-ai-search-integration.md) - Multi-source search in researcher agents

## Known Uses in Thala

- `workflows/research/graph.py`: Main graph construction
- `workflows/research/state.py`: State definitions with reducers
- `workflows/research/nodes/supervisor.py`: Diffusion algorithm implementation
- `workflows/research/subgraphs/researcher.py`: Parallel researcher agents
- `langchain_tools/deep_research.py`: LangChain tool wrapper

## References

- [LangGraph Send() Documentation](https://langchain-ai.github.io/langgraph/concepts/low_level/#send)
- [LangGraph Reducers](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers)
- [LangGraph Subgraphs](https://langchain-ai.github.io/langgraph/concepts/low_level/#subgraphs)
