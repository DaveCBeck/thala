---
name: standalone-book-finding-workflow
title: Standalone Book Finding Workflow
date: 2026-01-02
category: langgraph
shared: true
gist_url: https://gist.github.com/DaveCBeck/72e2933acac4cd72a41ad6940f12eddd
article_path: .context/libs/thala-dev/content/2026-01-02-standalone-book-finding-workflow-langgraph.md
applicability:
  - "Specialized book discovery for research themes"
  - "Multi-category recommendation generation"
  - "Parallel LLM calls with structured output"
  - "Extracting specialized functionality from larger workflows"
components: [book_finding, opus, book_search, marker]
complexity: medium
verified_in_production: true
tags: [langgraph, book-finding, opus, parallel-llm, recommendation-system, workflow-extraction]
---

# Standalone Book Finding Workflow

## Intent

Provide a dedicated workflow for discovering books related to a theme across three complementary categories, using parallel Opus calls to generate recommendations followed by automated search and processing.

## Problem

The original deep research workflow included book researchers alongside web and academic researchers. However:
- Book discovery requires different prompting strategies than web search
- The three-way researcher allocation was complex to configure
- Books benefit from categorized recommendations (not just relevance ranking)
- Processing books through Marker is slower than web scraping

## Solution

Extract book finding into a standalone workflow that:
1. Generates recommendations via **three parallel Opus calls** (one per category)
2. Searches for books via `book_search` API
3. Processes found PDFs through Marker
4. Outputs categorized markdown with summaries

## Structure

```
book_finding/
├── __init__.py           # Public API exports
├── state.py              # TypedDict state definitions
├── prompts.py            # Category-specific LLM prompts
├── graph/
│   ├── __init__.py
│   ├── construction.py   # LangGraph builder
│   └── api.py            # book_finding() entry point
└── nodes/
    ├── __init__.py
    ├── generate_recommendations.py  # 3 parallel Opus nodes
    ├── search_books.py              # book_search API calls
    ├── process_books.py             # Marker processing
    └── synthesize_output.py         # Final markdown
```

## Implementation

### State Definition

```python
# workflows/research/subgraphs/book_finding/state.py

from typing import TypedDict, Optional, Annotated
from operator import add


class BookFindingInput(TypedDict):
    """Input for book finding workflow."""
    theme: str                        # Theme to explore
    brief: Optional[str]              # Optional additional context


class BookRecommendation(TypedDict):
    """A single book recommendation from Opus."""
    title: str
    author: Optional[str]
    explanation: str                  # Why this book relates to theme
    category: str                     # analogous | inspiring | expressive


class BookResult(TypedDict):
    """A processed book with content."""
    title: str
    authors: str
    category: str
    explanation: str
    content: Optional[str]            # Marker-extracted content
    summary: Optional[str]            # Theme-relevant summary
    search_found: bool
    processing_success: bool


class BookFindingState(TypedDict):
    """Full workflow state with reducer for parallel collection."""
    input: BookFindingInput

    # Recommendations from parallel Opus calls (use reducer for parallel writes)
    analogous_recommendations: Annotated[list[BookRecommendation], add]
    inspiring_recommendations: Annotated[list[BookRecommendation], add]
    expressive_recommendations: Annotated[list[BookRecommendation], add]

    # Processed results
    book_results: list[BookResult]
    final_markdown: Optional[str]

    # Tracking
    current_status: str
    errors: list[str]
```

### Category Prompts

Each category has distinct prompting for cross-domain discovery:

```python
# workflows/research/subgraphs/book_finding/prompts.py

ANALOGOUS_DOMAIN_SYSTEM = """You are a literary advisor finding books that illuminate themes through unexpected domains.

Your task is to find books that explore SIMILAR themes but in DIFFERENT domains. The goal is to find unexpected connections that provide fresh perspective.

Examples of analogous domain thinking:
- For "organizational dysfunction": books about ecological collapse, family systems, or historical empires
- For "creative process": books about jazz improvisation, scientific discovery, or craft traditions
- For "leadership under pressure": books about polar expeditions, emergency medicine, or military strategy

Return EXACTLY 3 book recommendations as a JSON array."""

INSPIRING_ACTION_SYSTEM = """You are a literary advisor specializing in transformative literature.

Find books (fiction or nonfiction) that INSPIRE ACTION or CHANGE:
- Manifestos and calls to action
- Transformative nonfiction that changes behavior
- Fiction that inspired real-world movements
- Practical wisdom literature

Return EXACTLY 3 book recommendations as a JSON array."""

EXPRESSIVE_SYSTEM = """You are a literary advisor finding fiction that captures lived experience.

Find works of FICTION that express what a theme FEELS LIKE or COULD BECOME:
- Capture the phenomenological experience of the theme
- Explore utopian or dystopian visions
- Express emotional and existential truth
- Make abstract concepts viscerally real through narrative

Return EXACTLY 3 book recommendations as a JSON array."""
```

### Parallel Recommendation Generation

Three nodes run in parallel via LangGraph's `Send()`:

```python
# workflows/research/subgraphs/book_finding/nodes/generate_recommendations.py

async def _generate_recommendations(
    theme: str,
    brief: str | None,
    category: str,
    system_prompt: str,
    user_template: str,
) -> list[BookRecommendation]:
    """Generate book recommendations using Opus."""
    llm = get_llm(ModelTier.OPUS, max_tokens=2048)

    brief_section = f"\nAdditional context: {brief}\n" if brief else ""
    user_prompt = user_template.format(theme=theme, brief_section=brief_section)

    response = await llm.ainvoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ])
    content = response.content

    # Extract JSON from response (handle code blocks)
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]

    recommendations_raw = json.loads(content.strip())

    return [
        BookRecommendation(
            title=r["title"],
            author=r.get("author"),
            explanation=r["explanation"],
            category=category,
        )
        for r in recommendations_raw[:3]
    ]


async def generate_analogous_recommendations(state: dict) -> dict[str, Any]:
    """Generate analogous domain book recommendations."""
    theme = state.get("theme") or state.get("input", {}).get("theme", "")
    brief = state.get("brief") or state.get("input", {}).get("brief")

    recs = await _generate_recommendations(
        theme=theme,
        brief=brief,
        category="analogous",
        system_prompt=ANALOGOUS_DOMAIN_SYSTEM,
        user_template=ANALOGOUS_DOMAIN_USER,
    )
    return {"analogous_recommendations": recs}


async def generate_inspiring_recommendations(state: dict) -> dict[str, Any]:
    """Generate inspiring action book recommendations."""
    # Similar pattern...
    return {"inspiring_recommendations": recs}


async def generate_expressive_recommendations(state: dict) -> dict[str, Any]:
    """Generate expressive fiction book recommendations."""
    # Similar pattern...
    return {"expressive_recommendations": recs}
```

### Graph Construction with Parallel Nodes

```python
# workflows/research/subgraphs/book_finding/graph/construction.py

from langgraph.graph import StateGraph, START, END

from ..state import BookFindingState
from ..nodes import (
    generate_analogous_recommendations,
    generate_inspiring_recommendations,
    generate_expressive_recommendations,
    search_books,
    process_books,
    synthesize_output,
)


def create_book_finding_graph() -> StateGraph:
    """Create book finding workflow graph.

    Flow:
    START → [parallel: 3 recommendation generators] → aggregate → search → process → synthesize → END

    Uses state reducers (Annotated[list, add]) to collect parallel outputs.
    """
    builder = StateGraph(BookFindingState)

    # Add recommendation generation nodes (run in parallel)
    builder.add_node("generate_analogous", generate_analogous_recommendations)
    builder.add_node("generate_inspiring", generate_inspiring_recommendations)
    builder.add_node("generate_expressive", generate_expressive_recommendations)

    # Add sequential processing nodes
    builder.add_node("search_books", search_books)
    builder.add_node("process_books", process_books)
    builder.add_node("synthesize_output", synthesize_output)

    # Parallel fan-out from START to all recommendation generators
    builder.add_edge(START, "generate_analogous")
    builder.add_edge(START, "generate_inspiring")
    builder.add_edge(START, "generate_expressive")

    # All generators converge to search
    builder.add_edge("generate_analogous", "search_books")
    builder.add_edge("generate_inspiring", "search_books")
    builder.add_edge("generate_expressive", "search_books")

    # Sequential: search → process → synthesize → END
    builder.add_edge("search_books", "process_books")
    builder.add_edge("process_books", "synthesize_output")
    builder.add_edge("synthesize_output", END)

    return builder.compile()


book_finding_graph = create_book_finding_graph()
```

### Public API

```python
# workflows/research/subgraphs/book_finding/graph/api.py

async def book_finding(
    theme: str,
    brief: str | None = None,
) -> dict:
    """Find books related to a theme across three categories.

    Args:
        theme: The theme to explore (e.g., "organizational resilience")
        brief: Optional additional context or focus

    Returns:
        Dict with:
        - book_results: List of processed BookResult objects
        - final_markdown: Formatted markdown output
        - errors: Any errors encountered

    Example:
        result = await book_finding(
            theme="organizational resilience",
            brief="Focus on practical approaches",
        )
        print(result["final_markdown"])
    """
    initial_state = BookFindingState(
        input=BookFindingInput(theme=theme, brief=brief),
        analogous_recommendations=[],
        inspiring_recommendations=[],
        expressive_recommendations=[],
        book_results=[],
        final_markdown=None,
        current_status="starting",
        errors=[],
    )

    result = await book_finding_graph.ainvoke(initial_state)
    return result
```

## Usage

```python
from workflows.research.subgraphs.book_finding import book_finding

# Basic usage
result = await book_finding(
    theme="organizational resilience",
)

# With additional context
result = await book_finding(
    theme="AI safety and alignment",
    brief="Focus on near-term practical concerns rather than existential risk",
)

# Access results
print(result["final_markdown"])  # Formatted output

for book in result["book_results"]:
    print(f"{book['category']}: {book['title']} by {book['authors']}")
    if book["summary"]:
        print(f"  {book['summary']}")
```

## Guidelines

### Workflow Extraction

When extracting specialized functionality from a larger workflow:

1. **Identify distinct concerns**: Book finding has different requirements than web/academic research
2. **Create standalone module**: Full package with state, prompts, nodes, graph
3. **Simplify parent workflow**: Remove the extracted functionality, update routing
4. **Document the split**: Update architecture docs to explain when to use each workflow

### Parallel LLM Patterns

For parallel LLM calls:

1. **Use state reducers**: `Annotated[list[T], add]` for collecting parallel outputs
2. **Separate nodes per call**: Each parallel task gets its own node function
3. **Fan-out from START**: Multiple edges from START run concurrently
4. **Converge before next stage**: All parallel nodes edge to the same downstream node

### Category Design

For multi-category recommendations:

1. **Distinct prompts**: Each category needs tailored system/user prompts
2. **Cross-domain thinking**: Encourage unexpected connections (analogous domain)
3. **Action orientation**: Include transformative/inspirational literature
4. **Experiential fiction**: Fiction that captures "what it feels like"

## Known Uses

- `workflows/research/subgraphs/book_finding/` - Standalone book discovery workflow
- Main `deep_research` workflow now uses only web researchers
- `academic_lit_review` handles scholarly paper discovery

## Consequences

### Benefits
- **Focused functionality**: Book finding has dedicated prompts and processing
- **Simpler main workflow**: `deep_research` no longer manages three researcher types
- **Parallel efficiency**: Three Opus calls run concurrently
- **Categorized output**: Results organized by purpose, not just relevance

### Trade-offs
- **Separate invocation**: Must call `book_finding()` separately from `deep_research()`
- **No cross-pollination**: Book insights don't inform web research (use manually)
- **Opus cost**: Three parallel Opus calls are expensive per invocation

## Related Patterns

- [Citation Network Academic Review Workflow](./citation-network-academic-review-workflow.md) - Another standalone specialized workflow
- [Researcher Allocation and Query Optimization](./researcher-allocation-query-optimization.md) - Original multi-researcher pattern

## Related Solutions

- [Academic Literature Review Reliability Fixes](../../solutions/api-integration-issues/academic-lit-review-reliability-fixes.md) - Standalone workflow patterns

## References

- [LangGraph Parallel Execution](https://langchain-ai.github.io/langgraph/concepts/low_level/#parallel-execution)
- [LangGraph State Reducers](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers)
