---
name: researcher-allocation-query-optimization
title: "Researcher Allocation and Query Optimization"
date: 2025-12-27
category: langgraph
shared: true
gist_url: https://gist.github.com/DaveCBeck/c7b348bb0136261495240ca9709ff928
article_path: .context/libs/thala-dev/content/2025-12-27-researcher-allocation-langgraph.md
applicability:
  - "When multi-source research requires strategic allocation of specialized researchers based on topic type"
  - "When topics span STEM, humanities, and arts where different source types have varying relevance"
  - "When user-controlled research workflows need explicit resource allocation overrides"
components: [supervisor_decision, query_factory, allocation_parser, researcher_subgraph]
complexity: moderate
verified_in_production: false
stale: true
stale_date: 2026-01-29
stale_reason: "Allocation format changed from 3-digit (web/academic/book) to 1-digit (web count only); academic and book are standalone workflows"
related_solutions: []
tags: [researcher-allocation, query-optimization, specialized-prompts, supervisor-decision, langgraph, multi-agent, factory-pattern, priority-hierarchy]
---

> **STALE**: This documentation describes an older allocation system.
>
> **What changed:** The 3-digit allocation format ("210" = 2 web, 1 academic, 0 book) was replaced with a single-digit format for web-only allocation. Academic and book research are now invoked as standalone workflows, not through allocation.
> **Date:** 2026-01-29
> **Still valid:** The `create_generate_queries()` factory pattern and type-specific prompts (`GENERATE_WEB_QUERIES_SYSTEM`, etc.) still exist and work.
> **Current patterns:**
> - Web research allocation: Single digit (1-3 web researchers)
> - Academic: Invoke [Citation Network Academic Review](./citation-network-academic-review-workflow.md) directly
> - Books: Invoke [Standalone Book Finding](./standalone-book-finding-workflow.md) directly

# Researcher Allocation and Query Optimization

## Intent

Intelligently allocate specialized researchers (web, academic, book) through a priority-based system combining user specification, LLM topic-type analysis, and researcher-specific query optimization prompts to maximize source relevance and research quality.

## Motivation

Research topics vary widely in optimal source types:

1. **Topic diversity**: Current events need web sources; humanities need academic journals and books
2. **Source expertise**: Web queries differ from OpenAlex queries differ from book searches
3. **User control**: Power users know their domain and want explicit control
4. **LLM intelligence**: For unknown topics, LLM can analyze and allocate appropriately

This pattern solves all four by:
- Providing user-configurable allocation via simple "300" style strings
- Enabling LLM-decided allocation based on topic analysis
- Implementing type-specific query generation prompts
- Establishing a clear priority hierarchy (user > LLM > default)

## Applicability

Use this pattern when:
- Research spans multiple source types (web, academic, book)
- Topic type determines optimal source allocation
- Users need both automatic and manual control options
- Query quality varies by target search system

Do NOT use this pattern when:
- Single source type suffices (no allocation needed)
- All topics use identical query strategies
- User control adds unnecessary complexity
- Research volume doesn't justify optimization

## Structure

```
                      ┌─────────────────────────────┐
                      │     User Input Request      │
                      │  researcher_allocation="210"│
                      └──────────────┬──────────────┘
                                     │
                      ┌──────────────▼──────────────┐
                      │    parse_allocation()       │
                      │    Validates "210" format   │
                      │    Returns {web:2,acad:1}   │
                      └──────────────┬──────────────┘
                                     │
         PRIORITY 1: User-specified ─┴─ (skip LLM if user provided)
                                     │
                      ┌──────────────▼──────────────┐
                      │   SupervisorDecision LLM    │
                      │   web_researchers: 2        │
                      │   academic_researchers: 1   │
                      │   book_researchers: 0       │
                      │   allocation_reasoning: ... │
                      └──────────────┬──────────────┘
                                     │
         PRIORITY 2: LLM-decided ────┴─ (use if no user allocation)
                                     │
         PRIORITY 3: Default (1,1,1) ─┴─ (fallback)
                                     │
                      ┌──────────────▼──────────────┐
                      │   route_supervisor_action   │
                      │   Dispatch via Send()       │
                      └──────────────┬──────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │                           │                           │
         ▼                           ▼                           ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│ Web Researcher  │      │Academic Research│      │ Book Researcher │
│ create_generate │      │ create_generate │      │ create_generate │
│ _queries("web") │      │ _queries("acad")│      │ _queries("book")│
└────────┬────────┘      └────────┬────────┘      └────────┬────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│GENERATE_WEB_    │      │GENERATE_ACADEMIC│      │GENERATE_BOOK_   │
│QUERIES_SYSTEM   │      │_QUERIES_SYSTEM  │      │QUERIES_SYSTEM   │
│                 │      │                 │      │                 │
│• Google/Bing    │      │• OpenAlex terms │      │• Broad topics   │
│• Non-academic   │      │• Methodology    │      │• Foundational   │
│• Current events │      │• Domain-specific│      │• Classic works  │
└─────────────────┘      └─────────────────┘      └─────────────────┘
```

## Implementation

### Step 1: Define Allocation Parser

Parse user-friendly allocation strings into structured TypedDict:

```python
# workflows/research/state.py

from typing_extensions import TypedDict


class ResearcherAllocation(TypedDict):
    """Allocation of researchers by type."""

    web_count: int       # Number of web researchers (0-3)
    academic_count: int  # Number of academic researchers (0-3)
    book_count: int      # Number of book researchers (0-3)


def parse_allocation(allocation_str: str) -> ResearcherAllocation:
    """Parse allocation string like '111', '210', '300' into ResearcherAllocation.

    Format: 3-character string where each digit represents count of:
    - Position 0: web researchers
    - Position 1: academic researchers
    - Position 2: book researchers

    Total must not exceed 3 (MAX_CONCURRENT_RESEARCHERS).

    Examples:
        >>> parse_allocation("111")
        {'web_count': 1, 'academic_count': 1, 'book_count': 1}
        >>> parse_allocation("210")
        {'web_count': 2, 'academic_count': 1, 'book_count': 0}
        >>> parse_allocation("300")
        {'web_count': 3, 'academic_count': 0, 'book_count': 0}
    """
    if not allocation_str or len(allocation_str) != 3:
        raise ValueError(
            f"Allocation must be a 3-character string (e.g., '111', '210'), "
            f"got: {allocation_str!r}"
        )

    try:
        web = int(allocation_str[0])
        academic = int(allocation_str[1])
        book = int(allocation_str[2])
    except ValueError:
        raise ValueError(
            f"Allocation must contain only digits (0-3), got: {allocation_str!r}"
        )

    if not all(0 <= x <= 3 for x in [web, academic, book]):
        raise ValueError(
            f"Each allocation digit must be 0-3, got: {allocation_str!r}"
        )

    total = web + academic + book
    if total > 3:
        raise ValueError(
            f"Total allocation must not exceed 3, got {total} from '{allocation_str}'"
        )

    if total == 0:
        raise ValueError(
            f"Total allocation must be at least 1, got 0 from '{allocation_str}'"
        )

    return ResearcherAllocation(
        web_count=web,
        academic_count=academic,
        book_count=book,
    )
```

### Step 2: Extend SupervisorDecision with Allocation Fields

Add allocation fields to the Pydantic structured output schema:

```python
# workflows/research/state.py

from pydantic import BaseModel, Field
from typing import Optional


class SupervisorDecision(BaseModel):
    """Supervisor's structured decision for the next research step."""

    action: Literal["conduct_research", "refine_draft", "research_complete"]
    reasoning: str = Field(description="Brief explanation of why this action was chosen.")

    # Research questions (for conduct_research action)
    research_questions: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="1-3 specific research questions to investigate.",
    )

    # Researcher allocation (for conduct_research action)
    web_researchers: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Number of web researchers (0-3). Best for current events, tech trends, tools, products.",
    )
    academic_researchers: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Number of academic researchers (0-3). Best for peer-reviewed research, clinical studies.",
    )
    book_researchers: int = Field(
        default=1,
        ge=0,
        le=3,
        description="Number of book researchers (0-3). Best for foundational theory, historical context.",
    )
    allocation_reasoning: Optional[str] = Field(
        default=None,
        description="Brief explanation of why this researcher allocation was chosen based on topic type.",
    )

    # Draft refinement (for refine_draft action)
    draft_updates: Optional[str] = Field(default=None)
    remaining_gaps: list[str] = Field(default_factory=list)
```

### Step 3: Add Allocation Guidance to Supervisor Prompt

Guide the LLM to make topic-appropriate allocation decisions:

```python
# workflows/research/prompts/base.py

SUPERVISOR_ALLOCATION_GUIDANCE = """
<Researcher Allocation>
When choosing "conduct_research", allocate researchers based on topic suitability.
Total allocation must not exceed 3 researchers.

**Web researchers** (Firecrawl + Perplexity): Current events, technology trends,
product comparisons, tools/software, company info, news, practitioner blogs.
NOT for academic papers - those go to academic researcher.

**Academic researchers** (OpenAlex): Peer-reviewed research across ALL disciplines:
STEM, social sciences, humanities, arts. Includes journals in literature, philosophy,
musicology, art history, linguistics, etc.

**Book researchers** (book_search): Foundational theory, historical context,
comprehensive overviews, classic works, textbooks, literary criticism,
philosophy, art history, author studies.

Allocation guidelines:
- Current tech/tools/products → Favor web (e.g., web=2, academic=1, book=0)
- Scientific/clinical/medical → Favor academic (e.g., web=1, academic=2, book=0)
- Humanities/arts/literature → Academic + books (e.g., web=0, academic=2, book=1)
- Historical/theoretical/foundational → Include books (e.g., web=1, academic=1, book=1)
- Mixed or general topics → Balanced allocation (web=1, academic=1, book=1)
- Breaking news/current events → Web only (e.g., web=3, academic=0, book=0)
</Researcher Allocation>
"""
```

### Step 4: Create Type-Specific Query Generation Factory

Use a factory function to create researcher-specific query generators:

```python
# workflows/research/subgraphs/researcher_base.py

from typing import Any, Callable, Coroutine

# Map researcher types to their specialized prompts
RESEARCHER_QUERY_PROMPTS = {
    "web": GENERATE_WEB_QUERIES_SYSTEM,
    "academic": GENERATE_ACADEMIC_QUERIES_SYSTEM,
    "book": GENERATE_BOOK_QUERIES_SYSTEM,
}


def create_generate_queries(
    researcher_type: str = "web"
) -> Callable[[ResearcherState], Coroutine[Any, Any, dict[str, Any]]]:
    """Create a generate_queries node function for a specific researcher type.

    Args:
        researcher_type: One of "web", "academic", "book"

    Returns:
        Async function that generates queries optimized for the researcher type.
    """

    async def generate_queries(state: ResearcherState) -> dict[str, Any]:
        """Generate search queries using structured output with validation."""
        question = state["question"]
        language_config = state.get("language_config")

        llm = get_llm(ModelTier.HAIKU)
        structured_llm = llm.with_structured_output(SearchQueries)

        # Get researcher-specific base prompt
        base_prompt = RESEARCHER_QUERY_PROMPTS.get(
            researcher_type, GENERATE_WEB_QUERIES_SYSTEM
        )

        # Build language-aware prompt
        if language_config and language_config["code"] != "en":
            lang_name = language_config["name"]
            prompt = f"""{base_prompt}

Generate queries in {lang_name} to find {lang_name}-language sources.
Write queries naturally in {lang_name}.

Question: {question['question']}
"""
        else:
            prompt = f"""{base_prompt}

Question: {question['question']}
"""

        try:
            result: SearchQueries = await structured_llm.ainvoke(
                [{"role": "user", "content": prompt}]
            )

            # Validate queries against research context
            valid_queries = await validate_queries(
                queries=result.queries,
                research_question=question["question"],
                research_brief=question.get("brief"),
                draft_notes=question.get("context"),
            )

            if not valid_queries:
                logger.warning("All queries invalid, using fallback")
                valid_queries = [question["question"]]

            logger.debug(
                f"Generated {len(valid_queries)} {researcher_type} queries "
                f"for: {question['question'][:50]}..."
            )
            return {"search_queries": valid_queries}

        except Exception as e:
            logger.error(f"Failed to generate {researcher_type} queries: {e}")
            return {"search_queries": [question["question"]]}

    return generate_queries


# Backwards compatibility: default web query generator
async def generate_queries(state: ResearcherState) -> dict[str, Any]:
    """Generate search queries (default: web-optimized)."""
    generator = create_generate_queries("web")
    return await generator(state)
```

### Step 5: Define Type-Specific Query Prompts

Create specialized prompts for each researcher type:

```python
# workflows/research/prompts/base.py

# Web Researcher Query Generation (Firecrawl + Perplexity)
GENERATE_WEB_QUERIES_SYSTEM = """Generate 2-3 web search queries for general search engines.

Focus on finding recent, authoritative NON-ACADEMIC web sources:
- Official websites, documentation, and company pages
- News articles, journalism, and industry publications
- Blog posts from recognized experts and practitioners
- Forums, discussions, and community resources (Reddit, HN, Stack Overflow)
- Product pages, comparisons, and reviews

**For social science, arts, and cultural topics:**
- Popular criticism and reviews (film critics, book reviewers, art critics)
- Enthusiast communities and fan perspectives
- Practitioner insights (artists, musicians, writers discussing their craft)

AVOID academic sources - those are handled by the academic researcher.
Use natural language queries that work well with Google/Bing.
Include year references (2024, 2025) for current topics."""


# Academic Researcher Query Generation (OpenAlex)
GENERATE_ACADEMIC_QUERIES_SYSTEM = """Generate 2-3 search queries optimized for academic literature databases.

OpenAlex searches peer-reviewed research across ALL disciplines.

**For STEM topics:**
- Include methodology terms: "meta-analysis", "RCT", "longitudinal study"
- Use academic phrasing: "effects of X on Y", "relationship between X and Y"

**For humanities & social sciences:**
- Literature/Language: "literary analysis", "narrative theory", "discourse analysis"
- Arts: "aesthetic theory", "art criticism", "musicology", "film studies"
- Social Science: "qualitative study", "ethnography", "case study", "critical theory"
- Philosophy: "phenomenology", "hermeneutics", "epistemology"

Use domain-specific terminology from academic papers.
Avoid colloquial language, product names, and current events."""


# Book Researcher Query Generation (book_search)
GENERATE_BOOK_QUERIES_SYSTEM = """Generate 2-3 search queries optimized for book databases.

Books excel for foundational knowledge, theory, and comprehensive treatments.

**Best suited for:**
- Foundational theory and classic works in any field
- Comprehensive overviews and textbooks
- Historical context and development of ideas
- Philosophy, literary criticism, art history

**For humanities & arts topics:**
- Literature: author studies, genre analysis, literary movements
- Arts: art history, music theory, film criticism, aesthetics
- Philosophy: major philosophers, schools of thought

**Query strategies:**
- Use broad topic terms (books cover topics comprehensively)
- Include "introduction to", "handbook", "companion to" for overviews
- Use established terminology and movement names"""
```

### Step 6: Implement Priority Hierarchy in Supervisor

Handle the priority: user-specified > LLM-decided > default:

```python
# workflows/research/nodes/supervisor.py

async def supervisor(state: DeepResearchState) -> dict[str, Any]:
    """Supervisor node with allocation priority handling."""
    # ... (existing supervisor logic) ...

    if action == "conduct_research":
        # Determine researcher allocation with priority hierarchy
        # PRIORITY 1: User-specified (already in state from parse_allocation)
        current_allocation = state.get("researcher_allocation")

        if not current_allocation and action_data.get("llm_allocation"):
            # PRIORITY 2: Use LLM's allocation decision
            llm_alloc = action_data["llm_allocation"]
            total = (
                llm_alloc["web_count"]
                + llm_alloc["academic_count"]
                + llm_alloc["book_count"]
            )

            # Validate total doesn't exceed 3
            if 0 < total <= 3:
                current_allocation = llm_alloc
                logger.info(
                    f"Using LLM allocation: web={llm_alloc['web_count']}, "
                    f"academic={llm_alloc['academic_count']}, "
                    f"book={llm_alloc['book_count']} "
                    f"({action_data.get('allocation_reasoning', 'no reasoning')})"
                )

        # PRIORITY 3: Default (1,1,1) handled in route_supervisor_action

        result = {
            "pending_questions": questions,
            "diffusion": {...},
            "current_status": "conduct_research",
        }

        # Only update allocation if LLM set it (don't overwrite user-specified)
        if current_allocation and not state.get("researcher_allocation"):
            result["researcher_allocation"] = current_allocation

        return result
```

## Consequences

### Benefits

- **Topic-aware allocation**: LLM analyzes topic and allocates appropriate researchers
- **User override**: Power users can bypass LLM with explicit "300" allocation
- **Optimized queries**: Type-specific prompts improve search precision
- **Clear priority**: Predictable behavior (user > LLM > default)
- **Backwards compatible**: Default generator works without changes
- **Validation**: Both allocation strings and queries are validated

### Trade-offs

- **Prompt maintenance**: Three separate query prompts to maintain
- **Allocation overhead**: LLM must reason about allocation for each iteration
- **User learning curve**: Users must understand "210" notation
- **LLM variability**: Different runs may allocate differently for same topic

### Async Considerations

- **Parallel dispatch**: Allocated researchers run concurrently via `Send()`
- **Factory closure**: Each researcher type gets its own async query function
- **Validation overhead**: Query validation adds latency but improves quality

## Related Patterns

- [Specialized Researcher Pattern](./specialized-researcher-pattern.md) - Base pattern for web/academic/book researchers
- [Deep Research Workflow Architecture](./deep-research-workflow-architecture.md) - Supervisor orchestration and diffusion algorithm
- [Multi-Lingual Research Workflow](./multi-lingual-research-workflow.md) - Similar allocation concept for languages

## Known Uses in Thala

- `workflows/research/state.py`: `parse_allocation()`, `ResearcherAllocation`, `SupervisorDecision`
- `workflows/research/prompts/base.py`: `GENERATE_WEB_QUERIES_SYSTEM`, `GENERATE_ACADEMIC_QUERIES_SYSTEM`, `GENERATE_BOOK_QUERIES_SYSTEM`
- `workflows/research/subgraphs/researcher_base.py`: `create_generate_queries()` factory
- `workflows/research/nodes/supervisor.py`: Allocation priority handling
- `workflows/research/subgraphs/web_researcher.py`: Uses `create_generate_queries("web")`
- `workflows/research/subgraphs/academic_researcher.py`: Uses `create_generate_queries("academic")`
- `workflows/research/subgraphs/book_researcher.py`: Uses `create_generate_queries("book")`

## References

- [LangGraph Send() for Parallel Dispatch](https://python.langchain.com/docs/langgraph/how-tos/send)
- [Pydantic Field Constraints](https://docs.pydantic.dev/latest/concepts/fields/)
