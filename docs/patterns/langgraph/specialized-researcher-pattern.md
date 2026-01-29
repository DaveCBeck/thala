---
name: specialized-researcher-pattern
title: "Specialized Researcher Subgraphs with Source-Appropriate Evaluation"
date: 2025-12-23
category: langgraph
applicability:
  - "When research benefits from multiple source types with different evaluation criteria"
  - "When domain-specific quality signals matter (peer-review vs recency vs author credentials)"
  - "When parallel dispatch to heterogeneous subgraphs is needed with shared utilities"
components: [langgraph_node, langgraph_state, workflow_graph]
complexity: moderate
verified_in_production: false
stale: true
stale_date: 2026-01-29
stale_reason: "Architecture evolved: academic_researcher and book_researcher subgraphs extracted to standalone workflows; only web_researcher remains in web_research workflow"
related_solutions: []
tags: [langgraph, research, subgraphs, specialization, parallel-dispatch, send, allocation, source-evaluation]
shared: true
gist_url: https://gist.github.com/DaveCBeck/db3e35b14cd78ffe8836acebd82fd5d2
article_path: .context/libs/thala-dev/content/2025-12-23-specialized-researcher-subgraphs-langgraph.md
---

> **STALE**: This documentation describes an older architecture.
>
> **What changed:** The 3-way researcher architecture (web, academic, book subgraphs in one workflow) was refactored. Academic and book researchers are now standalone workflows. Only `web_researcher` remains as a subgraph in `web_research`.
> **Date:** 2026-01-29
> **Current patterns:**
> - Web researcher: `workflows/research/web_research/subgraphs/web_researcher.py` (still exists)
> - Academic: [Citation Network Academic Review](./citation-network-academic-review-workflow.md) (standalone)
> - Books: [Standalone Book Finding](./standalone-book-finding-workflow.md) (standalone)

# Specialized Researcher Subgraphs with Source-Appropriate Evaluation

## Intent

Replace a monolithic researcher with specialized subgraphs, each optimized for a specific source type (web, academic, books) with source-appropriate compression prompts and evaluation criteria, while sharing common utilities.

## Motivation

A single researcher handling all source types faces challenges:

1. **Conflicting evaluation criteria**: Web sources need recency checks; academic sources need citation counts; books need author credentials
2. **Source-specific compression**: Different prompts work better for different content types
3. **Allocation control**: Research may need more academic sources for some topics, more web sources for others
4. **Monolithic complexity**: A single file handling all cases becomes unwieldy

This pattern addresses these by:
- Three specialized subgraphs with source-specific prompts
- Configurable allocation per research iteration
- Shared utilities extracted to a common base module

## Applicability

Use this pattern when:
- Different source types require different evaluation criteria
- Research quality depends on source-appropriate compression
- Allocation needs to vary by topic or research depth
- Monolithic code is becoming hard to maintain

Do NOT use this pattern when:
- All sources can be evaluated uniformly
- Only one source type is relevant
- Simplicity is more important than source optimization

## Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Specialized Researcher Architecture                       │
└─────────────────────────────────────────────────────────────────────────────┘

                            supervisor
                                │
                    ┌───────────┼───────────┐
                    │           │           │
        conduct_research (allocation: web=1, academic=1, book=1)
                    │           │           │
                    ▼           ▼           ▼
            ┌───────────┐ ┌───────────┐ ┌───────────┐
            │   web     │ │ academic  │ │   book    │
            │researcher │ │researcher │ │researcher │
            └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
                  │             │             │
                  │  Firecrawl  │  OpenAlex   │  book_search
                  │  Perplexity │             │
                  │             │             │
                  ▼             ▼             ▼
            ┌─────────┐   ┌─────────┐   ┌─────────┐
            │Compress │   │Compress │   │Compress │
            │(recency │   │(citations│   │(author  │
            │authority)│   │evidence)│   │publisher)│
            └────┬────┘   └────┬────┘   └────┬────┘
                 │             │             │
                 └─────────────┼─────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  aggregate_findings  │
                    │  (reduce: add)       │
                    └──────────┬──────────┘
                               │
                               ▼
                           supervisor

┌─────────────────────────────────────────────────────────────────────────────┐
│                         researcher_base.py (shared)                          │
│  • TTL Cache (module-level singleton, 1h TTL, 200 items)                    │
│  • PDF processing (route to Marker service)                                  │
│  • Query generation (identical across types)                                 │
│  • Query validation (LLM-based relevance checking)                          │
│  • Parallel scraping (asyncio.gather with index tracking)                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Implementation

### Step 1: Define Researcher Types

Create enum and allocation types:

```python
# workflows/research/state.py

from enum import Enum

class ResearcherType(str, Enum):
    """Types of specialized researchers."""

    WEB = "web"           # Firecrawl + Perplexity
    ACADEMIC = "academic" # OpenAlex
    BOOK = "book"         # book_search


class ResearcherAllocation(TypedDict):
    """Allocation of researchers for a conduct_research action."""

    web_count: int       # Number of web researchers (default: 1)
    academic_count: int  # Number of academic researchers (default: 1)
    book_count: int      # Number of book researchers (default: 1)
```

### Step 2: Add Allocation to Workflow State

```python
# workflows/research/state.py

class DeepResearchState(TypedDict):
    # ... existing fields ...

    # Researcher allocation for specialized researchers
    researcher_allocation: Optional[ResearcherAllocation]  # {web_count, academic_count, book_count}
```

### Step 3: Extract Shared Utilities

Create a base module for common functionality:

```python
# workflows/research/subgraphs/researcher_base.py

import asyncio
import os
from cachetools import TTLCache

# Shared TTL cache for scraped content (all researchers share this)
_scrape_cache: TTLCache = TTLCache(
    maxsize=int(os.getenv("SCRAPE_CACHE_SIZE", "200")),
    ttl=int(os.getenv("SCRAPE_CACHE_TTL", "3600")),
)


def is_pdf_url(url: str) -> bool:
    """Check if URL points to a PDF file."""
    return url.lower().endswith(".pdf") or "pdf" in url.lower()


async def fetch_pdf_via_marker(url: str) -> str | None:
    """Fetch and process PDF via local Marker service."""
    # ... implementation ...


async def generate_queries(
    question: ResearchQuestion,
    language_config: Optional[LanguageConfig] = None,
) -> list[str]:
    """Generate search queries from research question.

    Identical for all researcher types.
    """
    llm = get_llm(ModelTier.HAIKU)
    structured_llm = llm.with_structured_output(SearchQueries)

    prompt = f"""Generate 2-3 search queries to research this question:

Question: {question['question']}
{f"Language: {language_config['name']}" if language_config else ""}

Make queries specific and likely to find authoritative sources.
"""

    result = await structured_llm.ainvoke([{"role": "user", "content": prompt}])
    return result.queries


async def validate_queries(
    queries: list[str],
    research_question: str,
) -> list[str]:
    """Validate queries are relevant to the research question.

    Identical for all researcher types.
    """
    # ... validation logic ...


async def scrape_urls_parallel(
    urls: list[str],
    max_scrapes: int = 10,
) -> list[tuple[str, str]]:
    """Scrape URLs in parallel using shared cache.

    Used by all researcher types.
    """
    tasks = [_scrape_single_url(url, i) for i, url in enumerate(urls[:max_scrapes])]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    # ... process results with index-based ordering ...
```

### Step 4: Create Specialized Web Researcher

```python
# workflows/research/subgraphs/web_researcher.py

from workflows.research.subgraphs.researcher_base import (
    generate_queries, validate_queries, scrape_urls_parallel, _scrape_cache
)

# Web-specific compression prompt (emphasizes recency, authority)
WEB_COMPRESS_SYSTEM = """You are compressing web research findings.

Evaluate sources based on:
1. **Recency**: Prefer recent content (within 1-2 years) unless historical context needed
2. **Domain authority**: Trust .gov, .edu, established news > blogs, forums
3. **Factual accuracy**: Cross-reference claims, note contradictions
4. **Bias detection**: Identify opinion vs fact, commercial interests

Output format (JSON):
{{
  "finding": "Clear answer with inline citations [1], [2]",
  "sources": [{{"url": "...", "title": "...", "relevance": "high/medium/low"}}],
  "confidence": 0.0-1.0,
  "gaps": ["What's still unclear"]
}}"""


async def execute_web_search(state: ResearcherState) -> dict:
    """Search using Firecrawl and Perplexity."""
    queries = state.get("search_queries", [])
    results = []

    for query in queries[:3]:
        # Firecrawl for general web
        firecrawl_results = await _search_firecrawl(query)
        results.extend(firecrawl_results)

        # Perplexity for synthesized answers
        perplexity_results = await _search_perplexity(query)
        results.extend(perplexity_results)

    return {"search_results": dedupe_by_url(results)}


async def compress_web_findings(state: ResearcherState) -> dict:
    """Compress using web-specific prompt."""
    llm = get_llm(ModelTier.SONNET)

    # Use web-specific compression prompt
    response = await invoke_with_cache(
        llm,
        system_prompt=WEB_COMPRESS_SYSTEM,
        user_prompt=format_findings_for_compression(state),
    )

    finding = parse_finding_json(response.content)
    finding["source_type"] = "web"

    return {"finding": finding}


# Build subgraph
def web_researcher_subgraph() -> StateGraph:
    builder = StateGraph(ResearcherState)
    builder.add_node("generate_queries", generate_queries_node)
    builder.add_node("search", execute_web_search)
    builder.add_node("scrape", scrape_pages)
    builder.add_node("compress", compress_web_findings)
    # ... edges ...
    return builder.compile()
```

### Step 5: Create Specialized Academic Researcher

```python
# workflows/research/subgraphs/academic_researcher.py

# Academic-specific compression prompt (emphasizes peer review, citations)
ACADEMIC_COMPRESS_SYSTEM = """You are compressing academic research findings.

Evaluate sources based on:
1. **Peer-review status**: Prioritize peer-reviewed journal articles
2. **Citation count**: Higher citations suggest more established findings
3. **Methodology quality**: Note study design, sample sizes, limitations
4. **Evidence hierarchy**: Meta-analysis > RCT > Observational > Case study
5. **Recency in field**: Note if findings may be outdated in fast-moving fields

Output format (JSON):
{{
  "finding": "Clear answer with inline citations [1], [2]",
  "sources": [{{"url": "...", "title": "...", "relevance": "high/medium/low", "cited_by": N}}],
  "confidence": 0.0-1.0,
  "gaps": ["Methodological limitations", "What needs more research"]
}}"""


async def execute_academic_search(state: ResearcherState) -> dict:
    """Search using OpenAlex for scholarly literature."""
    queries = state.get("search_queries", [])
    results = []

    for query in queries[:3]:
        openalex_results = await _search_openalex(query, limit=3)  # Default 3 articles
        results.extend(openalex_results)

    return {"search_results": dedupe_by_url(results)}


async def compress_academic_findings(state: ResearcherState) -> dict:
    """Compress using academic-specific prompt."""
    llm = get_llm(ModelTier.SONNET)

    response = await invoke_with_cache(
        llm,
        system_prompt=ACADEMIC_COMPRESS_SYSTEM,
        user_prompt=format_findings_for_compression(state),
    )

    finding = parse_finding_json(response.content)
    finding["source_type"] = "academic"

    return {"finding": finding}
```

### Step 6: Create Specialized Book Researcher

```python
# workflows/research/subgraphs/book_researcher.py

# Book-specific compression prompt (emphasizes author credentials, publisher)
BOOK_COMPRESS_SYSTEM = """You are compressing book research findings.

Evaluate sources based on:
1. **Author credentials**: Academic position, publication history, field expertise
2. **Publisher reputation**: Academic press > trade publisher > self-published
3. **Edition currency**: Is this the latest edition? Are there updates?
4. **Book's contribution**: Seminal work, synthesis of field, contrarian view?
5. **Depth vs breadth**: Note if book is comprehensive or focused

Output format (JSON):
{{
  "finding": "Clear answer with inline citations [1], [2]",
  "sources": [{{"url": "...", "title": "...", "author": "...", "relevance": "high/medium/low"}}],
  "confidence": 0.0-1.0,
  "gaps": ["Topics not covered", "Outdated aspects"]
}}"""


async def execute_book_search(state: ResearcherState) -> dict:
    """Search using book_search for long-form content."""
    queries = state.get("search_queries", [])
    results = []

    for query in queries[:2]:  # Fewer queries for books
        book_results = await _search_books(query, limit=1)  # Default 1 book per query
        results.extend(book_results)

    return {"search_results": dedupe_by_url(results)}


async def compress_book_findings(state: ResearcherState) -> dict:
    """Compress using book-specific prompt."""
    llm = get_llm(ModelTier.SONNET)

    response = await invoke_with_cache(
        llm,
        system_prompt=BOOK_COMPRESS_SYSTEM,
        user_prompt=format_findings_for_compression(state),
    )

    finding = parse_finding_json(response.content)
    finding["source_type"] = "book"

    return {"finding": finding}
```

### Step 7: Implement Allocation-Based Routing

Update supervisor routing to dispatch based on allocation:

```python
# workflows/research/graph.py

def route_supervisor_action(state: DeepResearchState) -> str | list[Send]:
    """Route based on supervisor's chosen action.

    For conduct_research, dispatches specialized researchers based on allocation.
    Default: 1 web + 1 academic + 1 book researcher.
    """
    current_status = state.get("current_status", "")

    if current_status == "conduct_research":
        pending = state.get("pending_questions", [])

        if not pending:
            return "final_report"

        # Get allocation (default: 1 of each type)
        allocation = state.get("researcher_allocation") or {
            "web_count": 1,
            "academic_count": 1,
            "book_count": 1,
        }

        language_config = state.get("primary_language_config")
        sends = []
        question_idx = 0

        # Dispatch web researchers
        for _ in range(allocation.get("web_count", 1)):
            if question_idx < len(pending):
                sends.append(Send("web_researcher", ResearcherState(
                    question=pending[question_idx],
                    language_config=language_config,
                    # ... other fields
                )))
                question_idx += 1

        # Dispatch academic researchers
        for _ in range(allocation.get("academic_count", 1)):
            if question_idx < len(pending):
                sends.append(Send("academic_researcher", ResearcherState(
                    question=pending[question_idx],
                    language_config=language_config,
                )))
                question_idx += 1

        # Dispatch book researchers
        for _ in range(allocation.get("book_count", 1)):
            if question_idx < len(pending):
                sends.append(Send("book_researcher", ResearcherState(
                    question=pending[question_idx],
                    language_config=language_config,
                )))
                question_idx += 1

        if not sends:
            return "final_report"

        logger.info(
            f"Launching {len(sends)} researchers "
            f"(web={allocation.get('web_count', 1)}, "
            f"academic={allocation.get('academic_count', 1)}, "
            f"book={allocation.get('book_count', 1)})"
        )

        return sends

    # ... other actions
```

### Step 8: Register Subgraphs in Main Graph

```python
# workflows/research/graph.py

from workflows.research.subgraphs.web_researcher import web_researcher_subgraph
from workflows.research.subgraphs.academic_researcher import academic_researcher_subgraph
from workflows.research.subgraphs.book_researcher import book_researcher_subgraph


def create_deep_research_graph():
    builder = StateGraph(DeepResearchState)

    # ... other nodes ...

    # Specialized researcher subgraphs
    builder.add_node("web_researcher", web_researcher_subgraph)
    builder.add_node("academic_researcher", academic_researcher_subgraph)
    builder.add_node("book_researcher", book_researcher_subgraph)

    builder.add_node("aggregate_findings", aggregate_researcher_findings)

    # Conditional edges for specialized researchers
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor_action,
        ["web_researcher", "academic_researcher", "book_researcher",
         "refine_draft", "final_report", "supervisor"],
    )

    # All researchers aggregate to same node
    builder.add_edge("web_researcher", "aggregate_findings")
    builder.add_edge("academic_researcher", "aggregate_findings")
    builder.add_edge("book_researcher", "aggregate_findings")

    return builder.compile()
```

## Source-Specific Evaluation Criteria

| Researcher | Sources | Primary Criteria | Secondary Criteria |
|------------|---------|------------------|-------------------|
| **Web** | Firecrawl, Perplexity | Recency, domain authority (.gov, .edu) | Factual accuracy, bias detection |
| **Academic** | OpenAlex | Peer-review status, citation count | Methodology quality, evidence hierarchy |
| **Book** | book_search | Author credentials, publisher | Edition currency, field contribution |

## Consequences

### Benefits

- **Source-appropriate evaluation**: Each researcher uses criteria matching source type
- **Configurable allocation**: Adjust web/academic/book ratio per topic
- **Shared utilities**: TTL cache, PDF processing, query generation reused
- **Maintainable code**: Smaller, focused files instead of monolithic researcher

### Trade-offs

- **More files**: Three subgraphs + base module vs single file
- **Routing complexity**: Allocation-based dispatch adds logic
- **Question distribution**: Questions assigned by position, not best-fit

## Related Patterns

- [Deep Research Workflow Architecture](./deep-research-workflow-architecture.md) - Base workflow this extends
- [Parallel AI Search Integration](../data-pipeline/parallel-ai-search-integration.md) - Similar parallel execution
- [Concurrent Scraping with TTL Cache](../async-python/concurrent-scraping-with-ttl-cache.md) - Shared cache pattern

## Known Uses in Thala

- `workflows/research/subgraphs/web_researcher.py`: Firecrawl + Perplexity researcher
- `workflows/research/subgraphs/academic_researcher.py`: OpenAlex researcher
- `workflows/research/subgraphs/book_researcher.py`: book_search researcher
- `workflows/research/subgraphs/researcher_base.py`: Shared utilities
- `workflows/research/graph.py`: Allocation-based routing

## References

- [LangGraph Send API](https://python.langchain.com/docs/langgraph/concepts/low_level/#send)
- [OpenAlex API](https://docs.openalex.org/)
