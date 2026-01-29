---
name: parallel-ai-search-integration
title: "Parallel AI Search Integration"
date: 2025-12-18
category: data-pipeline
applicability:
  - "When research requires aggregating results from heterogeneous search APIs"
  - "When academic sources must be combined with web search results"
  - "When source-specific metadata must survive through downstream processing"
  - "When fault tolerance is needed (one source failing should not block others)"
components: [asyncio, httpx, pydantic, langchain-tools]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [parallel-execution, search-aggregation, openalex, perplexity, firecrawl, book-search, asyncio-gather, fault-tolerance]
---

# Parallel AI Search Integration

## Intent

Aggregate search results from multiple heterogeneous sources (web search, AI-powered search, academic databases, book catalogs) in parallel, with fault tolerance and metadata preservation for downstream processing.

## Motivation

Comprehensive research requires diverse source types:

1. **Web search (Firecrawl)**: General web content, news, blogs
2. **AI-powered search (Perplexity)**: Synthesized answers with citations
3. **Academic search (OpenAlex)**: Peer-reviewed literature with rich metadata
4. **Book search**: Books and textbooks with format/availability metadata

Running these sequentially is slow. Running them in parallel requires:
- Fault tolerance (one source failing shouldn't break the pipeline)
- Unified result format across heterogeneous APIs
- Metadata preservation for source-specific downstream handling
- Deduplication when sources return overlapping results

## Applicability

Use this pattern when:
- Research tasks need both web and academic sources
- Individual source failures should not block results from other sources
- Source-specific metadata (DOI, authors, citations) must flow to downstream processing
- Latency matters and parallel execution is beneficial

Do NOT use this pattern when:
- Only one search source is needed
- Strict ordering of results by source is required
- API rate limits prevent parallel calls

## Structure

```
                    +------------------+
                    |  generate_queries |
                    +--------+---------+
                             |
                             v
                    +--------+---------+
                    | execute_searches  |
                    +--------+---------+
                             |
     +---------------+-------+-------+---------------+
     |               |               |               |
     v               v               v               v
+----------+   +------------+  +-----------+   +----------+
| Firecrawl |   | Perplexity |  | OpenAlex  |   | Books    |
| (web)     |   | (AI-search)|  | (academic)|   | (catalog)|
+----+-----+   +-----+------+  +-----+-----+   +----+-----+
     |               |               |               |
     +---------------+-------+-------+---------------+
                             |
                    +--------v---------+
                    |   Deduplicate    |
                    |    by URL        |
                    +--------+---------+
                             |
                             v
                    +--------+---------+
                    |  WebSearchResult  |
                    | (unified model)   |
                    +------------------+
```

## Implementation

### Step 1: Define Unified Result Model

A single model accommodates all sources while preserving source-specific data:

```python
from typing import Optional
from typing_extensions import TypedDict

class WebSearchResult(TypedDict):
    """A web search result from any source."""

    url: str
    title: str
    description: Optional[str]
    content: Optional[str]  # Scraped content if fetched
    source_metadata: Optional[dict]  # Source-specific structured data
```

The `source_metadata` field preserves rich data from academic sources (DOI, authors, citations) that web sources don't have.

### Step 2: Implement Source-Specific Adapters

Each source has an adapter that normalizes results to `WebSearchResult`:

```python
async def _search_firecrawl(query: str) -> list[WebSearchResult]:
    """Search using Firecrawl (general web)."""
    results = []
    try:
        result = await web_search.ainvoke({"query": query, "limit": MAX_RESULTS_PER_SOURCE})
        for r in result.get("results", []):
            results.append(
                WebSearchResult(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    description=r.get("description"),
                    content=None,
                    source_metadata=None,  # Web sources have no structured metadata
                )
            )
    except Exception as e:
        logger.warning(f"Firecrawl search failed for '{query}': {e}")
    return results


async def _search_openalex(query: str) -> list[WebSearchResult]:
    """Search using OpenAlex (academic sources)."""
    results = []
    try:
        result = await openalex_search.ainvoke({"query": query, "limit": MAX_RESULTS_PER_SOURCE})
        for r in result.get("results", []):
            # Format authors for human-readable description
            authors = r.get("authors", [])
            author_str = ", ".join(a.get("name", "") for a in authors[:3])
            if len(authors) > 3:
                author_str += " et al."

            # Build rich description from metadata
            description = f"{author_str}. " if author_str else ""
            if r.get("publication_date"):
                description += f"({r['publication_date'][:4]}). "
            if r.get("source_name"):
                description += f"{r['source_name']}. "
            description += f"Cited by {r.get('cited_by_count', 0)}."

            results.append(
                WebSearchResult(
                    url=r.get("url", ""),  # oa_url or DOI
                    title=r.get("title", ""),
                    description=description,
                    content=None,
                    source_metadata={  # Preserve all structured data
                        "source_type": "openalex",
                        "doi": r.get("doi"),
                        "oa_url": r.get("oa_url"),
                        "authors": r.get("authors", []),
                        "publication_date": r.get("publication_date"),
                        "cited_by_count": r.get("cited_by_count", 0),
                        "source_name": r.get("source_name"),
                        "abstract": r.get("abstract"),
                        "is_oa": r.get("is_oa", False),
                    },
                )
            )
    except Exception as e:
        logger.warning(f"OpenAlex search failed for '{query}': {e}")
    return results


async def _search_books(query: str) -> list[WebSearchResult]:
    """Search for books (catalog sources)."""
    results = []
    try:
        result = await book_search.ainvoke({"query": query, "limit": MAX_RESULTS_PER_SOURCE})
        for r in result.get("results", []):
            # Build description from authors + format + size
            description = f"{r.get('authors', 'Unknown')} · {r.get('format', '').upper()} · {r.get('size', '')}"
            if r.get("abstract"):
                description += f"\n{r['abstract']}"

            results.append(
                WebSearchResult(
                    url=r.get("url", ""),
                    title=r.get("title", ""),
                    description=description,
                    content=None,
                    source_metadata={
                        "source_type": "book",
                        "md5": r.get("md5"),
                        "authors": r.get("authors"),
                        "publisher": r.get("publisher"),
                        "language": r.get("language"),
                        "format": r.get("format"),
                        "size": r.get("size"),
                        "abstract": r.get("abstract"),
                    },
                )
            )
    except Exception as e:
        logger.warning(f"Book search failed for '{query}': {e}")
    return results
```

### Step 3: Implement Parallel Execution with Fault Tolerance

Use `asyncio.gather` with `return_exceptions=True` for parallel execution:

```python
import asyncio

MAX_RESULTS_PER_SOURCE = 5

async def execute_searches(state: ResearcherState) -> dict[str, Any]:
    """Execute web searches using multiple sources in parallel."""
    queries = state.get("search_queries", [])
    all_results = []

    for query in queries[:MAX_SEARCHES]:
        # Run all four sources in parallel for each query
        search_tasks = [
            _search_firecrawl(query),
            _search_perplexity(query),
            _search_openalex(query),
            _search_books(query),
        ]

        results_lists = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Collect results, handling exceptions from individual sources
        for results in results_lists:
            if isinstance(results, list):
                all_results.extend(results)
            elif isinstance(results, Exception):
                logger.warning(f"Search source failed: {results}")

    # Deduplicate by URL
    seen_urls = set()
    unique_results = []
    for r in all_results:
        url = r["url"]
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_results.append(r)

    logger.info(
        f"Found {len(unique_results)} unique results from {len(queries)} queries "
        f"across Firecrawl/Perplexity/OpenAlex/Books"
    )
    return {"search_results": unique_results[:15]}
```

### Step 4: Implement Lazy Singleton Clients

Each API client uses lazy initialization with cleanup registration:

```python
_perplexity_client = None


async def close_perplexity() -> None:
    """Close the global Perplexity client."""
    global _perplexity_client
    if _perplexity_client is not None:
        await _perplexity_client.aclose()
        _perplexity_client = None


def _get_perplexity():
    """Get Perplexity httpx client (lazy init)."""
    global _perplexity_client
    if _perplexity_client is None:
        import httpx
        from core.utils.async_http_client import register_cleanup

        api_key = os.environ.get("PERPLEXITY_API_KEY")
        if not api_key:
            raise ValueError("PERPLEXITY_API_KEY environment variable required")

        _perplexity_client = httpx.AsyncClient(
            base_url="https://api.perplexity.ai",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        register_cleanup("Perplexity", close_perplexity)
    return _perplexity_client
```

### Step 5: Preserve Metadata Through Pipeline

When compressing findings, preserve `source_metadata` for citation processing:

```python
async def compress_findings(state: ResearcherState) -> dict[str, Any]:
    """Compress research into structured finding."""
    search_results = state.get("search_results", [])

    # Build URL -> result map to preserve source_metadata
    url_to_result = {r.get("url", ""): r for r in search_results if r.get("url")}

    # After LLM compression, reconstruct sources with metadata
    finding_sources = []
    for s in finding_data.get("sources", []):
        url = s.get("url", "")
        original = url_to_result.get(url, {})
        finding_sources.append(
            WebSearchResult(
                url=url,
                title=s.get("title", original.get("title", "")),
                description=s.get("relevance", original.get("description", "")),
                content=original.get("content"),
                source_metadata=original.get("source_metadata"),  # Preserve!
            )
        )
```

### Step 6: Use Metadata in Citation Processing

Skip expensive Translation Server lookups for sources with rich metadata:

```python
async def process_citation(url: str, source_metadata: dict | None) -> str:
    """Process a citation, using source metadata when available."""

    if source_metadata and source_metadata.get("source_type") == "openalex":
        # Use OpenAlex metadata directly - skip Translation Server
        logger.info(f"Using OpenAlex metadata for: {url[:50]}...")

        enhanced_metadata = {
            "title": source_metadata.get("title"),
            "authors": [a.get("name") for a in source_metadata.get("authors", [])],
            "date": source_metadata.get("publication_date"),
            "publication_title": source_metadata.get("source_name"),
            "abstract": source_metadata.get("abstract"),
            "doi": source_metadata.get("doi"),
            "item_type": "journalArticle",
        }

        # Use DOI as canonical URL when available
        zotero_url = source_metadata.get("doi") or url
        return await _create_zotero_item(zotero_url, enhanced_metadata)

    # Fall back to Translation Server for web sources
    return await _process_via_translation_server(url)
```

## Consequences

### Benefits

- **Comprehensive results**: Combines web, AI-powered, academic, and book sources
- **Fault tolerance**: `return_exceptions=True` ensures partial results on source failures
- **Latency optimization**: Parallel execution reduces total search time
- **Metadata preservation**: `source_metadata` enables source-specific downstream handling
- **Cost optimization**: OpenAlex metadata bypasses Translation Server lookups

### Trade-offs

- **Deduplication by URL only**: Same content at different URLs appears twice
- **Sequential queries**: Queries run sequentially (parallel per-query, not across queries)
- **No rate limiting**: Relies on API timeouts, not explicit semaphores
- **Lazy init race condition**: Theoretical risk with concurrent first calls (mitigated by single-threaded event loop)

### Async Considerations

- `asyncio.gather` with `return_exceptions=True` is the canonical pattern for parallel execution with partial failure tolerance
- Lazy singleton clients avoid per-request connection overhead
- Sequential query execution provides natural rate limiting without explicit semaphores

## Related Patterns

- [Deep Research Workflow Architecture](../langgraph/deep-research-workflow-architecture.md) - Uses this pattern in researcher agents
- [Unified Scraping Service](./unified-scraping-service-fallback-chain.md) - For scraping search result URLs
- [Citation Processing with Zotero Integration](./citation-processing-zotero-integration.md) - Consumes source_metadata for citations

## Known Uses in Thala

- `langchain_tools/perplexity.py`: Perplexity search and fact-checking tools
- `langchain_tools/openalex.py`: OpenAlex academic search tool
- `langchain_tools/book_search.py`: Book catalog search with TTL caching
- `workflows/research/subgraphs/researcher.py`: Parallel search execution
- `workflows/research/nodes/process_citations.py`: Metadata-aware citation processing

## References

- [OpenAlex API](https://docs.openalex.org/)
- [Perplexity API](https://docs.perplexity.ai/)
- [asyncio.gather documentation](https://docs.python.org/3/library/asyncio-task.html#asyncio.gather)
