# OpenAlex

LangChain tools and query functions for searching academic literature via OpenAlex (240M+ scholarly works).

## Usage

### Tool: openalex_search

```python
from langchain_tools.openalex import openalex_search

# Basic search
result = await openalex_search.ainvoke({
    "query": "machine learning interpretability",
    "limit": 10,
})

# Advanced filtering
result = await openalex_search.ainvoke({
    "query": "neural networks",
    "limit": 20,
    "min_citations": 50,
    "from_year": 2020,
    "to_year": 2024,
    "language": "en",
})
```

### Query Functions

```python
from langchain_tools.openalex import (
    get_forward_citations,
    get_backward_citations,
    get_author_works,
    get_work_by_doi,
    get_works_by_dois,
    resolve_doi_to_openalex_id,
)

# Forward citations (who cites this paper)
citations = await get_forward_citations(
    work_id="10.1038/nature12345",
    limit=50,
    min_citations=10,
    from_year=2020,
)

# Backward citations (what this paper cites)
references = await get_backward_citations(
    work_id="W2741809807",
    limit=50,
)

# Author's works
works = await get_author_works(
    author_id="A2208157607",
    limit=20,
    min_citations=10,
)

# Single work by DOI
work = await get_work_by_doi("10.1038/nature12345")

# Batch fetch by DOIs
works = await get_works_by_dois([
    "10.1038/nature12345",
    "10.1126/science.abc123",
])

# DOI to OpenAlex ID resolution
openalex_id = await resolve_doi_to_openalex_id("10.1038/nature12345")
```

## Input/Output

### openalex_search Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | str | required | Search query |
| `limit` | int | 10 | Max results (1-50) |
| `min_citations` | int | None | Minimum citation count |
| `from_year` | int | None | Include works from this year onwards |
| `to_year` | int | None | Include works up to this year |
| `language` | str | None | ISO 639-1 code (en, es, zh, etc.) |

### OpenAlexSearchOutput

```python
{
    "query": str,
    "total_results": int,
    "results": [
        {
            "title": str,
            "url": str,              # OA URL or DOI
            "doi": str,              # Always preserved for citations
            "oa_url": str,           # Open access full text URL
            "abstract": str,
            "authors": [
                {
                    "name": str,
                    "institution": str,
                }
            ],
            "publication_date": str,
            "cited_by_count": int,
            "primary_topic": str,
            "source_name": str,      # Journal/venue
            "is_oa": bool,
            "oa_status": str,        # gold, green, hybrid, bronze, closed
            "language": str,
        }
    ],
}
```

## Key Features

### Inverted Index Reconstruction
OpenAlex stores abstracts in inverted index format for efficiency. The `_reconstruct_abstract()` function rebuilds readable text by mapping words to positions.

### Citation Network Analysis
- **Forward citations**: Uses `cites:{work_id}` filter to find citing papers
- **Backward citations**: Fetches `referenced_works` field and hydrates full metadata
- Supports filtering by citation count and publication year

### Persistent Caching
All queries cached for 30 days in shared persistent cache:
- Reduces API load
- Improves response times
- Maintains consistency across workflows

### Work ID Normalization
Accepts multiple identifier formats:
- DOI: `10.1038/nature12345`
- DOI URL: `https://doi.org/10.1038/nature12345`
- OpenAlex ID: `W2741809807`
- OpenAlex URL: `https://openalex.org/W2741809807`

### Batch Operations
`get_works_by_dois()` uses pipe-delimited filters for efficient bulk fetching with per-work caching.

## Architecture

```
tools.py           # LangChain tool decorator (openalex_search)
queries.py         # Query functions for citations, authors, DOI resolution
client.py          # Singleton httpx client with polite pool support
models.py          # Pydantic schemas for works, authors, citations
parsing.py         # Response transformation (inverted index, work parsing)
```

### Client Management
Global async client registered with cleanup handler:
```python
from langchain_tools.openalex.client import close_openalex

# Automatic cleanup on shutdown
await close_openalex()
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENALEX_EMAIL` | Optional | Email for polite pool (faster rate limits) |

### Rate Limits
- Without email: 10 req/sec (100k req/day)
- With email (polite pool): 100k req/day with higher burst

## Data Models

```python
from langchain_tools.openalex import (
    OpenAlexWork,              # Individual scholarly work
    OpenAlexAuthor,            # Author with institution
    OpenAlexSearchOutput,      # Search results
    OpenAlexCitationResult,    # Citation analysis
    OpenAlexAuthorWorksResult, # Author publication list
)
```

## Related Modules

- `langchain_tools/book_search/` - Book and textbook discovery
- `langchain_tools/web/` - Web search and scraping
- `workflows/research/` - Research workflows using OpenAlex
- `workflows/output/evening_reads/` - Academic literature curation
