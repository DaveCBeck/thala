---
module: book_finding
date: 2026-01-03
problem_type: api_integration_issue
component: search_books, book_search
symptoms:
  - "No results found for book recommendations"
  - "Book search returns empty for valid books"
  - "Cached empty results prevent retries"
  - "Language filter excludes books with unknown language"
root_cause: single_query_fragility
resolution_type: fallback_strategy
severity: medium
tags: [search, fallback, caching, language-filter, reliability]
---

# Search Reliability: Multi-Query Fallback

## Problem

Book search frequently returned no results for valid recommendations:

```
INFO: No results found for: "Thinking, Fast and Slow Daniel Kahneman"
INFO: No results found for: "The Innovator's Dilemma Clayton Christensen"
```

This occurred even for popular books that existed in the library.

## Root Cause

**Multiple issues compounding to reduce search reliability:**

1. **Single query strategy**: Only tried one query (title + author), which fails if:
   - Author name spelling varies from API records
   - Title includes subtitle that doesn't match
   - API only indexes partial title/author

2. **Caching empty results**: Temporary API failures cached as "no results":
   ```python
   # PROBLEMATIC: Cached even if empty
   _search_cache[cache_key] = output
   ```

3. **Strict language filtering**: Books with "Unknown" language were excluded:
   ```python
   # PROBLEMATIC: Excluded when language unknown
   if language is not None:
       if book_language != language:
           continue  # Filtered out even if book_language is "Unknown"
   ```

## Solution

**Multi-pronged reliability improvements:**

### 1. Multi-Query Fallback Strategy

Try multiple query variations in order of specificity:

```python
# workflows/research/subgraphs/book_finding/nodes/search_books.py

async def _search_single_book(
    recommendation_title: str,
    author: str | None,
    language: str | None = None,
) -> BookResult | None:
    """Search for a single recommended book with fallback strategies.

    Tries multiple query strategies:
    1. Title + Author (if author known) - most specific
    2. Title only - catches spelling variations in author
    3. Author only (if author known) - catches title variations
    """
    # Build list of queries to try
    queries = []
    if author:
        queries.append(f"{recommendation_title} {author}")
    queries.append(recommendation_title)
    if author:
        queries.append(author)

    all_results = []

    for query in queries:
        try:
            search_params = {
                "query": query,
                "limit": MAX_RESULTS_PER_SEARCH,
            }
            if language:
                search_params["language"] = language

            result = await book_search.ainvoke(search_params)
            books = result.get("results", [])

            if books:
                logger.debug(f"Query '{query}' found {len(books)} results")
                all_results.extend(books)
                # If we found results with title+author, that's good enough
                if len(queries) > 1 and query == queries[0]:
                    break
            else:
                logger.debug(f"No results for query: '{query}'")

        except Exception as e:
            logger.warning(f"Search failed for '{query}': {e}")
            continue

    if not all_results:
        logger.info(f"No results found for: {recommendation_title}")
        return None

    # Deduplicate by MD5
    seen_md5 = set()
    unique_results = []
    for book in all_results:
        md5 = book.get("md5", "")
        if md5 and md5 not in seen_md5:
            seen_md5.add(md5)
            unique_results.append(book)

    # Prioritize PDF format
    pdf_books = [b for b in unique_results if b.get("format", "").lower() == "pdf"]
    best_book = pdf_books[0] if pdf_books else unique_results[0]

    return BookResult(...)
```

### 2. Only Cache Non-Empty Results

Don't cache temporary failures:

```python
# langchain_tools/book_search.py

output = BookSearchOutput(results=books)

# Only cache non-empty results to avoid caching temporary failures
if books:
    _search_cache[cache_key] = output
```

### 3. Skip Language Filter for Unknown

Include books when language metadata is missing:

```python
# langchain_tools/book_search.py

for r in data.get("results", []):
    book_language = r.get("language", "Unknown")

    # Filter by language if specified (skip filtering if book language is unknown)
    if language is not None and book_language.lower().strip() != "unknown":
        if book_language.lower().strip() != language.lower().strip():
            continue
```

### 4. Add Concurrency Limiting

Prevent overwhelming the API:

```python
# workflows/research/subgraphs/book_finding/nodes/search_books.py

MAX_CONCURRENT_SEARCHES = 5

async def search_books(state: dict) -> dict[str, Any]:
    # Limit concurrency to avoid overwhelming the service
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)

    async def search_with_semaphore(rec: dict) -> BookResult | None:
        async with semaphore:
            return await _search_single_book(
                recommendation_title=rec["title"],
                author=rec.get("author"),
                language=language,
            )

    search_tasks = [search_with_semaphore(rec) for rec in all_recommendations]
    search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
```

## Query Strategy Order

| Order | Query Type | When Useful |
|-------|------------|-------------|
| 1 | Title + Author | Most specific, best match quality |
| 2 | Title only | Author name spelling differs |
| 3 | Author only | Title includes subtitle variations |

The strategy short-circuits: if query 1 returns results, queries 2-3 are skipped.

## Files Modified

- `workflows/research/subgraphs/book_finding/nodes/search_books.py` - Multi-query fallback, concurrency
- `langchain_tools/book_search.py` - Cache only non-empty, skip unknown language filter

## Prevention

When implementing search functionality:
1. **Multiple query strategies**: Try variations in specificity order
2. **Don't cache failures**: Only cache positive results
3. **Permissive filtering**: Handle missing metadata gracefully
4. **Concurrency limits**: Use semaphores to avoid API overload

## Testing

```python
async def test_multi_query_fallback():
    """Test that fallback queries find books when primary fails."""
    # Mock: first query returns nothing, second returns result
    with patch('langchain_tools.book_search.ainvoke') as mock:
        mock.side_effect = [
            {"results": []},  # Title + Author: no results
            {"results": [{"title": "Test", "md5": "abc"}]},  # Title only: found
        ]

        result = await _search_single_book(
            recommendation_title="Test Book",
            author="Test Author",
        )

        assert result is not None
        assert mock.call_count == 2  # Tried both queries


async def test_cache_only_non_empty():
    """Test that empty results are not cached."""
    _search_cache.clear()

    # First call: empty results
    await _search_books_internal(query="nonexistent", limit=10)
    assert "nonexistent" not in str(_search_cache.keys())  # Not cached

    # Second call should actually try API again
```

## Related Patterns

- [Multi-Source Paper Acquisition](./paper-acquisition-robustness.md) - Similar fallback strategy
- [Hash-Based Persistent Caching](../../patterns/data-pipeline/hash-based-persistent-caching.md) - Caching best practices

## References

- [Library Genesis API](https://wiki.mhut.org/content:libgen_api)
