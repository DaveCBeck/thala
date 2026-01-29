---
title: "Multi-Signal Completeness Calculation and Transient Error Retry Logic"
module: workflows/research
date: 2025-12-28
problem_type: workflow_reliability
component: research_workflow
symptoms:
  - "Workflows terminated prematurely with incomplete research"
  - "Single-metric completion checks gave false positives (e.g., iterations done = complete)"
  - "Transient API errors caused entire workflow failures"
  - "HTTP 502/503/504 from external services crashed long-running research"
  - "Connection resets during peak load caused data loss"
root_cause: "Single-signal completeness calculation was unreliable, and absence of retry logic for transient errors caused unnecessary workflow failures"
resolution_type: code_fix
severity: high
tags: [workflow-reliability, completeness-calculation, retry-logic, transient-errors, resilience]
---

# Multi-Signal Completeness Calculation and Transient Error Retry Logic

## Problem

The deep research workflow suffered from two related reliability issues that caused premature termination and unnecessary failures during long-running research tasks.

### Symptoms

1. **Premature workflow termination**: Research workflows reported 100% completeness after 2 iterations despite missing key findings
2. **False-positive completion**: Iteration-only metrics marked research "complete" when coverage was poor
3. **Transient error crashes**: HTTP 502/503/504 errors from web APIs crashed entire workflows
4. **Connection instability**: Network resets during scraping caused unrecoverable failures
5. **Lost progress**: Multi-hour research workflows lost all progress on temporary API hiccups

### Example of Premature Termination

```python
# Iteration 2/4 complete -> completeness = 50%
# But only 1 of 5 key questions answered with high confidence
# Workflow terminated because iteration threshold was reached
```

### Example of Transient Error Crash

```python
# Research 80% complete, 3 hours elapsed
# Firecrawl returns HTTP 503 (temporary overload)
# Entire workflow crashes, all progress lost
ERROR: Exception in researcher node: 503 Service Unavailable
```

## Root Cause

### Issue 1: Single-Signal Completeness

The original completeness calculation used only iteration progress:

```python
# BEFORE (unreliable)
def calculate_completeness(iteration: int, max_iterations: int) -> float:
    """Completeness based solely on iteration count."""
    return iteration / max_iterations
```

This approach failed because:
- **Iterations != Coverage**: Completing 4 iterations doesn't mean all questions were answered
- **No quality signal**: A workflow with 10 low-confidence findings was "complete" while missing key answers
- **No gap awareness**: Known research gaps weren't factored into completion decisions

### Issue 2: No Transient Error Handling

External API calls had no retry logic:

```python
# BEFORE (brittle)
async def scrape_page(url: str) -> str:
    response = await client.get(url)
    return response.text  # Fails immediately on any error
```

Transient errors (temporary server issues, network blips) were treated as permanent failures, causing:
- Immediate workflow termination on HTTP 5xx
- No recovery from connection resets
- Lost work on temporary API unavailability

## Solution

### Part 1: Multi-Signal Completeness Calculation

Replaced single-signal completeness with a weighted multi-signal formula.

```python
# workflows/research/nodes/supervisor.py

def calculate_completeness(
    findings: list["ResearchFinding"],
    key_questions: list[str],
    iteration: int,
    max_iterations: int,
    gaps_remaining: list[str] | None = None,
) -> float:
    """Calculate research completeness from multiple signals.

    Weighted formula balances progress, coverage, quality, and gaps:
    - 40%: Iteration progress (capped at 90% to prevent iteration-only completion)
    - 30%: Findings coverage (questions with high-confidence answers)
    - 20%: Average confidence (quality signal)
    - 10%: Gap penalty (reduces score based on known gaps)

    Args:
        findings: Research findings collected so far
        key_questions: Core questions that must be answered
        iteration: Current iteration number
        max_iterations: Maximum allowed iterations
        gaps_remaining: Known gaps in research coverage

    Returns:
        Completeness score between 0.0 and 1.0
    """
    if not findings:
        return 0.0

    gaps_remaining = gaps_remaining or []

    # Signal 1: Iteration progress (40%)
    # Cap at 90% - iterations alone should never complete research
    iteration_score = min(iteration / max(max_iterations, 1), 0.9)

    # Signal 2: Coverage - questions with high-confidence findings (30%)
    high_confidence_threshold = 0.7
    high_confidence_findings = sum(
        1 for f in findings
        if f.get("confidence", 0.5) >= high_confidence_threshold
    )
    total_questions = max(len(key_questions), 1)
    coverage_score = min(high_confidence_findings / total_questions, 1.0)

    # Signal 3: Average confidence across all findings (20%)
    avg_confidence = sum(f.get("confidence", 0.5) for f in findings) / len(findings)

    # Signal 4: Gap penalty - reduce score for known gaps (10%)
    # Each gap reduces this component by 20%
    gap_score = max(0.0, 1.0 - len(gaps_remaining) * 0.2)

    # Weighted combination
    completeness = (
        0.40 * iteration_score +
        0.30 * coverage_score +
        0.20 * avg_confidence +
        0.10 * gap_score
    )

    return min(completeness, 1.0)
```

**Key Design Decisions:**

1. **Iteration cap at 90%**: Prevents "iterations done = complete" false positives
2. **High-confidence threshold**: Only counts findings with confidence >= 0.7 as "answered"
3. **Gap penalty**: Known gaps actively reduce completeness score
4. **Weighted balance**: Coverage (30%) and confidence (20%) together outweigh iteration (40%)

### Part 2: Transient Error Detection

Added intelligent error classification to distinguish transient from permanent failures.

```python
# workflows/shared/retry.py

def _is_transient_error(error: Exception) -> bool:
    """Determine if an error is transient and worth retrying.

    Transient errors are temporary conditions that may resolve:
    - HTTP 502/503/504 (server overload, gateway issues)
    - Connection timeouts and resets
    - DNS resolution failures (temporary)

    Permanent errors should NOT be retried:
    - HTTP 400/401/403/404 (client errors, auth issues)
    - ValueError, TypeError (code bugs)
    - Rate limit exceeded (needs backoff, not immediate retry)
    """
    error_str = str(error).lower()

    # HTTP status codes indicating transient server issues
    transient_statuses = ["502", "503", "504"]

    # Error message patterns indicating transient conditions
    transient_indicators = [
        "timeout",
        "connection reset",
        "connection refused",
        "temporary failure",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "connection aborted",
        "read timed out",
        "name resolution failed",  # DNS issues
    ]

    # Check for transient HTTP status
    for status in transient_statuses:
        if status in error_str:
            return True

    # Check for transient error patterns
    for indicator in transient_indicators:
        if indicator in error_str:
            return True

    # Check exception type for network-related errors
    transient_types = (
        ConnectionError,
        TimeoutError,
        OSError,  # Includes socket errors
    )

    if isinstance(error, transient_types):
        return True

    return False
```

### Part 3: Exponential Backoff Retry

Implemented async retry with exponential backoff for transient errors.

```python
# workflows/shared/retry.py

import asyncio
import logging
from typing import TypeVar, Callable, Any

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Retry configuration
MAX_RETRY_ATTEMPTS = 3
RETRY_INITIAL_DELAY = 1.0  # seconds
RETRY_MAX_DELAY = 30.0  # seconds


async def with_retry(
    func: Callable[..., T],
    *args,
    max_attempts: int = MAX_RETRY_ATTEMPTS,
    initial_delay: float = RETRY_INITIAL_DELAY,
    max_delay: float = RETRY_MAX_DELAY,
    **kwargs,
) -> T:
    """Execute an async function with retry logic for transient errors.

    Uses exponential backoff: delay doubles after each failed attempt.
    Only retries on transient errors; permanent errors are re-raised immediately.

    Args:
        func: Async function to execute
        *args: Positional arguments for func
        max_attempts: Maximum retry attempts (default: 3)
        initial_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay cap in seconds (default: 30.0)
        **kwargs: Keyword arguments for func

    Returns:
        Result of successful function execution

    Raises:
        Exception: The last error if all retries fail, or immediately
                   if the error is not transient
    """
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await func(*args, **kwargs)

        except Exception as e:
            last_error = e

            # Don't retry non-transient errors
            if not _is_transient_error(e):
                logger.warning(
                    f"Non-transient error in {func.__name__}, not retrying: {e}"
                )
                raise

            # Don't sleep after final attempt
            if attempt == max_attempts:
                logger.error(
                    f"All {max_attempts} retry attempts failed for {func.__name__}: {e}"
                )
                raise

            # Calculate delay with exponential backoff
            delay = min(initial_delay * (2 ** (attempt - 1)), max_delay)

            logger.warning(
                f"Transient error in {func.__name__} (attempt {attempt}/{max_attempts}), "
                f"retrying in {delay:.1f}s: {e}"
            )

            await asyncio.sleep(delay)

    # Should never reach here, but satisfy type checker
    raise last_error  # type: ignore
```

### Part 4: Integration into Workflow Nodes

Applied retry logic to external API calls in researcher nodes.

```python
# workflows/research/subgraphs/researcher.py

from workflows.shared.retry import with_retry

async def execute_searches(state: ResearcherState) -> dict:
    """Execute web searches with retry logic."""
    results = []

    for query in state["search_queries"]:
        try:
            # Wrap external API call with retry
            result = await with_retry(
                firecrawl_search,
                query=query,
                max_results=5,
                max_attempts=3,
                initial_delay=2.0,  # Longer delay for rate-limited APIs
            )
            results.extend(result)

        except Exception as e:
            # Log but don't fail entire workflow
            logger.warning(f"Search failed after retries for '{query}': {e}")
            state.setdefault("errors", []).append({
                "node": "execute_searches",
                "query": query,
                "error": str(e),
            })

    return {"search_results": results}


async def scrape_pages(state: ResearcherState) -> dict:
    """Scrape pages with retry logic and graceful degradation."""
    content = []

    for url in state["search_results"][:5]:  # Limit scrapes
        try:
            # Wrap scraping with retry
            page_content = await with_retry(
                scrape_url,
                url=url,
                max_attempts=2,  # Fewer retries for scraping
                initial_delay=1.0,
            )
            content.append(page_content)

        except Exception as e:
            logger.warning(f"Scrape failed after retries for {url}: {e}")
            # Continue with partial results rather than failing

    return {"scraped_content": content}
```

### Part 5: Updated Supervisor Decision Logic

Integrated multi-signal completeness into supervisor decisions.

```python
# workflows/research/nodes/supervisor.py

async def supervisor(state: DeepResearchState) -> dict[str, Any]:
    """Supervisor with multi-signal completeness checking."""
    diffusion = state.get("diffusion", {})
    findings = state.get("research_findings", [])
    brief = state.get("research_brief", {})

    # Calculate completeness using multiple signals
    completeness = calculate_completeness(
        findings=findings,
        key_questions=brief.get("key_questions", []),
        iteration=diffusion.get("iteration", 0),
        max_iterations=diffusion.get("max_iterations", 4),
        gaps_remaining=diffusion.get("areas_to_explore", []),
    )

    # Update state with calculated completeness
    diffusion = {**diffusion, "completeness_score": completeness}

    # Termination conditions
    if completeness >= 0.95:
        logger.info(f"Research complete: {completeness:.1%} completeness")
        return {
            "diffusion": diffusion,
            "current_status": "research_complete",
        }

    if diffusion.get("iteration", 0) >= diffusion.get("max_iterations", 4):
        logger.info(
            f"Max iterations reached with {completeness:.1%} completeness"
        )
        return {
            "diffusion": diffusion,
            "current_status": "research_complete",
        }

    # Continue research...
```

## Prevention

### Completeness Calculation Guidelines

1. **Never use single signals for completion**: Always combine multiple indicators
2. **Cap iteration contribution**: Iterations should never exceed 50% of completeness weight
3. **Include quality signals**: Confidence scores and coverage metrics matter
4. **Track gaps explicitly**: Known gaps should actively reduce completeness
5. **Log completeness breakdown**: When debugging, log each signal's contribution

```python
# Debugging completeness issues
logger.debug(
    f"Completeness breakdown: "
    f"iteration={iteration_score:.2f} (40%), "
    f"coverage={coverage_score:.2f} (30%), "
    f"confidence={avg_confidence:.2f} (20%), "
    f"gaps={gap_score:.2f} (10%) -> {completeness:.2f}"
)
```

### Retry Logic Guidelines

1. **Classify errors correctly**: Only retry transient errors
2. **Use exponential backoff**: Start at 1-2s, cap at 30s
3. **Limit retry attempts**: 3 attempts is usually sufficient
4. **Log all retries**: Critical for debugging production issues
5. **Graceful degradation**: Continue with partial results when possible
6. **Don't retry rate limits**: Use separate rate-limit handling with longer delays

### Error Classification Checklist

| Error Type | Retry? | Action |
|------------|--------|--------|
| HTTP 502/503/504 | Yes | Exponential backoff |
| Connection timeout | Yes | Exponential backoff |
| Connection reset | Yes | Exponential backoff |
| HTTP 400 | No | Fix request |
| HTTP 401/403 | No | Fix auth |
| HTTP 404 | No | Skip resource |
| HTTP 429 (rate limit) | Maybe | Use rate-limit specific handler |
| ValueError/TypeError | No | Fix code |

## Files Modified

- `workflows/research/nodes/supervisor.py`: Multi-signal completeness calculation
- `workflows/shared/retry.py`: New retry utility module
- `workflows/research/subgraphs/researcher.py`: Retry integration for API calls

## Related Patterns

- [Deep Research Workflow Architecture](../../patterns/langgraph/deep-research-workflow-architecture.md) - Overall workflow structure
- [Research Query Generation Fixes](../llm-output/query-generation-supervisor-extraction-fixes.md) - Related supervisor improvements

## References

- [Exponential Backoff and Jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
