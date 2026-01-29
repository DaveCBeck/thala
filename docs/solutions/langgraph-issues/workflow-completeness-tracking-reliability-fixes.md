---
module: workflows/research
date: 2025-12-22
problem_type: langgraph_issue
component: workflow_graph
symptoms:
  - "Completeness score stuck at 0% throughout entire research workflow"
  - "Score only updated during refine_draft phase, not conduct_research or aggregate_findings"
  - "Transient HTTP errors (502/503/504) causing entire workflow failures"
  - "Workflow running to max iterations even when research sufficiently complete"
root_cause: logic_error
resolution_type: code_fix
severity: high
tags: [langgraph, state-management, completeness-tracking, retry-logic, exponential-backoff, resource-cleanup, transient-errors]
---

# Research Workflow Completeness Tracking and Reliability Fixes

## Problem

The research workflow had two critical reliability issues that combined to make it effectively broken:

1. **Completeness tracking stuck at 0%**: The workflow never reported progress, confusing users and preventing automatic completion.
2. **Transient errors causing failures**: Temporary HTTP errors (502/503/504) failed the entire workflow, losing hours of research.

### Symptoms

1. **Completeness always 0%**: Despite completing multiple research iterations with findings, the completeness score remained at 0% in logs and UI.

2. **Single-phase update**: Completeness only recalculated in `refine_draft` node, missing updates from `conduct_research` and `aggregate_findings`.

3. **Workflow failures on transient errors**: 502 gateway errors or connection timeouts caused complete workflow failure rather than retry.

4. **Max iterations always reached**: Workflow ran to `max_iterations` even when research was comprehensive, wasting API costs.

### Example Log Output (Before Fix)

```
INFO: Iteration 1/8, completeness: 0.0%
INFO: Completed research on 3 questions, found 12 sources
INFO: Iteration 2/8, completeness: 0.0%  # Still 0%!
INFO: Completed research on 2 questions, found 8 sources
...
INFO: Iteration 8/8, completeness: 0.0%  # Never updates
ERROR: 502 Bad Gateway from Firecrawl API
ERROR: Workflow failed: HTTPStatusError
```

## Root Cause

### Issue 1: Single-Phase Completeness Update

The completeness score was only calculated in `refine_draft`:

```python
# BEFORE (problematic) - Only one node updates completeness
async def refine_draft(state: DeepResearchState) -> dict:
    # ... draft refinement logic ...
    completeness = calculate_simple_completeness(state)
    return {"draft": new_draft, "completeness_score": completeness}

# Other nodes didn't update completeness
async def aggregate_findings(state: DeepResearchState) -> dict:
    return {"findings": new_findings}  # No completeness update!

async def supervisor(state: DeepResearchState) -> dict:
    # Used stale completeness_score from state
    return {"action": action}  # No completeness update!
```

Since `refine_draft` wasn't called on every iteration (only when supervisor chose that action), completeness stayed at the initial 0%.

### Issue 2: Single-Signal Completeness Formula

The original formula used only gap count:

```python
# BEFORE (problematic)
def calculate_simple_completeness(state: DeepResearchState) -> float:
    gaps = state.get("gaps_remaining", [])
    return max(0, 1.0 - len(gaps) * 0.2)  # 20% penalty per gap, unlimited
```

With 5+ gaps (common in early research), completeness was always 0%.

### Issue 3: No Retry for Transient Errors

All HTTP errors were treated as fatal:

```python
# BEFORE (problematic)
async def _scrape_firecrawl(self, url: str) -> ScrapeResult:
    response = await self.client.post("/scrape", json={"url": url})
    response.raise_for_status()  # Any error = failure
    return ScrapeResult(**response.json())
```

## Solution

### Step 1: Multi-Signal Completeness Calculation

Created a weighted formula using multiple signals:

```python
# workflows/research/state.py

def calculate_completeness(
    findings: list["ResearchFinding"],
    key_questions: list[str],
    iteration: int,
    max_iterations: int,
    gaps_remaining: list[str] | None = None,
) -> float:
    """Calculate research completeness from multiple signals.

    Weighted formula:
    - 40%: Iteration progress (capped at 90% contribution)
    - 30%: Findings coverage (questions with high-confidence answers)
    - 20%: Average confidence of findings
    - 10%: Gap penalty (reduces score for known gaps)

    This ensures:
    - Score increases during research (not stuck at 0%)
    - Score reflects quality (confidence)
    - Natural progression toward 85% threshold for completion
    """
    gaps_remaining = gaps_remaining or []

    # 1. Iteration progress (40% weight) - capped at 90% to prevent false completion
    iteration_score = min(iteration / max(max_iterations, 1), 0.9)

    # 2. Findings coverage (30% weight)
    total_questions = max(len(key_questions), 1)
    high_confidence_findings = sum(
        1 for f in findings if f.get("confidence", 0) > 0.5
    )
    coverage_score = min(high_confidence_findings / total_questions, 1.0)

    # 3. Average confidence (20% weight)
    if findings:
        avg_confidence = sum(f.get("confidence", 0.5) for f in findings) / len(findings)
    else:
        avg_confidence = 0.0

    # 4. Gap penalty (10% weight) - capped at 10 gaps, 5% each
    capped_gaps = min(len(gaps_remaining), 10)
    gap_score = max(0, 1.0 - capped_gaps * 0.05)  # 5% per gap, max 50% penalty

    # Weighted sum
    completeness = (
        0.40 * iteration_score
        + 0.30 * coverage_score
        + 0.20 * avg_confidence
        + 0.10 * gap_score
    )

    return min(completeness, 1.0)
```

### Step 2: Update Completeness in All Phases

Added completeness updates to every node that modifies research state:

```python
# workflows/research/nodes/supervisor.py

async def supervisor(state: DeepResearchState) -> dict:
    """Supervisor node with completeness tracking."""
    # Calculate current completeness
    completeness = calculate_completeness(
        findings=state.get("findings", []),
        key_questions=brief.get("key_questions", []),
        iteration=state.get("iteration", 1),
        max_iterations=state.get("max_iterations", 8),
        gaps_remaining=state.get("gaps_remaining"),
    )

    # Check for automatic completion at 85% threshold
    if completeness >= 0.85:
        logger.info(f"Completeness {completeness:.0%} >= 85%, marking complete")
        return {
            "action": "research_complete",
            "completeness_score": completeness,
        }

    # ... rest of supervisor logic ...
    return {
        "action": action,
        "completeness_score": completeness,  # Always update!
    }


# workflows/research/graph.py - aggregate_researcher_findings

def aggregate_researcher_findings(findings_list: list[dict]) -> dict:
    """Aggregate findings from parallel researchers."""
    all_findings = []
    for result in findings_list:
        all_findings.extend(result.get("findings", []))

    # Calculate updated completeness
    completeness = calculate_completeness(
        findings=all_findings,
        key_questions=current_questions,
        iteration=current_iteration,
        max_iterations=max_iterations,
    )

    return {
        "findings": all_findings,
        "completeness_score": completeness,  # Update after aggregation!
    }
```

### Step 3: Add Transient Error Detection and Retry

Added retry logic with exponential backoff:

```python
# core/scraping/service.py

MAX_RETRY_ATTEMPTS = 2
RETRY_INITIAL_DELAY = 2.0


def _is_transient_error(error: Exception) -> bool:
    """Check if error is transient and should be retried."""
    error_str = str(error).lower()
    error_type = type(error).__name__

    # HTTP status codes that are transient
    transient_statuses = ["502", "503", "504"]
    if any(status in error_str for status in transient_statuses):
        return True

    # Connection/timeout errors
    transient_indicators = [
        "timeout", "timed out", "connection reset",
        "connection refused", "temporary failure",
    ]
    if any(indicator in error_str for indicator in transient_indicators):
        return True

    # Common transient exception types
    transient_types = [
        "TimeoutError", "ClientConnectorError",
        "ServerTimeoutError", "asyncio.TimeoutError",
    ]
    if any(t in error_type for t in transient_types):
        return True

    return False


async def _with_retry(func, *args, **kwargs):
    """Execute function with retry logic for transient failures."""
    last_error = None

    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_error = e

            # Don't retry permanent errors
            if not _is_transient_error(e):
                raise

            # Don't retry on last attempt
            if attempt >= MAX_RETRY_ATTEMPTS:
                raise

            # Exponential backoff
            delay = RETRY_INITIAL_DELAY * (2 ** (attempt - 1))
            logger.info(
                f"Transient error on attempt {attempt}/{MAX_RETRY_ATTEMPTS}, "
                f"retrying in {delay}s: {e}"
            )
            await asyncio.sleep(delay)

    raise last_error
```

### Step 4: Apply Retry to Scraping Operations

Wrapped all external calls with retry:

```python
# core/scraping/service.py

async def scrape(self, url: str, include_links: bool = False) -> ScrapeResult:
    """Scrape URL with fallback chain and retry."""
    domain = _extract_domain(url)

    # Try Firecrawl basic with retry
    try:
        return await _with_retry(
            self._scrape_firecrawl, url, proxy=None, include_links=include_links
        )
    except SiteBlockedError:
        self._blocklist.add(domain)
    except Exception as e:
        logger.debug(f"Firecrawl basic failed: {e}")

    # Try Firecrawl stealth with retry
    try:
        return await _with_retry(
            self._scrape_firecrawl, url, proxy="stealth", include_links=include_links
        )
    except SiteBlockedError:
        self._blocklist.add(domain)
    except Exception as e:
        logger.debug(f"Firecrawl stealth failed: {e}")

    # Playwright fallback with retry
    return await _with_retry(self._scrape_playwright, url, include_links)
```

### Step 5: Add Completeness Threshold for Auto-Completion

Added 85% threshold in supervisor:

```python
# workflows/research/nodes/supervisor.py

COMPLETENESS_THRESHOLD = 0.85

async def supervisor(state: DeepResearchState) -> dict:
    completeness = calculate_completeness(...)

    # Auto-complete when sufficiently done
    if completeness >= COMPLETENESS_THRESHOLD:
        logger.info(
            f"Research {completeness:.0%} complete (>= {COMPLETENESS_THRESHOLD:.0%}), "
            f"transitioning to final report"
        )
        return {
            "action": "research_complete",
            "completeness_score": completeness,
        }

    # Continue research if below threshold
    # ...
```

## Prevention

### Completeness Tracking Guidelines

1. **Never use single signals** for completion decisions - combine iteration, coverage, confidence, gaps
2. **Cap iteration contribution** (90% max) to prevent iteration-only completion
3. **Include quality signals** (confidence scores) not just quantity
4. **Penalize gaps proportionally** - cap penalty to prevent 0% from many gaps
5. **Update in all relevant nodes** - any node that modifies state should update completeness
6. **Log completeness breakdown** for debugging

### Transient Error Handling Guidelines

1. **Classify errors before retrying**:
   | Error Type | Retry? | Examples |
   |------------|--------|----------|
   | Transient | Yes | 502, 503, 504, timeout, connection reset |
   | Permanent | No | 400, 401, 403, 404, validation errors |
   | Rate limit | Yes (longer delay) | 429, "rate limit" |

2. **Use exponential backoff**: 1s → 2s → 4s (cap at 30s)
3. **Limit retry attempts** to 2-3 to avoid infinite loops
4. **Log all retries** for production debugging
5. **Continue with partial results** when possible (graceful degradation)

## Files Modified

- `workflows/research/state.py`: Added `calculate_completeness()` function
- `workflows/research/graph.py`: Update completeness in `aggregate_researcher_findings`
- `workflows/research/nodes/supervisor.py`: Multi-phase completeness, 85% threshold
- `workflows/research/nodes/refine_draft.py`: Use new completeness calculation
- `core/scraping/service.py`: Added `_with_retry()`, `_is_transient_error()`
- `workflows/research/__init__.py`: Added `cleanup_research_resources()`

## Related Patterns

- [Deep Research Workflow Architecture](../../patterns/langgraph/deep-research-workflow-architecture.md) - Diffusion algorithm and state management
- [Unified Scraping Service](../../patterns/data-pipeline/unified-scraping-service-fallback-chain.md) - Fallback chain with retry

## References

- [LangGraph State Management](https://python.langchain.com/docs/langgraph/concepts/low_level/#state)
- [Exponential Backoff](https://cloud.google.com/iot/docs/how-tos/exponential-backoff)
