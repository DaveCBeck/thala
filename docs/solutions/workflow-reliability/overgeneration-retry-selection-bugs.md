---
title: "Over-Generation Retry and Selection Logic Bugs"
module: workflows/output/illustrate
date: 2026-02-09
problem_type: logic_errors
component: illustrate_workflow
symptoms:
  - "Retry successes silently dropped in finalize — only first-round results used"
  - "retry_count inflated beyond actual attempts, causing premature retry exhaustion"
  - "Vision pair comparison biased toward candidate 1 due to brief text in criteria"
  - "Retry brief_ids collided when max_retries > 1"
  - "Non-winning candidates held image bytes in memory indefinitely"
root_cause: "Multiple stateful bugs in the add-reducer accumulation pattern and retry round tracking"
resolution_type: code_fix
severity: high
tags: [retry-logic, selection-bugs, add-reducer, deduplication, langgraph, state-management, over-generation, memory-management]
shared: true
gist_url: https://gist.github.com/DaveCBeck/c4057f4a5651332e2cf333f4ec417128
article_path: .context/libs/thala-dev/content/generation/2026-02-09-over-generation-pair-selection-langgraph.md
---

# Over-Generation Retry and Selection Logic Bugs

## Problem

The illustrate workflow's over-generation feature (introduced in commit `57fa0cf`) had 15 code review findings spanning data loss, correctness issues, and code quality problems. The most severe bugs caused retry successes to be silently dropped and retry counts to inflate incorrectly.

### Symptoms

1. **Data loss - Retry successes dropped**: Locations that failed in round 1 and succeeded in retry round showed the "failed" result in final output, with retry success silently ignored
2. **Premature retry exhaustion**: `retry_count` grew beyond actual attempts, causing workflows to stop retrying after one round when `max_retries=3`
3. **Selection bias**: Vision comparison favored candidate 1 because criteria included the first candidate's brief text
4. **Brief ID collisions**: Retry candidates with `max_retries > 1` reused the same `brief_id` format, colliding with original entries
5. **Memory exhaustion risk**: Non-winning candidates held full image bytes (5-10 MB each) indefinitely
6. **Security risk**: `_download_image` had no size limit, enabling OOM attacks via malicious URLs

### Example: Retry Success Dropped (P1 Bug)

```python
# State after round 1
selection_results = [
    {"location_id": "loc_a", "quality_tier": "failed", ...},
]

# After retry round succeeds
selection_results = [
    {"location_id": "loc_a", "quality_tier": "failed", ...},    # from round 1
    {"location_id": "loc_a", "quality_tier": "good", ...},       # from retry
]

# BEFORE FIX: _select_winning_results kept first entry per location
# Result: "failed" entry won, retry success ignored
```

### Example: Retry Count Inflation (P2 Bug)

```python
# After round 1: location A fails
selection_results = [{"location_id": "A", "quality_tier": "failed"}]

# sync_after_selection in retry round
retry_count = dict(state.get("retry_count", {}))  # {"A": 1} from previous round
for s in selection_results:  # Still contains round 1 failure!
    if s["quality_tier"] == "failed":
        retry_count["A"] = retry_count.get("A", 0) + 1  # Now {"A": 2}

# Result: One actual retry round, but retry_count=2
# With max_retries=3, only one more retry allowed instead of two
```

## Root Cause Analysis

### Add Reducer Accumulation Pattern

The workflow uses `Annotated[list, add]` reducers for `generation_results` and `selection_results`. This means:
- Each node appends entries to the list
- Entries accumulate across all rounds (original + retry rounds)
- The state never "resets" between rounds

This accumulation is correct and intentional, but requires careful deduplication and counting logic.

### Bug Categories

**P1 - Data Loss**:
1. `_select_winning_results` iterated `selection_results` in order and kept the *first* entry per `location_id`, but retry-round entries appear *after* round-1 entries due to the `add` reducer. First-round failures won the slot.

**P2 - Correctness**:
2. `retry_count` incremented from `state.get("retry_count", {})` for each failed entry in the accumulated `selection_results`, causing over-inflation as old failures were re-counted
3. Selection criteria used `briefs_by_location[location_id][0].brief`, biasing vision comparison toward candidate 1
4. Retry `brief_id` format `{loc_id}_{candidate_index}` collided with original round when `max_retries > 1`
5. `brief_id` was patched in after `_generate_*` helpers returned, making control flow unclear
6. `_download_image` had no size limit on HTTP response body

**P3 - Quality/Cleanup**:
7. Manual dict manipulation for retry briefs instead of `model_copy`
8. Unused `enumerate` loop variable
9. No shared validator for `location_id` format
10. Image media type hardcoded as `image/png` instead of detected from bytes
11. Non-winning candidates retained full `image_bytes` after selection
12-15. Dead code from old architecture (4 result types, 3 nodes, 6 prompt constants)

## Solution

### Part 1: Fix Retry Success Dropping (P1)

Changed `_select_winning_results` to keep the *last* entry per `location_id` instead of first.

```python
# workflows/output/illustrate/nodes/finalize.py

def _select_winning_results(
    generation_results: list[ImageGenResult],
    selection_results: list[LocationSelection],
) -> list[ImageGenResult]:
    """Pick the winning ImageGenResult per location based on selection_results.

    For each location with a selection, find the generation result matching
    the selected_brief_id. Falls back to the last successful result if
    the selected brief_id can't be found.
    """
    # Build lookup: brief_id -> ImageGenResult
    results_by_brief_id: dict[str, ImageGenResult] = {}
    results_by_location: dict[str, list[ImageGenResult]] = defaultdict(list)
    for gen in generation_results:
        if gen["success"] and gen.get("image_bytes"):
            results_by_brief_id[gen["brief_id"]] = gen
            results_by_location[gen["location_id"]].append(gen)

    # BEFORE: kept first entry per location (bug: first-round "failed" wins slot)
    # seen = set()
    # for selection in selection_results:
    #     if selection["location_id"] not in seen:
    #         seen.add(selection["location_id"])
    #         # ... process

    # AFTER: keep last entry per location (retry success overrides failure)
    # Deduplicate selections: keep only the LAST entry per location_id.
    # selection_results uses an `add` reducer, so retry-round entries appear
    # after earlier rounds. Without dedup the first (often "failed") entry
    # claims the slot and the retry success is silently skipped.
    latest_selection: dict[str, LocationSelection] = {}
    for selection in selection_results:
        latest_selection[selection["location_id"]] = selection

    winners: list[ImageGenResult] = []
    selected_locations: set[str] = set()

    for loc_id, selection in latest_selection.items():
        selected_locations.add(loc_id)

        if selection["quality_tier"] == "failed":
            continue  # Both candidates failed, no image for this location

        brief_id = selection["selected_brief_id"]
        if brief_id and brief_id in results_by_brief_id:
            winners.append(results_by_brief_id[brief_id])
        elif loc_id in results_by_location:
            # Fallback: use the last successful result for this location
            logger.warning(f"Selected brief_id {brief_id} not found for {loc_id}, using fallback")
            winners.append(results_by_location[loc_id][-1])

    # Also include any successful results for locations without selection
    # (e.g., from retry rounds that bypass selection)
    for loc_id, results in results_by_location.items():
        if loc_id not in selected_locations:
            winners.append(results[-1])

    return winners
```

**Key Fix**: Build `latest_selection` dict by iterating all entries — last write per `location_id` wins. This ensures retry-round results override round-1 failures.

### Part 2: Fix Retry Count Inflation (P2)

Changed `sync_after_selection` to derive `retry_count` from scratch instead of incrementing from stale state.

```python
# workflows/output/illustrate/graph.py

def sync_after_selection(state: IllustrateState) -> dict:
    """Synchronization barrier after all selections complete.

    Updates retry_count for failed locations and clears image_bytes
    from non-winning generation results to free memory.
    """
    selection_results = state.get("selection_results", [])
    passed = sum(1 for s in selection_results if s["quality_tier"] != "failed")
    failed = sum(1 for s in selection_results if s["quality_tier"] == "failed")

    # BEFORE: incremented from existing state (stale across rounds)
    # retry_count = dict(state.get("retry_count", {}))
    # for s in selection_results:
    #     if s["quality_tier"] == "failed":
    #         loc = s["location_id"]
    #         retry_count[loc] = retry_count.get(loc, 0) + 1

    # AFTER: derived from scratch by counting accumulated failures
    # Derive retry counts directly from accumulated selection_results.
    # selection_results uses an add reducer, so it grows across rounds —
    # each failed round appends one "failed" entry per location.
    # Counting failed entries per location gives the exact retry count
    # without the over-inflation bug of the old increment-per-entry approach.
    retry_count: dict[str, int] = {}
    for s in selection_results:
        if s["quality_tier"] == "failed":
            loc = s["location_id"]
            retry_count[loc] = retry_count.get(loc, 0) + 1

    # Free memory: clear image_bytes from non-winning candidates...
    # (rest of function)
```

**Key Fix**: Start from empty `retry_count` dict instead of `state.get("retry_count", {})`. Each "failed" entry in accumulated list = one failed round, so count directly.

### Part 3: Fix Selection Criteria Bias (P2)

Changed `_build_selection_criteria` to use `ImageOpportunity` fields instead of first candidate's brief.

```python
# workflows/output/illustrate/graph.py

def _build_selection_criteria(
    opportunities: list[ImageOpportunity],
    editorial_notes: str,
    location_id: str,
) -> str:
    """Build neutral selection criteria from ImageOpportunity fields.

    Uses rationale + purpose (which describe *why* the location needs an
    image) rather than any single candidate's brief, so comparison is
    unbiased between candidates.
    """
    # BEFORE: used first candidate's brief text (biased)
    # criteria = briefs_by_location[location_id][0].brief

    # AFTER: use ImageOpportunity purpose/rationale (neutral)
    parts: list[str] = []
    for opp in opportunities:
        if opp.location_id == location_id:
            parts.append(f"Purpose: {opp.purpose}. {opp.rationale}")
            break

    if editorial_notes:
        parts.append(f"Editorial guidance: {editorial_notes}")

    return " | ".join(parts)
```

**Why This Matters**: The selection criteria are passed to the vision LLM to compare two candidates. Using candidate 1's brief implicitly biases the comparison toward candidate 1's approach. `ImageOpportunity.purpose` and `rationale` describe *why* the location needs an image (document-level context), not *how* any specific candidate addresses it.

### Part 4: Fix Brief ID Collision (P2)

Added round number to retry `brief_id` format.

```python
# workflows/output/illustrate/graph.py

def route_after_selection(state: IllustrateState) -> list[Send] | str:
    """Route failed locations to retry with cross-strategy fallback.

    Generates two retry candidates per failed location, switching image source type.
    """
    # ... setup ...

    sends = []
    for loc_id in failed_locations:
        # ... find plan and briefs ...

        # Generate two retry candidates with fallback image types
        for orig_brief in location_briefs[:2]:
            fallback_type = _FALLBACK_IMAGE_TYPE.get(orig_brief.image_type, "generated")
            retry_brief = orig_brief.model_copy(update={"image_type": fallback_type})

            # BEFORE: brief_id = f"{loc_id}_{orig_brief.candidate_index}"
            # Bug: collides with round 1 when max_retries > 1

            # AFTER: include round number to prevent collision
            round_num = retry_count.get(loc_id, 0)
            brief_id = f"{loc_id}_{orig_brief.candidate_index}_retry{round_num}"

            send_data = {
                "location": plan,
                "brief": retry_brief,
                "brief_id": brief_id,
                "document_context": document,
                "config": config,
                "visual_identity": visual_identity,
            }
            sends.append(Send("generate_candidate", send_data))

    if sends:
        logger.info(f"Routing to {len(sends)} retry generation nodes")
        return sends

    return "finalize"
```

**Key Fix**: Format is now `{loc_id}_{candidate_index}_retry{round_num}` instead of `{loc_id}_{candidate_index}`. With `max_retries=2`, location A gets:
- Round 1: `A_0`, `A_1`
- Retry round 1: `A_0_retry1`, `A_1_retry1`
- Retry round 2: `A_0_retry2`, `A_1_retry2`

### Part 5: Pass brief_id Into Helpers (P2)

Changed `_generate_*` helpers to accept `brief_id` parameter instead of patching it post-hoc.

```python
# workflows/output/illustrate/nodes/generate_additional.py

async def _generate_public_domain(
    location_id: str,
    plan: ImageLocationPlan,
    brief: str,
    document_context: str,
    brief_id: str = "",  # ← Added parameter
) -> dict:
    """Generate using public domain image search."""
    # ... generation logic ...

    return {
        "location_id": location_id,
        "brief_id": brief_id,  # ← Use parameter
        "success": True,
        "image_bytes": image_bytes,
        # ... rest of result
    }

# Similar changes to _generate_imagen and _generate_diagram
```

```python
# workflows/output/illustrate/nodes/generate_candidate.py

async def generate_candidate_node(state: dict) -> dict:
    """Generate a single image candidate."""
    brief_id = state["brief_id"]
    # ...

    # BEFORE: result returned without brief_id, patched after
    # result = await _generate_public_domain(...)
    # result["brief_id"] = brief_id

    # AFTER: pass brief_id directly
    result = await _generate_public_domain(
        location_id=location.location_id,
        plan=location,
        brief=brief.brief,
        document_context=document_context,
        brief_id=brief_id,
    )
```

**Why This Matters**: Clearer control flow and single source of truth. The caller owns the `brief_id` (including retry round tracking), so it should pass it in rather than patching results after the fact.

### Part 6: Add Streaming + Size Limit to _download_image (P2)

Added streaming with 20MB limit to prevent OOM attacks.

```python
# workflows/output/illustrate/nodes/generate_additional.py

MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB

async def _download_image(url: str) -> bytes:
    """Download image from URL with size limit to prevent memory exhaustion."""
    _validate_image_url(url)

    # BEFORE: No size limit
    # async with httpx.AsyncClient(timeout=30.0) as client:
    #     response = await client.get(url)
    #     response.raise_for_status()
    #     return response.content

    # AFTER: Streaming with size limit
    async with httpx.AsyncClient(timeout=30.0) as client:
        chunks: list[bytes] = []
        total = 0
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > MAX_IMAGE_SIZE:
                    raise ValueError(
                        f"Image exceeds size limit: >{MAX_IMAGE_SIZE} bytes ({MAX_IMAGE_SIZE // (1024 * 1024)} MB)"
                    )
                chunks.append(chunk)
        return b"".join(chunks)
```

**Security Impact**: Without this limit, a malicious URL returning a multi-GB response could exhaust process memory. 20MB is sufficient for high-quality images (typical generated images are 2-5MB).

### Part 7: Use model_copy for Retry Briefs (P3)

Replaced manual dict manipulation with Pydantic's `model_copy`.

```python
# workflows/output/illustrate/graph.py

# BEFORE: manual dict manipulation
retry_brief = CandidateBrief(
    location_id=orig_brief.location_id,
    candidate_index=orig_brief.candidate_index,
    image_type=fallback_type,
    brief=orig_brief.brief,
    # ... copy all other fields ...
)

# AFTER: model_copy with update
retry_brief = orig_brief.model_copy(update={"image_type": fallback_type})
```

### Part 8-11: Code Quality Fixes (P3)

- **Unused enumerate variable**: Removed loop index where only value was used
- **Location ID validator**: Added `@field_validator("location_id")` to `ImageLocationPlan` using shared helper
- **Image media type detection**: Changed `vision_comparison.py` to detect media type from bytes (JPEG starts with `\xff\xd8`, PNG with `\x89PNG`) instead of hardcoding `image/png`
- **Memory management**: Clear `image_bytes` from non-winning candidates after selection

```python
# workflows/output/illustrate/graph.py

def sync_after_selection(state: IllustrateState) -> dict:
    """Synchronization barrier after all selections complete."""
    # ... retry_count derivation ...

    # Free memory: clear image_bytes from non-winning candidates.
    # generation_results uses Annotated[list, add], so we cannot replace the
    # list — mutate entries in-place instead.
    #
    # Deduplicate selections (keep last per location, matching
    # _select_winning_results logic) to identify winning brief_ids.
    latest_selection: dict[str, dict] = {}
    for s in selection_results:
        latest_selection[s["location_id"]] = s

    winning_brief_ids: set[str | None] = set()
    for sel in latest_selection.values():
        if sel["quality_tier"] != "failed" and sel["selected_brief_id"]:
            winning_brief_ids.add(sel["selected_brief_id"])

    generation_results = state.get("generation_results", [])
    cleared = 0
    for gen in generation_results:
        if gen["brief_id"] not in winning_brief_ids and gen.get("image_bytes"):
            gen["image_bytes"] = b""
            cleared += 1

    logger.info(
        f"Selection sync: {passed} selected, {failed} failed, "
        f"{cleared} losers cleared"
    )
    return {"retry_count": retry_count}
```

**Memory Impact**: With 6 locations and 2 candidates each, 12 images generated. If selection picks 6 winners, the 6 losers hold ~30-60MB total. Clearing after selection frees this immediately rather than holding until workflow completion (potentially hours for long documents).

### Part 12-15: Dead Code Removal (P3)

Removed 4 result types, 3 nodes, and 6 prompt constants from the old architecture (pre-over-generation):

**Dead Nodes**:
- `generate_header_node` (271 lines)
- `generate_additional_node` (old version)
- `review_image_node` (194 lines)

**Dead Result Types**:
- `VisionReviewResult`
- `ImageCompareResult`
- `ImageReviewResult`

**Dead Prompt Constants**:
- 6 prompt templates for old review/compare flow

**Impact**: Removed 935 lines of dead code, reducing cognitive load and maintenance burden.

## Prevention

### Guidelines for Add Reducer Patterns with Retry Loops

1. **Always deduplicate accumulated lists**: When using `Annotated[list, add]` with retry loops, deduplication must keep the *last* entry per key, not first
2. **Derive counts from accumulated data**: Never increment counters from stale state — count entries in the accumulated list directly
3. **Include round number in retry IDs**: When retry rounds can repeat, include round number in generated IDs to prevent collision
4. **Pass context into helpers**: Helpers should receive all necessary context (like `brief_id`) as parameters, not rely on post-hoc patching
5. **Free memory proactively**: In workflows with large binary data, clear non-winning candidates as soon as selection completes

### Checklist for Vision Comparison Criteria

When building criteria for LLM-based selection between candidates:

- [ ] Criteria describe *why* the selection is needed (document-level purpose)
- [ ] Criteria do NOT include text from any specific candidate (avoid bias)
- [ ] Criteria are identical for both/all candidates being compared
- [ ] If using fields from planning stage, use neutral fields like `purpose`, `rationale`, `requirements`

### Security Checklist for HTTP Downloads

When downloading external resources:

- [ ] URL validation: HTTPS only, block localhost/private IPs
- [ ] Size limit: Cap at reasonable max (10-20MB for images)
- [ ] Streaming: Use `client.stream()` to enforce limit during download, not after
- [ ] Timeout: Set reasonable timeout (30s for images)
- [ ] Error handling: Catch and log all HTTP/network errors

## Trade-offs

| Decision | Choice | Trade-off |
|----------|--------|-----------|
| Deduplication strategy | Keep last per location | +Correct retry behavior, -Slightly more complex |
| Retry count derivation | Count from accumulated list | +Accurate, -Re-counts on every sync |
| Brief ID format | Include round number | +Prevents collision, -Longer IDs |
| Memory clearing | Mutate state in-place | +Free memory immediately, -Side effect in sync node |
| Image size limit | 20MB streaming | +Security, -Fails on rare oversized images |

## Files Modified

- `workflows/output/illustrate/nodes/finalize.py`: Fixed `_select_winning_results` deduplication
- `workflows/output/illustrate/graph.py`: Fixed retry_count derivation, selection criteria, brief_id format, memory clearing
- `workflows/output/illustrate/nodes/generate_additional.py`: Added streaming + size limit, passed brief_id into helpers
- `workflows/output/illustrate/nodes/generate_candidate.py`: Pass brief_id to helpers
- `workflows/output/illustrate/schemas.py`: Added location_id validator, removed dead types
- `workflows/shared/vision_comparison.py`: Detect image media type from bytes
- Removed: `nodes/generate_header.py`, `nodes/review_image.py`, 6 dead prompts

## Related Solutions and Patterns

- [Multi-Signal Completeness and Retry Logic](./multi-signal-completeness-and-retry-logic.md) - Related retry patterns for research workflow
- [Workflow State Truncation Fixes](./workflow-state-truncation-fixes.md) - State accumulation challenges in LangGraph
- [Two-Pass Planning Pattern](../../patterns/langgraph/two-pass-planning-pattern.md) - Architecture context for illustrate workflow

## Origin

These 15 findings were identified in code review and resolved in commit `de777b2` on the `feat/illustrate-overgeneration-selection` branch. The fixes span data loss (P1), correctness (P2), and code quality (P3) issues.
