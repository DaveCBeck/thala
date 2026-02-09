---
name: over-generation-pair-selection
title: Over-Generation with Per-Location Pair Selection
date: 2026-02-09
category: langgraph
applicability:
  - "Workflows where quality benefits from generating multiple alternatives and selecting the best"
  - "Image generation pipelines needing built-in redundancy without explicit retry loops"
  - "Fan-out patterns where candidates can be compared pairwise"
components: [workflow_graph, langgraph_node, langgraph_graph, llm_call]
complexity: moderate
verified_in_production: true
tags: [over-generation, pair-selection, vision-comparison, fan-out, fan-in, quality-tiers, cross-strategy-fallback, redundancy, send, candidate-generation]
---

# Over-Generation with Per-Location Pair Selection

## Intent

Replace single-generation with retry loops by generating multiple candidates per location upfront, then using vision-based pair comparison to select the best candidate at each location. This provides inherent redundancy and improves quality without complex retry logic.

## Problem

Single-generation approaches with review-based retry loops suffer from several issues:

1. **Sequential bottleneck**: Each retry waits for previous generation to fail, inflating latency
2. **Unpredictable quality**: Retries with feedback don't guarantee improvement
3. **Retry loop complexity**: State management for tracking retry counts, pending retries, and retry briefs
4. **Wasted context**: Failed generations provide no value to the final output
5. **Over-retry risk**: Max retry limits prevent infinite loops but waste API calls on persistently failing locations

## Solution

Generate two candidates per location from genuinely different briefs (written in Pass 2), then select the best using vision pair comparison. Failed locations trigger cross-strategy fallback (switching image source type) rather than retrying the same approach.

### Architecture

```
Document + Config
       |
       v
  Creative Direction (Pass 1)
       |
  Plan Briefs (Pass 2) — writes 2 CandidateBriefs per location
       |
  [Fan-out ~12] generate_candidate (unified node, routes by brief.image_type)
       |
  sync_after_generation (barrier)
       |
  [Fan-out ~6] select_per_location (vision pair comparison per location)
       |
  sync_after_selection (barrier, clear non-winning image_bytes)
       |
  [Conditional] retry_failed (cross-strategy fallback) or finalize
       |
  finalize (save files, insert into markdown)
       |
  END
```

### Key Elements

1. **Over-generation**: Two candidates per location, each from a different CandidateBrief (literal vs metaphorical, different visual interpretations)
2. **Unified generation node**: Single `generate_candidate` node replaces separate `generate_header`/`generate_additional` nodes, routes to public_domain/diagram/Imagen based on `brief.image_type`
3. **Per-location selection**: Groups candidates by `location_id`, runs vision pair comparison for 2-candidate locations, auto-selects for 1-candidate, marks "failed" for 0-candidate
4. **Quality tiers**: excellent (2 candidates, vision selected), acceptable (1 candidate auto-selected), failed (0 candidates)
5. **Cross-strategy fallback**: When both candidates fail, retry with alternate image source type via `_FALLBACK_IMAGE_TYPE` map
6. **Removed review node**: Per-location pair comparison subsumes individual vision review, `max_retries` reduced from 2 to 1

## Implementation

### State Changes

The state schema evolved to support over-generation and selection:

```python
# workflows/output/illustrate/state.py

class ImageGenResult(TypedDict):
    """Result from image generation attempt."""
    location_id: str
    brief_id: str  # "{location_id}_{candidate_index}" — groups results by brief
    success: bool
    image_bytes: bytes | None
    image_type: Literal["generated", "public_domain", "diagram"]
    prompt_or_query_used: str
    alt_text: str | None
    attribution: dict | None


class LocationSelection(TypedDict):
    """Result of per-location pair comparison."""
    location_id: str
    selected_brief_id: str | None  # None if both failed
    quality_tier: Literal["excellent", "acceptable", "failed"]
    reasoning: str


class IllustrateState(TypedDict, total=False):
    # Generation phase (parallel aggregation via add reducer)
    generation_results: Annotated[list[ImageGenResult], add]

    # Selection phase (parallel aggregation via add reducer)
    selection_results: Annotated[list[LocationSelection], add]

    # Retry tracking — counts derived from selection_results, not incremented per entry
    retry_count: Annotated[dict[str, int], merge_dicts]
```

**Key change**: `selection_results` replaces `review_results`, `pending_retries`, and `retry_briefs`. The `brief_id` field enables grouping candidates by location.

### Unified Generation Node

One node handles all generation types, routing based on `brief.image_type`:

```python
# workflows/output/illustrate/nodes/generate_candidate.py

async def generate_candidate_node(state: dict) -> dict:
    """Generate a single image candidate from a CandidateBrief.

    Receives brief + location plan, routes to the appropriate generator,
    and tags the result with brief_id for downstream grouping.
    """
    plan: ImageLocationPlan = state["location"]
    brief: CandidateBrief = state["brief"]
    brief_id: str = state["brief_id"]
    image_type = brief.image_type

    if image_type == "public_domain":
        result = await _generate_public_domain(location_id, plan, brief_text, document_context, brief_id)
    elif image_type == "diagram":
        result = await _generate_diagram(location_id, plan, brief_text, config, visual_identity, brief_id)
    elif image_type == "generated":
        result = await _generate_imagen(location_id, plan, brief_text, config, visual_identity, brief_id)

    return result
```

**Pattern**: Single node with internal routing simplifies graph structure and ensures consistent error handling across generation types.

### Fan-Out to Generation

Each `CandidateBrief` gets its own parallel generation task:

```python
# workflows/output/illustrate/graph.py

def route_after_analysis(state: IllustrateState) -> list[Send] | str:
    """Fan out one Send per CandidateBrief for parallel generation."""
    candidate_briefs = state.get("candidate_briefs", [])
    image_plan = state.get("image_plan", [])
    visual_identity = state.get("visual_identity")

    sends = []
    for brief in candidate_briefs:
        plan = _find_plan_by_id(image_plan, brief.location_id)
        brief_id = f"{brief.location_id}_{brief.candidate_index}"
        send_data = {
            "location": plan,
            "brief": brief,
            "brief_id": brief_id,
            "document_context": document,
            "config": config,
            "visual_identity": visual_identity,
        }
        sends.append(Send("generate_candidate", send_data))

    return sends or "finalize"
```

**Result**: For 6 locations with 2 candidates each, 12 parallel generation tasks execute simultaneously.

### Per-Location Selection Node

Groups candidates by location and applies vision pair comparison:

```python
# workflows/output/illustrate/nodes/select_per_location.py

async def select_per_location_node(state: dict) -> dict:
    """Select the best image candidate at a location using vision pair comparison.

    Handles three cases:
    - 0 successful candidates → quality_tier="failed"
    - 1 successful candidate → auto-select, quality_tier="acceptable"
    - 2 successful candidates → vision pair comparison, quality_tier="excellent"
    """
    location_id: str = state["location_id"]
    candidates: list[dict] = state["candidates"]  # successful ImageGenResults
    selection_criteria: str = state.get("selection_criteria", "")

    if not candidates:
        return {
            "selection_results": [
                LocationSelection(
                    location_id=location_id,
                    selected_brief_id=None,
                    quality_tier="failed",
                    reasoning="Both candidates failed generation",
                )
            ]
        }

    if len(candidates) == 1:
        return {
            "selection_results": [
                LocationSelection(
                    location_id=location_id,
                    selected_brief_id=candidates[0]["brief_id"],
                    quality_tier="acceptable",
                    reasoning="Auto-selected: only one candidate succeeded",
                )
            ]
        }

    # Vision pair comparison
    png_list = [c["image_bytes"] for c in candidates]
    best_idx = await vision_pair_select(png_list, selection_criteria=selection_criteria)
    selected = candidates[best_idx]

    return {
        "selection_results": [
            LocationSelection(
                location_id=location_id,
                selected_brief_id=selected["brief_id"],
                quality_tier="excellent",
                reasoning="Selected via vision pair comparison",
            )
        ]
    }
```

**Pattern**: Quality tier captures confidence in the selection. "excellent" means we had choices and picked the best; "acceptable" means we had no choice; "failed" means we need to retry.

### Grouping for Selection

Route from sync barrier to per-location selection nodes:

```python
# workflows/output/illustrate/graph.py

def route_to_selection(state: IllustrateState) -> list[Send] | str:
    """Group generation results by location_id, fan out pair comparison."""
    generation_results = state.get("generation_results", [])

    # Group by location_id
    by_location: dict[str, list[dict]] = defaultdict(list)
    for r in generation_results:
        by_location[r["location_id"]].append(r)

    # Build neutral selection criteria from opportunities + editorial notes
    opportunities = state.get("image_opportunities", [])
    editorial_notes = state.get("editorial_notes", "")

    sends = []
    for location_id, candidates in by_location.items():
        successful = [c for c in candidates if c["success"] and c.get("image_bytes")]
        sends.append(
            Send(
                "select_per_location",
                {
                    "location_id": location_id,
                    "candidates": successful,
                    "selection_criteria": _build_selection_criteria(
                        opportunities, editorial_notes, location_id
                    ),
                },
            )
        )

    return sends or "finalize"
```

**Key detail**: `selection_criteria` derives from `ImageOpportunity.rationale` and `purpose`, not from any single candidate's brief, ensuring unbiased comparison.

### Memory Management

After selection, clear non-winning candidates to free memory:

```python
# workflows/output/illustrate/graph.py

def sync_after_selection(state: IllustrateState) -> dict:
    """Synchronization barrier after all selections complete.

    Updates retry_count for failed locations and clears image_bytes
    from non-winning generation results to free memory.
    """
    selection_results = state.get("selection_results", [])

    # Derive retry counts from accumulated selection_results
    retry_count: dict[str, int] = {}
    for s in selection_results:
        if s["quality_tier"] == "failed":
            retry_count[s["location_id"]] = retry_count.get(s["location_id"], 0) + 1

    # Identify winning brief_ids
    latest_selection: dict[str, dict] = {}
    for s in selection_results:
        latest_selection[s["location_id"]] = s

    winning_brief_ids: set[str | None] = set()
    for sel in latest_selection.values():
        if sel["quality_tier"] != "failed" and sel["selected_brief_id"]:
            winning_brief_ids.add(sel["selected_brief_id"])

    # Clear image_bytes from losers
    generation_results = state.get("generation_results", [])
    cleared = 0
    for gen in generation_results:
        if gen["brief_id"] not in winning_brief_ids and gen.get("image_bytes"):
            gen["image_bytes"] = b""
            cleared += 1

    logger.info(f"Cleared {cleared} non-winning candidates")
    return {"retry_count": retry_count}
```

**Pattern**: Reducers prevent replacing the list, so mutate entries in-place. This prevents memory accumulation across retry rounds.

### Cross-Strategy Fallback

Failed locations retry with alternate image source type:

```python
# workflows/output/illustrate/graph.py

_FALLBACK_IMAGE_TYPE = {
    "public_domain": "generated",
    "generated": "public_domain",
    "diagram": "generated",
}


def route_after_selection(state: IllustrateState) -> list[Send] | str:
    """Route failed locations to retry with cross-strategy fallback."""
    selection_results = state.get("selection_results", [])
    retry_count = state.get("retry_count", {})
    candidate_briefs = state.get("candidate_briefs", [])

    # Find failed locations eligible for retry
    failed_locations = [
        s["location_id"]
        for s in selection_results
        if s["quality_tier"] == "failed" and retry_count.get(s["location_id"], 0) <= config.max_retries
    ]

    if not failed_locations:
        return "finalize"

    sends = []
    for loc_id in failed_locations:
        location_briefs = [b for b in candidate_briefs if b.location_id == loc_id]

        # Generate two retry candidates with fallback image types
        for orig_brief in location_briefs[:2]:
            fallback_type = _FALLBACK_IMAGE_TYPE.get(orig_brief.image_type, "generated")
            retry_brief = orig_brief.model_copy(update={"image_type": fallback_type})

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

    return sends if sends else "finalize"
```

**Rationale**: If public domain search returns no suitable results, switching to Imagen may succeed. If Imagen generation fails, trying public domain may find a match. This is more effective than retrying the same approach with feedback.

### Retry Count Derivation

Retry counts are derived from `selection_results`, not incremented per entry:

```python
# sync_after_selection (excerpt)

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
```

**Gotcha**: Previously, retry counts were incremented in sync nodes. With parallel selection nodes writing to the same dict via a merge reducer, this caused over-inflation (each selection node incremented the count independently). Deriving from accumulated results fixes this.

### Finalize: Selecting Winners

Final images are selected from generation results based on selection results:

```python
# workflows/output/illustrate/nodes/finalize.py

def _select_winning_results(
    generation_results: list[ImageGenResult],
    selection_results: list[LocationSelection],
) -> list[ImageGenResult]:
    """Pick the winning ImageGenResult per location based on selection_results."""
    # Build lookup: brief_id -> ImageGenResult
    results_by_brief_id: dict[str, ImageGenResult] = {}
    for gen in generation_results:
        if gen["success"] and gen.get("image_bytes"):
            results_by_brief_id[gen["brief_id"]] = gen

    # Deduplicate selections: keep only the LAST entry per location_id
    latest_selection: dict[str, LocationSelection] = {}
    for selection in selection_results:
        latest_selection[selection["location_id"]] = selection

    winners: list[ImageGenResult] = []
    for loc_id, selection in latest_selection.items():
        if selection["quality_tier"] == "failed":
            continue

        brief_id = selection["selected_brief_id"]
        if brief_id and brief_id in results_by_brief_id:
            winners.append(results_by_brief_id[brief_id])

    return winners
```

**Key insight**: Deduplication is critical. `selection_results` accumulates across retry rounds, so without dedup, the first (often "failed") entry claims the slot and retry successes are ignored.

## Key Design Decisions

### Why Two Candidates Per Location?

| Count | Benefits | Costs |
|-------|----------|-------|
| 1 | Lowest cost, simplest | No redundancy, no comparison |
| 2 | Pair comparison, 2x redundancy | 2x generation cost |
| 3+ | More choices | Diminishing returns, N-1 vision calls |

Two provides redundancy without excessive cost. Vision pair comparison achieves 80.6% selection accuracy (per MLLM-as-a-Judge research).

### Why Unified Generation Node?

| Approach | Structure | Benefits | Costs |
|----------|-----------|----------|-------|
| Separate nodes | `generate_header`, `generate_additional` | Clear separation | Duplicate code, routing complexity |
| Unified node | Single `generate_candidate` | Shared error handling, simpler graph | Internal routing logic |

Unified node reduces duplication and simplifies the graph shape. The brief carries `image_type`, so routing is straightforward.

### Why Quality Tiers?

Quality tiers enable observability and fallback decisions:

- **excellent** (2 candidates, vision selected): High confidence in quality
- **acceptable** (1 candidate auto-selected): Lower confidence, but better than nothing
- **failed** (0 candidates): Needs retry

This is more informative than binary pass/fail and enables selective retries.

### Why Cross-Strategy Fallback?

Retrying the same generation approach with feedback often fails in the same way. Switching image source types provides genuine diversity:

- Public domain fails → try Imagen (no matching images in corpus)
- Imagen fails → try public domain (prompt interpretation issue)
- Diagram fails → try Imagen (diagram complexity issue)

This is similar to the multi-source fallback pattern but applied at retry time.

### Why Derive Retry Count from Results?

Previously, retry counts were incremented in sync nodes. With parallel selection nodes writing to a shared dict via merge reducer, each node independently incremented the count, causing over-inflation (e.g., 6 selection nodes each incrementing `retry_count[loc_id]` results in count 6 after one round).

Deriving counts from accumulated `selection_results` is idempotent and correct: count the number of "failed" entries for each location.

### Why Remove Review Node?

The per-location pair comparison selects between two genuinely different candidates. This subsumes the role of the old review node, which assessed a single candidate and decided whether to retry. Pair comparison is both cheaper (one call per location instead of one per candidate) and more effective (comparative judgment vs absolute judgment).

### Brief ID Collision Fix

Original implementation used `brief_id = "{location_id}_{candidate_index}"`, which collided across retry rounds. Fixed by including round number: `brief_id = "{loc_id}_{candidate_index}_retry{round_num}"`.

This enables correct winner selection in `finalize` when the same location succeeds on retry.

## Consequences

### Benefits

- **Built-in redundancy**: Two candidates per location provide inherent fault tolerance
- **Higher quality**: Vision pair comparison selects better results than heuristics
- **Simpler retry logic**: No state tracking for pending retries, retry briefs
- **Parallel efficiency**: All candidates generate simultaneously, not sequentially
- **Cross-strategy diversity**: Fallback to alternate image sources when both candidates fail
- **Observability**: Quality tiers provide insight into workflow confidence
- **Memory efficiency**: Non-winning candidates cleared after selection

### Trade-offs

- **Higher generation cost**: 2x candidates means 2x API calls for generation
- **Vision comparison cost**: One vision call per location for pair comparison
- **More complex state**: `brief_id` field, quality tiers, selection results
- **Max retries reduced**: From 2 to 1, relying on over-generation for quality
- **Brief writing cost**: Pass 2 writes two briefs per location (more LLM work)

### Cost Analysis

For 6 locations with 2 candidates each:

- **Generation**: 12 parallel calls (2 per location)
- **Selection**: 6 vision calls (1 per location)
- **Retry (if needed)**: 2 more calls per failed location

Total for all locations succeeding first try: 12 generation + 6 vision = 18 calls vs 6 generation + 6 review + possible retries for single-generation approach.

The over-generation approach front-loads cost for higher quality and reliability.

## Known Uses

- `workflows/output/illustrate/graph.py` — Graph construction with unified generation node and per-location selection
- `workflows/output/illustrate/nodes/generate_candidate.py` — Unified generation node with routing
- `workflows/output/illustrate/nodes/select_per_location.py` — Per-location pair comparison
- `workflows/output/illustrate/state.py` — State schema with `brief_id`, quality tiers, selection results
- `workflows/output/illustrate/nodes/finalize.py` — Winner selection from accumulated results
- `workflows/shared/vision_comparison.py` — Tournament-style vision pair comparison

## Related Patterns

- [Document Illustration Workflow](./document-illustration-workflow.md) — The full workflow using this pattern
- [Parallel Candidate Vision Selection](../llm-interaction/parallel-candidate-vision-selection.md) — The underlying vision pair comparison technique
- [Two-Pass LLM Planning](../llm-interaction/two-pass-llm-planning.md) — How two candidates per location are planned

## References

- Commit `57fa0cf` — feat(illustrate): over-generation with per-location pair selection
- Commit `de777b2` — fix(illustrate): resolve 15 code review findings (retry logic, security, dead code)
- MLLM-as-a-Judge research: pair comparison achieves 80.6% accuracy vs 55.7% for scoring
- Files:
  - `workflows/output/illustrate/graph.py` — Graph construction
  - `workflows/output/illustrate/state.py` — State schema changes
  - `workflows/output/illustrate/nodes/generate_candidate.py` — Unified generation node
  - `workflows/output/illustrate/nodes/select_per_location.py` — Per-location selection
  - `workflows/output/illustrate/nodes/finalize.py` — Winner selection
  - `workflows/shared/vision_comparison.py` — Vision pair comparison
