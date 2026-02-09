---
name: vision-editorial-curation
title: Vision-Based Editorial Image Curation
date: 2026-02-09
category: llm-interaction
applicability:
  - "Document illustration workflows with over-generation where surplus images must be cut"
  - "Multi-image visual content requiring document-level coherence assessment"
  - "Quality gating that depends on holistic context rather than individual image evaluation"
components: [llm_call, vision_model, structured_output]
complexity: moderate
verified_in_production: true
tags: [vision, editorial, curation, multi-image, document-level, coherence, sonnet, over-generation, quality-gate]
shared: true
gist_url: https://gist.github.com/DaveCBeck/05862ebff5aabcd73ea3f3fc69fda5a6
article_path: .context/libs/thala-dev/content/generation/2026-02-09-vision-editorial-curation.md
---

# Vision-Based Editorial Image Curation

## Intent

After over-generation produces N+2 images, use a single vision LLM call to evaluate all non-header images holistically in document context and cut the weakest 2, ensuring the retained set works as a cohesive whole with proper pacing, variety, and visual identity consistency.

## Problem

Per-location evaluation (including pair comparison) optimizes for individual image quality at each location but cannot assess document-level coherence:

1. **No pacing awareness**: Per-location selection cannot detect clustering or uneven distribution across the document
2. **No variety enforcement**: Cannot ensure adjacent images differ in type or visual approach
3. **No identity consistency**: Cannot compare visual coherence across the full set (e.g., all images matching style/palette)
4. **Surplus removal requires full context**: After over-generating N+2 images and selecting winners, cutting the weakest 2 requires comparing ALL images simultaneously

## Solution

Use a single SONNET vision call that receives all non-header images in document order, evaluates each on four criteria (visual coherence, pacing contribution, variety contribution, individual quality), ranks all images by overall contribution, and marks the bottom N for cutting with specific reasons.

### How It Differs from Pair Comparison

| Pattern | Scope | Purpose | Use Case |
|---------|-------|---------|----------|
| **Pair Comparison** | 2 candidates at ONE location | Select best candidate for a specific position | Choose between literal vs. metaphorical interpretations |
| **Editorial Curation** | ALL retained images across ALL locations | Ensure document-level coherence and cut surplus | Remove weakest images from over-generated set |

Pair comparison asks "which is better for THIS spot?" Editorial curation asks "which images work least well as part of the WHOLE?"

## Architecture

```
Select Per Location (pair comparison — per location)
    ↓
Assemble Document (gather winning images + metadata)
    ↓
Editorial Review (SONNET vision — all images at once)
    ↓
Finalize (apply cuts, save files)
```

**Key transition**: After per-location selection produces N+2 winners, assembly gathers them into a single list. Editorial review evaluates this complete set and identifies cuts. Finalize applies the cuts before file operations.

## Implementation

### State Schema

```python
# workflows/output/illustrate/state.py

class AssembledImage(TypedDict):
    """An image that won per-location selection, ready for editorial review."""
    location_id: str
    image_bytes: bytes
    image_type: Literal["generated", "public_domain", "diagram"]
    purpose: Literal["header", "illustration", "diagram"]
    # ... other metadata

class IllustrateState(TypedDict, total=False):
    # Assembly phase
    assembled_images: list[AssembledImage]  # Winners from selection phase

    # Editorial review output
    editorial_review_result: dict  # EditorialReviewResult.model_dump()
```

### Structured Output Schemas

```python
# workflows/output/illustrate/schemas.py

class EditorialImageEvaluation(BaseModel):
    """Evaluation of a single image in the document context."""

    model_config = ConfigDict(extra="forbid")

    location_id: str
    contribution_rank: int = Field(
        ge=1,
        description="1=strongest contribution to the article, N=weakest",
    )
    visual_coherence: int = Field(
        ge=1,
        le=5,
        description="1=clashes with surrounding images, 5=perfectly complements the visual identity",
    )
    pacing_contribution: int = Field(
        ge=1,
        le=5,
        description="1=poorly placed or clustered, 5=ideal spacing and document flow",
    )
    variety_contribution: int = Field(
        ge=1,
        le=5,
        description="1=redundant with neighbors, 5=adds needed visual diversity",
    )
    individual_quality: int = Field(
        ge=1,
        le=5,
        description="1=poor technical quality or context mismatch, 5=excellent standalone quality",
    )
    cut_reason: str | None = Field(
        default=None,
        description="If this image should be cut, explain why (1-2 sentences)",
    )


class EditorialReviewResult(BaseModel):
    """Full editorial review output."""

    model_config = ConfigDict(extra="forbid")

    evaluations: list[EditorialImageEvaluation]
    cut_location_ids: list[str] = Field(description="The location_ids to remove")
    editorial_summary: str = Field(description="Overall assessment for logging")
```

**Design choices**:
- `extra="forbid"` prevents LLM from adding unexpected fields
- Four distinct evaluation dimensions (1-5 scales) force explicit reasoning
- `contribution_rank` (ge=1) provides total ordering independent of scores
- `cut_reason` only populated for images marked for cutting

### Multimodal Message Construction

```python
# workflows/output/illustrate/nodes/editorial_review.py

async def editorial_review_node(state: IllustrateState) -> dict:
    """Vision-based editorial review of the full illustrated document."""
    assembled_images = state.get("assembled_images", [])
    visual_identity = state.get("visual_identity")

    # Filter to non-header images only
    non_header_images = [img for img in assembled_images if img["purpose"] != "header"]

    # Compute adaptive cut count
    cuts_count = _compute_cuts_count(non_header_images, image_opportunities)

    # Build multimodal message
    user_prompt = EDITORIAL_USER.format(
        n_images=len(non_header_images),
        cuts_count=cuts_count,
        primary_style=visual_identity.primary_style,
        color_palette=", ".join(visual_identity.color_palette),
        mood=visual_identity.mood,
    )

    content_parts: list[dict] = [{"type": "text", "text": user_prompt}]

    for img in non_header_images:
        image_bytes = img["image_bytes"]
        if len(image_bytes) > MAX_IMAGE_SIZE:
            logger.warning("Skipping oversized image '%s'", img["location_id"])
            continue

        media_type = detect_media_type(image_bytes)
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Interleave image and label
        content_parts.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": b64,
            },
        })
        content_parts.append({
            "type": "text",
            "text": f"Image above is '{img['location_id']}' ({img['image_type']}, {img['purpose']})",
        })

    llm = get_llm(tier=ModelTier.SONNET).with_structured_output(EditorialReviewResult)
    response = await llm.ainvoke([
        {"role": "system", "content": EDITORIAL_SYSTEM},
        {"role": "user", "content": content_parts},
    ])

    return {"editorial_review_result": response.model_dump()}
```

**Pattern**: Interleaved `[image, label text]` pairs ensure each image is immediately identified by its `location_id` and metadata, reducing ambiguity in multi-image evaluation.

### Adaptive Cut Count

```python
# workflows/output/illustrate/nodes/editorial_review.py

def _compute_cuts_count(
    non_header_images: list[AssembledImage],
    image_opportunities: list[ImageOpportunity],
) -> int:
    """Compute how many images to cut.

    Rule: never cut below the target N.
    Target N = total non-header opportunities - 2 (the over-generation surplus).
    Cuts = min(2, non_header_count - target_non_header_count).
    """
    non_header_opportunities = [opp for opp in image_opportunities if opp.purpose != "header"]
    target_non_header = max(0, len(non_header_opportunities) - 2)

    surplus = len(non_header_images) - target_non_header
    return max(0, min(2, surplus))
```

**Rationale**: If over-generation only produced 5 images instead of the target 6, cutting 2 would leave 3 (below target). The adaptive formula caps cuts at actual surplus.

### Cut ID Validation and Deduplication

```python
# workflows/output/illustrate/nodes/editorial_review.py (continued)

# Validate cut count doesn't exceed requested
valid_location_ids = {img["location_id"] for img in non_header_images}
cut_ids = list(dict.fromkeys(cid for cid in response.cut_location_ids if cid in valid_location_ids))
cut_ids = cut_ids[:cuts_count]  # Never cut more than requested

result = response.model_copy(update={"cut_location_ids": cut_ids})
```

**Key details**:
- Filter to valid location IDs (prevents hallucinated IDs)
- `dict.fromkeys()` deduplicates while preserving order
- Cap at requested `cuts_count` (prevents over-cutting)

### Evaluation Criteria Prompt

```python
# workflows/output/illustrate/nodes/editorial_review.py

EDITORIAL_USER = """This article was intentionally illustrated with {cuts_count} more images than needed so you can select the strongest set. You will evaluate {n_images} non-header images and cut the {cuts_count} that contribute least to the overall article.

Evaluate each image on:
1. Visual coherence — Does it match the style/identity of the other images?
2. Pacing contribution — Is it well-placed? Does it avoid clustering?
3. Variety contribution — Is it different from its neighbors? Different type than adjacent?
4. Individual quality — Is it technically good and contextually relevant?

Visual identity for this article:
- Style: {primary_style}
- Palette: {color_palette}
- Mood: {mood}

Rank ALL {n_images} images from strongest to weakest contribution, then mark the bottom {cuts_count} for removal. For each cut, explain why.

The images are shown below with their location IDs."""
```

**Design choices**:
- Explicit visual identity context (from creative direction pass) enables coherence judgments
- Four evaluation dimensions force consideration of multiple quality aspects
- "Rank ALL images" mitigates positional bias by requiring total ordering
- "Mark bottom N" for cutting with reasons ensures cuts are justified

### Visual Identity Context

The prompt includes visual identity parameters from the creative direction pass:

```python
# From creative_direction_node output
visual_identity = VisualIdentity(
    primary_style="editorial watercolor illustration",
    color_palette=["warm amber", "deep teal", "ivory"],
    mood="contemplative, intellectual, accessible",
    lighting="soft diffused natural light",
    avoid=["photorealistic faces", "neon colors"],
)
```

This shared context enables the editorial reviewer to assess coherence across the full image set using the same visual criteria that guided individual image generation.

### Image Size Limit

```python
# workflows/output/illustrate/nodes/generate_additional.py

MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5 MB

# In editorial_review_node:
if len(image_bytes) > MAX_IMAGE_SIZE:
    logger.warning(
        "Skipping oversized image '%s' (%d bytes, limit %d) from editorial review",
        img["location_id"], len(image_bytes), MAX_IMAGE_SIZE
    )
    continue
```

**Rationale**: Anthropic vision API has per-image size limits. Skipping oversized images prevents API errors. In practice, this rarely happens (images are typically <1MB after PNG conversion).

### Memory Management

```python
# workflows/output/illustrate/nodes/editorial_review.py

return {
    "editorial_review_result": result.model_dump(),
    "assembled_images": [],  # Clear to free memory
}
```

**Pattern**: After editorial review completes, clear `assembled_images` from state to free memory. The finalize node uses `selection_results` and `generation_results` to reconstruct winners, not `assembled_images`.

### Integration with Finalize

```python
# workflows/output/illustrate/nodes/finalize.py

def _determine_status(
    image_plan: list[ImageLocationPlan],
    final_images: list[ImageGenResult],
    editorial_review_result: dict | None,
) -> Literal["complete", "partial", "failed"]:
    """Determine workflow completion status.

    Editorially cut locations are excluded from 'expected' count so
    cuts don't cause false 'partial' status.
    """
    editorial_cuts = set()
    if editorial_review_result:
        editorial_cuts = set(editorial_review_result.get("cut_location_ids", []))

    # Expected locations = all planned locations MINUS editorial cuts
    expected_locations = {
        p["location_id"] for p in image_plan if p["location_id"] not in editorial_cuts
    }

    actual_locations = {img["location_id"] for img in final_images if img["success"]}

    if not expected_locations:
        return "failed"
    if actual_locations >= expected_locations:
        return "complete"
    if actual_locations:
        return "partial"
    return "failed"
```

**Key insight**: Without adjusting the expected count for editorial cuts, a workflow that successfully cut 2 images would be marked "partial" (e.g., 4/6 images) instead of "complete" (4/4 after cuts).

### Conditional Graph Routing

```python
# workflows/output/illustrate/graph.py

def route_after_assembly(state: IllustrateState) -> str:
    """Route to editorial review or skip directly to finalize."""
    config = state.get("config", {})
    enable_editorial = config.get("enable_editorial_review", True)

    if not enable_editorial:
        logger.info("Editorial review disabled, skipping to finalize")
        return "finalize"

    assembled_images = state.get("assembled_images", [])
    non_header_images = [img for img in assembled_images if img["purpose"] != "header"]

    if len(non_header_images) <= 2:
        logger.info("Insufficient images for editorial review, skipping")
        return "finalize"

    return "editorial_review"
```

**Configuration**: `enable_editorial_review: bool = True` allows disabling the editorial review pass for testing or cost-sensitive workflows.

### Fail-Open Error Handling

```python
# workflows/output/illustrate/nodes/editorial_review.py

try:
    llm = get_llm(tier=ModelTier.SONNET).with_structured_output(EditorialReviewResult)
    response = await llm.ainvoke([
        {"role": "system", "content": EDITORIAL_SYSTEM},
        {"role": "user", "content": content_parts},
    ])
    # ... validation ...
except Exception as e:
    logger.warning(f"Editorial review failed, keeping all images: {e}")
    result = EditorialReviewResult(
        evaluations=[],
        cut_location_ids=[],
        editorial_summary="Editorial review failed. All images kept.",
    )
```

**Rationale**: Editorial review is a quality enhancement, not a blocking requirement. If the vision call fails, the workflow continues with all generated images rather than failing entirely.

## Key Design Decisions

### Why Single Vision Call vs. Per-Image Evaluation?

| Approach | API Calls | Context | Cost |
|----------|-----------|---------|------|
| Per-image evaluation | N calls | No awareness of other images | N × vision call cost |
| Single vision call | 1 call | All images in one context | 1 × vision call cost |

Single-call approach is cheaper and provides full document context necessary for coherence/pacing/variety judgments.

### Why Four Evaluation Dimensions?

The four criteria (visual coherence, pacing contribution, variety contribution, individual quality) force the LLM to consider orthogonal aspects of image fit:

- **Visual coherence**: Matches style/palette/identity
- **Pacing contribution**: Spacing and document flow
- **Variety contribution**: Diversity from neighbors
- **Individual quality**: Standalone technical quality

Without explicit dimensions, the LLM tends to focus only on individual quality, ignoring document-level concerns.

### Why Adaptive Cut Count?

If over-generation produces fewer images than expected (e.g., some locations fail), cutting a fixed 2 images would drop below the target. The adaptive formula ensures cuts never reduce the set below the original target:

```
Target N = 6 non-header images
Over-generated to N+2 = 8 images
Expected surplus = 2, cuts_count = 2

If only 5 images succeed:
Surplus = 5 - 6 = -1
cuts_count = max(0, min(2, -1)) = 0  # No cuts
```

### Why Exclude Headers from Editorial Review?

Header images serve a distinct purpose (visual anchor for the article) and are not comparable to inline illustrations. Editorial review focuses on non-header images where pacing, variety, and coherence are relevant.

### Why dict.fromkeys() for Deduplication?

LLMs sometimes repeat location IDs in the cut list. `dict.fromkeys(cut_ids)` deduplicates while preserving order (unlike `set()` which loses order):

```python
# LLM output: ["section_2", "section_4", "section_2"]
cut_ids = list(dict.fromkeys(response.cut_location_ids))
# Result: ["section_2", "section_4"]
```

## Consequences

### Benefits

- **Holistic quality gating**: Document-level coherence assessment impossible with per-image review
- **Single API call**: Cheaper than per-image evaluation (1 call for N images vs. N calls)
- **Explicit reasoning**: Four evaluation dimensions + ranking + cut reasons provide transparency
- **Visual identity consistency**: Using creative direction context enables style/palette coherence checks
- **Adaptive cutting**: Never cuts below target N regardless of generation success rate
- **Fail-open resilience**: Workflow continues with all images if editorial review fails

### Trade-offs

- **One additional Sonnet vision call**: Adds cost per workflow (but cheaper than N per-image calls)
- **Memory spike**: All images must be base64-encoded simultaneously (mitigated by MAX_IMAGE_SIZE limit and clearing `assembled_images` after review)
- **Positional bias risk**: Multi-image evaluation may favor images shown first (mitigated by explicit ranking requirement)
- **No individual image feedback**: Unlike per-location review, editorial review doesn't provide improvement suggestions for individual images

### Cost Analysis

For 6 non-header images in editorial review:

- **Single vision call**: 1 × Sonnet vision call with 6 images
- **Alternative (per-image)**: 6 × Sonnet vision calls

Editorial curation is ~6× cheaper than hypothetical per-image document-level evaluation.

## Known Uses

- `workflows/output/illustrate/nodes/editorial_review.py` — Main implementation
- `workflows/output/illustrate/schemas.py` — `EditorialImageEvaluation`, `EditorialReviewResult`
- `workflows/output/illustrate/graph.py` — `route_after_assembly()` conditional routing
- `workflows/output/illustrate/nodes/finalize.py` — `_determine_status()` excludes editorial cuts from expected count
- `workflows/output/illustrate/config.py` — `enable_editorial_review` configuration flag

## Related Patterns

- [Over-Generation with Per-Location Pair Selection](../langgraph/over-generation-pair-selection.md) — The over-generation architecture that produces N+2 images for editorial curation
- [Parallel Candidate Vision Selection](./parallel-candidate-vision-selection.md) — The pair comparison pattern used per-location before editorial review
- [Document Illustration Workflow](../langgraph/document-illustration-workflow.md) — The full workflow integrating both patterns

## References

- Commit `c56a75b` — feat(illustrate): editorial review — vision-based full-document curation
- Commit `de777b2` — fix(illustrate): resolve 15 code review findings — retry logic, security, dead code
- Files:
  - `workflows/output/illustrate/nodes/editorial_review.py` — Implementation
  - `workflows/output/illustrate/schemas.py` — Structured output schemas
  - `workflows/output/illustrate/graph.py` — Graph routing
  - `workflows/output/illustrate/nodes/finalize.py` — Status computation with editorial cuts
