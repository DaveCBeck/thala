---
name: document-illustration-workflow
title: Document Illustration with Multi-Source Generation
date: 2026-01-26
category: langgraph
applicability:
  - "Workflows requiring parallel generation from multiple sources with quality review"
  - "Fan-out/fan-in patterns with conditional retry loops"
  - "Iterative refinement loops with vision-based assessment"
components: [workflow_graph, langgraph_node, langgraph_graph, llm_call]
complexity: complex
verified_in_production: true
tags: [fan-out, fan-in, send, parallel-execution, vision-review, retry-loop, multi-source, diagram-refinement, quality-assessment, sync-barrier]
---

# Document Illustration with Multi-Source Generation

## Intent

Create a workflow that intelligently illustrates markdown documents using multiple image sources (public domain, AI-generated, SVG diagrams) with parallel generation, vision-based review, and conditional retry loops.

## Problem

Document illustration workflows face several challenges:

1. **Multiple image sources** with different APIs, quality characteristics, and costs
2. **Parallel generation** needed for performance but requires careful state aggregation
3. **Quality control** needs vision-based assessment with retry capability
4. **Different image types** require specialized handling (diagrams need refinement loops)
5. **Graceful degradation** when preferred sources fail

## Solution

A LangGraph workflow with:
- **Fan-out/fan-in** using `Send()` for parallel image generation
- **Sync barrier nodes** for coordination between phases
- **Vision review** with structured output for quality assessment
- **Conditional retry loop** for failed images within the same graph
- **Nested refinement loop** (outside main graph) for SVG diagram quality

### Workflow Architecture

```
START
  |
  v
analyze_document (Sonnet plans all image locations)
  |
  v
[conditional] --> Send("generate_header", {...}) for headers
            \--> Send("generate_additional", {...}) for others
            \--> "finalize" if no images planned
  |
  v (all generations complete)
sync_after_generation (barrier node)
  |
  v
[conditional] --> Send("review_image", {...}) for each success
            \--> "finalize" if review disabled
  |
  v (all reviews complete)
sync_after_review (barrier, update retry counts)
  |
  v
[conditional] --> Send("generate_*", {..., retry_brief}) for retries
            \--> "finalize" if no eligible retries
  |
  v
finalize (save files, insert into markdown)
  |
  v
END
```

## Implementation

### State Definition with Reducers

The critical pattern: **every field written by parallel branches needs a reducer**.

```python
# workflows/output/illustrate/state.py

from operator import add
from typing import Annotated, Literal
from typing_extensions import TypedDict


def merge_dicts(left: dict, right: dict) -> dict:
    """Reducer that merges dictionaries from parallel nodes."""
    result = dict(left) if left else {}
    if right:
        result.update(right)
    return result


class IllustrateState(TypedDict, total=False):
    """Main workflow state for document illustration.

    Uses Annotated[list[...], add] for parallel aggregation of results
    from nodes invoked via Send().
    """

    # Input (no reducer needed - set once at start)
    input: IllustrateInput
    config: IllustrateConfig

    # Analysis phase (no reducer - single node writes)
    extracted_title: str
    image_plan: list[ImageLocationPlan]

    # Generation phase - PARALLEL writes require reducers
    generation_results: Annotated[list[ImageGenResult], add]

    # Review phase - PARALLEL writes require reducers
    review_results: Annotated[list[ImageReviewResult], add]

    # Retry tracking - PARALLEL writes require reducers
    retry_count: Annotated[dict[str, int], merge_dicts]
    pending_retries: Annotated[list[str], add]
    retry_briefs: Annotated[dict[str, str], merge_dicts]

    # Final output (single node writes)
    final_images: list[FinalImage]
    illustrated_document: str

    # Workflow metadata - PARALLEL writes require reducers
    errors: Annotated[list[WorkflowError], add]
    status: Literal["success", "partial", "failed"]
```

**Key insight**: The `merge_dicts` reducer handles dictionary fields like `retry_count` and `retry_briefs` that multiple parallel nodes might update.

### Fan-Out with Send()

Route to parallel generation nodes based on analysis:

```python
# workflows/output/illustrate/graph.py

from langgraph.types import Send


def route_after_analysis(state: IllustrateState) -> list[Send] | str:
    """Route to generation after analysis.

    If analysis failed or no images planned, go to finalize.
    Otherwise, fan out to generate nodes.
    """
    if state.get("status") == "failed":
        return "finalize"

    image_plan = state.get("image_plan", [])
    if not image_plan:
        return "finalize"

    config = state.get("config") or IllustrateConfig()
    document = state["input"]["markdown_document"]

    sends = []

    for plan in image_plan:
        # Header uses special node with public domain preference
        if plan.purpose == "header":
            sends.append(
                Send(
                    "generate_header",
                    {
                        "location": plan,
                        "document_context": document,
                        "config": config,
                    },
                )
            )
        else:
            sends.append(
                Send(
                    "generate_additional",
                    {
                        "location": plan,
                        "document_context": document,
                        "config": config,
                    },
                )
            )

    return sends
```

**Pattern**: Conditional edges can return either:
- A string (single node name)
- A list of `Send()` objects for parallel execution

### Sync Barrier Nodes

Empty nodes that act as synchronization points:

```python
def sync_after_generation(state: IllustrateState) -> dict:
    """Synchronization barrier after all generations complete."""
    generation_results = state.get("generation_results", [])
    successful = sum(1 for r in generation_results if r["success"])
    logger.info(
        f"Generation sync: {successful}/{len(generation_results)} successful"
    )
    return {}  # No state changes, just synchronization


def sync_after_review(state: IllustrateState) -> dict:
    """Synchronization barrier and retry preparation after review.

    Updates retry_count for any pending retries.
    """
    pending_retries = state.get("pending_retries", [])
    retry_count = dict(state.get("retry_count", {}))

    for loc_id in pending_retries:
        retry_count[loc_id] = retry_count.get(loc_id, 0) + 1

    return {"retry_count": retry_count}
```

**Pattern**: Barrier nodes ensure all parallel branches complete before proceeding. They can also perform aggregation logic.

### Conditional Retry Loop

Route back to generation for failed images:

```python
def route_after_review(state: IllustrateState) -> list[Send] | str:
    """Route to retry generation or finalize."""
    config = state.get("config") or IllustrateConfig()
    pending_retries = state.get("pending_retries", [])
    retry_count = state.get("retry_count", {})
    retry_briefs = state.get("retry_briefs", {})
    image_plan = state.get("image_plan", [])
    document = state["input"]["markdown_document"]

    # Filter to retries within limit
    eligible_retries = [
        loc_id
        for loc_id in pending_retries
        if retry_count.get(loc_id, 0) <= config.max_retries
    ]

    if not eligible_retries:
        return "finalize"

    sends = []
    for loc_id in eligible_retries:
        plan = _find_plan_by_id(image_plan, loc_id)
        if not plan:
            continue

        retry_brief = retry_briefs.get(loc_id)

        # Use appropriate generation node
        if plan.purpose == "header":
            sends.append(
                Send(
                    "generate_header",
                    {
                        "location": plan,
                        "document_context": document,
                        "config": config,
                        "is_retry": True,
                        "retry_brief": retry_brief,  # Vision feedback
                    },
                )
            )
        else:
            sends.append(
                Send(
                    "generate_additional",
                    {
                        "location": plan,
                        "document_context": document,
                        "config": config,
                        "is_retry": True,
                        "retry_brief": retry_brief,
                    },
                )
            )

    return sends if sends else "finalize"
```

**Key insight**: The same generation nodes are reused for retries. The `retry_brief` contains vision model feedback for improved generation.

### Graph Construction

Wire up nodes with proper edge routing:

```python
def create_illustrate_graph() -> StateGraph:
    """Create the illustrate workflow graph."""
    builder = StateGraph(IllustrateState)

    # Add nodes
    builder.add_node("analyze_document", analyze_document_node)
    builder.add_node("generate_header", generate_header_node)
    builder.add_node("generate_additional", generate_additional_node)
    builder.add_node("sync_after_generation", sync_after_generation)
    builder.add_node("review_image", review_image_node)
    builder.add_node("sync_after_review", sync_after_review)
    builder.add_node("finalize", finalize_node)

    # Entry point
    builder.add_edge(START, "analyze_document")

    # After analysis, fan out to generation
    builder.add_conditional_edges(
        "analyze_document",
        route_after_analysis,
        ["generate_header", "generate_additional", "finalize"],
    )

    # All generation nodes converge to sync
    builder.add_edge("generate_header", "sync_after_generation")
    builder.add_edge("generate_additional", "sync_after_generation")

    # After generation sync, route to review or finalize
    builder.add_conditional_edges(
        "sync_after_generation",
        route_to_review,
        ["review_image", "finalize"],
    )

    # All review nodes converge to sync
    builder.add_edge("review_image", "sync_after_review")

    # After review sync, route to retry or finalize
    builder.add_conditional_edges(
        "sync_after_review",
        route_after_review,
        ["generate_header", "generate_additional", "finalize"],
    )

    # Finalize to end
    builder.add_edge("finalize", END)

    return builder.compile()
```

### Multi-Source Generation with Fallback

Header generation tries public domain first:

```python
# workflows/output/illustrate/nodes/generate_header.py

async def generate_header_node(state: dict) -> dict:
    """Generate header image: try public domain first, fallback to Imagen."""
    plan: ImageLocationPlan = state["location"]
    config: IllustrateConfig = state.get("config") or IllustrateConfig()
    is_retry: bool = state.get("is_retry", False)
    retry_brief: str | None = state.get("retry_brief")

    # Use retry brief if this is a retry
    brief = retry_brief or plan.brief

    # Step 1: Try public domain (unless this is a retry)
    if config.header_prefer_public_domain and not is_retry:
        try:
            pd_result = await get_image(query=plan.search_query or brief[:100])
            image_bytes = await _download_image(pd_result.url)

            # Vision evaluation: is this image "apposite"?
            is_apposite, reasoning = await _evaluate_pd_appositeness(
                image_bytes=image_bytes,
                document_context=state["document_context"],
                query=plan.search_query,
                criteria=brief,
            )

            if is_apposite:
                return {
                    "generation_results": [
                        ImageGenResult(
                            location_id=plan.location_id,
                            success=True,
                            image_bytes=image_bytes,
                            image_type="public_domain",
                            # ... other fields
                        )
                    ]
                }
        except NoResultsError:
            pass  # Fall through to Imagen

    # Step 2: Generate with Imagen
    image_bytes, prompt_used = await generate_article_header(
        custom_prompt=brief,
        aspect_ratio=config.imagen_aspect_ratio,
    )

    return {
        "generation_results": [
            ImageGenResult(
                location_id=plan.location_id,
                success=bool(image_bytes),
                image_bytes=image_bytes,
                image_type="generated",
                prompt_or_query_used=prompt_used or brief,
                # ... other fields
            )
        ]
    }
```

### Vision Review with Retry Brief

Review node determines if retry is needed:

```python
# workflows/output/illustrate/nodes/review_image.py

async def review_image_node(state: dict) -> dict:
    """Vision review of a generated image."""
    gen_result: ImageGenResult = state["generation_result"]
    plan: ImageLocationPlan = state["location"]

    # Get structured review from vision model
    review = await get_structured_output(
        output_schema=VisionReviewResult,
        user_prompt=[
            {"type": "text", "text": VISION_REVIEW_USER.format(...)},
            {"type": "image", "source": {...}},
        ],
        system_prompt=VISION_REVIEW_SYSTEM,
        tier=ModelTier.SONNET,
    )

    if review.recommendation == "retry":
        return {
            "review_results": [
                ImageReviewResult(
                    location_id=gen_result["location_id"],
                    passed=False,
                    severity="substantive",
                    issues=review.issues,
                    improved_brief=review.improved_brief,
                )
            ],
            "pending_retries": [gen_result["location_id"]],
            "retry_briefs": {gen_result["location_id"]: review.improved_brief},
        }

    # ... handle other recommendations
```

### Diagram Quality Refinement Loop

Separate from main graph, called during diagram generation:

```python
# workflows/shared/diagram_utils/refinement.py

async def refine_diagram_quality(
    svg_content: str,
    analysis: DiagramAnalysis,
    config: DiagramConfig,
) -> tuple[str, DiagramQualityAssessment | None, list[float]]:
    """Iteratively refine diagram until quality threshold met.

    Loop logic:
    1. Convert SVG to PNG
    2. Assess quality with vision model (7 criteria)
    3. If meets threshold or max iterations reached, exit
    4. If no improvement for 2 consecutive rounds, exit
    5. Generate feedback and regenerate SVG
    6. Loop back to step 1
    """
    max_iterations = config.max_refinement_iterations  # Default 3
    threshold = config.quality_threshold  # Default 4.7

    current_svg = svg_content
    quality_history: list[float] = []
    best_svg = svg_content
    best_score = 0.0
    consecutive_no_improvement = 0

    for iteration in range(max_iterations):
        # Convert SVG to PNG for assessment
        png_bytes = convert_svg_to_png(current_svg, dpi=config.dpi)
        if not png_bytes:
            break

        # Assess quality on 7 criteria
        assessment = await assess_diagram_quality(
            svg_content=current_svg,
            png_bytes=png_bytes,
            analysis=analysis,
            config=config,
        )
        if not assessment:
            break

        current_score = assessment.overall_score
        quality_history.append(current_score)

        # Track best result
        if current_score > best_score:
            best_svg = current_svg
            best_score = current_score
            consecutive_no_improvement = 0
        else:
            consecutive_no_improvement += 1

        # Exit if threshold met
        if assessment.meets_threshold:
            return current_svg, assessment, quality_history

        # Exit if no improvement for 2 consecutive rounds
        if consecutive_no_improvement >= 2:
            break

        # Regenerate with feedback
        improved_svg = await _regenerate_svg_with_feedback(
            svg_content=current_svg,
            assessment=assessment,
            analysis=analysis,
            config=config,
        )
        if not improved_svg:
            break

        current_svg = improved_svg

    return best_svg, best_assessment, quality_history
```

**7 Quality Criteria**:
1. `text_legibility` - Font sizes, contrast
2. `overlap_free` - No element overlaps
3. `visual_hierarchy` - Clear importance levels
4. `spacing_balance` - Even whitespace distribution
5. `layout_logic` - Natural reading flow
6. `shape_appropriateness` - Correct shapes for content
7. `completeness` - All key elements present

## Key Design Patterns

### 1. Reducer Selection Guide

| State Field Type | Parallel Writes? | Recommended Reducer |
|------------------|------------------|---------------------|
| `list[T]` | Yes | `Annotated[list[T], add]` |
| `dict[str, T]` | Yes | `Annotated[dict[str, T], merge_dicts]` |
| Scalar | No | No reducer (last-write-wins) |
| Scalar | Yes | Custom reducer or restructure |

### 2. Sync Barrier Pattern

```
[Fan-out via Send()] --> node_a, node_b, node_c
                              |       |       |
                              +-------+-------+
                                      |
                                      v
                              sync_barrier_node
                                      |
                                      v
                              [Next phase]
```

### 3. Conditional Retry Pattern

```python
def route_after_review(state) -> list[Send] | str:
    eligible = [id for id in pending if retry_count[id] <= max_retries]
    if not eligible:
        return "finalize"  # Exit condition
    return [Send("generate", {..., is_retry=True}) for id in eligible]
```

### 4. Multi-Source Fallback Chain

```
Preferred Source (e.g., public domain)
         |
         | (not available or not suitable)
         v
Fallback Source (e.g., AI generated)
         |
         | (failed)
         v
Return failure result (graceful degradation)
```

## Trade-offs

| Aspect | Benefit | Cost |
|--------|---------|------|
| Parallel generation | Faster total execution | More complex state management |
| Vision review | Quality assurance | Additional API calls |
| Retry loop | Higher success rate | Longer execution for failures |
| Sync barriers | Clean phase separation | Additional graph nodes |
| Separate refinement loop | Simpler main graph | Nested async complexity |

## Related Patterns

- **Fan-Out/Fan-In**: Core pattern for parallel execution
- **Sync Barrier**: Coordination between parallel phases
- **Conditional Retry**: Quality-based regeneration loops
- **Multi-Source Fallback**: Preference ordering with graceful degradation
- **Nested Quality Loop**: Iterative refinement outside main graph

## References

- Commit: `6909ba2` - feat(illustrate): add document illustration workflow
- Files:
  - `/home/dave/thala/workflows/output/illustrate/graph.py` - Main workflow graph
  - `/home/dave/thala/workflows/output/illustrate/state.py` - State with reducers
  - `/home/dave/thala/workflows/shared/diagram_utils/refinement.py` - Quality loop
  - `/home/dave/thala/workflows/shared/diagram_utils/quality_assessment.py` - Vision assessment
