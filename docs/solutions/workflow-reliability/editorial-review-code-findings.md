---
title: "Editorial Review Code Findings"
module: workflows/output/illustrate
date: 2026-02-09
problem_type: logic_errors
component: editorial_review
symptoms:
  - "_determine_status reports 'partial' even when all non-cut images are present"
  - "Exception details leaked into editorial_summary state field"
  - "Oversized images cause memory spikes during base64 encoding"
  - "Duplicated utility functions across 3+ files"
root_cause: "Missing editorial-cut awareness in status logic, plus standard code review findings from new feature implementation"
resolution_type: code_fix
severity: medium
tags: [editorial-review, code-review, DRY, schemas, security, dead-code, type-safety, status-determination]
---

# Editorial Review Code Findings

## Problem

The editorial review feature (introduced in commit `c56a75b`) added vision-based image curation, allowing the LLM to cut unnecessary images after full-document review. Code review identified 15 findings spanning correctness issues, security gaps, DRY violations, and dead code.

### Symptoms

1. **Status determination bug**: After editorial review cuts images, `_determine_status` still compared against the original plan count → workflow reported "partial" even when all non-cut images were successfully generated
2. **Information leak**: Exception messages (potentially containing API keys) interpolated directly into the `editorial_summary` state field
3. **Memory exhaustion risk**: Large images (10+ MB) base64-encoded without size checks, causing memory spikes during editorial review
4. **Code duplication**: PNG/JPEG detection logic copy-pasted across 3 files
5. **Type safety**: Wrong type annotation allowed runtime attribute access errors
6. **Schema robustness**: Missing Field descriptions, constraints, and extra="forbid" on editorial output schemas

### Example: Status Determination Bug (P2 Finding #084)

```python
# After editorial review cuts 2 images from a 10-image plan
image_plan = [...10 original locations...]
final_images = [...8 successfully generated images...]
cut_location_ids = ["loc_3", "loc_7"]

# BEFORE FIX: _determine_status compared 8 against 10
status = _determine_status(image_plan, final_images, errors)
# Returns "partial" even though all 8 expected images present

# AFTER FIX: subtract cuts from expected count
status = _determine_status(image_plan, final_images, errors, cut_location_ids)
# Returns "success" because 8 >= (10 - 2)
```

### Example: Information Leak (P2 Finding #093)

```python
# BEFORE FIX: exception details leaked into state
except Exception as e:
    return {
        "editorial_summary": f"Editorial review failed: {e}",  # Could expose API keys
    }

# AFTER FIX: generic message
except Exception as e:
    logger.error(f"Editorial review failed: {e}")
    return {
        "editorial_summary": "Editorial review failed due to an error.",
    }
```

## Root Cause Analysis

The editorial review feature introduced a new filtering step that removes images from the plan after initial generation. This created a disconnect:

1. **Status determination unaware of cuts**: The finalize node's `_determine_status` helper compared `len(final_images)` against `len(image_plan)`, but didn't account for locations removed by editorial review
2. **Missing guardrails**: As a new feature, the implementation hadn't yet been hardened with size limits, proper error handling, and schema constraints
3. **Copy-paste development**: Utility functions (image type detection) were duplicated rather than extracted during initial implementation

The remaining findings were standard code review issues: dead code from removed functionality, missing type constraints, and schema robustness gaps.

## Solution

### Part 1: Fix Status Determination (P2 Finding #084)

Added `cut_location_ids` parameter to `_determine_status` to subtract cuts from expected count.

```python
# workflows/output/illustrate/nodes/finalize.py

def _determine_status(
    image_plan: list[ImageLocationPlan],
    final_images: list[dict],
    errors: list[dict],
    cut_location_ids: list[str] | None = None,
) -> str:
    """Determine workflow completion status.

    Args:
        image_plan: Original planned image locations
        final_images: Successfully generated images
        errors: Generation errors
        cut_location_ids: Location IDs removed by editorial review

    Returns:
        "success" if all expected images present, else "partial"
    """
    # Calculate expected count after editorial cuts
    if cut_location_ids:
        expected_count = len(image_plan) - len(cut_location_ids)
    else:
        expected_count = len(image_plan)

    # Success if we have all expected images
    if len(final_images) >= expected_count:
        return "success"

    # Partial if we have some but not all
    return "partial"
```

**Updated call site** in `finalize_node`:

```python
# workflows/output/illustrate/nodes/finalize.py

async def finalize_node(state: IllustrateState) -> dict:
    """Select winning images and build final markdown."""
    # ... selection logic ...

    # Extract cut location IDs from editorial review
    editorial_results = state.get("editorial_results", [])
    cut_location_ids = []
    for result in editorial_results:
        if "cut_location_ids" in result:
            cut_location_ids.extend(result["cut_location_ids"])

    # Determine status with cut awareness
    status = _determine_status(
        image_plan,
        final_images,
        errors,
        cut_location_ids=cut_location_ids,
    )

    return {
        "workflow_status": status,
        "final_images": final_images,
        # ...
    }
```

**Why This Matters**: Editorial review is an optimization that reduces unnecessary images. Without this fix, workflows that successfully generated all needed images would incorrectly report "partial" completion, causing confusion and triggering unnecessary retry logic.

### Part 2: Extract Shared Utilities (P2 Findings #085, #089)

Created `workflows/output/illustrate/utils.py` with shared utility functions.

**#085 — Image type detection** (duplicated across 3 files):

```python
# workflows/output/illustrate/utils.py

def detect_media_type(data: bytes) -> str:
    """Detect image media type from byte signature.

    Args:
        data: Image bytes

    Returns:
        "image/png" or "image/jpeg"
    """
    # PNG signature: 89 50 4E 47 0D 0A 1A 0A
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"

    # JPEG signature: FF D8 FF
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"

    # Default to JPEG (most generated images)
    return "image/jpeg"
```

**#089 — Selection result filtering** (private function imported cross-module):

```python
# workflows/output/illustrate/utils.py

def select_winning_results(
    generation_results: list[dict],
    selection_results: list[dict],
) -> list[dict]:
    """Select the winning ImageGenResult per location.

    Uses the last selection per location (retry rounds override earlier rounds).
    Falls back to last successful result if selected brief_id not found.

    Args:
        generation_results: All generation attempts across rounds
        selection_results: All selection results across rounds

    Returns:
        List of winning ImageGenResult dicts
    """
    from collections import defaultdict

    # Build lookups
    results_by_brief_id: dict[str, dict] = {}
    results_by_location: dict[str, list[dict]] = defaultdict(list)

    for gen in generation_results:
        if gen["success"] and gen.get("image_bytes"):
            results_by_brief_id[gen["brief_id"]] = gen
            results_by_location[gen["location_id"]].append(gen)

    # Deduplicate selections: keep last per location
    latest_selection: dict[str, dict] = {}
    for selection in selection_results:
        latest_selection[selection["location_id"]] = selection

    # Select winners
    winners: list[dict] = []
    selected_locations: set[str] = set()

    for loc_id, selection in latest_selection.items():
        selected_locations.add(loc_id)

        if selection["quality_tier"] == "failed":
            continue

        brief_id = selection["selected_brief_id"]
        if brief_id and brief_id in results_by_brief_id:
            winners.append(results_by_brief_id[brief_id])
        elif loc_id in results_by_location:
            # Fallback: last successful result
            winners.append(results_by_location[loc_id][-1])

    # Include unselected locations (e.g., from retry bypass)
    for loc_id, results in results_by_location.items():
        if loc_id not in selected_locations:
            winners.append(results[-1])

    return winners
```

**Updated imports** in `finalize.py` and `assemble_document.py`:

```python
# workflows/output/illustrate/nodes/finalize.py
from workflows.output.illustrate.utils import select_winning_results

# workflows/output/illustrate/nodes/assemble_document.py
from workflows.output.illustrate.utils import detect_media_type
```

### Part 3: Add Image Size Limit (P2 Finding #088)

Added 20MB size check before base64 encoding in editorial review.

```python
# workflows/output/illustrate/nodes/editorial_review.py

MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20 MB

async def editorial_review_node(state: IllustrateState) -> dict:
    """Perform vision-based editorial review of all images."""
    # ... setup ...

    # Prepare image context for vision model
    image_contexts = []
    for img in final_images:
        image_bytes = img.get("image_bytes", b"")

        # Skip oversized images to prevent memory exhaustion
        if len(image_bytes) > MAX_IMAGE_SIZE:
            logger.warning(
                "Skipping oversized image '%s' in editorial review: %d bytes (limit: %d)",
                img["location_id"],
                len(image_bytes),
                MAX_IMAGE_SIZE,
            )
            continue

        # Safe to base64 encode
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        # ...
```

**Why 20MB**: Typical generated images are 2-5MB. Public domain images can reach 10MB. 20MB provides headroom while preventing memory exhaustion from anomalous files.

### Part 4: Fix Exception Leak (P2 Finding #093)

Removed exception interpolation from state fields.

```python
# workflows/output/illustrate/nodes/editorial_review.py

async def editorial_review_node(state: IllustrateState) -> dict:
    """Perform vision-based editorial review of all images."""
    try:
        # ... editorial review logic ...

        return {
            "editorial_results": [result],
            "editorial_summary": summary,
        }

    except Exception as e:
        # BEFORE: leaked exception message
        # return {"editorial_summary": f"Editorial review failed: {e}"}

        # AFTER: log details, return generic message
        logger.error(f"Editorial review failed: {e}")
        return {
            "editorial_summary": "Editorial review failed due to an error.",
        }
```

**Security Impact**: Exception messages can contain sensitive data (API keys in auth errors, file paths, internal IPs). Logging captures details for debugging while state contains only safe user-facing messages.

### Part 5: Fix Type Annotation (P2 Finding #090)

Corrected `_compute_cuts_count` parameter type.

```python
# workflows/output/illustrate/nodes/editorial_review.py

# BEFORE: wrong type allowed attribute access errors
def _compute_cuts_count(
    cut_location_ids: list[str],
    opportunities: list[dict],  # ← Should be list[ImageOpportunity]
) -> int:
    """Count cuts by purpose category."""
    # ...
    for opp in opportunities:
        if opp.location_id in cut_set:  # ← Would fail if dict passed
            # ...

# AFTER: correct type
def _compute_cuts_count(
    cut_location_ids: list[str],
    opportunities: list[ImageOpportunity],
) -> int:
    """Count cuts by purpose category."""
    # ... (uses opp.location_id, opp.purpose safely)
```

### Part 6: Deduplicate cut_location_ids (P2 Finding #091)

Added deduplication before count capping.

```python
# workflows/output/illustrate/nodes/editorial_review.py

# BEFORE: LLM could return duplicate location IDs
cut_location_ids = review_result.cut_location_ids[:max_cuts]

# AFTER: deduplicate first, then cap
cut_location_ids = list(dict.fromkeys(review_result.cut_location_ids))[:max_cuts]
```

**Why dict.fromkeys**: Preserves order (unlike `set()`) while removing duplicates. If LLM returns `["A", "B", "A", "C"]`, result is `["A", "B", "C"]`.

### Part 7: Extract Test Helpers (P2 Finding #092)

Moved 5 duplicated factory functions to shared conftest.

```python
# tests/unit/workflows/illustrate/conftest.py

import pytest

@pytest.fixture
def sample_image_plan():
    """Factory for ImageLocationPlan test fixtures."""
    from workflows.output.illustrate.schemas import ImageLocationPlan
    def _factory(location_id="loc_1", **kwargs):
        defaults = {
            "location_id": location_id,
            "position_context": "Introduction paragraph",
            "rationale": "Test rationale",
            "purpose": "clarify-concept",
        }
        return ImageLocationPlan(**{**defaults, **kwargs})
    return _factory

@pytest.fixture
def sample_generation_result():
    """Factory for ImageGenResult test fixtures."""
    def _factory(location_id="loc_1", brief_id="brief_1", **kwargs):
        defaults = {
            "location_id": location_id,
            "brief_id": brief_id,
            "success": True,
            "image_bytes": b"fake_image_data",
            "media_type": "image/png",
        }
        return {**defaults, **kwargs}
    return _factory

@pytest.fixture
def sample_selection_result():
    """Factory for LocationSelection test fixtures."""
    def _factory(location_id="loc_1", **kwargs):
        defaults = {
            "location_id": location_id,
            "quality_tier": "good",
            "selected_brief_id": f"{location_id}_0",
        }
        return {**defaults, **kwargs}
    return _factory

@pytest.fixture
def sample_editorial_result():
    """Factory for EditorialReviewResult test fixtures."""
    def _factory(**kwargs):
        from workflows.output.illustrate.schemas import EditorialReviewResult
        defaults = {
            "overall_assessment": "Good coverage",
            "cut_location_ids": [],
            "keep_all": True,
        }
        return EditorialReviewResult(**{**defaults, **kwargs})
    return _factory

@pytest.fixture
def sample_image_opportunity():
    """Factory for ImageOpportunity test fixtures."""
    def _factory(location_id="loc_1", **kwargs):
        from workflows.output.illustrate.schemas import ImageOpportunity
        defaults = {
            "location_id": location_id,
            "position_context": "Test context",
            "rationale": "Test rationale",
            "purpose": "clarify-concept",
        }
        return ImageOpportunity(**{**defaults, **kwargs})
    return _factory
```

**Impact**: Eliminated 50+ lines of duplicated test setup code across `test_editorial_review.py` and `test_overgeneration.py`.

### Part 8: Remove Dead Code (P2 Finding #094)

Removed `_build_assembled_markdown` function and `assembled_document` state field.

```python
# workflows/output/illustrate/state.py

# BEFORE: 10-20MB string built but never used
class IllustrateState(TypedDict, total=False):
    # ...
    assembled_document: str  # ← Removed

# AFTER: removed field
```

```python
# workflows/output/illustrate/nodes/assemble_document.py

# BEFORE: Built full markdown, stored in state
def _build_assembled_markdown(
    document_markdown: str,
    final_images: list[dict],
) -> str:
    """Build complete markdown with inline images."""
    # ... 50 lines building 10-20MB string ...
    return assembled_markdown

async def assemble_document_node(state: IllustrateState) -> dict:
    assembled = _build_assembled_markdown(document, images)
    return {"assembled_document": assembled}  # Never consumed

# AFTER: function and return field removed entirely
```

**Memory Impact**: For a 50-page document with 20 images, this removes a 15MB string from workflow state that was never consumed by downstream nodes.

### Part 9: Schema Robustness (P3 Findings #087, #095, #096)

**#087 — Added Field descriptions for scoring fields**:

```python
# workflows/output/illustrate/schemas.py

class ImageScoringDetail(BaseModel):
    """Detailed scoring for a single image candidate."""

    # BEFORE: no descriptions on scoring fields
    # visual_appeal: int
    # relevance_to_context: int
    # information_clarity: int
    # narrative_contribution: int

    # AFTER: explicit guidance for LLM
    visual_appeal: int = Field(
        description="Visual quality and aesthetic appeal (1-5, where 5 is excellent)",
        ge=1,
        le=5,
    )
    relevance_to_context: int = Field(
        description="How well the image fits the surrounding text (1-5, where 5 is perfect fit)",
        ge=1,
        le=5,
    )
    information_clarity: int = Field(
        description="Clarity and effectiveness of information presentation (1-5, where 5 is very clear)",
        ge=1,
        le=5,
    )
    narrative_contribution: int = Field(
        description="Contribution to document narrative flow (1-5, where 5 is essential)",
        ge=1,
        le=5,
    )
```

**#095 — Added constraint to contribution_rank**:

```python
# workflows/output/illustrate/schemas.py

class ImageScoringDetail(BaseModel):
    # BEFORE: could be 0 or negative
    contribution_rank: int

    # AFTER: must be >= 1
    contribution_rank: int = Field(
        ge=1,
        description="Rank of this image's contribution (1 = most important)",
    )
```

**#096 — Added extra="forbid" to editorial schemas**:

```python
# workflows/output/illustrate/schemas.py

class EditorialReviewResult(BaseModel):
    """Result of editorial review with cut recommendations."""

    # AFTER: reject extra fields from LLM
    model_config = ConfigDict(extra="forbid")

    overall_assessment: str
    cut_location_ids: list[str]
    # ...

class ImageScoringDetail(BaseModel):
    """Detailed scoring for a single image candidate."""

    model_config = ConfigDict(extra="forbid")

    visual_appeal: int = Field(...)
    # ...
```

**Why This Matters**: Structured output parsing silently ignores extra fields by default. With `extra="forbid"`, the LLM gets validation errors if it returns unexpected fields, prompting it to fix the output format.

### Part 10: Improve Editorial Prompt Context (P3 Finding #097)

Added over-generation context to editorial review prompt.

```python
# workflows/output/illustrate/nodes/editorial_review.py

EDITORIAL_REVIEW_PROMPT = """
You are an expert editor reviewing a fully illustrated academic article.

The document was intentionally illustrated with {overgen_count} extra images
beyond the {base_count} core locations, using an over-generation strategy to
ensure comprehensive visual coverage. Your role is to curate the final set by
identifying any images that are:

1. Redundant (similar concept already shown)
2. Tangential (interesting but not essential to the narrative)
3. Low-impact (doesn't enhance understanding)

Review all {total_count} images in the context of the complete document.
Recommend cuts to reach the optimal set of {base_count} images.
"""
```

**Why This Matters**: Without context about intentional surplus, the LLM might assume all images are meant to stay and be reluctant to cut. Explaining the over-generation strategy helps it understand that cutting is expected and desirable.

### Part 11: Remove Defensive Fallback (P3 Finding #099)

Removed unnecessary hasattr/get pattern in editorial review.

```python
# workflows/output/illustrate/nodes/editorial_review.py

async def editorial_review_node(state: IllustrateState) -> dict:
    """Perform vision-based editorial review."""

    # BEFORE: defensive dict fallback (dead code path)
    if hasattr(state, "visual_identity"):
        identity = state.visual_identity
    else:
        identity = state.get("visual_identity")  # Never executed

    # AFTER: direct attribute access
    visual_identity = state.get("visual_identity", {})
```

**Why Dead Code**: `state` is always a TypedDict with `.get()` method. The `hasattr` check was defensive programming from copy-pasting code that handled multiple state types, but editorial review only receives `IllustrateState`.

## Findings Summary by Priority

### P2 — Correctness & Security (9 findings)

| Finding | Issue | Fix |
|---------|-------|-----|
| #084 | Status ignored editorial cuts | Added `cut_location_ids` param to `_determine_status` |
| #085 | `detect_media_type` duplicated 3x | Extracted to `utils.py` |
| #088 | No image size limit | Added 20MB guard before base64 encoding |
| #089 | Cross-module import of private function | Moved `select_winning_results` to `utils.py` |
| #090 | Wrong type annotation | Changed `list[dict]` to `list[ImageOpportunity]` |
| #091 | Duplicate cut IDs not deduplicated | Added `dict.fromkeys()` dedup |
| #092 | Test helpers duplicated | Extracted to `conftest.py` |
| #093 | Exception leak in state | Generic message, log details |
| #094 | `assembled_document` dead code | Removed function, state field |

### P3 — Schema Robustness & Quality (6 findings)

| Finding | Issue | Fix |
|---------|-------|-----|
| #087 | Missing Field descriptions | Added descriptions to 4 scoring fields |
| #095 | `contribution_rank` missing ge=1 | Added constraint |
| #096 | No extra="forbid" on schemas | Added to 2 editorial schemas |
| #097 | Prompt lacks overgen context | Added "intentionally illustrated with N extra" |
| #099 | Defensive hasattr/get pattern | Direct attribute access |

## Prevention

### Guidelines for Status Determination After Filtering

When workflows include nodes that filter or remove items from the plan:

1. **Thread the filtering through**: Pass filter results (like `cut_location_ids`) to status determination functions
2. **Document the adjustment**: Add comments explaining why expected counts are reduced
3. **Test both paths**: Write tests for status determination with and without filtering applied

### Checklist for Security-Sensitive State Fields

When returning state fields that may be user-visible:

- [ ] Never interpolate exception objects directly (no `f"Error: {e}"`)
- [ ] Log detailed errors with full context for debugging
- [ ] Return generic user-facing messages in state
- [ ] Review all state fields for potential information leaks (paths, IPs, keys)

### DRY Extraction Trigger Points

Extract shared utilities when:

- [ ] Same logic appears in 2+ files (not just copy-paste, but identical purpose)
- [ ] Function is imported across module boundaries (especially if marked private)
- [ ] Logic requires maintenance (like magic byte detection, URL validation)

### Schema Robustness Checklist

For structured LLM output schemas:

- [ ] All numeric fields have `ge=`, `le=` constraints
- [ ] Scoring fields include `Field(description=...)` explaining scale
- [ ] Root model has `model_config = ConfigDict(extra="forbid")`
- [ ] List fields specify expected item types (not generic `list`)

## Files Modified

- `workflows/output/illustrate/nodes/editorial_review.py`: Size limit, exception handling, type fix, prompt context
- `workflows/output/illustrate/nodes/finalize.py`: Status determination, import update
- `workflows/output/illustrate/nodes/assemble_document.py`: Dead code removal, import update
- `workflows/output/illustrate/schemas.py`: Field descriptions, constraints, extra="forbid"
- `workflows/output/illustrate/state.py`: Removed `assembled_document` field
- `workflows/output/illustrate/utils.py`: Created with shared utilities
- `workflows/output/illustrate/graph.py`: Removed dead references
- `tests/unit/workflows/illustrate/conftest.py`: Created with shared fixtures
- `tests/unit/workflows/illustrate/test_editorial_review.py`: Updated to use shared fixtures
- `tests/unit/workflows/illustrate/test_overgeneration.py`: Updated to use shared fixtures

## Related Solutions and Patterns

- [Over-Generation Retry and Selection Logic Bugs](./overgeneration-retry-selection-bugs.md) - Related findings from over-generation implementation
- [Parallel Candidate Vision Selection](../../patterns/llm-interaction/parallel-candidate-vision-selection.md) - Vision comparison pattern used in editorial review
- [Document Illustration Workflow](../../patterns/langgraph/document-illustration-workflow.md) - Overall workflow architecture
- [Over-Generation Pair Selection](../../patterns/langgraph/over-generation-pair-selection.md) - Over-generation strategy that editorial review curates

## Origin

These 15 findings were identified in code review of the editorial review feature and resolved in commit `de777b2` on the `feat/illustrate-editorial-review` branch. The fixes span correctness (P2) and schema robustness (P3) issues.
