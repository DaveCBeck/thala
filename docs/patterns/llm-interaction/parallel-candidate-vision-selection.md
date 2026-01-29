---
name: parallel-candidate-vision-selection
title: Parallel Candidate Generation with Vision-Based Selection
date: 2026-01-26
category: llm-interaction
applicability:
  - "Generating visual content where quality is subjective and benefits from comparison"
  - "LLM outputs where retry loops produce inconsistent quality improvements"
  - "Tasks where multiple valid outputs can be generated and best selected"
components: [llm_call, async_task, structured_output]
complexity: moderate
verified_in_production: true
tags: [svg, vision, parallel, candidate-selection, asyncio, diagram, quality]
shared: true
gist_url: https://gist.github.com/DaveCBeck/04f092252437959539ad84e2d2e06a48
article_path: .context/libs/thala-dev/content/2026-01-29-parallel-candidate-vision-selection.md
---

# Parallel Candidate Generation with Vision-Based Selection

## Intent

Replace retry-based generation with a parallel "choose-best" flow that generates multiple candidates simultaneously, then uses vision-capable LLMs to compare and select the best one.

## Problem

Traditional retry-based approaches for LLM-generated visual content have several issues:

1. **Sequential bottleneck**: Each retry waits for the previous to fail validation
2. **Inconsistent quality**: Retries don't guarantee improvement; may oscillate
3. **Heuristic selection**: Programmatic checks (overlap detection) miss visual quality
4. **Wasted computation**: Failed attempts provide no benefit to final output

Example of problematic retry pattern:

```python
# Old approach: retry on failure
for attempt in range(max_retries):
    svg = await generate_svg(analysis)
    overlap = check_overlaps(svg)
    if not overlap.has_overlaps:
        return svg
    # Retry with feedback about overlaps
    svg = await regenerate_with_feedback(svg, overlap)
# May still have overlaps after all retries
```

## Solution

Generate N candidates in parallel, then use a vision-capable LLM to visually compare the rendered outputs and select the best one:

1. **Parallel generation**: Use `asyncio.gather()` to generate N candidates simultaneously
2. **Quality metrics collection**: Check overlaps and convert to PNG for each candidate
3. **Vision-based selection**: Send all rendered PNGs to a vision model for comparison
4. **Improvement phase**: Apply minor refinements to the selected candidate
5. **Graceful degradation**: Fall back to heuristic selection if vision fails

## Structure

```
workflows/shared/diagram_utils/
├── __init__.py         # Public exports
├── schemas.py          # DiagramCandidate, DiagramConfig, DiagramResult
├── generation.py       # generate_svg_diagram() - single candidate generation
├── selection.py        # generate_candidates(), select_and_improve()
├── overlap.py          # Programmatic overlap detection
├── conversion.py       # SVG to PNG conversion
└── core.py             # generate_diagram() - main entry point
```

## Implementation

### Data Structures

```python
# workflows/shared/diagram_utils/schemas.py

from dataclasses import dataclass
from pydantic import BaseModel, Field


class DiagramConfig(BaseModel):
    """Configuration for diagram generation."""

    width: int = Field(default=800, description="SVG width in pixels")
    height: int = Field(default=600, description="SVG height in pixels")
    dpi: int = Field(default=150, description="DPI for PNG conversion")
    background_color: str = Field(default="#ffffff", description="Background color")
    num_candidates: int = Field(
        default=3, description="Number of SVG candidates to generate in parallel"
    )


class OverlapCheckResult(BaseModel):
    """Result of text overlap validation."""

    has_overlaps: bool = Field(description="Whether any text elements overlap")
    overlap_pairs: list[tuple[str, str]] = Field(
        default_factory=list, description="Pairs of overlapping text labels"
    )


@dataclass
class DiagramCandidate:
    """A candidate SVG with its quality metrics."""

    svg_content: str
    png_bytes: bytes
    overlap_check: OverlapCheckResult
    candidate_id: int  # 1, 2, or 3


@dataclass
class DiagramResult:
    """Result of diagram generation."""

    svg_bytes: bytes | None
    png_bytes: bytes | None
    analysis: DiagramAnalysis | None
    overlap_check: OverlapCheckResult | None
    generation_attempts: int  # Number of candidates generated
    selected_candidate: int | None = None  # Which candidate was chosen
    selection_rationale: str | None = None  # Why it was chosen
    improvements_made: list[str] | None = None  # What was improved
    success: bool = False
    error: str | None = None
```

### Parallel Candidate Generation

```python
# workflows/shared/diagram_utils/selection.py

import asyncio
import logging

from .conversion import convert_svg_to_png
from .generation import generate_svg_diagram
from .overlap import check_text_overlaps
from .schemas import DiagramAnalysis, DiagramCandidate, DiagramConfig

logger = logging.getLogger(__name__)


async def generate_candidates(
    analysis: DiagramAnalysis,
    config: DiagramConfig,
    num_candidates: int = 3,
) -> list[DiagramCandidate]:
    """Generate N SVG candidates in parallel, check overlaps, convert to PNG.

    Args:
        analysis: DiagramAnalysis from content analysis
        config: Diagram configuration
        num_candidates: Number of candidates to generate (default 3)

    Returns:
        List of DiagramCandidate objects with SVG, PNG, and overlap info
    """
    # Generate SVGs in parallel
    tasks = [generate_svg_diagram(analysis, config) for _ in range(num_candidates)]
    svg_results = await asyncio.gather(*tasks, return_exceptions=True)

    candidates = []
    for i, svg_result in enumerate(svg_results):
        # Handle exceptions gracefully
        if isinstance(svg_result, Exception):
            logger.warning(f"Candidate {i + 1} generation failed: {svg_result}")
            continue

        # Handle None results
        if not svg_result:
            logger.warning(f"Candidate {i + 1} returned None")
            continue

        svg_content = svg_result

        # Check overlaps (programmatic quality check)
        overlap = check_text_overlaps(svg_content)

        # Convert to PNG for visual comparison
        png_bytes = convert_svg_to_png(
            svg_content,
            dpi=config.dpi,
            background_color=config.background_color,
        )

        if not png_bytes:
            logger.warning(f"Candidate {i + 1} PNG conversion failed")
            continue

        candidates.append(
            DiagramCandidate(
                svg_content=svg_content,
                png_bytes=png_bytes,
                overlap_check=overlap,
                candidate_id=i + 1,
            )
        )

    logger.info(f"Generated {len(candidates)} valid candidates out of {num_candidates}")
    return candidates
```

### Vision-Based Selection

```python
# workflows/shared/diagram_utils/selection.py (continued)

import base64
import re

from .prompts import SVG_IMPROVEMENT_SYSTEM, SVG_SELECTION_SYSTEM


async def select_and_improve(
    candidates: list[DiagramCandidate],
    analysis: DiagramAnalysis,
    config: DiagramConfig,
) -> tuple[str, int, str] | None:
    """Use Sonnet with vision to select best candidate and make improvements.

    Two-phase approach:
    1. Show all candidate images + overlap analysis, ask for selection
    2. Provide selected candidate's SVG for improvement

    Args:
        candidates: List of DiagramCandidate objects
        analysis: Original diagram analysis
        config: Diagram configuration

    Returns:
        Tuple of (improved_svg, selected_id, rationale) or None on failure
    """
    from workflows.shared.llm_utils import ModelTier, get_llm

    if not candidates:
        logger.error("No candidates to select from")
        return None

    # If only one candidate, skip selection phase
    if len(candidates) == 1:
        logger.info("Only one candidate, skipping selection")
        return (
            candidates[0].svg_content,
            candidates[0].candidate_id,
            "Only one candidate available",
        )

    try:
        llm = get_llm(tier=ModelTier.SONNET, max_tokens=8000)

        # Build multimodal content with all candidates
        content_parts = []

        for candidate in candidates:
            # Format overlap description for context
            overlap_desc = "No overlaps detected"
            if candidate.overlap_check.has_overlaps:
                pairs = candidate.overlap_check.overlap_pairs[:3]
                overlap_desc = (
                    f"{len(candidate.overlap_check.overlap_pairs)} overlaps: "
                    + "; ".join([f'"{t1}" / "{t2}"' for t1, t2 in pairs])
                )
                if len(candidate.overlap_check.overlap_pairs) > 3:
                    overlap_desc += (
                        f" (and {len(candidate.overlap_check.overlap_pairs) - 3} more)"
                    )

            # Add text describing this candidate
            content_parts.append({
                "type": "text",
                "text": f"**Candidate {candidate.candidate_id}**\n"
                        f"Overlap Analysis: {overlap_desc}",
            })

            # Add the rendered image
            b64_png = base64.b64encode(candidate.png_bytes).decode("utf-8")
            content_parts.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": b64_png,
                },
            })

        # Add selection instruction
        content_parts.append({
            "type": "text",
            "text": "Which candidate is best? State your choice (1, 2, or 3) "
                    "and explain briefly why.",
        })

        # Phase 1: Visual selection
        selection_response = await llm.ainvoke([
            {"role": "system", "content": SVG_SELECTION_SYSTEM},
            {"role": "user", "content": content_parts},
        ])

        selection_text = (
            selection_response.content
            if isinstance(selection_response.content, str)
            else str(selection_response.content)
        ).strip()

        # Extract selected candidate ID from response
        match = re.search(r"[Cc]andidate\s*(\d)", selection_text)
        selected_id = int(match.group(1)) if match else 1

        # Find the selected candidate
        selected_candidate = next(
            (c for c in candidates if c.candidate_id == selected_id),
            candidates[0],  # Fallback to first
        )

        rationale = selection_text

        # Phase 2: Improve the selected candidate's SVG
        improvement_response = await llm.ainvoke([
            {"role": "system", "content": SVG_IMPROVEMENT_SYSTEM},
            {
                "role": "user",
                "content": f"Here is the SVG for candidate {selected_id}. "
                           f"Make minor improvements to spacing and alignment:\n\n"
                           f"{selected_candidate.svg_content}",
            },
        ])

        improved_svg = (
            improvement_response.content
            if isinstance(improvement_response.content, str)
            else str(improvement_response.content)
        ).strip()

        # Clean up response - extract SVG if wrapped in markdown
        if improved_svg.startswith("```"):
            lines = improved_svg.split("\n")
            improved_svg = "\n".join(
                lines[1:-1] if lines[-1] == "```" else lines[1:]
            )

        if not improved_svg.startswith("<svg"):
            svg_start = improved_svg.find("<svg")
            svg_end = improved_svg.rfind("</svg>")
            if svg_start != -1 and svg_end > svg_start:
                improved_svg = improved_svg[svg_start : svg_end + 6]
            else:
                logger.warning("Could not extract improved SVG, using original")
                improved_svg = selected_candidate.svg_content

        logger.info(
            f"Selected candidate {selected_id}, improved SVG ({len(improved_svg)} chars)"
        )
        return (improved_svg, selected_id, rationale)

    except Exception as e:
        logger.error(f"Selection and improvement failed: {e}")
        # Graceful degradation: return candidate with fewest overlaps
        best = min(candidates, key=lambda c: len(c.overlap_check.overlap_pairs))
        return (
            best.svg_content,
            best.candidate_id,
            f"Fallback selection due to error: {e}",
        )
```

### Main Entry Point

```python
# workflows/shared/diagram_utils/core.py

async def generate_diagram(
    title: str,
    content: str,
    config: DiagramConfig | None = None,
) -> DiagramResult:
    """Generate a diagram from content (main entry point).

    Full pipeline:
    1. Analyze content to determine diagram type and elements
    2. Generate multiple SVG candidates in parallel
    3. Check text overlaps and convert to PNG for each candidate
    4. Use vision model to select best candidate and improve it
    5. Convert final improved SVG to PNG
    """
    config = config or DiagramConfig()

    # Stage 1: Analyze content
    analysis = await analyze_content_for_diagram(title, content)
    if not analysis or not analysis.should_generate:
        return DiagramResult(
            svg_bytes=None, png_bytes=None, analysis=analysis,
            overlap_check=None, generation_attempts=0,
            success=False, error="Analysis determined diagram not needed",
        )

    # Stage 2: Generate multiple candidates in parallel
    candidates = await generate_candidates(
        analysis=analysis,
        config=config,
        num_candidates=config.num_candidates,
    )

    if not candidates:
        return DiagramResult(
            svg_bytes=None, png_bytes=None, analysis=analysis,
            overlap_check=None, generation_attempts=config.num_candidates,
            success=False, error="All SVG generation attempts failed",
        )

    # Stage 3: Select best candidate and improve using vision
    selection_result = await select_and_improve(
        candidates=candidates,
        analysis=analysis,
        config=config,
    )

    if not selection_result:
        # Graceful degradation
        best_candidate = min(
            candidates, key=lambda c: len(c.overlap_check.overlap_pairs)
        )
        svg_content = best_candidate.svg_content
        selected_id = best_candidate.candidate_id
        rationale = "Fallback: selected candidate with fewest overlaps"
    else:
        svg_content, selected_id, rationale = selection_result

    # Stage 4: Final conversion
    final_overlap_check = check_text_overlaps(svg_content)
    svg_bytes = svg_content.encode("utf-8")
    png_bytes = convert_svg_to_png(
        svg_content, dpi=config.dpi, background_color=config.background_color
    )

    return DiagramResult(
        svg_bytes=svg_bytes,
        png_bytes=png_bytes,
        analysis=analysis,
        overlap_check=final_overlap_check,
        generation_attempts=len(candidates),
        selected_candidate=selected_id,
        selection_rationale=rationale,
        success=True,
    )
```

### Prompts for Selection

```python
# workflows/shared/diagram_utils/prompts.py

SVG_SELECTION_SYSTEM = """You are reviewing 3 candidate diagrams and will:
1. Select the best one based on: clarity, layout, readability, minimal overlaps
2. Make minor editorial improvements to the selected SVG

## Evaluation Criteria
- **Layout**: Well-balanced use of space, logical flow
- **Readability**: Text is clear and appropriately sized
- **Overlaps**: Fewer overlaps is better (see overlap reports)
- **Aesthetics**: Clean, professional appearance

## Output Format
First, briefly state which candidate you selected (1, 2, or 3) and why (1-2 sentences).
Then output ONLY the improved SVG code starting with <svg and ending with </svg>.
No markdown code fences."""


SVG_IMPROVEMENT_SYSTEM = """You are improving an SVG diagram. Make minor editorial
improvements while preserving the structure.

## Allowed Improvements
- Adjust text positions to fix any remaining overlaps
- Tweak spacing/margins for better balance
- Fix alignment issues
- Minor color adjustments for contrast

## NOT Allowed
- Changing the fundamental structure or content
- Adding or removing major elements
- Changing the diagram type

Output ONLY the improved SVG code starting with <svg and ending with </svg>.
No explanation, no markdown code fences."""
```

## Usage

### Basic Usage

```python
from workflows.shared.diagram_utils import generate_diagram, DiagramConfig

# Generate with defaults (3 candidates)
result = await generate_diagram(
    title="Data Processing Pipeline",
    content="The pipeline has three stages: ingestion, transformation, output...",
)

if result.success:
    # Save the diagram
    with open("diagram.png", "wb") as f:
        f.write(result.png_bytes)
    print(f"Selected candidate {result.selected_candidate}: {result.selection_rationale}")
```

### Custom Configuration

```python
config = DiagramConfig(
    width=1200,
    height=800,
    num_candidates=5,  # Generate more candidates for higher quality
    dpi=200,           # Higher resolution output
    primary_color="#1e40af",
)

result = await generate_diagram(
    title="System Architecture",
    content=architecture_description,
    config=config,
)
```

### Inspecting Selection Details

```python
result = await generate_diagram(title, content)

if result.success:
    print(f"Generated {result.generation_attempts} valid candidates")
    print(f"Selected: Candidate {result.selected_candidate}")
    print(f"Rationale: {result.selection_rationale}")

    if result.improvements_made:
        print(f"Improvements: {', '.join(result.improvements_made)}")

    if result.overlap_check and result.overlap_check.has_overlaps:
        print(f"Warning: {len(result.overlap_check.overlap_pairs)} unresolved overlaps")
```

## Guidelines

### Number of Candidates

| Use Case | Candidates | Rationale |
|----------|------------|-----------|
| Quick preview | 2 | Minimum for comparison |
| Standard generation | 3 | Good balance of quality vs cost |
| High-quality output | 5 | More options for selection |
| Critical diagrams | 7+ | Diminishing returns beyond this |

### Model Selection

| Phase | Model | Rationale |
|-------|-------|-----------|
| Content analysis | HAIKU | Simple classification task |
| SVG generation | SONNET | Needs code generation ability |
| Visual selection | SONNET | Vision capability required |
| Improvement | SONNET | Code modification |

### Graceful Degradation

Always provide fallback behavior:

```python
try:
    # Primary: vision-based selection
    selection_result = await select_and_improve(candidates, analysis, config)
except Exception as e:
    # Fallback: heuristic selection (fewest overlaps)
    best = min(candidates, key=lambda c: len(c.overlap_check.overlap_pairs))
    selection_result = (best.svg_content, best.candidate_id, f"Fallback: {e}")
```

### Observability

Track selection decisions for analysis:

```python
@dataclass
class DiagramResult:
    # ... other fields ...
    selected_candidate: int | None = None    # Which was chosen
    selection_rationale: str | None = None   # Why it was chosen
    improvements_made: list[str] | None = None  # What was improved
```

## Known Uses

- `workflows/shared/diagram_utils/selection.py` - SVG diagram candidate generation and selection
- `workflows/shared/diagram_utils/core.py` - Main diagram generation pipeline

## Consequences

### Benefits

- **Better quality**: Visual comparison produces better results than heuristics alone
- **Faster generation**: Parallel generation vs sequential retries
- **More intelligent selection**: LLM compares visually, considering subjective qualities
- **Observability**: Track which candidate won and why for analysis
- **Graceful degradation**: Falls back to heuristic selection if vision fails
- **Deterministic improvements**: Two-phase approach separates selection from improvement

### Trade-offs

- **Higher cost**: Generating N candidates costs N times the single generation
- **Vision model required**: Selection phase requires vision-capable model (Sonnet)
- **Complexity**: More moving parts than simple retry loop
- **Memory usage**: Holding N PNG images in memory during selection

### Cost Analysis

For 3 candidates:
- Generation: 3x base cost (parallel, not sequential)
- Selection: 1x vision call (includes all images)
- Improvement: 1x text call

Total: ~4x single generation, but produces higher quality output with better success rate.

## Related Patterns

- [Batch API Cost Optimization](./batch-api-cost-optimization.md) - For batching many LLM calls
- [Anthropic Claude Extended Thinking](./anthropic-claude-extended-thinking.md) - For complex reasoning in selection

## Related Solutions

- [SVG Overlap Detection](../../solutions/llm-output/svg-overlap-detection.md) - Programmatic overlap checking
- [LLM Vision Integration](../../solutions/llm-issues/vision-model-integration.md) - Vision model usage

## References

- Commit `6e171c7` - Original implementation: "refactor(diagram-utils): modularize into package with choose-best flow"
- [Anthropic Vision API](https://docs.anthropic.com/en/docs/vision)
- [asyncio.gather documentation](https://docs.python.org/3/library/asyncio-task.html#asyncio.gather)
