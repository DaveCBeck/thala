---
name: two-pass-llm-planning
title: Two-Pass LLM Planning with Visual Identity
date: 2026-02-08
category: llm-interaction
applicability:
  - "Multi-image generation requiring visual consistency across outputs"
  - "Planning tasks where a single LLM call conflates strategy with execution"
  - "Workflows that benefit from separating 'what to do' from 'how to do it'"
components: [llm_call, structured_output, langgraph_node]
complexity: moderate
verified_in_production: true
tags: [two-pass, planning, visual-identity, structured-output, prompt-engineering, creative-direction, brief-generation, image-generation, pydantic, separation-of-concerns]
---

# Two-Pass LLM Planning with Visual Identity

## Intent

Split a monolithic LLM planning call into two sequential passes: a strategic pass (creative direction) that establishes constraints and identity, followed by a tactical pass (brief writing) that operates within those constraints. This produces more consistent, higher-quality outputs by preventing a single call from conflating strategy with execution.

## Problem

When a single LLM call must simultaneously:

1. **Analyze** a document's tone, themes, and structure
2. **Decide** a visual style that unifies all outputs
3. **Identify** the best locations for images
4. **Write** detailed generation briefs for each location

The result suffers from several failure modes:

- **Style drift**—Each brief independently invents its own style, producing inconsistent images
- **Cognitive overload**—The model tries to do too much in one call, producing shallow results
- **No cross-location awareness**—Briefs don't account for variety or pacing across the article
- **Wasted context**—The same document is reanalyzed for each decision instead of building on prior analysis

## Solution

Split the planning into two sequential LLM calls with distinct roles:

### Pass 1: Creative Direction (Strategy)

The LLM reads the full document and produces:

- **Visual Identity**—palette, mood, style, lighting, avoid-list
- **Image Opportunity Map**—N+2 candidate locations (more than needed, for selection)
- **Editorial Notes**—tone, pacing, variety guidance

This pass thinks like a magazine art director, not a technical illustrator.

### Pass 2: Brief Planning (Execution)

The LLM reads the document again, plus the visual identity and selected opportunities from Pass 1. It produces:

- **Up to two candidate briefs per location**—genuinely different approaches
- **Visual identity references** in every brief
- **Cross-location variety** enforcement

This pass thinks like a staff writer given a creative brief.

### Data Flow

```
Document + Config
       |
       v
  ┌─────────────────────┐
  │  Creative Direction  │  Pass 1: Sonnet
  │  (art director)      │
  └──────────┬──────────┘
             │
   VisualIdentity + ImageOpportunity[] + editorial_notes
             │
       ┌─────┴─────┐
       │ Selection  │  Code: filter strong > stretch, cap at target_count
       └─────┬─────┘
             │
  ┌──────────┴──────────┐
  │    Plan Briefs       │  Pass 2: Sonnet
  │    (staff writer)    │
  └──────────┬──────────┘
             │
   CandidateBrief[] → ImageLocationPlan[] (backward compat)
             │
       [Fan-out to generation]
```

## Implementation

### Schemas (Pydantic Structured Output)

```python
# workflows/output/illustrate/schemas.py

class VisualIdentity(BaseModel):
    """Consistent visual style across all images in one article."""
    primary_style: str       # e.g., "editorial watercolor illustration"
    color_palette: list[str] # Three to five descriptive names: ["warm amber", "deep teal", "ivory"]
    mood: str                # e.g., "contemplative, intellectual, accessible"
    lighting: str            # e.g., "soft diffused natural light"
    avoid: list[str]         # e.g., ["photorealistic faces", "neon colors"]
    palette_hex: list[str]   # Resolved hex codes for diagram injection

class ImageOpportunity(BaseModel):
    """A candidate location for an image, identified in Pass 1."""
    location_id: str         # "header", "section_1", etc.
    insertion_after_header: str
    purpose: Literal["header", "illustration", "diagram"]
    suggested_type: Literal["generated", "public_domain", "diagram"]
    strength: Literal["strong", "stretch"]  # Selection priority
    rationale: str
    diagram_subtype: DiagramSubtype | None

class CandidateBrief(BaseModel):
    """A single brief for one candidate at a location, from Pass 2."""
    location_id: str
    candidate_index: int     # One or two (up to two per location)
    image_type: Literal["generated", "public_domain", "diagram"]
    brief: str               # Full brief text
    relationship_to_text: Literal["literal", "metaphorical", "explanatory", "evocative"]
    visual_identity_references: str  # How this brief uses the palette/mood/style
```

### Pass 1: Creative Direction Node

```python
# workflows/output/illustrate/nodes/creative_direction.py

async def creative_direction_node(state: IllustrateState) -> dict:
    """Pass 1: visual identity + opportunity map."""
    target_image_count = (1 if config.generate_header_image else 0) + config.additional_image_count
    extra_opportunity_count = target_image_count + 2  # Overgenerate for selection

    result = await invoke(
        tier=ModelTier.SONNET,
        system=CREATIVE_DIRECTION_SYSTEM,
        user=CREATIVE_DIRECTION_USER.format(
            title=title,
            document=document,
            target_image_count=target_image_count,
            extra_opportunity_count=extra_opportunity_count,
            generate_header=config.generate_header_image,
        ),
        schema=CreativeDirectionResult,
    )

    # Resolve descriptive colors to hex for diagram injection
    vi = result.visual_identity
    vi.palette_hex = resolve_palette_hex(vi.color_palette)

    return {
        "extracted_title": result.document_title,
        "visual_identity": vi,
        "image_opportunities": result.image_opportunities,
        "editorial_notes": result.editorial_notes,
    }
```

### Opportunity Selection (Code, Not LLM)

Between passes, deterministic code selects which opportunities to brief:

```python
def _select_opportunities(opportunities, target_count, config):
    """Prefer 'strong' over 'stretch', always include header if configured."""
    selected = []
    header_opps = [o for o in opportunities if o.purpose == "header"]
    non_header = [o for o in opportunities if o.purpose != "header"]

    if config.generate_header_image and header_opps:
        selected.append(header_opps[0])

    remaining = target_count - len(selected)
    strong = [o for o in non_header if o.strength == "strong"]
    stretch = [o for o in non_header if o.strength == "stretch"]

    selected.extend(strong[:remaining])
    remaining = target_count - len(selected)
    if remaining > 0:
        selected.extend(stretch[:remaining])

    return selected
```

### Pass 2: Brief Planning Node

```python
# workflows/output/illustrate/nodes/plan_briefs.py

async def plan_briefs_node(state: IllustrateState) -> dict:
    """Pass 2: candidate briefs with visual identity context."""
    result = await invoke(
        tier=ModelTier.SONNET,
        system=PLAN_BRIEFS_SYSTEM,
        user=PLAN_BRIEFS_USER.format(
            document=document,
            visual_identity_text=build_visual_identity_context(visual_identity),
            opportunities_text=json.dumps([o.model_dump() for o in selected], indent=2),
            editorial_notes=editorial_notes,
        ),
        schema=PlanBriefsResult,
    )

    # Convert to backward-compatible ImageLocationPlan
    image_plan = _briefs_to_image_plan(result.candidate_briefs, selected, config)
    return {"candidate_briefs": result.candidate_briefs, "image_plan": image_plan}
```

### Visual Identity Propagation

The visual identity flows downstream into generation prompts:

```python
def build_visual_identity_context(vi, *, for_imagen=False):
    """Inject visual identity into generation prompts.

    for_imagen=True omits the 'avoid' list because Imagen has no
    negative_prompt parameter and embedding 'avoid X' in positive
    prompts paradoxically causes generation of X.
    """
    if not vi:
        return ""
    base = (
        "\n## Visual Identity (apply to this image)\n"
        f"- Style: {vi.primary_style}\n"
        f"- Color palette: {', '.join(vi.color_palette)}\n"
        f"- Mood: {vi.mood}\n"
        f"- Lighting: {vi.lighting}"
    )
    if not for_imagen:
        base += f"\n- AVOID: {', '.join(vi.avoid)}"
    return base + "\n"
```

### Color Resolution for Diagrams

Descriptive palette names are mapped to hex codes for diagram styling:

```python
COMMON_COLOR_MAP = {
    "warm amber": "#F5A623", "deep teal": "#1A6B6A",
    "ivory": "#FFFFF0", "soft blue": "#6B9BD2",
    # ... More than 25 common colors
}

def resolve_palette_hex(palette: list[str]) -> list[str]:
    """Map descriptive names to hex, fallback to neutral palette."""
    hex_colors = [COMMON_COLOR_MAP.get(c.lower().strip()) for c in palette]
    return [h for h in hex_colors if h] or ["#4A90D9", "#7B68EE", "#2E8B57"]
```

## Key Design Decisions

### Why Two Passes Instead of One?

| Aspect | Single Pass | Two Passes |
|--------|-------------|------------|
| Style consistency | Each brief invents own style | All briefs reference shared identity |
| Location selection | Fixed, no filtering | Overgenerate + filter by strength |
| Cross-location variety | Hope the LLM remembers | Explicit editorial notes |
| Brief quality | Shallow (cognitive overload) | Deep (focused task) |
| Cost | One Sonnet call | Two Sonnet calls (approximately 2x planning cost) |
| Latency | Lower | Higher (sequential) |

### Why Overgenerate Opportunities?

Pass 1 generates `target_count + 2` opportunities. This gives the selection step room to do the following:

- Drop "stretch" locations that would produce weak images
- Prefer "strong" locations with clear visual potential
- Always include header when configured

### Why Separate Selection from LLM?

The opportunity selection between passes is deterministic code, not an LLM call because it is faster and cheaper, provides predictable behavior, is easy to test, and is config-driven (header preference, count limits).

### Imagen Avoid-List Gotcha

Imagen 4 has no `negative_prompt` parameter. Including "avoid X" in the positive prompt paradoxically causes generation of X (the model attends to the subject noun). The `for_imagen=True` flag omits the avoid list for Imagen prompts while keeping it for LLM-consumed prompts (diagrams, briefs).

## Consequences

### Benefits

- **Visual consistency**—All images share palette, mood, style
- **Better location selection**—Overgenerate and filter beats fixed-count planning
- **Higher brief quality**—Each pass has a focused task
- **Downstream propagation**—Visual identity flows into generation, review, and retry
- **Observability**—`candidate_briefs` retained in state for LangSmith tracing

### Trade-Offs

- **Higher planning cost**—Two Sonnet calls instead of one
- **Higher planning latency**—Sequential passes (approximately three to five seconds added)
- **More state fields**—`visual_identity`, `image_opportunities`, `editorial_notes`, `candidate_briefs`
- **Schema complexity**—More Pydantic models to maintain

## Known Uses

- `workflows/output/illustrate/nodes/creative_direction.py`—Pass 1
- `workflows/output/illustrate/nodes/plan_briefs.py`—Pass 2
- `workflows/output/illustrate/prompts.py`—System/user prompts + `build_visual_identity_context()`
- `workflows/output/illustrate/schemas.py`—`VisualIdentity`, `ImageOpportunity`, `CandidateBrief`, `CreativeDirectionResult`, `PlanBriefsResult`
- `workflows/output/illustrate/graph.py`—Graph wiring: `creative_direction → plan_briefs → generation`

## Applicability Beyond Images

This pattern generalizes to any multi-output LLM workflow that needs consistency:

- **Multi-chapter writing**—Pass 1 establishes voice, themes, narrative arc. Pass 2 writes individual chapters.
- **Multi-slide presentations**—Pass 1 defines visual theme and storyline. Pass 2 designs individual slides.
- **Multi-email campaigns**—Pass 1 defines brand voice and campaign strategy. Pass 2 writes individual emails.

The key insight is that whenever you need N outputs that should feel like they come from the same source, establish the identity first, then execute within it.

## Related Patterns

- [Document Illustration Workflow](../langgraph/document-illustration-workflow.md)—The LangGraph workflow that uses this pattern
- [Structured Imagen Prompts](./structured-imagen-prompts.md)—Brief-to-prompt conversion downstream
- [Parallel Candidate Vision Selection](./parallel-candidate-vision-selection.md)—Multi-candidate generation downstream
- [Central LLM Broker Routing](./central-llm-broker-routing.md)—Model tier routing used by both passes

## References

- Commit: `cc870ae`—feat(illustrate): two-pass planning with visual identity
- Commit: `e7c0d34`—fix(illustrate): resolve nine code review findings
- Files:
  - `workflows/output/illustrate/nodes/creative_direction.py`
  - `workflows/output/illustrate/nodes/plan_briefs.py`
  - `workflows/output/illustrate/prompts.py`
  - `workflows/output/illustrate/schemas.py`
  - `workflows/output/illustrate/state.py`
  - `workflows/output/illustrate/graph.py`
