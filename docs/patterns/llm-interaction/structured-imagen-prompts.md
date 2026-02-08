---
name: structured-imagen-prompts
title: Structured Imagen Prompts via Pydantic Schema
date: 2026-02-08
category: llm-interaction
applicability:
  - "AI image generation where prompt structure significantly affects output quality"
  - "Pipelines converting natural language descriptions to API-optimized prompts"
  - "Any LLM-to-API bridge where the target API has specific prompt best practices"
components: [llm_call, structured_output, pydantic]
complexity: simple
verified_in_production: true
tags: [imagen, prompt-engineering, pydantic, structured-output, image-generation, google]
---

# Structured Imagen Prompts via Pydantic Schema

## Intent

Convert unstructured image briefs into Imagen-optimized prompts by decomposing them into structured components via LLM + Pydantic schema, then reassembling with must-have elements front-loaded.

## Problem

1. **Unstructured briefs underperform**: Free-form text prompts to Imagen produce inconsistent results
2. **Element ordering matters**: Imagen pays most attention to the beginning of the prompt; important elements buried mid-sentence get de-prioritized
3. **Missing components**: Briefs often omit composition, lighting, or style — leading to generic outputs
4. **Negative instructions backfire**: "No text" in Imagen prompts often *adds* text

## Solution

A two-step pipeline:

1. **Decompose**: LLM converts the unstructured brief into a Pydantic schema with named fields
2. **Reassemble**: Build the prompt string with must-have elements first

```python
# workflows/shared/imagen_prompts.py

class ImagenPromptStructure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_subject: str = Field(
        description="The single most important element. Be very specific.",
    )
    composition: str = Field(
        description="Camera angle, framing. E.g., 'close-up', 'aerial view'",
    )
    key_elements: list[str] = Field(
        max_length=4,
        description="2-3 additional required elements, in priority order.",
    )
    style_and_mood: str = Field(
        description="Visual style, lighting, color temperature.",
    )
    context_setting: str = Field(
        description="Background and environment.",
    )
```

### Prompt Assembly (Front-Loading)

```python
def build_imagen_prompt(structure: ImagenPromptStructure) -> str:
    """Build prompt with must-have elements front-loaded."""
    parts = [
        structure.primary_subject,       # Most important — goes first
        ", ".join(structure.key_elements),  # Required elements
        structure.composition,            # Framing/angle
        structure.context_setting,        # Background
        structure.style_and_mood,         # Style last (least priority)
    ]
    return ", ".join(p for p in parts if p)
```

### Full Pipeline

```python
async def structure_brief_for_imagen(brief: str) -> str:
    """Convert unstructured brief → structured components → optimized prompt."""
    try:
        structure = await invoke(
            tier=ModelTier.HAIKU,  # Fast, cheap — this is a simple conversion
            system=STRUCTURE_SYSTEM,
            user=STRUCTURE_USER.format(brief=brief),
            schema=ImagenPromptStructure,
            config=InvokeConfig(max_tokens=500, batch_policy=BatchPolicy.PREFER_SPEED),
        )
        return build_imagen_prompt(structure)
    except Exception:
        return brief  # Graceful fallback to raw brief
```

### Integration with Image Generation

```python
# workflows/shared/image_utils.py

async def generate_article_header(title, content, custom_prompt=None, ...):
    if custom_prompt:
        prompt = await structure_brief_for_imagen(custom_prompt)
    else:
        prompt = await generate_image_prompt(title, content)

    # Generate multiple candidates, select best via vision comparison
    response = await client.aio.models.generate_images(
        model=IMAGEN_MODEL, prompt=prompt,
        config=types.GenerateImagesConfig(number_of_images=sample_count, ...),
    )
```

## Key Design Decisions

### Why Haiku for Structuring

The brief-to-structure conversion is a simple transformation, not creative work. Haiku is:
- Fast enough to not add meaningful latency
- Cheap (this runs for every Imagen generation)
- Accurate for structured extraction tasks

### Why `extra="forbid"` on the Schema

Prevents the LLM from adding unexpected fields. Without this, the LLM might return creative but unhelpful additions that break the assembly logic.

### Front-Loading Order

Per Google's Imagen prompt guide, the model pays most attention to the beginning. The ordering reflects priority:
1. **Primary subject** — what must be in the image
2. **Key elements** — supporting visual elements
3. **Composition** — how it's framed
4. **Context** — where it is
5. **Style** — how it looks (least priority, most flexible)

### Graceful Fallback

If structuring fails (LLM error, schema validation failure), the raw brief is used directly. This ensures image generation never fails due to the structuring step.

## System Prompt Rules

```python
STRUCTURE_SYSTEM = """...
RULES:
- The primary_subject MUST be the single most important visual element
- key_elements should have 2-3 items maximum, ordered by importance
- composition should use photography terminology (close-up, wide shot, etc.)
- style_and_mood should specify lighting, color palette, and visual style
- context_setting describes the background/environment
- Be SPECIFIC and CONCRETE — avoid abstract descriptions
- Use positive framing — describe what to include, not what to avoid"""
```

The "positive framing" rule is critical: Imagen interprets "no text" as "add text." The system prompt instructs the LLM to describe what to include, never what to avoid.

## Known Uses

- `workflows/shared/imagen_prompts.py` — Schema and builder
- `workflows/shared/image_utils.py:generate_article_header()` — Integration point

## Consequences

### Benefits

- **Consistent quality**: Every prompt has all required components
- **Optimal ordering**: Must-haves always front-loaded
- **Cheap**: Haiku call adds negligible cost
- **Transparent**: Structured output is loggable and debuggable
- **Graceful**: Falls back to raw brief on any failure

### Trade-offs

- **Extra LLM call**: One additional Haiku call per generation
- **Schema rigidity**: Fixed fields may not suit all image types
- **Loss of nuance**: Decomposition may lose subtle relationships in the original brief

## Related Patterns

- [Document Illustration Workflow](../langgraph/document-illustration-workflow.md) — Orchestrates image generation
- [Parallel Candidate Vision Selection](./parallel-candidate-vision-selection.md) — Selects best from multiple generated images
- [LLM Factory Pattern](./llm-factory-pattern.md) — `invoke()` used for structured output

## References

- Commit: `feeaa1b` — feat(illustrate): quick wins for image quality
- Google Imagen prompt guide (prompt ordering recommendations)
- Files:
  - `workflows/shared/imagen_prompts.py`
  - `workflows/shared/image_utils.py`
