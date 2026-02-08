---
name: validate-repair-render-loop
title: Validate-Repair-Render Loop for LLM-Generated Code
date: 2026-02-08
category: llm-interaction
applicability:
  - "LLM-generated code that must pass a validator before use (Mermaid, DOT, SVG, SQL, etc.)"
  - "Scenarios where LLM output has predictable syntax errors fixable with targeted feedback"
  - "Pipelines where rendering is expensive and should only happen on validated input"
components: [llm_call, validation, async_task]
complexity: moderate
verified_in_production: true
tags: [mermaid, graphviz, dot, validation, repair, rendering, sanitization, xss-prevention, llm-code-generation]
---

# Validate-Repair-Render Loop for LLM-Generated Code

## Intent

Generate structured code (Mermaid, Graphviz DOT, SVG) via LLM, validate syntax, repair errors with targeted LLM feedback, sanitize for security, and render to a visual output — with bounded retry attempts.

## Problem

LLM-generated code (diagrams, queries, markup) frequently contains syntax errors:

1. **Syntax fragility**: LLMs produce valid-looking but subtly broken code (unclosed subgraphs, unquoted special characters, invalid edge syntax)
2. **Wasted rendering**: Passing broken code to renderers wastes time and may produce silent failures
3. **Security risks**: LLM-generated code can contain injections (XSS in Mermaid via PhantomJS, file-access attributes in Graphviz DOT)
4. **Unbounded retries**: Naive retry loops may never converge

## Solution

A 3-stage loop with bounded attempts:

```
LLM Generate → Sanitize → Validate → [Repair if invalid] → Render
                                ↑           |
                                └───────────┘ (up to 2 repairs)
```

1. **Generate**: LLM produces code from instructions
2. **Sanitize**: Reject dangerous patterns before they reach the renderer
3. **Validate**: Attempt to parse/render as a validation step
4. **Repair**: On failure, feed error messages back to LLM for targeted fix
5. **Render**: Convert validated code to final output (PNG)

## Implementation

### Mermaid Engine

```python
# workflows/shared/diagram_utils/mermaid.py

# PhantomJS XSS mitigation (CVE-2019-17221)
_DANGEROUS_HTML_PATTERN = re.compile(
    r'<\s*(?:script|iframe|object|embed|link|style|img\b[^>]*\bonerror)[^>]*>',
    re.IGNORECASE,
)
_DANGEROUS_ATTR_PATTERN = re.compile(r'\bon\w+\s*=', re.IGNORECASE)

def _sanitize_mermaid_code(code: str) -> str:
    if _DANGEROUS_HTML_PATTERN.search(code) or _DANGEROUS_ATTR_PATTERN.search(code):
        raise ValueError("Mermaid code contains potentially dangerous HTML content")
    return code

async def generate_mermaid_diagram(analysis, config, custom_instructions=""):
    # Step 1: LLM generates Mermaid code
    mermaid_code = await _llm_generate_mermaid(instructions)

    # Step 1.5: Sanitize before it reaches PhantomJS
    mermaid_code = _sanitize_mermaid_code(mermaid_code)

    # Step 2: Validate + repair loop (initial + 2 repair attempts)
    for attempt in range(3):
        is_valid, errors = await asyncio.to_thread(_validate_mermaid, mermaid_code)
        if is_valid:
            break
        if attempt < 2:
            repaired = await _llm_repair_mermaid(mermaid_code, errors)
            if repaired:
                mermaid_code = _sanitize_mermaid_code(repaired)  # Re-sanitize repairs

    # Step 3: Render to PNG
    png_bytes = await _render_mermaid_to_png(mermaid_code, width=config.width)
```

### Graphviz Engine

```python
# workflows/shared/diagram_utils/graphviz_engine.py

# Block file-access attributes
_FORBIDDEN_DOT_ATTRS = re.compile(
    r'\b(image|shapefile|fontpath|imagepath)\s*=', re.IGNORECASE
)

def _sanitize_dot_code(code: str) -> str:
    if _FORBIDDEN_DOT_ATTRS.search(code):
        raise ValueError("DOT code contains forbidden file-access attributes")
    return code

async def generate_graphviz_diagram(analysis, config, custom_instructions=""):
    dot_code = await _llm_generate_dot(instructions)

    for attempt in range(3):
        png_bytes, error = await _render_dot_to_png(dot_code, config)
        if png_bytes:
            return DiagramResult(success=True, png_bytes=png_bytes, ...)
        if attempt < 2:
            repaired = await _llm_repair_dot(dot_code, error)
            if repaired:
                dot_code = repaired
```

### Repair Prompts

The repair LLM call receives the specific error message, enabling targeted fixes:

```python
MERMAID_REPAIR_SYSTEM = """You are fixing a Mermaid diagram that has syntax errors.
Common fixes:
- Add quotes around labels with special characters
- Close unclosed subgraphs with 'end'
- Fix edge syntax (use --> not ->)
- Remove invalid characters from node IDs
Output ONLY the corrected Mermaid code, no explanation."""

# User message includes both the error and the code:
user = f"Fix this Mermaid diagram:\n\nERRORS:\n{errors}\n\nCODE:\n{code}"
```

## Key Design Decisions

### Sanitize Before Validate

Sanitization happens *before* the code reaches any renderer or validator. This prevents:
- **PhantomJS XSS** (CVE-2019-17221): Mermaid supports HTML in node labels; LLM-generated `<script>` tags would execute in PhantomJS
- **Graphviz file access**: DOT `image=` and `shapefile=` attributes can read arbitrary files from disk

### Re-Sanitize After Repair

The repair LLM might introduce new dangerous patterns. Sanitization runs again on repaired code.

### Bounded Attempts (3 total)

- 1 initial generation + 2 repair attempts = 3 maximum
- Prevents unbounded loops on fundamentally broken diagrams
- Each repair attempt gets the specific error message for convergence

### Thread Offloading for Sync Renderers

Both `mmdc` and `graphviz.Source()` are sync operations. They run via `asyncio.to_thread()` to avoid blocking the event loop.

## Guidelines

| Aspect | Recommendation |
|--------|---------------|
| Max repair attempts | 2 (diminishing returns beyond this) |
| Sanitization | Always before rendering AND after repair |
| Error feedback | Include exact error message, not just "fix it" |
| Validation method | Use the renderer itself (cheapest validation) |
| Thread safety | Offload sync renderers to threads |

## Known Uses

- `workflows/shared/diagram_utils/mermaid.py` — Mermaid validate-repair-render
- `workflows/shared/diagram_utils/graphviz_engine.py` — Graphviz validate-repair-render
- `workflows/shared/diagram_utils/validation.py` — SVG validation and entity sanitization

## Consequences

### Benefits

- **Higher success rate**: Most LLM syntax errors are fixable with one repair pass
- **Security**: Dangerous code never reaches renderers
- **Bounded cost**: Maximum 3 LLM calls per diagram (generate + 2 repairs)
- **Specific feedback**: Error messages guide targeted fixes, not random regeneration

### Trade-offs

- **Latency**: Failed validation adds 1-2 LLM round-trips
- **Complexity**: Sanitization rules must be maintained per engine
- **PhantomJS dependency**: mmdc uses abandoned PhantomJS; migration to Playwright-based CLI noted for future

## Related Patterns

- [Parallel Candidate Vision Selection](./parallel-candidate-vision-selection.md) — Uses this loop within multi-candidate generation
- [Diagram Engine Registry and Routing](./diagram-engine-registry-routing.md) — Routes to the engine that uses this loop
- [Document Illustration Workflow](../langgraph/document-illustration-workflow.md) — Orchestrates the full pipeline

## References

- Commit: `b5336d9` — feat(illustrate): diagram engine overhaul
- Commit: `9e43702` — fix(illustrate): resolve 14 code review findings
- CVE-2019-17221 — PhantomJS arbitrary code execution
- Files:
  - `workflows/shared/diagram_utils/mermaid.py`
  - `workflows/shared/diagram_utils/graphviz_engine.py`
  - `workflows/shared/diagram_utils/validation.py`
