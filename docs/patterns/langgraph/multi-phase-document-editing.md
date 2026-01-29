---
name: multi-phase-document-editing
title: "Multi-Phase Document Editing with Optimized Verification"
date: 2026-01-16
category: langgraph
shared: true
gist_url: https://gist.github.com/DaveCBeck/39f7eedb0c4d9ab27add2fc1382bc027
article_path: .context/libs/thala-dev/content/2026-01-16-multi-phase-document-editing-langgraph.md
applicability:
  - "Academic document editing requiring structure improvements"
  - "Documents with citations needing fact-checking and validation"
  - "Multi-stage editing with quality vs speed tradeoffs"
  - "Workflows needing pre-screening to reduce verification cost"
components: [structure_phase, enhancement_phase, verification_phase, polish_phase, pre_screening, citation_cache]
complexity: complex
verified_in_production: true
related_solutions: []
tags: [document-editing, multi-phase, fact-checking, citation-validation, pre-screening, caching, quality-tiers]
---

# Multi-Phase Document Editing with Optimized Verification

## Intent

Improve document quality through four sequential phases (structure, enhancement, verification, polish) with pre-screening and caching optimizations to reduce computational cost by ~50%.

## Motivation

Document editing has different concerns at different stages:
- **Structure**: Organization, flow, redundancy
- **Enhancement**: Evidence integration, depth
- **Verification**: Accuracy, citation validity
- **Polish**: Sentence-level flow, transitions

Processing all sections through all phases is wasteful. This pattern applies pre-screening to skip sections that don't need expensive verification, and caches citation validation to eliminate redundant API calls.

## Applicability

Use this pattern when:
- Documents need structural improvements and content enhancement
- Citations require validation against a reference database (Zotero)
- Quality tiers need to scale from quick drafts to publication-ready
- Verification cost justifies pre-screening optimization

Do NOT use this pattern when:
- Simple formatting edits only
- No citations to validate
- Single-pass editing is sufficient
- Cost constraints prevent multi-phase processing

## Structure

```
Document Input
       │
       ▼
╔══════════════════════════════════════════════════════╗
║  PHASE 1: STRUCTURE (iterative)                      ║
║  ┌─────────┐   ┌──────────┐   ┌─────────────────┐   ║
║  │  Parse  │ → │ Analyze  │ → │ Plan + Execute  │   ║
║  └─────────┘   └──────────┘   └────────┬────────┘   ║
║                                        ▼            ║
║                              ┌───────────────────┐   ║
║               iterate if     │ Verify Structure  │   ║
║               not complete ←─┤ (iteration check) │   ║
║                              └───────────────────┘   ║
╚══════════════════════════════════════════════════════╝
       │
       ▼ (if citations detected)
╔══════════════════════════════════════════════════════╗
║  PHASE 2: ENHANCEMENT (iterative)                    ║
║  ┌───────────────┐   ┌─────────────────────────┐    ║
║  │ Screen Sects  │ → │ Enhance (parallel)      │    ║
║  └───────────────┘   └─────────────────────────┘    ║
║                              │                       ║
║                              ▼                       ║
║                      ┌───────────────────┐          ║
║       iterate if     │ Review Coherence  │          ║
║       not complete ←─┤ (iteration check) │          ║
║                      └───────────────────┘          ║
╚══════════════════════════════════════════════════════╝
       │
       ▼ (if citations detected)
╔══════════════════════════════════════════════════════╗
║  PHASE 3: VERIFICATION (optimized)                   ║
║  ┌──────────────────┐   ┌─────────────────────┐     ║
║  │ Pre-validate     │ → │ Screen for Fact-   │     ║
║  │ Citations (cache)│   │ Check (~50% skip)  │     ║
║  └──────────────────┘   └─────────────────────┘     ║
║                                 │                    ║
║                    ┌────────────┴────────────┐       ║
║                    ▼                         ▼       ║
║           ┌────────────────┐      ┌─────────────┐   ║
║           │ Fact-Check     │      │ Reference-  │   ║
║           │ (parallel)     │      │ Check       │   ║
║           └────────────────┘      └─────────────┘   ║
║                    │                         │       ║
║                    └────────────┬────────────┘       ║
║                                 ▼                    ║
║                    ┌─────────────────────────┐       ║
║                    │ Apply Verified Edits    │       ║
║                    └─────────────────────────┘       ║
╚══════════════════════════════════════════════════════╝
       │
       ▼
╔══════════════════════════════════════════════════════╗
║  PHASE 4: POLISH                                     ║
║  ┌───────────────┐   ┌─────────────────────────┐    ║
║  │ Screen Sects  │ → │ Polish (section-level)  │    ║
║  └───────────────┘   └─────────────────────────┘    ║
╚══════════════════════════════════════════════════════╝
       │
       ▼
   Final Document
```

## Implementation

### Step 1: Define Quality Presets

```python
# quality_presets.py

from typing import TypedDict


class EditingQualitySettings(TypedDict):
    max_structure_iterations: int
    max_enhance_iterations: int
    max_polish_edits: int
    use_opus_for_analysis: bool
    use_opus_for_generation: bool
    use_perplexity_fact_check: bool
    coherence_threshold: float
    fact_check_max_tool_calls: int
    reference_check_max_tool_calls: int
    word_count_tolerance: float


QUALITY_PRESETS: dict[str, EditingQualitySettings] = {
    "test": {
        "max_structure_iterations": 1,
        "max_enhance_iterations": 1,
        "max_polish_edits": 3,
        "use_opus_for_analysis": False,
        "use_opus_for_generation": False,
        "use_perplexity_fact_check": False,
        "coherence_threshold": 0.60,
        "fact_check_max_tool_calls": 5,
        "reference_check_max_tool_calls": 3,
        "word_count_tolerance": 0.25,
    },
    "standard": {
        "max_structure_iterations": 3,
        "max_enhance_iterations": 3,
        "max_polish_edits": 10,
        "use_opus_for_analysis": True,
        "use_opus_for_generation": False,
        "use_perplexity_fact_check": True,
        "coherence_threshold": 0.75,
        "fact_check_max_tool_calls": 15,
        "reference_check_max_tool_calls": 8,
        "word_count_tolerance": 0.20,
    },
    "high_quality": {
        "max_structure_iterations": 5,
        "max_enhance_iterations": 5,
        "max_polish_edits": 20,
        "use_opus_for_analysis": True,
        "use_opus_for_generation": True,
        "use_perplexity_fact_check": True,
        "coherence_threshold": 0.85,
        "fact_check_max_tool_calls": 20,
        "reference_check_max_tool_calls": 12,
        "word_count_tolerance": 0.10,
    },
}
```

### Step 2: Pre-Screening for Verification

```python
# nodes/fact_check.py

async def screen_sections_for_fact_check(state: dict) -> dict[str, Any]:
    """Pre-screen sections to determine which need fact-checking.

    Uses lightweight Haiku model to categorize sections, reducing
    expensive fact-checking by ~50%.
    """
    document_model = state["document_model"]
    leaf_sections = document_model.get_leaf_sections()

    # Build compact summary for screening
    sections_summary_parts = []
    for section in leaf_sections:
        content = document_model.get_section_content(
            section.section_id, include_subsections=False
        )
        preview = content[:150].replace("\n", " ").strip()
        sections_summary_parts.append(
            f"- {section.section_id}: \"{section.heading}\"\n  Preview: {preview}..."
        )

    # Use Haiku for fast, cheap screening
    result: ScreeningResult = await get_structured_output(
        output_schema=ScreeningResult,
        user_prompt=SCREENING_USER.format(sections="\n".join(sections_summary_parts)),
        system_prompt=SCREENING_SYSTEM,
        tier=ModelTier.HAIKU,
    )

    return {
        "screened_sections": result.sections_to_check,
        "screening_skipped": result.sections_to_skip,
        "screening_summary": result.summary,
    }
```

### Step 3: Citation Caching

```python
# nodes/reference_check.py

# Module-level cache
_citation_validation_cache: dict[str, dict] = {}


async def pre_validate_citations(state: dict) -> dict[str, Any]:
    """Pre-validate all unique citations in the document.

    Caches citation existence checks to avoid redundant validation
    when the same citation appears in multiple sections.
    """
    document_model = state["document_model"]

    # Collect all unique citations across all sections
    all_citations = set()
    for section in document_model.get_all_sections():
        content = document_model.get_section_content(section.section_id)
        citations = extract_section_citations(content)
        all_citations.update(citations)

    # Validate each citation once
    validated = {}
    for citation_key in all_citations:
        if citation_key in _citation_validation_cache:
            validated[citation_key] = _citation_validation_cache[citation_key]
            continue

        content = await get_paper_content.ainvoke({"zotero_key": citation_key})
        exists = bool(content and "not found" not in content.lower())
        result = {
            "exists": exists,
            "content_preview": content[:500] if content else "",
        }
        validated[citation_key] = result
        _citation_validation_cache[citation_key] = result

    return {"citation_cache": validated}


def route_to_reference_check_sections(state: dict) -> list[Send] | str:
    """Route only to sections with valid citations."""
    citation_cache = state.get("citation_cache", {})
    sections_with_citations = state.get("sections_with_citations", [])

    sections_to_check = []
    for section_id, citations in sections_with_citations:
        # Filter out citations known to be invalid
        valid_citations = [
            c for c in citations
            if citation_cache.get(c, {}).get("exists", True)
        ]
        if valid_citations:
            sections_to_check.append((section_id, valid_citations))

    if not sections_to_check:
        return "apply_verified_edits"

    return [
        Send("reference_check_section", {
            "section_id": section_id,
            "citations": citations,
        })
        for section_id, citations in sections_to_check
    ]
```

### Step 4: State with Phase Tracking

```python
# state.py

from operator import add
from typing import Annotated, Optional
from typing_extensions import TypedDict


class EditingState(TypedDict, total=False):
    # Input
    input: dict
    document_model: Any

    # Structure phase
    structure_iteration: int
    max_structure_iterations: int
    completed_edits: Annotated[list[dict], add]

    # Enhancement phase (conditional)
    has_citations: bool
    enhance_iteration: int
    max_enhance_iterations: int
    section_enhancements: Annotated[list[dict], add]

    # Verification phase (conditional, optimized)
    screened_sections: list[str]
    screening_skipped: list[str]
    citation_cache: dict[str, dict]
    fact_check_results: Annotated[list[dict], add]
    pending_edits: Annotated[list[dict], add]

    # Polish phase
    polish_results: Annotated[list[dict], add]

    # Output
    final_document: Optional[str]
    status: Optional[str]
    errors: Annotated[list[dict], add]
```

### Step 5: Conditional Phase Routing

```python
# graph/construction.py

def route_after_structure(state: dict) -> str:
    """Route to enhancement or polish based on citation presence."""
    if state.get("has_citations", False):
        return "screen_for_enhancement"
    return "screen_for_polish"


def route_after_verification(state: dict) -> str:
    """Route based on verification results."""
    pending_edits = state.get("pending_edits", [])
    if pending_edits:
        return "apply_verified_edits"
    return "screen_for_polish"


def create_editing_graph() -> StateGraph:
    builder = StateGraph(EditingState)

    # Phase 1: Structure
    builder.add_edge(START, "parse_document")
    builder.add_edge("parse_document", "analyze_structure")
    builder.add_edge("analyze_structure", "plan_edits")
    builder.add_conditional_edges("plan_edits", route_to_edit_workers, [...])
    builder.add_edge("assemble_edits", "verify_structure")
    builder.add_conditional_edges("verify_structure", check_structure_complete,
        ["analyze_structure", "detect_citations"])

    # Phase 2: Enhancement (conditional)
    builder.add_conditional_edges("detect_citations", route_after_structure,
        ["screen_for_enhancement", "screen_for_polish"])
    builder.add_edge("screen_for_enhancement", "enhance_router")
    builder.add_conditional_edges("enhance_router", route_to_enhance_sections, [...])
    builder.add_edge("review_coherence", check_enhancement_complete,
        ["screen_for_enhancement", "pre_validate_citations"])

    # Phase 3: Verification (conditional, optimized)
    builder.add_edge("pre_validate_citations", "screen_for_fact_check")
    builder.add_conditional_edges("screen_for_fact_check", route_to_fact_check, [...])
    builder.add_edge("fact_check_collector", "reference_check_router")
    builder.add_conditional_edges("reference_check_router", route_to_reference_check, [...])
    builder.add_edge("apply_verified_edits", "screen_for_polish")

    # Phase 4: Polish
    builder.add_conditional_edges("screen_for_polish", route_to_polish, [...])
    builder.add_edge("polish_collector", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()
```

## Consequences

### Benefits

- **Cost reduction**: Pre-screening reduces verification by ~50%
- **Cache efficiency**: Citation caching eliminates redundant API calls
- **Quality flexibility**: Five quality tiers from quick drafts to publication
- **Conditional phases**: Skip enhancement/verification for citation-free documents
- **Parallel execution**: Workers run concurrently within each phase

### Trade-offs

- **Complexity**: Four phases with conditional routing adds complexity
- **State size**: Caching increases state memory usage
- **Screening errors**: Pre-screening may occasionally skip sections that need checking
- **Phase ordering**: Edits in later phases may conflict with earlier changes

### Alternatives

- **Single-pass editing**: Faster but lower quality
- **Human-in-the-loop**: Manual verification instead of automated fact-checking
- **External verification service**: Dedicated fact-checking API

## Related Patterns

- [Section Rewriting and Citation Validation](./section-rewriting-citation-validation.md) - Section-level editing approach
- [Multi-Loop Supervision System](./multi-loop-supervision-system.md) - Supervision loop architecture
- [Phased Pipeline Architecture](../data-pipeline/phased-pipeline-architecture-gpu-queue.md) - Phase-based concurrency

## Known Uses in Thala

- `workflows/enhance/editing/graph/construction.py` - Graph construction
- `workflows/enhance/editing/nodes/fact_check.py` - Pre-screening implementation
- `workflows/enhance/editing/nodes/reference_check.py` - Citation caching
- `workflows/enhance/editing/quality_presets.py` - Quality tier configuration

## References

- [LangGraph Conditional Edges](https://langchain-ai.github.io/langgraph/concepts/low_level/#conditional-edges)
- [Pydantic for Structured Output](https://docs.pydantic.dev/)
