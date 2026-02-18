---
name: data-grounded-methodology-transparency
title: "Data-Grounded Methodology Transparency"
date: 2026-02-18
category: llm-interaction
applicability:
  - "When LLM-generated methodology sections must report real pipeline metrics without fabrication"
  - "When an automated workflow produces measurable outputs (counts, thresholds, stages) that need honest documentation"
  - "When editorial or post-processing changes should be transparently disclosed in the final output"
components: [llm_call, transparency_report, template_prompting, editorial_summary]
complexity: moderate
verified_in_production: true
related_solutions:
  - "editorial-review-code-findings"
related_patterns:
  - "validate-repair-render-loop"
  - "quality-tier-word-count-parameterization"
  - "citation-network-academic-review-workflow"
tags: [transparency, methodology, anti-hallucination, data-grounding, literature-review, structured-prompts, editorial-disclosure]
---

# Data-Grounded Methodology Transparency

## Intent

Prevent LLM fabrication in methodology sections by collecting real pipeline metrics into a structured report, injecting them into constrained prompts, and disclosing significant post-processing changes — ensuring every number in the final output traces back to actual workflow data.

## Problem

When LLMs write methodology sections, they routinely fabricate:

1. **False database claims**: Reporting searches across "Web of Science, Scopus, and PubMed" when only OpenAlex was used
2. **Invented counts**: Claiming "1,247 records were screened" when the actual number was 83
3. **Fabricated processes**: Describing Boolean search strategies, PRISMA screening stages, or inter-rater reliability checks that never happened
4. **Round-number hallucination**: Converting real metrics into plausible-sounding round numbers (e.g., 47 papers becomes "approximately 50")
5. **Invisible editing**: Post-synthesis editorial enhancement changes citation counts and word lengths without disclosure

These fabrications undermine the core purpose of a methodology section: reproducibility and honesty.

## Solution

A three-layer transparency architecture:

```
┌─ Layer 1: Data Collection ─────────────────────────┐
│  collect_transparency_report(state) → TypedDict     │
│  Pure extraction from workflow state — no LLM calls  │
└─────────────────────────────────────────────────────┘
           │
           ▼
┌─ Layer 2: Prompt Grounding ────────────────────────┐
│  render_transparency_for_prompt(report) → dict      │
│  Pre-render complex data into template-ready strings │
│  + XML-delimited <transparency_data> in prompt       │
│  + Anti-hallucination system prompt constraints       │
└─────────────────────────────────────────────────────┘
           │
           ▼
┌─ Layer 3: Editorial Disclosure ────────────────────┐
│  summarise_editorial_changes(enhancements) → str    │
│  Rule-based significance filtering, no LLM calls    │
│  Appended to methodology after enhancement phase     │
└─────────────────────────────────────────────────────┘
```

### Layer 1: Structured Data Collection

A `TransparencyReport` TypedDict aggregates real metrics from workflow state:

```python
class TransparencyReport(TypedDict, total=False):
    # Discovery
    search_queries: list[str]
    keyword_paper_count: int
    citation_paper_count: int
    raw_results_count: int

    # Diffusion (citation expansion)
    diffusion_stages: list[DiffusionStageReport]
    total_discovered: int
    total_rejected: int
    saturation_reason: str

    # Quality filters applied
    min_citations_filter: int
    recency_years: int
    relevance_threshold: float

    # Processing outcomes
    papers_processed_count: int
    papers_failed_count: int
    metadata_only_count: int
    fallback_substitutions_count: int

    # Clustering
    clustering_method: str
    cluster_count: int

    # Access limitations
    access_limitation_note: str
```

Key design decisions:

- **`total=False`**: All fields optional for backward compatibility with older checkpoints
- **Pure extraction**: `collect_transparency_report()` uses only `.get()` with defaults — no LLM calls, no computation beyond counting
- **Hardcoded access limitations**: Known constraints (e.g., "OpenAlex as sole database") are stated as constants, not left for the LLM to characterise

```python
def collect_transparency_report(state: AcademicLitReviewState) -> TransparencyReport:
    """Aggregate real pipeline metrics from workflow state."""
    keyword_paper_count = len(state.get("keyword_papers", []))
    citation_paper_count = len(state.get("citation_papers", []))

    diffusion_stages = []
    for stage in state.get("diffusion", {}).get("stages", []):
        diffusion_stages.append(DiffusionStageReport(
            stage_number=stage.get("stage_number", 0),
            forward_found=stage.get("forward_papers_found", 0),
            new_relevant=len(stage.get("new_relevant", [])),
            # ... real counts, not estimates
        ))

    return TransparencyReport(
        keyword_paper_count=keyword_paper_count,
        diffusion_stages=diffusion_stages,
        access_limitation_note=_ACCESS_LIMITATION_NOTE,  # hardcoded constant
        # ...
    )
```

### Layer 2: Constrained Prompt Templates

The system prompt enforces anti-fabrication rules:

```python
def get_methodology_system_prompt(target_words):
    return f"""You are documenting the methodology for a systematic literature review.

Write a methodology section using ONLY the factual data provided in the user message.

STRICT CONSTRAINTS:
- Never mention databases that are not listed in the data
- Never invent Boolean search queries or screening processes not described
- Never claim PRISMA compliance unless explicitly stated
- Every number you write must come directly from the provided data
- If a pipeline stage is missing from the data, omit it — do not fabricate

Target length: {word_range} words
Style: Precise, process-honest, AI-neutral academic tone"""
```

The user prompt wraps real data in XML delimiters:

```
<transparency_data>
SOURCE DATABASE: OpenAlex

SEARCH QUERIES USED:
- machine learning clinical trials
- deep learning drug discovery

DISCOVERY:
- Keyword search results: 47 papers (from 312 candidates, filtered by relevance >= 0.6)
- Citation network expansion: 23 papers

PROCESSING OUTCOMES:
- Full-text analysis: 52 papers
- Metadata-only analysis: 18 papers
- Failed retrieval: 3 papers

ACCESS LIMITATIONS:
This review used OpenAlex as its sole discovery database...
</transparency_data>
```

A pre-rendering step converts complex structures into prose fragments before template substitution:

```python
def render_transparency_for_prompt(report: TransparencyReport) -> dict[str, str]:
    """Pre-render complex data into template-ready strings."""
    # Diffusion stages → readable sentences
    # Fallback substitutions → count + reason breakdown
    # Expert papers → conditionally included only if non-zero
    # All values sanitised for template safety
    return {"search_queries_formatted": ..., "keyword_paper_count": str(...), ...}
```

Sanitisation escapes curly braces and XML closing tags in LLM-derived content (e.g., clustering rationale) to prevent template injection:

```python
def _sanitise_for_template(value: str) -> str:
    value = value.replace("{", "{{").replace("}", "}}")
    value = value.replace("</transparency_data>", "&lt;/transparency_data&gt;")
    return value
```

### Layer 3: Editorial Disclosure

After the synthesis phase, an editing workflow may restructure sections, add/remove citations, or change word counts. These changes are disclosed through rule-based significance filtering:

```python
# Significance thresholds
SIGNIFICANT_WORD_DELTA_PCT = 0.20   # 20% length change
LOW_CONFIDENCE_THRESHOLD = 0.7      # Below this = restructured
MIN_CITATION_CHANGES = 1            # Any citation change is significant

def summarise_editorial_changes(section_enhancements: list[dict]) -> str:
    """Summarise significant editorial changes — no LLM calls."""
    # Count citation additions/removals, low-confidence sections, word deltas
    # Return empty string if no significant changes → methodology unchanged
    # Otherwise: "Following initial synthesis, the review underwent automated
    #   editorial enhancement. Significant changes included the addition of
    #   12 citations across 3 sections..."
```

The summary is inserted into the methodology section via regex heading detection — pure string manipulation, no LLM round-trip.

## Consequences

### Benefits

- **Verifiable methodology**: Every number in the final output traces to a `TransparencyReport` field populated from real state
- **No round-number hallucination**: The LLM receives "47 papers" and cannot round to "approximately 50" because the system prompt forbids extrapolation
- **Backward compatibility**: `total=False` TypedDict + `.get()` defaults mean older checkpoint states still work
- **Zero additional LLM cost**: Both the data collection and editorial disclosure layers are pure Python — only the methodology writing itself uses an LLM call
- **Editorial honesty**: Readers know when and how the review was modified after initial synthesis

### Trade-offs

- **Prompt length**: The transparency data block adds ~300-500 tokens to the methodology prompt
- **Maintenance coupling**: New pipeline stages require updating both the state schema and `collect_transparency_report()`
- **Hardcoded limitations**: The access limitation note is a constant string — if the pipeline adds new data sources, this must be updated manually

## Implementation

### Key Files

| File | Role |
|------|------|
| `workflows/research/academic_lit_review/synthesis/types.py` | `TransparencyReport` and `DiffusionStageReport` TypedDicts |
| `workflows/research/academic_lit_review/synthesis/transparency.py` | `collect_transparency_report()` and `render_transparency_for_prompt()` |
| `workflows/research/academic_lit_review/synthesis/nodes/writing/prompts.py` | Anti-hallucination system prompt + `METHODOLOGY_USER_TEMPLATE` |
| `workflows/research/academic_lit_review/synthesis/nodes/writing/drafting.py` | Integration point: renders report into prompt vars |
| `workflows/enhance/editing/transparency.py` | `summarise_editorial_changes()` for editorial disclosure |
| `core/task_queue/workflows/lit_review_full/phases/methodology.py` | `append_editorial_summary()` — inserts disclosure into final report |

### Data Flow

```
AcademicLitReviewState
    │
    ├─► collect_transparency_report(state) → TransparencyReport
    │       │
    │       └─► render_transparency_for_prompt(report) → dict[str, str]
    │               │
    │               └─► METHODOLOGY_USER_TEMPLATE.format(**vars) → prompt
    │                       │
    │                       └─► LLM (Sonnet, anti-hallucination system prompt)
    │                               │
    │                               └─► methodology_draft
    │
    └─► [after enhancement phase]
            │
            └─► summarise_editorial_changes(section_enhancements) → str
                    │
                    └─► append_editorial_summary(final_report, enhance_result)
                            │
                            └─► final report with "### Editorial Process" appended
```

### State Propagation

Transparency data is collected once in the synthesis entry node and propagated through the `SynthesisState`:

```python
# In synthesis graph state_init
transparency_report = collect_transparency_report(parent_state)
return {"transparency_report": transparency_report, ...}
```

This ensures the same real metrics are available to both the methodology writer and the PRISMA documentation generator.

## Known Uses

- **Academic literature review workflow**: Methodology section writing (`synthesis/nodes/writing/drafting.py`)
- **PRISMA documentation**: Search process documentation (`synthesis/nodes/documentation_nodes.py`) uses the same `TransparencyReport` for consistent numbers
- **Editorial enhancement phase**: Post-editing disclosure (`core/task_queue/workflows/lit_review_full/phases/methodology.py`)

## Related Patterns

- **[Quality Tier Word Count Parameterization](../langgraph/quality-tier-word-count-parameterization.md)**: Injects structural parameters (word targets) into prompts — this pattern extends the approach to pipeline metrics
- **[Validate-Repair-Render Loop](validate-repair-render-loop.md)**: Validates LLM output post-generation; this pattern prevents fabrication pre-generation by constraining inputs
- **[Citation Network Academic Review Workflow](../langgraph/citation-network-academic-review-workflow.md)**: The parent workflow that produces the state data consumed by this pattern
