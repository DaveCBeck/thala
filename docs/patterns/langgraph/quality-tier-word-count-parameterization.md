---
name: quality-tier-word-count-parameterization
title: "Quality Tier Word Count Parameterization: Section-Proportional Allocation"
date: 2026-01-18
category: langgraph
applicability:
  - "Multi-section document generation with quality tiers"
  - "Workflows where section lengths must sum to a target total"
  - "LLM prompts with word count guidance that varies by quality setting"
  - "Document structures with fixed proportional relationships between sections"
components: [section_proportions, target_calculation, prompt_generators, quality_presets]
complexity: low
verified_in_production: true
related_solutions: []
tags: [quality-tiers, word-count, prompts, parameterization, document-generation, sections]
---

# Quality Tier Word Count Parameterization: Section-Proportional Allocation

## Intent

Calculate section-specific word count targets as proportions of the total quality tier target, ensuring that LLM guidance for individual sections always sums to the overall document target.

## Motivation

When generating multi-section documents with quality tiers (quick, standard, comprehensive), hardcoded word counts in prompts cause conflicts:

**The Problem:**
```
# Quick tier: target_word_count = 3,000

# But individual prompts requested:
Introduction: 800-1000 words   (hardcoded)
Methodology: 600-800 words     (hardcoded)
Section 1: 1200-1800 words     (hardcoded)
Section 2: 1200-1800 words     (hardcoded)
Section 3: 1200-1800 words     (hardcoded)
Section 4: 1200-1800 words     (hardcoded)
Discussion: 1000-1200 words    (hardcoded)
Conclusions: 500-700 words     (hardcoded)
─────────────────────────────────────────
Total: 8,700-12,500 words      ← 3x tier target!
```

**Result:** LLM receives conflicting guidance—tier target says 3,000 words but section prompts request 8,700+.

This pattern solves the problem by making section targets proportional to the tier total.

## Applicability

Use this pattern when:
- Documents have multiple sections with distinct purposes
- Quality tiers define different overall lengths
- LLM prompts include word count guidance
- Section lengths have natural proportional relationships

Do NOT use this pattern when:
- Fixed section lengths are required regardless of tier
- Single-section documents
- No quality tier variation needed

## Structure

```
Quality Preset (e.g., "standard")
       │
       ▼
┌─────────────────────────────────────────────────┐
│  target_word_count: 12,000                      │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  SECTION_PROPORTIONS                            │
│  ┌─────────────────────────────────────────┐   │
│  │ introduction: 8%                         │   │
│  │ methodology: 6%                          │   │
│  │ thematic_total: 70% (÷ theme_count)     │   │
│  │ discussion: 9%                           │   │
│  │ conclusions: 5%                          │   │
│  │ abstract: 2%                             │   │
│  └─────────────────────────────────────────┘   │
│                      = 100%                     │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  get_section_targets(12000, theme_count=4)     │
│  ┌─────────────────────────────────────────┐   │
│  │ introduction: 960 words                  │   │
│  │ methodology: 720 words                   │   │
│  │ thematic_section: 2,100 words (each)    │   │
│  │ discussion: 1,080 words                  │   │
│  │ conclusions: 600 words                   │   │
│  │ abstract: 240 words                      │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│  get_introduction_system_prompt(12000)         │
│  "Target length: 816-1,104 words" (±15%)       │
└─────────────────────────────────────────────────┘
```

## Implementation

### Step 1: Define Section Proportions

```python
# prompts.py

# Section proportions of total word count
# Must sum to 1.0 (100%)
SECTION_PROPORTIONS = {
    "introduction": 0.08,      # 8%
    "methodology": 0.06,       # 6%
    "thematic_total": 0.70,    # 70% (divided by theme count)
    "discussion": 0.09,        # 9%
    "conclusions": 0.05,       # 5%
    "abstract": 0.02,          # 2%
}
# Total: 0.08 + 0.06 + 0.70 + 0.09 + 0.05 + 0.02 = 1.0

# Default total word count if not specified by quality tier
DEFAULT_TARGET_WORDS = 12000
```

### Step 2: Create Target Calculation Function

```python
# prompts.py

def get_section_targets(total_words: int, theme_count: int = 4) -> dict[str, int]:
    """Calculate word targets for each section based on total target.

    Args:
        total_words: Total target word count for the entire document
        theme_count: Number of thematic sections (affects per-theme target)

    Returns:
        Dictionary mapping section names to their word count targets
    """
    theme_count = max(1, theme_count)  # Avoid division by zero
    thematic_per_section = int(
        total_words * SECTION_PROPORTIONS["thematic_total"] / theme_count
    )

    return {
        "introduction": int(total_words * SECTION_PROPORTIONS["introduction"]),
        "methodology": int(total_words * SECTION_PROPORTIONS["methodology"]),
        "thematic_section": thematic_per_section,
        "discussion": int(total_words * SECTION_PROPORTIONS["discussion"]),
        "conclusions": int(total_words * SECTION_PROPORTIONS["conclusions"]),
        "abstract": int(total_words * SECTION_PROPORTIONS["abstract"]),
    }
```

### Step 3: Create Word Range Helper

```python
# prompts.py

def _word_range(target: int, variance: float = 0.15) -> str:
    """Format a word count target as a range (e.g., '850-1,050').

    Adds ±15% flexibility to prevent artificial rigidity while
    keeping targets aligned with the tier total.
    """
    low = int(target * (1 - variance))
    high = int(target * (1 + variance))
    return f"{low:,}-{high:,}"
```

### Step 4: Create Parameterized Prompt Generators

```python
# prompts.py

def get_introduction_system_prompt(target_words: int = DEFAULT_TARGET_WORDS) -> str:
    """Generate introduction system prompt with appropriate word target."""
    word_target = int(target_words * SECTION_PROPORTIONS["introduction"])
    return f"""You are an academic writer drafting the introduction for a systematic literature review.

Write a compelling introduction that:
1. Establishes the importance of the research topic
2. Provides background context
3. States the research questions being addressed
4. Outlines the scope and boundaries of the review
5. Previews the thematic structure

Target length: {_word_range(word_target)} words
Style: Academic, third-person, objective tone

Do NOT include citations in the introduction."""


def get_thematic_section_system_prompt(
    target_words: int = DEFAULT_TARGET_WORDS,
    theme_count: int = 4,
) -> str:
    """Generate thematic section prompt with per-section target.

    Args:
        target_words: Total target for entire document
        theme_count: Number of themes to divide the 70% budget across
    """
    theme_count = max(1, theme_count)
    word_target = int(
        target_words * SECTION_PROPORTIONS["thematic_total"] / theme_count
    )
    return f"""You are writing a thematic section for an academic literature review.

Guidelines:
1. Start with an overview paragraph introducing the theme
2. Synthesize findings across multiple papers (not paper-by-paper)
3. Identify patterns, agreements, and contradictions
4. Use inline citations: [@CITATION_KEY] format

Target length: {_word_range(word_target)} words
Style: Academic, analytical, synthesizing (not just summarizing)"""


# Similarly: get_methodology_system_prompt, get_discussion_system_prompt,
#            get_conclusions_system_prompt
```

### Step 5: Update Quality Presets

```python
# quality_presets.py

from typing import TypedDict


class QualitySettings(TypedDict):
    max_stages: int
    max_papers: int
    target_word_count: int  # The single source of truth
    min_citations_filter: int
    # ... other settings


QUALITY_PRESETS: dict[str, QualitySettings] = {
    "test": QualitySettings(
        max_stages=1,
        max_papers=5,
        target_word_count=2000,  # Sections scale proportionally
        min_citations_filter=0,
        # ...
    ),
    "quick": QualitySettings(
        max_stages=2,
        max_papers=50,
        target_word_count=8000,
        min_citations_filter=5,
        # ...
    ),
    "standard": QualitySettings(
        max_stages=3,
        max_papers=100,
        target_word_count=12000,
        min_citations_filter=10,
        # ...
    ),
    "comprehensive": QualitySettings(
        max_stages=4,
        max_papers=200,
        target_word_count=17500,
        min_citations_filter=10,
        # ...
    ),
    "high_quality": QualitySettings(
        max_stages=5,
        max_papers=300,
        target_word_count=25000,
        min_citations_filter=10,
        # ...
    ),
}
```

### Step 6: Pass Target to Prompt Generators

```python
# nodes/writing/drafting.py

async def write_intro_methodology_node(state: SynthesisState) -> dict[str, Any]:
    """Write introduction and methodology sections."""
    quality_settings = state.get("quality_settings", {})

    # Get total target from quality settings
    target_words = quality_settings.get("target_word_count", DEFAULT_TARGET_WORDS)

    # Generate prompts with proportional targets
    intro_system = get_introduction_system_prompt(target_words)
    method_system = get_methodology_system_prompt(target_words)

    # Use prompts with LLM...
```

```python
# nodes/writing/revision.py

async def write_thematic_sections_node(state: SynthesisState) -> dict[str, Any]:
    """Write thematic sections with theme-count-aware targets."""
    quality_settings = state.get("quality_settings", {})
    clusters = state.get("clusters", [])

    target_words = quality_settings.get("target_word_count", DEFAULT_TARGET_WORDS)
    theme_count = len(clusters)

    # Each theme gets its share of the 70% thematic budget
    thematic_system = get_thematic_section_system_prompt(target_words, theme_count)

    # Use prompt for each theme...
```

## Complete Example

```python
# Calculate targets for standard tier with 5 themes
from prompts import get_section_targets, SECTION_PROPORTIONS

targets = get_section_targets(total_words=12000, theme_count=5)

print(f"Introduction: {targets['introduction']} words")      # 960
print(f"Methodology: {targets['methodology']} words")        # 720
print(f"Per-theme: {targets['thematic_section']} words")     # 1,680
print(f"Discussion: {targets['discussion']} words")          # 1,080
print(f"Conclusions: {targets['conclusions']} words")        # 600
print(f"Abstract: {targets['abstract']} words")              # 240

# Verify total
total = (
    targets['introduction'] +
    targets['methodology'] +
    targets['thematic_section'] * 5 +
    targets['discussion'] +
    targets['conclusions'] +
    targets['abstract']
)
print(f"Total: {total} words")  # 12,000 (matches tier target exactly)
```

### Quality Tier Scaling Example

```python
# Show how sections scale across tiers

for tier, preset in QUALITY_PRESETS.items():
    total = preset["target_word_count"]
    intro = int(total * SECTION_PROPORTIONS["introduction"])
    theme = int(total * SECTION_PROPORTIONS["thematic_total"] / 4)

    print(f"{tier:15} total={total:6} intro={intro:5} per_theme={theme:5}")

# Output:
# test            total= 2000 intro=  160 per_theme=  350
# quick           total= 8000 intro=  640 per_theme= 1400
# standard        total=12000 intro=  960 per_theme= 2100
# comprehensive   total=17500 intro= 1400 per_theme= 3062
# high_quality    total=25000 intro= 2000 per_theme= 4375
```

## Consequences

### Benefits

- **No conflicting guidance**: Section targets always sum to tier total
- **Automatic scaling**: All sections scale proportionally with tier
- **Single source of truth**: `target_word_count` in quality preset controls everything
- **Consistent proportions**: Document structure stays balanced across tiers
- **Flexible ranges**: ±15% variance prevents over-rigid word counts

### Trade-offs

- **Fixed ratios**: Introduction is always 8% regardless of topic
- **Rounding errors**: Integer division may lose a few words
- **Theme dependency**: Per-theme targets vary with theme count

### Alternatives

- **Absolute targets**: Fixed word counts per section (causes conflicts)
- **Per-tier overrides**: Different proportions per quality tier (more complex)
- **No guidance**: Let LLM decide lengths (inconsistent results)

## Related Patterns

- [Unified Quality Tier System](./unified-quality-tier-system.md) - Quality preset architecture
- [Multi-Phase Document Editing](./multi-phase-document-editing.md) - Document generation workflow
- [Synthesis Workflow Orchestration](./synthesis-workflow-orchestration.md) - Uses this pattern

## Known Uses in Thala

- `workflows/research/academic_lit_review/synthesis/nodes/writing/prompts.py` - Section proportion constants and prompt generators
- `workflows/research/academic_lit_review/quality_presets.py` - Academic tier word counts
- `workflows/wrappers/synthesis/quality_presets.py` - Synthesis tier word counts (1.5x academic)
- `workflows/wrappers/synthesis/prompts.py` - Synthesis section allocation

## References

- [Academic Literature Review Structure](https://library.unimelb.edu.au/ace/academic-writing/literature-reviews)
- [PRISMA Guidelines](http://www.prisma-statement.org/)
