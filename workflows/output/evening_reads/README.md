# Evening Reads Workflow

Transforms academic literature reviews into a 4-part Substack-style series: 1 overview + 3 deep-dives.

## Overview

Each output article is:
- **Standalone**: Can be read independently
- **Distinct**: No significant content overlap between pieces
- **Sourced**: Preserves `[@KEY]` citations from the input

## Output Structure

| Article | Target Words | Purpose |
|---------|--------------|---------|
| Overview | 2,000-3,000 | Big-picture synthesis, references deep-dives |
| Deep-Dive 1-3 | 2,500-3,500 each | Focused exploration of specific theme |

## Structural Approaches

Each deep-dive is assigned one of three narrative approaches at planning time:

- **puzzle**: Opens with mystery/anomaly, unfolds as investigation
- **finding**: Leads with striking quantitative result, explores implications
- **contrarian**: Steelmans assumption, then complicates with evidence

## Usage

```python
from workflows.output.evening_reads import evening_reads_graph

result = await evening_reads_graph.ainvoke({
    "input": {
        "literature_review": "Your literature review markdown..."
    }
})

# Access final outputs
for output in result["final_outputs"]:
    print(f"{output['id']}: {output['title']} ({output['word_count']} words)")

# Check status
if result["status"] == "success":
    print("All references resolved")
```

## Pipeline

```
extract_citations → plan_content → enrich_sources (parallel)
                                 ↓
                  write_deep_dive (3x parallel via Send)
                                 ↓
              collect_drafts → write_overview → generate_images
                                 ↓
                          format_references → END
```

## Image Generation

Header images are generated for each article using a two-step process:
1. Sonnet generates an optimized image prompt based on article content
2. Google Imagen generates the image from that prompt

## Model Usage

| Node | Model | Tokens |
|------|-------|--------|
| plan_content | OPUS | 4,096 |
| write_deep_dive | OPUS | 14,000 |
| write_overview | OPUS | 12,000 |
| generate_image_prompt | SONNET | 500 |

## Input Requirements

The literature review should:
- Be in markdown format
- Contain `[@KEY]` citations (will be preserved and formatted)
- Have enough depth to support 3 distinct themes
