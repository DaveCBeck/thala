# Enhance

A three-phase document enhancement workflow that combines content supervision, structural editing, and fact verification. The workflow deepens theoretical foundations, expands literature coverage, polishes structure and flow, then verifies factual claims and citations.

## Description

The enhance module orchestrates three specialized subworkflows:

1. **Supervision**: Analyzes documents for theoretical gaps and literature coverage, runs targeted literature searches, and integrates new perspectives
2. **Editing**: Improves document structure, generates missing content, removes redundancy, enhances sections with citations, and polishes flow
3. **Fact-Check**: Verifies factual claims, validates citation references, and applies corrections

Together, these workflows transform draft documents into publication-ready academic literature reviews.

## Subworkflows

### [Supervision](./supervision/README.md)
Content enhancement through iterative analysis and literature integration. Runs two loops:
- **Loop 1**: Identifies and fills theoretical gaps with focused literature searches
- **Loop 2**: Expands perspective by discovering adjacent literature bases

### [Editing](./editing/README.md)
Structural editing through multi-phase analysis and enhancement:
- **Structure**: Reorganizes sections, generates intros/conclusions, removes redundancy
- **Enhancement**: Strengthens arguments with evidence from paper corpus
- **Polish**: Improves sentence-level flow and transitions

### [Fact-Check](./fact_check/README.md)
Verification of claims and citations:
- **Fact-Check**: Verifies factual claims using corpus and Perplexity
- **Reference-Check**: Validates citations exist and support their claims
- **Apply Edits**: Applies verified corrections automatically

## Usage

### Full Enhancement Pipeline

```python
from workflows.enhance import enhance_report

result = await enhance_report(
    report=markdown_text,
    topic="Machine Learning in Healthcare",
    research_questions=["How effective is ML for diagnosis?"],
    quality="standard",
    loops="all",
    run_editing=True,
    run_fact_check=True,
)

enhanced_document = result["final_report"]
print(f"Status: {result['status']}")
```

### Supervision Only

```python
result = await enhance_report(
    report=markdown_text,
    topic="Attention mechanisms in transformers",
    research_questions=["How do attention mechanisms improve performance?"],
    quality="standard",
    loops="all",
    run_editing=False,
)
```

### Editing Only

```python
result = await enhance_report(
    report=markdown_text,
    topic="Transformer architectures",
    research_questions=["What are key architectural innovations?"],
    quality="standard",
    loops="none",
    run_editing=True,
)
```

### Loop Selection

```python
# Both supervision loops (default)
result = await enhance_report(..., loops="all")

# Loop 1 only (theoretical depth)
result = await enhance_report(..., loops="one")

# Loop 2 only (literature expansion)
result = await enhance_report(..., loops="two")

# Skip to editing
result = await enhance_report(..., loops="none")
```

### Quality Settings

```python
# Quick - for drafts and testing
result = await enhance_report(..., quality="quick")

# Standard - recommended for most documents
result = await enhance_report(..., quality="standard")

# Comprehensive - thorough literature coverage
result = await enhance_report(..., quality="comprehensive")

# High quality - final publication output
result = await enhance_report(..., quality="high_quality")
```

### With Existing Paper Corpus

```python
result = await enhance_report(
    report=markdown_text,
    topic="Neural architecture search",
    research_questions=["What are efficient NAS methods?"],
    paper_corpus=existing_papers,
    paper_summaries=existing_summaries,
    zotero_keys=existing_keys,
    quality="standard",
)

# Access updated corpus
updated_corpus = result["paper_corpus"]
updated_summaries = result["paper_summaries"]
```

## Return Values

The `enhance_report()` function returns a dictionary containing:

| Field | Type | Description |
|-------|------|-------------|
| `final_report` | str | Enhanced markdown document |
| `status` | str | `"success"`, `"partial"`, or `"failed"` |
| `supervision_result` | dict \| None | Results from supervision phase |
| `editing_result` | dict \| None | Results from editing phase |
| `fact_check_result` | dict \| None | Results from fact-check phase |
| `paper_corpus` | dict | Merged paper corpus (DOI → metadata) |
| `paper_summaries` | dict | Merged paper summaries (DOI → summary) |
| `zotero_keys` | dict | Citation keys for bibliography (DOI → Zotero key) |
| `errors` | list | Any errors encountered during processing |

## Examples

### Basic Enhancement

```python
from workflows.enhance import enhance_report

result = await enhance_report(
    report=open("draft.md").read(),
    topic="Deep learning optimization",
    research_questions=[
        "What are state-of-the-art optimization methods?",
        "How do they compare in practice?",
    ],
    quality="standard",
)

with open("enhanced.md", "w") as f:
    f.write(result["final_report"])
```

### Iterative Enhancement

```python
# First pass: supervision only
result1 = await enhance_report(
    report=draft,
    topic=topic,
    research_questions=questions,
    quality="standard",
    loops="all",
    run_editing=False,
)

# Second pass: editing on supervised content
result2 = await enhance_report(
    report=result1["final_report"],
    topic=topic,
    research_questions=questions,
    quality="high_quality",
    loops="none",
    run_editing=True,
    paper_corpus=result1["paper_corpus"],
    paper_summaries=result1["paper_summaries"],
    zotero_keys=result1["zotero_keys"],
)
```

### Error Handling

```python
result = await enhance_report(
    report=markdown_text,
    topic=topic,
    research_questions=questions,
)

if result["status"] == "success":
    print("Enhancement completed successfully")
elif result["status"] == "partial":
    print(f"Enhancement completed with {len(result['errors'])} errors")
    for error in result["errors"]:
        print(f"  {error['phase']}: {error.get('error')}")
else:
    print("Enhancement failed")
    print(result["errors"])
```

## Configuration

### Maximum Iterations

Control how many iterations each supervision loop can run:

```python
result = await enhance_report(
    ...,
    max_iterations_per_loop=5,  # Default: 3 (increase for deeper exploration)
)
```

### LangSmith Tracing

Pass LangGraph configuration for distributed tracing:

```python
result = await enhance_report(
    ...,
    config={"run_name": "enhance_ml_healthcare", "tags": ["production"]},
)
```

## Quality Tier Comparison

| Setting | Supervision | Editing | Use Case |
|---------|-------------|---------|----------|
| `quick` | 2 diffusion stages, 25 papers* | 2 iterations, Sonnet only | Drafts, testing |
| `standard` | 3 diffusion stages, 50 papers* | 3 iterations, Opus analysis | Most documents |
| `comprehensive` | 4 diffusion stages, 100 papers* | 4 iterations, Opus analysis + generation | Thorough coverage |
| `high_quality` | 5 diffusion stages, 150 papers* | 5 iterations, Opus analysis + generation | Final publication |

*Paper counts reflect the default 0.5x supervision factor. The base quality presets define max_papers (50/100/200/300), but supervision uses half by default.

See subworkflow READMEs for detailed quality settings.
