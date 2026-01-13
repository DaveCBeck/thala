# Workflow Configuration Guide

> **Updated:** 2026-01-13
> **Status:** Quality tiers standardized across all workflows

## Unified Quality System

All workflows now use a unified 5-tier quality system defined in `workflows/shared/quality_config.py`:

```python
QualityTier = Literal["test", "quick", "standard", "comprehensive", "high_quality"]
```

| Tier | Description | Typical Duration |
|------|-------------|------------------|
| `test` | Minimal processing for testing | ~1 min |
| `quick` | Fast results with limited depth | ~5 min |
| `standard` | Balanced quality and speed (default) | ~15 min |
| `comprehensive` | Thorough processing | ~30 min |
| `high_quality` | Maximum depth and quality | 45+ min |

---

## Workflow Entry Points

### Academic Literature Review

**Location:** `workflows/research/academic_lit_review/graph/api.py`

```python
async def academic_lit_review(
    topic: str,
    research_questions: list[str],
    quality: QualityTier = "standard",
    language: str = "en",
    date_range: Optional[tuple[int, int]] = None,
) -> dict[str, Any]
```

### Web Research (Deep Research)

**Location:** `workflows/research/web_research/graph/api.py`

```python
async def deep_research(
    query: str,
    quality: QualityTier = "standard",
    max_iterations: int = None,
    clarification_responses: dict[str, str] = None,
    language: str = None,
) -> dict
```

### Book Finding

**Location:** `workflows/research/book_finding/graph/api.py`

```python
async def book_finding(
    theme: str,
    brief: Optional[str] = None,
    quality: QualityTier = "standard",
    language: str = "en",
) -> dict[str, Any]
```

### Multi-Language Research

**Location:** `workflows/wrappers/multi_lang/graph/api.py`

```python
async def multi_lang_research(
    topic: str,
    mode: Literal["set_languages", "all_languages"] = "set_languages",
    languages: Optional[list[str]] = None,
    research_questions: Optional[list[str]] = None,
    brief: Optional[str] = None,
    workflows: Optional[dict[str, bool]] = None,
    quality: QualityTier = "standard",
    per_language_quality: Optional[dict[str, dict]] = None,
    extend_to_all_30: bool = False,
) -> MultiLangResult
```

### Supervised Literature Review

**Location:** `workflows/wrappers/supervised_lit_review/api.py`

```python
async def supervised_lit_review(
    topic: str,
    research_questions: list[str],
    quality: QualityTier = "standard",
    language: str = "en",
    date_range: Optional[tuple[int, int]] = None,
    supervision_loops: str = "all",
) -> dict[str, Any]
```

### Document Processing

**Location:** `workflows/document_processing/graph.py`

```python
async def process_document(
    source: str,
    title: str = None,
    item_type: str = "document",
    langs: list[str] = None,
    extra_metadata: dict = None,
    use_batch_api: bool = True,
) -> dict[str, Any]
```

Note: Document processing does not use quality tiers - PDF OCR uses fixed "balanced" quality internally.

---

## Quality Presets by Workflow

Each workflow defines its own quality presets that map to the unified tiers:

### Academic Literature Review

**Location:** `workflows/research/academic_lit_review/quality_presets.py`

| Tier | max_stages | max_papers | target_word_count | min_citations_filter |
|------|------------|------------|-------------------|---------------------|
| test | 1 | 5 | 500 | 0 |
| quick | 2 | 50 | 3,000 | 5 |
| standard | 3 | 100 | 6,000 | 10 |
| comprehensive | 4 | 200 | 10,000 | 10 |
| high_quality | 5 | 300 | 12,500 | 10 |

### Web Research

**Location:** `workflows/research/web_research/graph/config.py`

| Tier | max_iterations | recursion_limit |
|------|----------------|-----------------|
| test | 1 | 25 |
| quick | 2 | 50 |
| standard | 4 | 100 |
| comprehensive | 8 | 200 |
| high_quality | 12 | 300 |

### Book Finding

**Location:** `workflows/research/book_finding/state.py`

| Tier | recommendations_per_category | max_concurrent_downloads | use_opus |
|------|------------------------------|-------------------------|----------|
| test | 1 | 1 | False |
| quick | 2 | 2 | False |
| standard | 3 | 3 | True |
| comprehensive | 5 | 5 | True |
| high_quality | 7 | 7 | True |

---

## Standard Return Structure

All workflows return a standardized structure:

```python
{
    "final_report": str,                                    # Main output document
    "status": Literal["success", "partial", "failed"],      # Execution status
    "langsmith_run_id": str,                                # Tracing ID
    "errors": list[dict],                                   # Accumulated errors
    "source_count": int,                                    # Sources processed
    "started_at": datetime,                                 # Start timestamp
    "completed_at": datetime,                               # End timestamp
}
```

---

## Importing Quality Types

```python
# Import the unified type
from workflows.shared.quality_config import QualityTier

# Or from the wrappers module (re-exports)
from workflows.shared.wrappers.quality import QualityTier, get_quality_tiers
```
