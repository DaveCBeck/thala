---
name: language-detection-workflow-integration
title: "Language Detection Integration in Document Workflows"
date: 2026-01-12
category: data-pipeline
applicability:
  - "Document processing workflows handling multi-lingual content"
  - "Academic paper acquisition needing language filtering"
  - "Workflows requiring dual records (original + translation)"
  - "Systems with language-dependent processing overhead"
components: [document_processing, paper_processor, language_detector, diffusion_engine]
complexity: medium
verified_in_production: true
tags: [language-detection, multi-lingual, langdetect, document-processing, paper-acquisition, translation]
---

# Language Detection Integration in Document Workflows

## Intent

Add automatic language detection and verification to document processing workflows, enabling language-aware filtering, dual records for non-English content, and processing overhead adjustments.

## Problem

Multi-lingual document workflows face several challenges:
- **Mislabeled content**: Papers claimed as English may actually be in other languages
- **Processing failures**: LLM extraction prompts fail on non-target-language content
- **Missing translations**: Non-English papers need both original and translated records
- **Resource estimation**: Non-English processing requires more resources due to filtering losses

## Solution

Implement a two-tier language detection system:
1. **Early detection**: Detect language from L0 document content (fast, broad)
2. **Late verification**: Verify language from extracted short_summary (accurate, enriched)

### Architecture

```
Document Processing Workflow:
┌─────────────────────────────────────────────────────────────────┐
│ resolve_input → create_stub → update_store                      │
│                                    │                            │
│                                    ▼                            │
│                          ┌─────────────────┐                    │
│                          │ LANGUAGE DETECT │ ← NEW              │
│                          │ (from L0 text)  │                    │
│                          └────────┬────────┘                    │
│                                   │                             │
│                    ┌──────────────┴──────────────┐              │
│                    ▼                             ▼              │
│           generate_summary              check_metadata          │
│                    │                             │              │
│                    └──────────────┬──────────────┘              │
│                                   ▼                             │
│                          save_short_summary                     │
└─────────────────────────────────────────────────────────────────┘

Paper Processing (Academic Lit Review):
┌─────────────────────────────────────────────────────────────────┐
│ papers_to_process                                               │
│         │                                                       │
│         ▼                                                       │
│ ┌───────────────────┐                                           │
│ │ LANGUAGE VERIFY   │ ← NEW                                     │
│ │ (from short_sum)  │                                           │
│ └─────────┬─────────┘                                           │
│           │                                                     │
│     ┌─────┴─────┐                                               │
│     │           │                                               │
│ ✓ Match    ✗ Mismatch                                           │
│     │           │                                               │
│ continue    reject → language_rejected_dois                     │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation

### Step 1: Shared Language Detection Module

```python
# workflows/shared/language/detection.py

from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Minimum text length for reliable detection
MIN_TEXT_LENGTH = 50

# Normalize regional variants to base language codes
LANGUAGE_CODE_NORMALIZATION = {
    "zh-cn": "zh",
    "zh-tw": "zh",
    "zh-hk": "zh",
    "pt-br": "pt",
    "pt-pt": "pt",
}


def detect_language(
    text: str,
    min_text_length: int = MIN_TEXT_LENGTH,
) -> tuple[Optional[str], float]:
    """Detect language using langdetect library.

    Args:
        text: Text to analyze
        min_text_length: Minimum chars for reliable detection

    Returns:
        (language_code: ISO 639-1, confidence: 0.0-1.0)
    """
    if not text or len(text.strip()) < min_text_length:
        return None, 0.0

    try:
        from langdetect import detect_langs

        results = detect_langs(text)
        if results:
            top_result = results[0]
            lang_code = LANGUAGE_CODE_NORMALIZATION.get(
                top_result.lang, top_result.lang
            )
            return lang_code, top_result.prob
    except Exception as e:
        logger.warning(f"Language detection failed: {e}")

    return None, 0.0


def verify_language_match(
    text: str,
    target_language: str,
    confidence_threshold: float = 0.7,
) -> tuple[bool, Optional[str], float]:
    """Verify text matches expected language.

    Args:
        text: Text to verify
        target_language: Expected ISO 639-1 code (e.g., "en")
        confidence_threshold: Minimum confidence to accept

    Returns:
        (is_match: bool, detected_language: str, confidence: float)
    """
    detected, confidence = detect_language(text)

    if detected is None:
        # Can't detect - assume match to avoid false rejections
        return True, None, 0.0

    is_match = detected == target_language and confidence >= confidence_threshold
    return is_match, detected, confidence
```

### Step 2: Document Processing Node

```python
# workflows/document_processing/nodes/language_detector.py

from typing import Any
from langsmith import traceable

from workflows.shared.language import detect_language
from core.stores import get_store_manager


MAX_DETECTION_SAMPLE = 5000  # First N chars for detection


@traceable(run_type="chain", name="DetectLanguage")
async def detect_document_language(state: dict) -> dict[str, Any]:
    """Detect original language from L0 content.

    Updates L0 store record with language_code field.
    Returns original_language and confidence for downstream use.
    """
    try:
        processing_result = state.get("processing_result", {})
        markdown = processing_result.get("markdown", "")

        if not markdown:
            return {
                "original_language": "en",
                "original_language_confidence": 0.0,
            }

        # Detect from sample (first 5000 chars)
        sample = markdown[:MAX_DETECTION_SAMPLE]
        detected_lang, confidence = detect_language(sample)

        # Persist to L0 record
        store_records = state.get("store_records", {})
        l0_record = store_records.get("L0")
        if detected_lang and l0_record:
            store_manager = get_store_manager()
            await store_manager.es_stores.store.update(
                l0_record["id"],
                {"language_code": detected_lang},
                compression_level=0,
            )

        return {
            "original_language": detected_lang or "en",
            "original_language_confidence": confidence,
        }

    except Exception as e:
        # Graceful fallback - don't block pipeline
        return {
            "original_language": "en",
            "original_language_confidence": 0.0,
            "errors": [{"node": "language_detector", "error": str(e)}],
        }
```

### Step 3: Paper Language Verification

```python
# workflows/academic_lit_review/paper_processor/language_verification.py

import logging
from typing import Any

from workflows.shared.language import verify_language_match

logger = logging.getLogger(__name__)


async def verify_paper_languages(
    papers_to_process: list[str],
    paper_summaries: dict[str, dict],
    target_language: str,
    confidence_threshold: float = 0.7,
) -> dict[str, Any]:
    """Verify papers match target language, rejecting mismatches.

    Args:
        papers_to_process: DOIs to verify
        paper_summaries: Dict of DOI -> summary data (with short_summary)
        target_language: Expected ISO 639-1 code
        confidence_threshold: Minimum confidence to accept

    Returns:
        {
            "verified_papers": [...],      # DOIs that passed
            "rejected_dois": [...],        # DOIs that failed
            "verification_stats": {...},   # Breakdown by detected language
        }
    """
    verified = []
    rejected = []
    stats = {"total": 0, "passed": 0, "rejected": 0, "by_language": {}}

    for doi in papers_to_process:
        stats["total"] += 1
        summary_data = paper_summaries.get(doi, {})
        short_summary = summary_data.get("short_summary", "")

        if not short_summary:
            # No summary to verify - assume OK
            verified.append(doi)
            stats["passed"] += 1
            continue

        is_match, detected, confidence = verify_language_match(
            short_summary, target_language, confidence_threshold
        )

        if is_match:
            verified.append(doi)
            stats["passed"] += 1
        else:
            rejected.append(doi)
            stats["rejected"] += 1
            logger.info(
                f"Rejected {doi}: detected={detected} "
                f"(conf={confidence:.2f}), expected={target_language}"
            )

        # Track by detected language
        if detected:
            stats["by_language"][detected] = stats["by_language"].get(detected, 0) + 1

    return {
        "verified_papers": verified,
        "rejected_dois": rejected,
        "verification_stats": stats,
    }
```

### Step 4: Non-English Overhead Multiplier

```python
# workflows/academic_lit_review/diffusion_engine/api.py

# Non-English papers require more initial papers because
# ~30-50% will be filtered during language verification
NON_ENGLISH_OVERHEAD_MULTIPLIER = 1.5


def calculate_discovery_targets(
    max_papers: int,
    target_language: str,
) -> int:
    """Calculate discovery targets accounting for language filtering."""
    if target_language != "en":
        # Request more papers to compensate for filtering losses
        return int(max_papers * NON_ENGLISH_OVERHEAD_MULTIPLIER)
    return max_papers
```

### Step 5: Dual L1 Records for Non-English

```python
# workflows/document_processing/nodes/save_short_summary.py

async def save_short_summary(state: dict) -> dict:
    """Save short summary, creating dual records for non-English."""
    original_language = state.get("original_language", "en")
    short_summary = state.get("short_summary", "")

    store_manager = get_store_manager()

    # Save L1 record (always in original language)
    l1_record = await store_manager.es_stores.store.create(
        content=short_summary,
        language_code=original_language,
        compression_level=1,
        source_ids=[state["store_records"]["L0"]["id"]],
    )

    result = {"l1_record": l1_record}

    # For non-English: also save English translation as separate L1
    if original_language != "en" and state.get("english_translation"):
        l1_english = await store_manager.es_stores.store.create(
            content=state["english_translation"],
            language_code="en",
            compression_level=1,
            source_ids=[state["store_records"]["L0"]["id"]],
            metadata={"translated_from": original_language},
        )
        result["l1_english_record"] = l1_english

    return result
```

## Configuration

Language detection requires the `langdetect` library:

```
# requirements.txt
langdetect>=1.0.9
```

### Quality Presets with Language Support

```python
# workflows/academic_lit_review/quality_presets.py

QUALITY_PRESETS = {
    "test": {
        "max_papers": 5,
        "language_verification": False,  # Skip for fast tests
    },
    "quick": {
        "max_papers": 50,
        "language_verification": True,
        "language_confidence_threshold": 0.6,  # More lenient
    },
    "standard": {
        "max_papers": 100,
        "language_verification": True,
        "language_confidence_threshold": 0.7,
    },
    "high_quality": {
        "max_papers": 300,
        "language_verification": True,
        "language_confidence_threshold": 0.8,  # More strict
    },
}
```

## Consequences

### Benefits

- **Accurate filtering**: Mislabeled papers caught before expensive LLM extraction
- **Dual records**: Non-English content preserved in original and translated forms
- **Resource efficiency**: Overhead multiplier ensures target paper counts are met
- **Graceful degradation**: Detection failures default to permissive behavior

### Trade-offs

- **Additional dependency**: Requires `langdetect` library
- **Processing overhead**: ~100ms per document for detection
- **False positives**: Short or mixed-language text may be misclassified
- **Storage increase**: Non-English documents create dual L1 records

## Related Patterns

- [Monolithic-to-Modular Refactoring](./monolithic-to-modular-refactoring.md) - Code organization (commit 9b2a513 also applies this)
- [Multi-Language Workflow Orchestration](../langgraph/multi-language-workflow-orchestration.md) - Runtime multi-language execution

## Known Uses

- `workflows/document_processing/nodes/language_detector.py` - L0 detection
- `workflows/academic_lit_review/paper_processor/language_verification.py` - Paper verification
- `workflows/academic_lit_review/diffusion_engine/api.py` - Overhead calculation
- `workflows/shared/language/detection.py` - Core detection utilities

## References

- [langdetect Library](https://pypi.org/project/langdetect/)
- [ISO 639-1 Language Codes](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
