# Workflow Return Field Dependencies

This document lists all code that depends on non-standard workflow return fields.
These need to be updated to use the state store (`load_workflow_state()`) instead.

---

## 1. ACADEMIC_LIT_REVIEW

**Simplified return:** `{final_report, status, langsmith_run_id, errors, source_count, started_at, completed_at}`

### Removed Fields and Dependencies

| Field | Consumer | Usage | Priority |
|-------|----------|-------|----------|
| `paper_corpus` | `testing/test_academic_lit_review.py:71,78,154,430` | Quality metrics: papers_discovered count | HIGH |
| `paper_summaries` | `testing/test_academic_lit_review.py:79,168-170,465` | Quality metrics: papers_processed count | HIGH |
| `clusters` | `testing/test_academic_lit_review.py:88-95,183-198` | Quality metrics: cluster analysis | HIGH |
| `references` | `testing/test_academic_lit_review.py:72,104-105,209-216` | Quality metrics: reference count | HIGH |
| `diffusion` | `testing/test_academic_lit_review.py:98-101,173-180` | Quality metrics: saturation stage | MEDIUM |
| `zotero_keys` | `testing/test_academic_lit_review.py:225,230-231` | Display: Zotero items created | LOW |
| `elasticsearch_ids` | `testing/test_academic_lit_review.py:226,232-233` | Display: ES records created | LOW |
| `prisma_documentation` | `testing/test_academic_lit_review.py:219-222` | Display: PRISMA compliance doc | LOW |
| `citation_keys` | N/A | Not used externally | NONE |
| `quality_metrics` | N/A | Not used externally | NONE |
| `final_review` | N/A | Duplicate of `final_report` | NONE |

---

## 2. SUPERVISED_LIT_REVIEW

**Simplified return:** `{final_report, status, langsmith_run_id, errors, source_count, started_at, completed_at}`

### Removed Fields and Dependencies

| Field | Consumer | Usage | Priority |
|-------|----------|-------|----------|
| `final_review_v2` | `testing/test_supervised_lit_review.py:188-193,257,264-268,541-556` | Display: supervised review output | HIGH (but duplicates `final_report`) |
| `review_loop1-4` | `testing/test_supervised_lit_review.py:180-186,524-538` | Display: intermediate loop outputs | MEDIUM |
| `supervision` | `testing/test_supervised_lit_review.py:85,88-98,283-294` | Quality metrics: loops_run, completion_reason | HIGH |
| `human_review_items` | `testing/test_supervised_lit_review.py:101-114,297-300,559-562` | Quality: items for human review | HIGH |
| `paper_corpus` | `testing/test_supervised_lit_review.py:117-119,303-309` | Quality metrics: papers discovered | HIGH |
| `paper_summaries` | `testing/test_supervised_lit_review.py:133-135,303-309` | Quality metrics: papers processed | HIGH |
| `clusters` | `testing/test_supervised_lit_review.py:148-163,318-328` | Quality metrics: cluster analysis | HIGH |
| `final_review` | N/A | Duplicate of base review | NONE |

---

## 3. DEEP_RESEARCH (web_research)

**Simplified return:** `{final_report, status, langsmith_run_id, errors, source_count, started_at, completed_at}`

### Removed Fields and Dependencies

| Field | Consumer | Usage | Priority |
|-------|----------|-------|----------|
| `research_findings` | `testing/test_research_workflow.py:71,79,199-212` | Quality metrics: findings count, confidence | HIGH |
| `citations` | `testing/test_research_workflow.py:74-75,108-109` | Quality metrics: citation count | HIGH |
| `diffusion` | `testing/test_research_workflow.py:94-100,187-196` | Quality metrics: completeness, iteration | MEDIUM |
| `draft_report` | `testing/test_research_workflow.py:215-223` | Display: intermediate draft | LOW |
| `research_brief` | `testing/test_research_workflow.py:158-172` | Display: research planning | LOW |
| `memory_context` | `testing/test_research_workflow.py:175-178` | Display: memory search results | LOW |
| `memory_findings` | N/A | Internal use only | NONE |
| `translated_report` | `testing/test_research_workflow.py:103-105,258-268` | Output: translated version | MEDIUM |
| `store_record_id` | `testing/test_research_workflow.py:245,250-251` | Display: saved record ID | LOW |

---

## 4. BOOK_FINDING

**Simplified return:** `{final_report, status, langsmith_run_id, errors, source_count, started_at, completed_at}`

### Removed Fields and Dependencies

| Field | Consumer | Usage | Priority |
|-------|----------|-------|----------|
| `processed_books` | `testing/test_book_finding.py:77,82,90,158-175` | Quality metrics: books processed | HIGH |
| `analogous_recommendations` | `testing/test_book_finding.py:67,71,134-143` | Quality metrics: category count | HIGH |
| `inspiring_recommendations` | `testing/test_book_finding.py:68,72,145-149` | Quality metrics: category count | HIGH |
| `expressive_recommendations` | `testing/test_book_finding.py:69,73,151-155` | Quality metrics: category count | HIGH |
| `search_results` | `testing/test_book_finding.py:79,81,86-87,163` | Quality metrics: search count | HIGH |
| `processing_failed` | `testing/test_book_finding.py:78,83,159-182` | Display: failed books | LOW |
| `final_markdown` | N/A | Duplicate of `final_report` | NONE |

---

## 5. MULTI_LANG

**Simplified return:** `{final_report, status, langsmith_run_id, errors, source_count, started_at, completed_at}`

### Removed Fields and Dependencies

| Field | Consumer | Usage | Priority |
|-------|----------|-------|----------|
| `language_results` | `testing/test_multi_lang_academic.py:123-145` | Display: per-language outputs | HIGH |
| `synthesis` | `testing/test_multi_lang_academic.py:161-168,195` | Output (duplicates `final_report`) | NONE |
| `comparative` | `testing/test_multi_lang_academic.py:151-158,194` | Output: cross-language analysis | HIGH |
| `sonnet_analysis` | `testing/test_multi_lang_academic.py:197-202` | Display: analysis insights | MEDIUM |
| `integration_steps` | N/A | Not used externally | NONE |
| `synthesis_record_id` | N/A | Internal tracking | LOW |
| `comparative_record_id` | N/A | Internal tracking | LOW |
| `per_language_record_ids` | N/A | Internal tracking | LOW |

---

## 6. WRAPPED_RESEARCH

**Simplified return:** `{final_report, status, langsmith_run_id, errors, source_count, started_at, completed_at}`

### Removed Fields and Dependencies

| Field | Consumer | Usage | Priority |
|-------|----------|-------|----------|
| `web_result` | `testing/test_wrapped_research.py:108,152,548-556` | Display: web research output | HIGH |
| `academic_result` | `testing/test_wrapped_research.py:109,153,559-567` | Display: academic output | HIGH |
| `book_result` | `testing/test_wrapped_research.py:110,154,570-578` | Display: book output | HIGH |
| `combined_summary` | `testing/test_wrapped_research.py:113-120,176-182,581-589` | Output (duplicates `final_report`) | NONE |
| `top_of_mind_ids` | `testing/test_wrapped_research.py:123-129,186-187` | Display: saved record IDs | MEDIUM |

---

## Migration Strategy

### For Test Files

All test files need to be updated to load detailed state from the state store:

```python
# Before
paper_corpus = result.get("paper_corpus", {})

# After
from workflows.shared.workflow_state_store import load_workflow_state

state = load_workflow_state("academic_lit_review", result["langsmith_run_id"])
paper_corpus = state.get("paper_corpus", {}) if state else {}
```

### Files to Update

1. `testing/test_academic_lit_review.py` - 38 references
2. `testing/test_supervised_lit_review.py` - 48 references
3. `testing/test_research_workflow.py` - 23 references
4. `testing/test_book_finding.py` - 18 references
5. `testing/test_multi_lang_academic.py` - 8 references
6. `testing/test_wrapped_research.py` - 15 references

### Note: Internal Workflow Fields

Some fields are used **internally** within workflow nodes (e.g., `web_result` in wrapped workflow nodes).
These are stored in workflow STATE and are NOT affected by simplifying the RETURN value.
The workflow graph still has access to full state through the state dict passed between nodes.
