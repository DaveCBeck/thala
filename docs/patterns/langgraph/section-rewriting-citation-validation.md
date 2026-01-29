---
name: section-rewriting-citation-validation
title: "Section Rewriting and Citation Validation Pattern"
date: 2026-01-13
category: langgraph
shared: true
gist_url: https://gist.github.com/DaveCBeck/68d5b0ec0053fda0a4f9a1136c745c73
article_path: .context/libs/thala-dev/content/2026-01-13-section-rewriting-citation-validation-langgraph.md
applicability:
  - "Document editing workflows needing structural fixes"
  - "LLM-driven document restructuring where edit operations fail"
  - "Citation-heavy documents requiring strict validation"
  - "Supervision loops with multi-phase refinement"
components: [loop3_structure, loop4_editing, loop5_factcheck, citation_validation, section_rewriter]
complexity: high
verified_in_production: true
tags: [supervision, section-rewriting, citation-validation, zotero, edit-operations, llm-editing]
---

# Section Rewriting and Citation Validation Pattern

## Intent

Replace structured edit operations with direct LLM section rewriting for structural fixes, combined with strict Zotero-based citation validation to ensure all references are traceable.

## Problem

### Edit Operation Failures

Structured edit specifications (find/replace, delete_paragraph, trim_redundancy) consistently failed:
- **Validation failures**: LLMs couldn't reliably specify exact `replacement_text`
- **Retry complexity**: Failed edits led to retry loops with diminishing returns
- **Specification gap**: Difference between "identifying an issue" and "specifying how to fix it" was large

### Citation Integrity Issues

Citations introduced during editing lacked validation:
- **Synthetic keys**: Metadata-only papers used generated keys (not verifiable)
- **No source of truth**: Corpus keys weren't authoritative
- **Silent failures**: Invalid citations passed through undetected

## Solution

### Insight: LLMs Are Better at Rewriting Than Editing

Instead of having LLMs generate structured edit operations, have them directly rewrite affected sections:

```
OLD (Edit Operations):
Phase A: Identify issues
Phase B: Generate StructuralEdit specs → Validate → Retry if fails → Apply

NEW (Section Rewriting):
Phase A: Identify issues (diagnosis only)
Phase B: Rewrite sections directly → Apply rewrites
```

### Architecture

```
Loop 3 (Structure) - Section Rewriting:
┌─────────────────────────────────────────────────────────────────┐
│ number_paragraphs → phase_a_identify_issues                     │
│                              │                                  │
│                     [route_after_phase_a]                       │
│                      /              \                           │
│               issues found      no issues                       │
│                    │                │                           │
│                    ▼                ▼                           │
│         phase_b_rewrite_sections   pass_through                 │
│                    │                │                           │
│                    ▼                │                           │
│            verify_architecture      │                           │
│                    │                │                           │
│            [coherence >= 0.8?]      │                           │
│              /          \           │                           │
│           yes            no         │                           │
│            │         (continue)     │                           │
│            ▼              │         ▼                           │
│         finalize ←────────┴─────────┘                           │
└─────────────────────────────────────────────────────────────────┘

Citation Validation Flow:
┌─────────────────────────────────────────────────────────────────┐
│ edited_section                                                  │
│      │                                                          │
│      ▼                                                          │
│ extract_citations([@KEY] format)                                │
│      │                                                          │
│      ▼                                                          │
│ verify_zotero_citations_batch(verify_all=True)                  │
│      │                                                          │
│   ┌──┴──┐                                                       │
│   │     │                                                       │
│ valid  invalid → strip_invalid_citations() → TODO markers       │
│   │                                                             │
│   ▼                                                             │
│ accept edit                                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation

### Section Rewriting

```python
# workflows/supervised_lit_review/supervision/loops/loop3/section_rewriter.py

async def rewrite_section_for_issue(
    issue: StructuralIssue,
    paragraph_mapping: dict[int, str],
    topic: str,
    zotero_keys: dict[str, str] | None = None,
) -> SectionRewriteResult:
    """Rewrite a section to fix a structural issue.

    Instead of generating edit operations, directly rewrite the section
    with context to maintain coherence.
    """
    # Extract section with surrounding context (3 paragraphs each side)
    context_before, section, context_after, start, end = extract_section_with_context(
        paragraph_mapping,
        issue.affected_paragraphs,
        context_size=3,
    )

    # Build rewriting prompt
    prompt = SECTION_REWRITE_USER.format(
        issue_id=issue.issue_id,
        issue_type=issue.issue_type,
        severity=issue.severity,
        description=issue.description,
        suggested_resolution=issue.suggested_resolution,
        affected_paragraphs=issue.affected_paragraphs,
        context_before=context_before,
        section_content=section,
        context_after=context_after,
    )

    # Use Sonnet for rewriting (good enough, faster/cheaper than Opus)
    llm = get_llm(ModelTier.SONNET, max_tokens=8000)
    rewritten = await llm.ainvoke([
        {"role": "system", "content": SECTION_REWRITE_SYSTEM},
        {"role": "user", "content": prompt},
    ])

    # Generate change summary for audit trail (Haiku - minimal cost)
    summary_llm = get_llm(ModelTier.HAIKU, max_tokens=500)
    changes_summary = await generate_change_summary(
        section, rewritten.content, summary_llm
    )

    return SectionRewriteResult(
        issue_id=issue.issue_id,
        original_paragraphs=list(range(start, end + 1)),
        rewritten_content=rewritten.content,
        changes_summary=changes_summary,
        confidence=0.8,
    )
```

### Issue Types for Rewriting

| Issue Type | Description | Rewriting Approach |
|------------|-------------|-------------------|
| `content_sprawl` | Same topic scattered across 3+ sections | Consolidate into unified section |
| `premature_detail` | Technical content before foundations | Reorder with transitions |
| `orphaned_content` | Disconnected paragraph | Add transitional phrases |
| `redundant_framing` | Multiple introductions | Keep most informative, remove duplicates |
| `redundant_paragraphs` | >60% content overlap | Merge, eliminate repetition |
| `logical_gap` | Missing connecting tissue | Add bridging sentences |
| `consolidate` | (NEW) Gather from 3+ locations | Reorganize into coherent whole |

### Citation Validation with Zotero

```python
# workflows/supervised_lit_review/supervision/utils/citation_validation/validator.py

async def validate_edit_citations_with_zotero(
    original_section: str,
    edited_section: str,
    corpus_keys: set[str],
    zotero_client: ZoteroStore,
    verify_all: bool = True,  # NEW: Zotero as source of truth
) -> tuple[bool, list[str], set[str]]:
    """Validate citations in edited content against Zotero.

    Args:
        verify_all: If True, verify ALL citations against Zotero.
                   Zotero becomes authoritative source of truth.
    """
    edited_citations = extract_citations(edited_section)
    verified_keys = set()
    invalid_citations = []

    if verify_all:
        # Verify ALL citations against Zotero
        citations_to_verify = edited_citations

        if citations_to_verify:
            verification_results = await verify_zotero_citations_batch(
                citations_to_verify, zotero_client
            )

            for key, exists in verification_results.items():
                if exists:
                    verified_keys.add(key)
                elif key not in corpus_keys:
                    # Not in Zotero AND not in corpus - invalid
                    invalid_citations.append(f"{key} (not in Zotero)")
                else:
                    # In corpus but not Zotero - trust corpus as fallback
                    verified_keys.add(key)

    is_valid = len(invalid_citations) == 0
    return is_valid, invalid_citations, verified_keys
```

### Zotero Stubs for Metadata-Only Papers

```python
# workflows/academic_lit_review/paper_processor/nodes.py

async def create_zotero_stubs_for_papers(
    papers: list[PaperMetadata],
) -> dict[str, str]:
    """Create real Zotero records for metadata-only papers.

    Replaces synthetic keys with verifiable Zotero keys.
    """
    store_manager = get_store_manager()
    zotero_keys = {}

    for paper in papers:
        doi = paper.get("doi")
        if not doi:
            continue

        zotero_item = ZoteroItemCreate(
            itemType="journalArticle",
            fields={
                "title": paper.get("title", "Unknown"),
                "DOI": doi,
                "abstractNote": paper.get("abstract", "")[:2000],
                "date": str(paper.get("year", "")),
                "publicationTitle": paper.get("venue", ""),
            },
            tags=[
                ZoteroTag(tag="metadata-only", type=1),
                ZoteroTag(tag="academic-lit-review", type=1),
            ],
            creators=[
                {"creatorType": "author", "name": name}
                for name in paper.get("authors", [])
            ],
        )

        try:
            zotero_key = await store_manager.zotero.add(zotero_item)
            zotero_keys[doi] = zotero_key
        except Exception as e:
            # Fallback only if Zotero creation fails
            fallback = doi.replace("/", "_").replace(".", "")[:20].upper()
            zotero_keys[doi] = fallback
            logger.warning(f"Zotero stub creation failed for {doi}: {e}")

    return zotero_keys
```

### Paper Summaries with Inline Citations

```python
# workflows/academic_lit_review/synthesis/citation_utils.py

def format_papers_with_keys(
    dois: list[str],
    paper_summaries: dict[str, dict],
    zotero_keys: dict[str, str],
) -> str:
    """Format papers with Zotero keys for LLM prompts.

    Raises:
        ValueError: If any paper missing Zotero key (fail-fast).
    """
    formatted = []

    for doi in dois:
        summary = paper_summaries.get(doi)
        if not summary:
            continue

        key = zotero_keys.get(doi)
        if not key:
            raise ValueError(
                f"Paper {doi} has no Zotero key. "
                f"Document processing may have failed."
            )

        # Format with inline citation key
        paper_text = f"""
[@{key}] {summary.get('title')} ({summary.get('year', 'n.d.')})
Authors: {', '.join(a['name'] for a in summary.get('authors', [])[:3])}
Short Summary: {summary.get('short_summary', 'N/A')}
"""
        formatted.append(paper_text)

    return "\n".join(formatted)
```

## Section Rewriting Prompt

```python
SECTION_REWRITE_SYSTEM = """You are an expert academic editor fixing a specific structural issue.

Critical Rules:
1. FIX the specific issue described - nothing more, nothing less
2. PRESERVE all citations in exact [@KEY] format
   - [@Smith2020], [@ABC12345], [@jones_2019_climate]
   - Never convert to (Author, Year) or [1] style
3. MAINTAIN document's voice and style
4. DO NOT add new factual claims - only restructure/consolidate/clarify
5. DO NOT rewrite context paragraphs - only the section between markers
6. KEEP similar length (±30% unless issue requires expansion/reduction)

Output: ONLY the rewritten section content - no explanations, no meta-commentary.
"""
```

## Model Tier Strategy

| Phase | Model | Rationale |
|-------|-------|-----------|
| Phase A (Diagnosis) | Opus | Full analysis, architecture assessment |
| Phase B (Rewriting) | Sonnet | Good enough for text transformation, faster |
| Change Summaries | Haiku | Simple summarization, minimal cost |
| Architecture Verification | Sonnet | Coherence scoring |
| Citation Batch Verification | (API) | Zotero API calls, no LLM |

## Consequences

### Benefits

- **No validation failures**: LLMs generate valid text, not edit specs
- **Simpler code path**: Removed validate_edits, retry loops, fallback paths
- **Better structural fixes**: Holistic rewriting handles complex issues
- **Citation integrity**: All citations verifiable against Zotero
- **Audit trail**: Change summaries for each rewrite
- **Fail-fast**: Missing Zotero keys caught early

### Trade-offs

- **Pure move operations skipped**: Can't fix by rewriting in place
- **Context tokens**: 3 paragraphs each side adds token usage
- **Style drift**: Rewrites may slightly alter voice if LLM not careful
- **Zotero dependency**: Requires Zotero API access for validation

## Related Patterns

- [Multi-Loop Supervision System](./multi-loop-supervision-system.md) - Full supervision architecture
- [Iterative Theoretical Depth Supervision](./iterative-theoretical-depth-supervision.md) - Loop 1 implementation
- [Citation Processing with Zotero Integration](../data-pipeline/citation-processing-zotero-integration.md) - Zotero metadata

## Known Uses

- `workflows/supervised_lit_review/supervision/loops/loop3/section_rewriter.py` - Main rewriter
- `workflows/supervised_lit_review/supervision/loops/loop3/graph.py` - Loop 3 graph
- `workflows/supervised_lit_review/supervision/utils/citation_validation/validator.py` - Validation
- `workflows/academic_lit_review/paper_processor/nodes.py` - Zotero stub creation

## References

- [Anthropic Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) - Two-agent patterns
- [LangGraph Conditional Edges](https://langchain-ai.github.io/langgraph/concepts/conditional_edges/)
