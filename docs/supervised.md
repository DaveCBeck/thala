# Supervised Lit Review Refactoring

> **Goal:** Make each supervision loop standalone, passing only standardized outputs + original call parameters between them.

## Target Interface

Each loop should accept/return:

```python
# Input (standardized)
{
    "current_review": str,                    # The document
    "topic": str,                             # Original topic
    "research_questions": list[str],          # Original RQs
    "quality": QualityTier,                   # Quality tier
    "source_count": int,                      # Sources so far
    "exclude_dois": set[str],                 # DOIs to skip (for paper fetching)
}

# Output (standardized)
{
    "current_review": str,                    # Modified document
    "changes_summary": str,                   # What changed
    "papers_added": dict[str, PaperSummary],  # New papers (DOI → summary)
    "citations_added": dict[str, str],        # New citations (DOI → zotero_key)
    "issues_flagged": list[dict],             # For human review
}
```

---

## Loop Status

| Loop | Status | Notes |
|------|--------|-------|
| Loop 1 (Theoretical depth) | **Done** | New `run_loop1_standalone()` |
| Loop 2 (Literature expansion) | **Done** | New `run_loop2_standalone()` |
| Loop 3 (Structure/cohesion) | **Done** | Simplified interface, removed unused params |
| Loop 4 (Section editing) | **Done** | Simplified interface, removed citation tracking |
| Loop 4.5 (Cohesion check) | Pending | Already standalone |
| Loop 5 (Fact checking) | **Done** | Simplified interface, tools query ES directly |

---

## State Persistence (All Loops)

Each loop should save its full state to disk (when `THALA_MODE=dev`) for analysis:

```python
save_workflow_state(
    workflow_name="supervision_loop{N}",
    run_id=str(run_id),
    state={
        "input": {...},      # Inputs received
        "output": {...},     # Summary of outputs
        "final_state": ...,  # Full internal state
    },
)
```

Location: `~/.thala/workflow_states/supervision_loop{N}/{run_id}.json`

---

## ES Integration (Loops 4 & 5) ✓

**Completed:**
- Loops 4/5 now query ES directly for paper content
- `SupervisionStoreQuery` queries ES by zotero_key (no paper_summaries needed)
- `search_papers` tool uses hybrid ES/Chroma search over full corpus
- `get_paper_content` fetches from ES with Zotero API fallback
- `paper_summaries` removed from loop inputs entirely

**Future enhancements (not blocking):**
- More sophisticated paper queries ("Find claims about X", "Get methodology details")
- Better semantic search for contradicting evidence

---

## Loop 1 Analysis

### Implemented Changes ✓

**New standalone signature:**
```python
async def run_loop1_standalone(
    review: str,
    topic: str,
    research_questions: list[str],
    max_iterations: int = 3,
    source_count: int = 0,
    quality_settings: dict | None = None,
    config: dict | None = None,
) -> Loop1Result
```

**Files modified:**
- `prompts/loop1_supervision.py` - Removed `{cluster_summary}`, renamed `corpus_size` → `source_count`
- `nodes/analyze_review.py` - Removed cluster handling, uses direct params
- `nodes/expand_topic.py` - Uses `exclude_dois` directly
- `graph.py` - New `Loop1State`, `Loop1Result`, `run_loop1_standalone()`
- `orchestration/nodes.py` - Updated `run_loop1_node` to use new interface
- `orchestration/orchestrator.py` - Updated `run_supervision_configurable` to use new interface

### Loop 1 Output (Final)

```python
@dataclass
class Loop1Result:
    current_review: str       # Modified document
    changes_summary: str      # "Explored 2 theoretical gaps: X, Y"
    issues_explored: list[str]  # Topics that were expanded
```

**Dropped from output:**
- `iterations` - inferred from `len(issues_explored)`
- `completion_reason` - debugging only
- `citations_added` - redundant with papers_added
- `papers_added` - papers go to ES/Zotero; downstream loops will query directly

**Dropped from input:**
- `exclude_dois` - paper processing pipeline checks ES/Zotero internally for duplicates

---

## Loop 2 Analysis

### What Loop 2 Does

Identifies missing literature bases (theoretical perspectives, methodological approaches) and runs **nested `academic_lit_review` workflows** to expand coverage. This is heavyweight - each identified base triggers a full discovery→diffusion→processing pipeline.

### Original Interface

```python
async def run_loop2_standalone(
    review: str,
    paper_corpus: dict,          # DOI -> PaperMetadata
    paper_summaries: dict,       # DOI -> PaperSummary
    zotero_keys: dict,           # DOI -> Zotero key
    input_data: LitReviewInput,  # Contains topic + research_questions
    quality_settings: QualitySettings,
    max_iterations: int = 3,
    config: dict | None = None,
) -> dict
```

**Problems:**
- `paper_corpus`, `paper_summaries`, `zotero_keys` passed in and merged with new papers on output
- `exclude_dois` filtering used to avoid re-fetching papers already in corpus
- `max_iterations` as separate param instead of derived from quality
- Returns merged dicts, coupling downstream loops to this output

### Implemented Changes ✓

**New standalone signature:**
```python
@traceable(run_type="chain", name="Loop2_LiteratureExpansion")
async def run_loop2_standalone(
    review: str,
    topic: str,
    research_questions: list[str],
    quality_settings: dict,        # max_stages used for iterations
    config: dict | None = None,    # tracing only
) -> Loop2Result
```

**New output:**
```python
@dataclass
class Loop2Result:
    current_review: str
    changes_summary: str      # "Expanded 2 literature bases: X, Y"
    explored_bases: list[str] # Bases that were expanded
```

**Dropped from input:**
- `paper_corpus` - papers go to ES/Zotero via nested workflow
- `paper_summaries` - same
- `zotero_keys` - same
- `input_data` - flattened to `topic` + `research_questions`
- `max_iterations` - derived from `quality_settings["max_stages"]`
- `exclude_dois` - ES/Zotero handles duplicates internally

**Dropped from output:**
- `paper_corpus` - papers already in ES/Zotero
- `paper_summaries` - same
- `zotero_keys` - same
- `added_zotero_keys` - same
- `iteration`, `is_complete`, `errors`, `iterations_failed` - debugging only

**Files modified:**
- `loops/loop2/graph.py` - New `Loop2State`, `Loop2Result`
- `loops/loop2/utils.py` - New `run_loop2_standalone()` with state persistence
- `loops/loop2/analyzer.py` - Uses flat params instead of `input` dict
- `loops/loop2/integrator.py` - Removed exclude_dois filtering and corpus merging
- `orchestration/nodes.py` - Updated `run_loop2_node` to use new interface
- `orchestration/orchestrator.py` - Updated `run_supervision_configurable`

---

## Loop 3 Analysis

### What Loop 3 Does

Two-phase structural editing to improve document organization:
- **Phase A**: Identify structural issues (diagnosis) - uses Opus with extended thinking
- **Phase B**: Rewrite affected sections to fix issues (section-rewrite approach)

The section-rewrite approach replaced a previous edit-specification approach, eliminating the "identifies issues but can't generate valid edits" failure mode.

### Original Interface

```python
async def run_loop3_standalone(
    review: str,
    input_data: LitReviewInput,        # For topic + research_questions
    max_iterations: int = 3,
    config: dict | None = None,
    zotero_keys: dict[str, str] | None = None,      # UNUSED
    zotero_key_sources: dict[str, dict] | None = None,  # UNUSED
) -> dict
```

**Problems:**
- `zotero_keys`, `zotero_key_sources` passed but never actually used internally
- `input_data` only used for `topic` field (research_questions never accessed)
- `max_iterations` as separate param instead of derived from quality
- No state persistence
- Legacy fields (`edit_manifest`, `applied_edits`) cluttering state

### Implemented Changes ✓

**New standalone signature:**
```python
@traceable(run_type="chain", name="Loop3_StructureCohesion")
async def run_loop3_standalone(
    review: str,
    topic: str,
    quality_settings: dict,   # max_stages used for iterations
    config: dict | None = None,
) -> Loop3Result
```

**New output:**
```python
@dataclass
class Loop3Result:
    current_review: str
    changes_summary: str      # "Resolved 3 structural issues"
    iterations_used: int
```

**Dropped from input:**
- `input_data` - flattened to just `topic: str`
- `max_iterations` - derived from `quality_settings["max_stages"] + 1` (extra iteration for structural work)
- `zotero_keys` - was passed but never used
- `zotero_key_sources` - was passed but never used

**Dropped from state:**
- `input: LitReviewInput` - replaced with `topic: str`
- `zotero_keys`, `zotero_key_sources` - removed entirely
- `edit_manifest`, `applied_edits` - legacy fields removed

**Files modified:**
- `loops/loop3/graph.py` - New `Loop3State`, `Loop3Result`, updated `run_loop3_standalone()` with state persistence
- `loops/loop3/analyzer.py` - Uses `topic` directly from state
- `loops/loop3/section_rewriter.py` - Removed unused `zotero_keys` param
- `loops/loop3/utils.py` - Simplified `finalize_node` to remove legacy field checks
- `loops/loop3/__init__.py` - Export `Loop3Result`
- `orchestration/nodes.py` - Updated `run_loop3_node` to use new interface
- `orchestration/orchestrator.py` - Updated `run_supervision_configurable`

---

## Loop 4 Analysis

### What Loop 4 Does

Section-level deep editing with parallel processing and tool access. Has multiple phases:
1. **Split**: Divides document into sections (with section type classification)
2. **Parallel Edit**: Each section edited with Opus + paper tools (5-way parallel)
3. **Resolve TODOs**: Attempts to resolve `<!-- TODO: -->` markers using tools
4. **Reassemble**: Reconstructs document from edited sections
5. **Holistic Review**: Evaluates coherence, flags sections for re-editing
6. **Iterate/Finalize**: Either re-edits flagged sections or finalizes

### Current Interface

```python
async def run_loop4_standalone(
    review: str,
    paper_summaries: dict,              # DOI -> PaperSummary
    input_data: LitReviewInput,         # Only uses .topic
    zotero_keys: dict[str, str],        # DOI -> Zotero key
    zotero_key_sources: dict[str, dict] | None = None,  # Provenance tracking
    max_iterations: int = 3,
    config: dict | None = None,
    verify_zotero: bool = True,
) -> dict
```

### Loop4State Fields

| Field | Used Internally | Used by Orchestrator | Notes |
|-------|-----------------|---------------------|-------|
| `current_review` | ✓ Split, edit, reassemble | ✓ Main output | Core |
| `paper_summaries` | ✓ Tools, prompts, validation | ✗ | **Key coupling** |
| `zotero_keys` | ✓ Build corpus_keys for validation | ✗ | DOI→key mapping |
| `zotero_key_sources` | ✓ Track verified keys | ✓ Updated with new verifications | Provenance |
| `input` | ✓ Only `.topic` used | ✗ | Should flatten |
| `sections` | ✓ Internal | ✗ | |
| `section_results` | ✓ Internal | ✗ Returned in dict | |
| `editor_notes` | ✓ Holistic review input | ✗ | |
| `holistic_result` | ✓ Internal routing | ✗ Returned in dict | |
| `flagged_sections` | ✓ Internal routing | ✗ | |
| `flagged_reasons` | ✓ Internal routing | ✗ | |
| `iteration` | ✓ Loop control | ✓ As `iterations` | |
| `max_iterations` | ✓ Loop control | ✗ | Should derive |
| `is_complete` | ✓ Internal | ✗ | |
| `verify_zotero` | ✓ Controls validation | ✗ | |
| `verified_citation_keys` | ✓ Accumulates verified | ✓ Returned for Loop 5 | **Must preserve** |

### Citation Handling Complexity

Loop 4's citation handling has multiple interconnected pieces:

**Input sources:**
1. `zotero_keys: dict[str, str]` - DOI → Zotero key from initial corpus
2. `zotero_key_sources: dict[str, dict]` - Citation key → metadata tracking where citations came from

**Validation flow:**
1. Build `corpus_keys = set(zotero_keys.values()) | set(zotero_key_sources.keys())`
2. For each section edit, extract new citations from edited text
3. If `verify_zotero=True`: verify new citations against Zotero API
4. Strip/TODO-mark citations that fail verification
5. Accumulate newly verified keys in `verified_citation_keys`

**Output used by orchestrator:**
- `verified_citation_keys: set[str]` - Keys verified against Zotero
- Orchestrator adds these to `zotero_key_sources` with `CITATION_SOURCE_LOOP4`

### Problems Identified

1. **`paper_summaries` coupling** - Entire corpus passed as dict; should query ES
2. **`input_data` wrapper** - Only `.topic` used; should flatten to `topic: str`
3. **`max_iterations` hardcoded** - Should derive from `quality_settings`
4. **Complex provenance tracking** - `zotero_key_sources` tracks citation sources across loops; adds complexity for unclear benefit
5. **Dict return** - Returns untyped dict; should use dataclass
6. **No state persistence** - Unlike Loops 1-3, doesn't save state for analysis
7. **`SupervisionStoreQuery` wraps dict** - Doesn't actually query ES; just wraps passed-in `paper_summaries`

### How Paper Tools Work (Current)

`create_paper_tools(paper_summaries, store_query)` creates two tools:

1. **`search_papers(query, limit)`** - Hybrid semantic+keyword search
   - Uses in-memory `paper_summaries` dict (not ES)
   - Returns DOI, title, year, authors, relevance, zotero_key

2. **`get_paper_content(doi, max_chars)`** - Fetch L2 compressed content
   - Uses `store_query.get_paper_content()` which DOES query ES
   - Falls back to `paper_summaries` metadata if ES lookup fails

**Key insight:** The tools partially use ES (for content) but search is in-memory.

### Implemented Changes ✓

**New standalone signature:**
```python
@dataclass
class Loop4Result:
    current_review: str
    changes_summary: str
    iterations_used: int

@traceable(run_type="chain", name="Loop4_SectionEditing")
async def run_loop4_standalone(
    review: str,
    topic: str,
    quality_settings: dict[str, Any],
    config: dict | None = None,
) -> Loop4Result
```

**Changes made:**
- Flattened `input_data: LitReviewInput` → `topic: str`
- **Removed `paper_summaries` from input entirely** - tools now query ES/Chroma directly
- Removed `zotero_keys` and `verify_zotero` from input entirely
- Removed `zotero_key_sources` from input entirely
- Removed `verified_citation_keys` from output
- **Removed all citation validation** - Loop 4 now focuses purely on section editing; citation validation to be handled by Loop 5
- **Removed paper summaries from prompt** - LLM now uses tools exclusively
- Derived `max_iterations` from `quality_settings["max_stages"]` (minimum 2 enforced)
- Created `Loop4Result` dataclass
- Added state persistence via `save_workflow_state()`

**Paper tools refactored:**
- `search_papers` now searches full ES/Chroma corpus without filtering
- `get_paper_content` queries ES by zotero_key, falls back to Zotero API for metadata
- `SupervisionStoreQuery` no longer needs paper_summaries - queries ES directly

**Files modified:**
- `loops/loop4_editing.py` - Simplified `Loop4State`, `Loop4Result`, removed paper_summaries dependency
- `store_query.py` - Refactored to query ES by zotero_key directly
- `tools/paper_search/searcher.py` - Updated `create_paper_tools` to not need paper_summaries
- `tools/paper_search/sources.py` - Updated search functions to return full results without filtering
- `prompts/loop4_editing.py` - Simplified user prompt (removed paper_summaries placeholder)
- `orchestration/nodes.py` - Updated `run_loop4_node`
- `orchestration/orchestrator.py` - Updated `run_supervision_configurable`

---

## New Utility: core/stores/utils.py

Added Zotero/ES verification utilities for use by Loop 5 (and other components):

```python
@dataclass
class KeyVerificationResult:
    zotero_key: str
    exists_in_zotero: bool
    es_record_id: Optional[UUID] = None

async def verify_zotero_keys(
    keys: list[str],
    zotero_client: Optional[ZoteroStore] = None,
    es_stores: Optional[ElasticsearchStores] = None,
    concurrency: int = 10,
) -> list[KeyVerificationResult]

async def verify_zotero_keys_batch(
    keys: set[str], ...
) -> dict[str, KeyVerificationResult]
```

---

## Loop 5 Analysis

### What Loop 5 Does

Fact and reference checking with tool access:
1. **Split**: Divides document into sections
2. **Fact Check**: Verifies factual claims using paper tools + web search
3. **Reference Check**: Validates citations support claims
4. **Validate Edits**: Checks edit find/replace strings are valid
5. **Apply Edits**: Applies validated edits, then verifies ALL citations against Zotero
6. **Citation Resolution**: Invalid citations → LLM resolves (find replacement, remove, or rewrite)
7. **Flag Issues**: Collects ambiguous claims and unaddressed TODOs
8. **Finalize**: Strips remaining TODO markers

### Implemented Changes ✓

**New standalone signature:**
```python
@dataclass
class Loop5Result:
    current_review: str
    changes_summary: str
    human_review_items: list[str]

@traceable(run_type="chain", name="Loop5_FactReferenceCheck")
async def run_loop5_standalone(
    review: str,
    topic: str,
    quality_settings: dict[str, Any],
    config: dict | None = None,
) -> Loop5Result
```

**Changes made:**
- **Removed `paper_summaries` from input** - tools query ES/Chroma directly
- **Removed `zotero_keys` from input** - all citations verified against Zotero API directly
- **Removed `verify_zotero` config** - always verify
- Removed `max_iterations` - derived from `quality_settings["max_stages"]`
- Removed `verify_todos_enabled` - always enabled
- **Removed paper context from prompts** - LLM uses tools exclusively
- **New citation resolution**: Invalid citations → LLM uses tools to fix (no TODOs)
- Created `Loop5Result` dataclass
- Added state persistence via `save_workflow_state()`

**Citation verification flow:**
1. Extract all `[@KEY]` citations from document
2. Batch verify against Zotero API
3. Invalid keys → `resolve_invalid_citations()`:
   - LLM searches for replacement papers
   - Removes unnecessary citations
   - Rewrites claims if no supporting paper exists
   - Applies fixes directly (no TODO markers)

**Files modified:**
- `loops/loop5/graph.py` - Simplified state and interface
- `loops/loop5/fact_checking.py` - Removed paper_summaries usage
- `loops/loop5/reference_checking.py` - Removed zotero_keys usage
- `loops/loop5/result_processing.py` - New citation verification flow
- `loops/loop5/citation_resolution.py` - **NEW**: LLM-based citation fixing
- `loops/loop5/utils.py` - Removed unused functions
- `prompts/loop5_factcheck.py` - Removed paper_summaries placeholders
- `orchestration/nodes.py` - Updated `run_loop5_node`
