# Workflow Restructuring Plan

## Overview

Restructure the workflows folder to properly separate independent workflows that are currently misplaced as "subgraphs" of research.

### Current Structure
```
workflows/
├── research/
│   ├── subgraphs/
│   │   ├── academic_lit_review/     # Independent, not a subgraph
│   │   │   ├── supervision/         # To be extracted
│   │   │   └── ...
│   │   ├── book_finding/            # Independent, not a subgraph
│   │   ├── researcher_base/         # Shared utilities
│   │   └── web_researcher.py        # Actual subgraph (stays)
│   └── ...
└── ...
```

### Target Structure
```
workflows/
├── academic_lit_review/             # Standalone (no supervision)
│   ├── graph/
│   ├── clustering/
│   ├── diffusion_engine/
│   ├── paper_processor/
│   ├── synthesis/
│   └── ...
├── supervised_lit_review/           # Wrapper with supervision loops
│   ├── graph/
│   ├── supervision/                 # Moved from academic_lit_review
│   │   ├── loops/
│   │   ├── orchestration/
│   │   ├── mini_review/
│   │   └── ...
│   └── ...
├── book_finding/                    # Standalone
│   ├── graph/
│   └── nodes/
├── research/                        # Web research workflow
│   ├── subgraphs/
│   │   ├── researcher_base/         # Stays for shared utilities
│   │   └── web_researcher.py        # Stays (actual subgraph)
│   └── ...
└── ...
```

---

## Execution Steps

### Phase 1: Move academic_lit_review (without supervision)

1. **Create new directory**: `workflows/academic_lit_review/`

2. **Copy structure** (excluding supervision/):
   - `state.py`
   - `__init__.py`
   - `graph/` (api.py, construction.py, state_init.py, phases/ minus supervision.py)
   - `keyword_search.py`
   - `citation_network.py`
   - `citation_graph.py`
   - `diffusion_engine/`
   - `paper_processor/`
   - `clustering/`
   - `synthesis/`
   - `utils/`

3. **Modify graph/construction.py**: Remove supervision node and edge
   - Remove: `builder.add_node("supervision", supervision_phase_node)`
   - Change: `builder.add_edge("synthesis", END)` (was supervision)

4. **Update all internal imports**:
   - From: `workflows.research.subgraphs.academic_lit_review`
   - To: `workflows.academic_lit_review`

### Phase 2: Create supervised_lit_review

1. **Create new directory**: `workflows/supervised_lit_review/`

2. **Move supervision/ folder** from academic_lit_review

3. **Create wrapper graph** that:
   - Imports and runs `academic_lit_review` graph
   - Applies supervision loops to the output

4. **Update supervision imports**:
   - From: `workflows.research.subgraphs.academic_lit_review.supervision`
   - To: `workflows.supervised_lit_review.supervision`

5. **Update mini_review imports**:
   - It calls academic_lit_review phases, so update to new location

### Phase 3: Move book_finding

1. **Move entire directory**:
   - From: `workflows/research/subgraphs/book_finding/`
   - To: `workflows/book_finding/`

2. **Update all internal imports**:
   - From: `workflows.research.subgraphs.book_finding`
   - To: `workflows.book_finding`

### Phase 4: Update external references

Files to update:
- `workflows/multi_lang/builtin_workflows.py`
- `workflows/wrapped/nodes/run_parallel_research.py`
- `workflows/wrapped/nodes/run_book_finding.py`
- `testing/test_academic_lit_review.py`
- `testing/test_book_finding.py`
- `workflows/research/GRAPH.md`

### Phase 5: Cleanup

1. Remove old directories from `workflows/research/subgraphs/`
2. Update `workflows/__init__.py` if needed
3. Verify no broken imports

---

## Import Pattern Changes

### academic_lit_review
```python
# Before
from workflows.research.subgraphs.academic_lit_review.state import ...
from workflows.research.subgraphs.academic_lit_review.clustering import ...

# After
from workflows.academic_lit_review.state import ...
from workflows.academic_lit_review.clustering import ...
```

### supervised_lit_review (supervision module)
```python
# Before
from workflows.research.subgraphs.academic_lit_review.supervision import ...

# After
from workflows.supervised_lit_review.supervision import ...
```

### book_finding
```python
# Before
from workflows.research.subgraphs.book_finding.state import ...

# After
from workflows.book_finding.state import ...
```

---

## Files Affected

### academic_lit_review internal (bulk replace): ~50 files
### supervision internal (bulk replace): ~15 files
### book_finding internal (bulk replace): ~8 files
### External references: ~6 files

---

## Risk Mitigation

1. **Backup**: Git will preserve history
2. **Incremental**: Move one workflow at a time
3. **Verify**: Run imports check after each phase
4. **Test**: Run existing tests after completion
