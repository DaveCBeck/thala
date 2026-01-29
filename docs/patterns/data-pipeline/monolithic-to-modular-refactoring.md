---
name: monolithic-to-modular-refactoring
title: Monolithic-to-Modular Refactoring Pattern
date: 2025-12-31
category: data-pipeline
applicability:
  - "Files exceeding 500+ lines that handle multiple concerns"
  - "Modules with multiple classes or logical groupings"
  - "Code with circular import risks"
  - "Large files that slow down IDE navigation and testing"
components: [core_stores, langchain_tools, workflows, shared_utilities]
complexity: medium
verified_in_production: true
tags: [refactoring, modularization, package-structure, backward-compatibility, imports]
---

# Monolithic-to-Modular Refactoring Pattern

## Intent

Transform large monolithic Python files (500+ lines) into organized package structures with focused modules, while maintaining complete backward compatibility through `__init__.py` re-exports.

## Problem

Large monolithic files create several issues:
- **Navigation difficulty**: Finding specific functions in 1000+ line files
- **Circular imports**: Tangled dependencies between functions
- **Testing complexity**: Can't test individual components in isolation
- **Merge conflicts**: Multiple developers editing the same large file
- **IDE performance**: Slow analysis and autocomplete

## Solution

Convert `module.py` → `module/` package with:
1. **Focused submodules**: Each handles a single concern
2. **Re-exporting `__init__.py`**: Maintains existing import paths
3. **Clear naming conventions**: Submodule names reflect their responsibility
4. **Minimal cross-module dependencies**: Each submodule is as self-contained as possible

## Structure

```
# BEFORE: Monolithic file
workflows/research/
├── graph.py          # 464 lines - construction, routing, aggregation, API
├── state.py          # 497 lines - 8 different state types
└── prompts/
    └── base.py       # 806 lines - all prompts in one file

# AFTER: Organized packages
workflows/research/
├── graph/
│   ├── __init__.py       # Re-exports for backward compatibility
│   ├── construction.py   # Graph building
│   ├── routing.py        # Conditional routing logic
│   ├── aggregation.py    # Result aggregation
│   ├── api.py            # Public API functions
│   └── config.py         # Configuration constants
├── state/
│   ├── __init__.py
│   ├── input_types.py
│   ├── researcher_state.py
│   ├── supervisor_state.py
│   └── workflow_state.py
└── prompts/
    ├── __init__.py
    ├── supervision.py
    ├── queries.py
    ├── reporting.py
    └── compression.py
```

## Implementation

### Step 1: Analyze the Monolithic File

Identify logical groupings by function responsibility:

```python
# Example: llm_utils.py (569 lines) contains:
# - Model tier definitions (ModelTier enum, get_llm)
# - Text processing (extract_json, extract_structured)
# - Caching utilities (invoke_with_cache, CacheManager)

# Group into:
# llm_utils/models.py       - ModelTier, get_llm
# llm_utils/text_processors.py - extract_json, extract_structured
# llm_utils/caching.py      - invoke_with_cache, CacheManager
```

### Step 2: Create Package Structure

Create the directory and move code:

```bash
# Create package directory
mkdir -p workflows/shared/llm_utils

# Create submodules (one file per concern)
touch workflows/shared/llm_utils/models.py
touch workflows/shared/llm_utils/text_processors.py
touch workflows/shared/llm_utils/caching.py
touch workflows/shared/llm_utils/__init__.py
```

### Step 3: Split Code into Submodules

Move related functions to their submodules:

```python
# workflows/shared/llm_utils/models.py
"""Model tier definitions and LLM factory."""

import logging
from enum import Enum
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class ModelTier(str, Enum):
    """Model tiers for different complexity tasks."""
    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"
    DEEPSEEK = "deepseek"


def get_llm(
    tier: ModelTier = ModelTier.SONNET,
    max_tokens: int = 4096,
    thinking_budget: Optional[int] = None,
) -> ChatAnthropic | ChatOpenAI:
    """Factory function for LLM instances."""
    # Implementation...
```

```python
# workflows/shared/llm_utils/text_processors.py
"""Text extraction and processing utilities."""

import json
import logging
import re
from typing import Any

from .models import ModelTier, get_llm

logger = logging.getLogger(__name__)


async def extract_json(
    text: str,
    prompt: str,
    schema_hint: str,
) -> dict[str, Any]:
    """Extract JSON from text using LLM."""
    # Implementation...


async def extract_structured(
    text: str,
    prompt: str,
    schema: dict,
    tier: ModelTier = ModelTier.SONNET,
) -> dict:
    """Extract structured data using Anthropic tool use."""
    # Implementation...
```

```python
# workflows/shared/llm_utils/caching.py
"""LLM caching utilities for cost reduction."""

import hashlib
import logging
from typing import Any

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


async def invoke_with_cache(
    llm: Any,
    system_prompt: str,
    user_prompt: str,
) -> AIMessage:
    """Invoke LLM with prompt caching for cost reduction."""
    # Implementation...
```

### Step 4: Create Backward-Compatible `__init__.py`

Re-export everything from the original module's public API:

```python
# workflows/shared/llm_utils/__init__.py
"""
LLM utilities package.

Re-exports all public symbols for backward compatibility.
Existing imports like `from workflows.shared.llm_utils import get_llm`
continue to work unchanged.
"""

from .models import (
    ModelTier,
    get_llm,
    ANTHROPIC_MODELS,
    OPENAI_MODELS,
)

from .text_processors import (
    extract_json,
    extract_structured,
    clean_json_string,
)

from .caching import (
    invoke_with_cache,
    CacheManager,
    get_cache_key,
)

__all__ = [
    # Models
    "ModelTier",
    "get_llm",
    "ANTHROPIC_MODELS",
    "OPENAI_MODELS",
    # Text processors
    "extract_json",
    "extract_structured",
    "clean_json_string",
    # Caching
    "invoke_with_cache",
    "CacheManager",
    "get_cache_key",
]
```

### Step 5: Update Internal Imports

Within the package, use relative imports:

```python
# workflows/shared/llm_utils/text_processors.py

# Use relative imports within the package
from .models import ModelTier, get_llm

# NOT absolute imports (creates coupling)
# from workflows.shared.llm_utils.models import ModelTier
```

### Step 6: Remove Original File

Once all imports are verified working:

```bash
# Delete original monolithic file
rm workflows/shared/llm_utils.py

# The package __init__.py at the same path handles all imports
```

## Real Example: Clustering Module

The `clustering.py` (1146 lines) was split into 11 focused modules:

```python
# workflows/research/subgraphs/academic_lit_review/clustering/__init__.py
"""
Clustering package for academic literature review.

Provides:
- BERTopic-based statistical clustering
- LLM-based semantic clustering
- Opus synthesis for final themes
- Cluster analysis and formatting
"""

from .schemas import (
    LLMTheme,
    LLMTopicSchema,
    ThematicCluster,
    ClusterAnalysis,
)

from .bertopic_clustering import (
    run_bertopic_clustering,
    create_bertopic_model,
)

from .llm_clustering import (
    run_llm_clustering,
    LLMTopicSchemaOutput,
)

from .synthesis import (
    synthesize_clusters,
    OpusSynthesisOutput,
)

from .analysis import (
    analyze_cluster,
    ClusterAnalysisOutput,
)

from .graph import create_clustering_subgraph

from .api import cluster_papers

__all__ = [
    # Schemas
    "LLMTheme",
    "LLMTopicSchema",
    "ThematicCluster",
    "ClusterAnalysis",
    # BERTopic
    "run_bertopic_clustering",
    "create_bertopic_model",
    # LLM clustering
    "run_llm_clustering",
    "LLMTopicSchemaOutput",
    # Synthesis
    "synthesize_clusters",
    "OpusSynthesisOutput",
    # Analysis
    "analyze_cluster",
    "ClusterAnalysisOutput",
    # Graph
    "create_clustering_subgraph",
    # API
    "cluster_papers",
]
```

## Guidelines

### When to Modularize

| File Size | Logical Groupings | Recommendation |
|-----------|-------------------|----------------|
| < 300 lines | 1-2 | Keep as single file |
| 300-500 lines | 2-3 | Consider splitting |
| 500+ lines | 3+ | Split into package |
| 1000+ lines | Any | Definitely split |

### Naming Conventions

| Submodule Type | Naming Pattern | Example |
|----------------|----------------|---------|
| Core logic | `core.py` | `supervisor/core.py` |
| Type definitions | `types.py`, `schemas.py` | `state/types.py` |
| Constants | `constants.py`, `config.py` | `clustering/constants.py` |
| Utilities | `utils.py` | `process_citations/utils.py` |
| Public API | `api.py` | `synthesis/api.py` |
| LangGraph construction | `graph.py` | `diffusion_engine/graph.py` |
| Domain-specific | Descriptive name | `citation_formatter.py` |

### Import Organization

```python
# In __init__.py, organize imports by category:

# 1. Type definitions and schemas
from .types import ...
from .schemas import ...

# 2. Core functionality
from .core import ...

# 3. Specialized modules
from .specific_feature import ...

# 4. Public API (high-level functions)
from .api import ...

# 5. Graph construction (if LangGraph)
from .graph import ...
```

### Testing Strategy

After modularization, tests can target specific submodules:

```python
# tests/test_clustering/test_bertopic.py
from workflows.research.subgraphs.academic_lit_review.clustering.bertopic_clustering import (
    run_bertopic_clustering,
    create_bertopic_model,
)

# tests/test_clustering/test_synthesis.py
from workflows.research.subgraphs.academic_lit_review.clustering.synthesis import (
    synthesize_clusters,
)
```

## Known Uses

This pattern was applied to:

| Original File | Lines | Package Structure |
|--------------|-------|-------------------|
| `core/stores/elasticsearch.py` | 522 | `elasticsearch/` (4 submodules) |
| `langchain_tools/openalex.py` | 274+ | `openalex/` (5 submodules) |
| `workflows/shared/llm_utils.py` | 569 | `llm_utils/` (3 submodules) |
| `workflows/research/graph.py` | 464 | `graph/` (5 submodules) |
| `workflows/research/state.py` | 497 | `state/` (8 submodules) |
| `clustering.py` | 1146 | `clustering/` (11 submodules) |
| `synthesis.py` | 1000 | `synthesis/` (8 submodules) |
| `diffusion_engine.py` | 823 | `diffusion_engine/` (8 submodules) |

## Consequences

### Benefits
- **Focused modules**: Each file has a single responsibility
- **Easier navigation**: IDE can quickly jump to specific functions
- **Isolated testing**: Test submodules independently
- **Reduced conflicts**: Developers work on separate files
- **Better imports**: Clearer dependency graph

### Trade-offs
- **More files**: Directory tree is deeper
- **Import management**: `__init__.py` must stay in sync
- **Refactoring effort**: Initial split takes time

## Related Patterns

- [Workflow Modularization Pattern](../langgraph/workflow-modularization.md) - LangGraph-specific modularization
- [Quality Tier Standardization](../llm-interaction/quality-tier-standardization.md) - Consistent configuration patterns

## References

- [Python Packaging User Guide](https://packaging.python.org/en/latest/)
- [PEP 328 - Imports: Multi-Line and Absolute/Relative](https://peps.python.org/pep-0328/)
