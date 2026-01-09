# Workflows

LangGraph-based research and knowledge workflows. Each workflow uses composable subgraphs, quality tiers, and checkpointing for resilience.

## Available Workflows

| Workflow | Purpose | Entry Point |
|----------|---------|-------------|
| `academic_lit_review` | PhD-equivalent literature reviews | `academic_lit_review()` |
| `supervised_lit_review` | Multi-loop supervision with editing | `run_supervised_lit_review()` |
| `web_research` | Web research with supervisor/researcher agents | `web_research()` |
| `book_finding` | Book discovery across three categories | `book_finding()` |
| `document_processing` | PDF/document extraction and summarization | `process_document()` |
| `multi_lang` | Cross-language research with synthesis | `multi_lang_research()` |
| `wrapped` | Orchestrates web + academic + books | `wrapped_research()` |

## Quality Tiers

All workflows support quality presets controlling depth/cost trade-offs:

| Tier | Use Case |
|------|----------|
| `test` | Development/CI |
| `quick` | Fast results, limited depth |
| `standard` | Balanced (default) |
| `comprehensive` | Thorough coverage |
| `high_quality` | Maximum depth (lit review only) |

```python
from workflows.academic_lit_review import academic_lit_review

result = await academic_lit_review(
    topic="transformer architectures",
    quality="standard",
    language="en"
)
```

## Academic Literature Review

Produces comprehensive literature reviews through:
1. **Keyword Search** - Multi-strategy paper discovery
2. **Citation Network** - Forward/backward expansion
3. **Diffusion Engine** - Citation network with saturation detection
4. **Paper Processing** - Structured summaries
5. **Clustering** - Thematic grouping (BERTopic + LLM)
6. **Synthesis** - Coherent narrative with citations

Supports 30+ languages via ISO 639-1 codes.

## Web Research

Self-Balancing Diffusion Algorithm:
1. Clarify intent with user
2. Search memory stores
3. Create research brief
4. Supervisor coordinates parallel researchers
5. Iterative web research with query validation
6. Final report synthesis
7. Save findings to stores

## Book Finding

Discovers books in three categories:
- **Analogous domain** - Theme from different fields
- **Inspiring action** - Change-oriented books
- **Expressive fiction** - Theme-capturing narrative

## Shared Utilities (`shared/`)

```
shared/
├── llm_utils/           # Structured output, model tiers
├── language/            # Multi-language support (30+ languages)
├── persistent_cache.py  # File-based cache with TTL
├── checkpointing.py     # State serialization for resumption
├── text_utils.py        # Chunking, word counting
├── batch_processor.py   # Async batch processing
└── wrappers/            # Workflow registry, result types
```

### Structured Output

Auto-selects best strategy for LLM outputs:

```python
from workflows.shared.llm_utils import get_structured_output, ModelTier

result = await get_structured_output(
    output_schema=MySchema,
    user_prompt="Analyze this",
    tier=ModelTier.SONNET,
)
```

### Checkpointing

```python
from workflows.shared.checkpointing import save_checkpoint, load_checkpoint

save_checkpoint(state, "after_processing")
state = load_checkpoint("after_processing")
```
