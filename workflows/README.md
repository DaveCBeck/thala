# Workflows

LangGraph-based research and knowledge workflows. Each workflow uses composable subgraphs and quality tiers.

## Available Workflows

| Workflow | Purpose | Entry Point | Docs |
|----------|---------|-------------|------|
| `academic_lit_review` | PhD-equivalent literature reviews | `academic_lit_review()` | [README](research/academic_lit_review/README.md) |
| `web_research` | Web research with supervisor/researcher agents | `web_research()` | [README](research/web_research/README.md) |
| `book_finding` | Book discovery across three categories | `book_finding()` | [README](research/book_finding/README.md) |
| `document_processing` | PDF/document extraction and summarization | `process_document()` | [README](document_processing/README.md) |
| `substack_review` | Essay generation from literature reviews | `substack_review()` | [README](output/substack_review/README.md) |
| `enhance` | Two-phase document enhancement | `enhance()` | [README](enhance/README.md) |
| `multi_lang` | Cross-language research with synthesis | `multi_lang_research()` | — |

### Enhancement Subworkflows

| Subworkflow | Purpose | Docs |
|-------------|---------|------|
| `enhance.supervision` | Multi-loop paper corpus expansion | [README](enhance/supervision/README.md) |
| `enhance.editing` | Structural editing with fact-checking | [README](enhance/editing/README.md) |

### Utilities

| Module | Purpose | Docs |
|--------|---------|------|
| `shared` | LLM utils, caching, batch processing, language support | [README](shared/README.md) |

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


## Shared Utilities (`shared/`)

```
shared/
├── llm_utils/           # Structured output, model tiers
├── language/            # Multi-language support (30+ languages)
├── persistent_cache.py  # File-based cache with TTL
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
