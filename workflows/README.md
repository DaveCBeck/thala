# Workflows

LangGraph-based research and knowledge workflows. Each workflow uses composable subgraphs and quality tiers.

## Available Workflows

| Workflow | Purpose | Entry Point | Docs |
|----------|---------|-------------|------|
| `academic_lit_review` | PhD-equivalent literature reviews | `academic_lit_review()` | [README](research/academic_lit_review/README.md) |
| `web_research` | Web research with supervisor/researcher agents | `deep_research()` | [README](research/web_research/README.md) |
| `book_finding` | Book discovery across three categories | `book_finding()` | [README](research/book_finding/README.md) |
| `document_processing` | PDF/document extraction and summarization | `process_document()` | [README](document_processing/README.md) |
| `evening_reads` | 4-part article series from literature reviews | `evening_reads_graph` | [README](output/evening_reads/README.md) |
| `illustrate` | Document illustration with images and diagrams | `illustrate_graph` | [README](output/illustrate/README.md) |
| `enhance` | Three-phase document enhancement | `enhance_report()` | [README](enhance/README.md) |
| `multi_lang` | Cross-language research with synthesis | `multi_lang_research()` | [README](wrappers/multi_lang/README.md) |
| `synthesis` | Multi-workflow orchestration for comprehensive reports | `synthesis()` | — |

### Enhancement Subworkflows

| Subworkflow | Purpose | Docs |
|-------------|---------|------|
| `enhance.supervision` | Multi-loop paper corpus expansion | [README](enhance/supervision/README.md) |
| `enhance.editing` | Structural editing and flow polishing | [README](enhance/editing/README.md) |
| `enhance.fact_check` | Fact verification and citation validation | [README](enhance/fact_check/README.md) |

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
├── llm_utils/           # Structured output, model tiers, caching
├── language/            # Multi-language support (30+ languages)
├── batch_processor/     # Async batch processing with Anthropic API
├── diagram_utils/       # SVG diagram generation and validation
├── image_utils.py       # Image handling and processing
├── token_utils.py       # Token counting and context management
├── metadata_utils.py    # Metadata extraction utilities
├── persistent_cache.py  # File-based cache with TTL
├── text_utils.py        # Chunking, word counting
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
