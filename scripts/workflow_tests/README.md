# Workflow Test Scripts

CLI test scripts for Thala workflows with LangSmith tracing.

## Prerequisites

```bash
./services/services.sh up
./services/services.sh status
```

Ensure `THALA_MODE=dev` in `.env` for LangSmith tracing.

## Directory Structure

```
scripts/workflow_tests/
├── README.md                  # This file
├── test_*.py                  # Test scripts
├── test_data/                 # Output files
│   └── *.md, *.json           # Results
├── traces/                    # LangSmith trace exports
└── utils/                     # Shared utilities
```

## Scripts

### `test_research_workflow.py`
Run a full research workflow with LangSmith tracing.

```bash
python -m scripts.workflow_tests.test_research_workflow "your topic" [depth]
python -m scripts.workflow_tests.test_research_workflow "AI agents" quick
python -m scripts.workflow_tests.test_research_workflow "AI agents" comprehensive
python -m scripts.workflow_tests.test_research_workflow  # default topic, standard depth
```

Depth options: `quick`, `standard` (default), `comprehensive`

Outputs results to `scripts/workflow_tests/test_data/` and displays the `langsmith_run_id` for inspection.

### `test_document_processing.py`
Test the document processing workflow.

```bash
python -m scripts.workflow_tests.test_document_processing
```

### `test_academic_lit_review.py`
Run an academic literature review workflow.

```bash
python -m scripts.workflow_tests.test_academic_lit_review "transformer architectures" quick
python -m scripts.workflow_tests.test_academic_lit_review "AI in drug discovery" standard

# Multilingual support (29 languages)
python -m scripts.workflow_tests.test_academic_lit_review "machine learning" standard --language es
```

Quality options: `test`, `quick`, `standard`, `comprehensive`, `high_quality`
Language: ISO 639-1 code (e.g., `en`, `es`, `zh`, `ja`, `de`, `fr`)

### `test_multi_lang_academic.py`
Run academic literature review across multiple languages with cross-language synthesis.

```bash
# Specific languages
python -m scripts.workflow_tests.test_multi_lang_academic "AI ethics" quick --languages en,es,de

# Major 10 languages (en, zh, es, de, fr, ja, pt, ru, ar, ko)
python -m scripts.workflow_tests.test_multi_lang_academic "climate policy" quick --languages major

# With custom research questions
python -m scripts.workflow_tests.test_multi_lang_academic "topic" standard --languages en,zh -q "Question 1?" "Question 2?"
```

Quality options: `quick`, `standard`, `comprehensive`
Languages: comma-separated codes or `major` for top 10

**Outputs** (saved to `scripts/workflow_tests/test_data/`):
- `multilang-{lang}-{datetime}.md` - Per-language reports (translated to English)
- `multilang-comparative-{datetime}.md` - Cross-language analysis
- `multilang-final-{datetime}.md` - Integrated synthesis

Uses the multi_lang workflow with `workflows={"academic": True}`.

### `test_book_finding.py`
Run the book finding workflow to discover relevant books across three categories.

```bash
python -m scripts.workflow_tests.test_book_finding "organizational resilience"
python -m scripts.workflow_tests.test_book_finding "creative leadership" standard
python -m scripts.workflow_tests.test_book_finding "liderazgo creativo" quick --language es
```

Quality options: `quick`, `standard`, `comprehensive`
Language: ISO 639-1 code (e.g., `en`, `es`, `zh`, `ja`, `de`, `fr`)

### `test_editing_workflow.py`
Test the structural editing workflow on documents.

```bash
python -m scripts.workflow_tests.test_editing_workflow --quality standard
python -m scripts.workflow_tests.test_editing_workflow --quality comprehensive --topic "custom topic"
```

### `test_evening_reads_illustrated.py`
Test the combined evening reads + illustration workflow.

```bash
python -m scripts.workflow_tests.test_evening_reads_illustrated --output-dir /tmp/output
```

### `test_synthesis_workflow.py`
Run the complete multi-phase synthesis workflow.

```bash
python -m scripts.workflow_tests.test_synthesis_workflow "AI ethics" quick
python -m scripts.workflow_tests.test_synthesis_workflow "topic" standard --language es
```

### Trace Analysis

```bash
# Export trace to JSON
python -m scripts.traces.retrieve_langsmith_run <run_id> --json > trace.json

# Analyze trace structure
python -m scripts.traces.analyze_trace_structure <trace_file>
```

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `THALA_MODE=dev` | Enable LangSmith tracing |
| `LANGSMITH_API_KEY` | LangSmith authentication |
| `LANGSMITH_PROJECT` | Project name (default: `thala-dev`) |
