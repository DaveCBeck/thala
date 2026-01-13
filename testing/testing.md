# Testing

Test scripts for Thala workflows with LangSmith tracing.

## Prerequisites

```bash
./services/services.sh up
./services/services.sh status
```

Ensure `THALA_MODE=dev` in `.env` for LangSmith tracing.

## Directory Structure

```
testing/
├── testing.md                  # This file
├── test_*.py                   # Test scripts
├── test_data/                  # Output files
│   └── *.md, *.json            # Results
├── logs/                       # Timestamped logs
└── traces/                     # LangSmith trace exports
```

## Scripts

### `test_research_workflow.py`
Run a full research workflow with LangSmith tracing.

```bash
python -m testing.test_research_workflow "your topic" [depth]
python -m testing.test_research_workflow "AI agents" quick
python -m testing.test_research_workflow "AI agents" comprehensive
python -m testing.test_research_workflow  # default topic, standard depth
```

Depth options: `quick`, `standard` (default), `comprehensive`

Outputs results to `testing/test_data/` and displays the `langsmith_run_id` for inspection.

### `test_document_processing.py`
Test the document processing workflow.

```bash
python -m testing.test_document_processing
```

### `test_academic_lit_review.py`
Run an academic literature review workflow.

```bash
python -m testing.test_academic_lit_review "transformer architectures" quick
python -m testing.test_academic_lit_review "AI in drug discovery" standard

# Multilingual support (29 languages)
python -m testing.test_academic_lit_review "machine learning" standard --language es
```

Quality options: `test`, `quick`, `standard`, `comprehensive`, `high_quality`
Language: ISO 639-1 code (e.g., `en`, `es`, `zh`, `ja`, `de`, `fr`)

### `test_supervised_lit_review.py`
Run academic literature review with multi-loop supervision for enhanced quality.

```bash
python -m testing.test_supervised_lit_review "transformer architectures" quick
python -m testing.test_supervised_lit_review "AI in drug discovery" standard --loops all
python -m testing.test_supervised_lit_review "machine learning" standard --loops two
python -m testing.test_supervised_lit_review "topic" standard --language es
```

Quality options: `test`, `quick`, `standard`, `comprehensive`, `high_quality`
Supervision loops: `none`, `one`, `two`, `three`, `four`, `all` (default: `all`)

Supervision loops applied after the base lit review:
- Loop 1: Citation coverage & missing sources
- Loop 2: Structural editing & logical flow
- Loop 3: Fact-checking & claim verification
- Loop 4: Gap analysis & completeness

### `test_multi_lang_academic.py`
Run academic literature review across multiple languages with cross-language synthesis.

```bash
# Specific languages
python -m testing.test_multi_lang_academic "AI ethics" quick --languages en,es,de

# Major 10 languages (en, zh, es, de, fr, ja, pt, ru, ar, ko)
python -m testing.test_multi_lang_academic "climate policy" quick --languages major

# With custom research questions
python -m testing.test_multi_lang_academic "topic" standard --languages en,zh -q "Question 1?" "Question 2?"
```

Quality options: `quick`, `standard`, `comprehensive`
Languages: comma-separated codes or `major` for top 10

**Outputs** (saved to `testing/test_data/`):
- `multilang-{lang}-{datetime}.md` - Per-language reports (translated to English)
- `multilang-comparative-{datetime}.md` - Cross-language analysis
- `multilang-final-{datetime}.md` - Integrated synthesis

Uses the multi_lang workflow with `workflows={"academic": True}`.

### `test_book_finding.py`
Run the book finding workflow to discover relevant books across three categories.

```bash
python -m testing.test_book_finding "organizational resilience"
python -m testing.test_book_finding "creative leadership" standard
python -m testing.test_book_finding "liderazgo creativo" quick --language es
```

Quality options: `quick`, `standard`, `comprehensive`
Language: ISO 639-1 code (e.g., `en`, `es`, `zh`, `ja`, `de`, `fr`)

### `test_wrapped_research.py`
Run the wrapped research workflow that orchestrates web, academic, and book research.

```bash
python -m testing.test_wrapped_research "AI agents in creative work" quick
python -m testing.test_wrapped_research "Impact of LLMs" standard
```

Quality options: `quick`, `standard`, `comprehensive`

**Outputs:** Saves 4 markdown files (web, academic, books, combined) plus JSON result and analysis.

### `retrieve_langsmith_run.py`
Inspect a LangSmith run - supervisor decisions, research questions, findings.

```bash
python -m scripts.traces.retrieve_langsmith_run <run_id>
python -m scripts.traces.retrieve_langsmith_run <run_id> -v      # verbose
python -m scripts.traces.retrieve_langsmith_run <run_id> --json  # JSON output
```

## Workflow

1. Run a test workflow
2. Copy the `LangSmith Run ID` from output
3. Inspect with `retrieve_langsmith_run.py`

Or use the [LangSmith UI](https://smith.langchain.com/) directly.

## Test Utilities

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
