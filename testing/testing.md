# Testing

## Prerequisites

Start services and confirm they're running before executing test scripts:

```bash
./services/services.sh up
./services/services.sh status
```

## Scripts

### `test_research_workflow.py`
Run a full research workflow with LangSmith tracing.

```bash
.venv/bin/python3 testing/test_research_workflow.py "your topic" [depth]
.venv/bin/python3 testing/test_research_workflow.py "AI agents" quick
.venv/bin/python3 testing/test_research_workflow.py "AI agents" comprehensive
.venv/bin/python3 testing/test_research_workflow.py  # default topic, standard depth
```

Depth options: `quick`, `standard` (default), `comprehensive`

Outputs results to `testing/test_data/` and displays the `langsmith_run_id` for inspection.

### `test_document_processing.py`
Test the document processing workflow.

```bash
.venv/bin/python3 testing/test_document_processing.py
```

### `test_academic_lit_review.py`
Run an academic literature review workflow with checkpointing support.

```bash
# Full run with automatic checkpoints
.venv/bin/python3 testing/test_academic_lit_review.py "transformer architectures" quick
.venv/bin/python3 testing/test_academic_lit_review.py "AI in drug discovery" standard

# Multilingual support (29 languages)
.venv/bin/python3 testing/test_academic_lit_review.py "machine learning" standard --language es

# Named checkpoints for iterative testing
.venv/bin/python3 testing/test_academic_lit_review.py "topic" quick --checkpoint-prefix mytest
```

Quality options: `quick`, `standard`, `comprehensive`, `high_quality`
Language: ISO 639-1 code (e.g., `en`, `es`, `zh`, `ja`, `de`, `fr`)

**Checkpointing for fast iteration:**

The workflow saves checkpoints after expensive phases (diffusion, processing). Resume from these to iterate on later phases without re-running discovery/processing:

```bash
# Resume from after-processing (fastest - only runs clustering + synthesis)
.venv/bin/python3 testing/test_academic_lit_review.py --resume-from processing --checkpoint-prefix mytest

# Resume from after-diffusion (runs processing + clustering + synthesis)
.venv/bin/python3 testing/test_academic_lit_review.py --resume-from diffusion --checkpoint-prefix mytest

# Original behavior without checkpoints
.venv/bin/python3 testing/test_academic_lit_review.py "topic" quick --no-checkpoint
```

Checkpoints are saved to `testing/test_data/checkpoints/`.

### `test_book_finding.py`
Run the book finding workflow to discover relevant books across three categories.

```bash
.venv/bin/python3 testing/test_book_finding.py "organizational resilience"
.venv/bin/python3 testing/test_book_finding.py "creative leadership" standard
.venv/bin/python3 testing/test_book_finding.py "liderazgo creativo" quick --language es
```

Quality options: `quick`, `standard`, `comprehensive`
Language: ISO 639-1 code (e.g., `en`, `es`, `zh`, `ja`, `de`, `fr`)

### `test_wrapped_research.py`
Run the wrapped research workflow that orchestrates web, academic, and book research.

```bash
# Full run with automatic checkpoints
.venv/bin/python3 testing/test_wrapped_research.py "AI agents in creative work" quick
.venv/bin/python3 testing/test_wrapped_research.py "Impact of LLMs" standard

# Named checkpoints for iterative testing
.venv/bin/python3 testing/test_wrapped_research.py "topic" quick --checkpoint-prefix mytest
```

Quality options: `quick`, `standard`, `comprehensive`

**Checkpointing for fast iteration:**

The workflow saves checkpoints after expensive phases (parallel research, book finding). Resume from these to iterate on later phases:

```bash
# Resume from after-parallel (runs books + summary + save)
.venv/bin/python3 testing/test_wrapped_research.py --resume-from parallel --checkpoint-prefix mytest

# Resume from after-books (runs summary + save only)
.venv/bin/python3 testing/test_wrapped_research.py --resume-from books --checkpoint-prefix mytest

# Original behavior without manual checkpoints
.venv/bin/python3 testing/test_wrapped_research.py "topic" quick --no-checkpoint
```

Checkpoints are saved to `testing/test_data/checkpoints/wrapped/`.

**Outputs:** Saves 4 markdown files (web, academic, books, combined) plus JSON result and analysis.

### `retrieve_langsmith_run.py`
Inspect a LangSmith run - supervisor decisions, research questions, findings.

```bash
.venv/bin/python3 testing/retrieve_langsmith_run.py <run_id>
.venv/bin/python3 testing/retrieve_langsmith_run.py <run_id> -v      # verbose
.venv/bin/python3 testing/retrieve_langsmith_run.py <run_id> --json  # JSON output
```

## Workflow

1. Run a test workflow
2. Copy the `LangSmith Run ID` from output
3. Inspect with `retrieve_langsmith_run.py`

Or use the [LangSmith UI](https://smith.langchain.com/) directly.
