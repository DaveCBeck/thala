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
