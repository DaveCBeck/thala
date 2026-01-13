"""Formatting utilities for supervisor output."""


def _format_findings_summary(findings: list) -> str:
    """Format research findings into a summary string."""
    if not findings:
        return "No findings yet."

    summaries = []
    for f in findings:
        summary = (
            f"- **{f.get('question_id', 'Q?')}**: {f.get('finding', '')[:200]}... "
            f"(confidence: {f.get('confidence', 0):.1f})"
        )
        summaries.append(summary)

    return "\n".join(summaries)
