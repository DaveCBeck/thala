"""Supervisor — Python-managed research coordination loop.

Uses invoke_via_cli for planning and report generation (no tools).
Dispatches parallel researcher subagents for actual web research.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone

from workflows.shared.llm_utils import ModelTier
from workflows.shared.llm_utils.cli_backend import invoke_via_cli

from .prompts import BRIEF_SYSTEM, PLAN_SYSTEM, REPORT_SYSTEM, get_today
from .researcher import run_researcher, ResearchFinding

logger = logging.getLogger(__name__)


async def run_research(
    query: str,
    max_iterations: int = 3,
    max_researcher_turns: int = 20,
    recency_info: str = "",
) -> dict:
    """Run the full research workflow.

    Returns:
        {"final_report": str, "source_count": int, "errors": list, ...}
    """
    started_at = datetime.now(timezone.utc)
    errors = []
    all_findings: list[ResearchFinding] = []

    # Phase 1: Generate research brief
    logger.info(f"Generating research brief for: {query[:80]}...")
    brief = await _generate_brief(query)
    if not brief:
        return _error_result("Failed to generate research brief", started_at)

    logger.info(
        f"Brief: {brief.get('topic', 'Unknown')}, "
        f"{len(brief.get('key_questions', []))} key questions"
    )

    # Phase 2: Research loop
    for iteration in range(max_iterations):
        logger.info(f"Research iteration {iteration + 1}/{max_iterations}")

        plan = await _plan_iteration(brief, all_findings, iteration, max_iterations)

        if plan is None:
            errors.append({"node": "plan", "error": "Planning failed"})
            continue

        if plan == "RESEARCH_COMPLETE":
            logger.info("Supervisor signaled research complete")
            break

        questions = plan.get("questions", [])
        if not questions:
            logger.warning("No questions generated, completing")
            break

        # Dispatch researchers in parallel
        logger.info(f"Dispatching {len(questions)} researchers")
        findings, dispatch_errors = await _dispatch_researchers(
            questions, brief, recency_info, max_researcher_turns
        )
        all_findings.extend(findings)
        errors.extend(dispatch_errors)
        logger.info(
            f"Iteration {iteration + 1}: {len(findings)} new findings, "
            f"{len(all_findings)} total"
        )

    # Phase 3: Generate final report
    logger.info(f"Generating final report from {len(all_findings)} findings")
    report = await _generate_report(query, brief, all_findings)

    if not report:
        report = _fallback_report(query, all_findings)
        errors.append({"node": "report", "error": "Report generation failed, using fallback"})

    return {
        "final_report": report,
        "status": "success" if not errors else "partial",
        "source_count": len(all_findings),
        "errors": errors,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc),
    }


async def _generate_brief(query: str) -> dict | None:
    """Generate a structured research brief."""
    try:
        response = await invoke_via_cli(
            tier=ModelTier.OPUS,
            system=BRIEF_SYSTEM.format(date=get_today()),
            user_prompt=query,
        )
        return _parse_json(response.content)
    except Exception as e:
        logger.error(f"Brief generation failed: {e}")
        return None


async def _plan_iteration(
    brief: dict,
    findings: list[ResearchFinding],
    iteration: int,
    max_iterations: int,
) -> dict | str | None:
    """Ask the supervisor what to research next."""
    findings_text = _format_findings(findings) if findings else "No findings yet."

    user_prompt = f"""## Research Brief
{json.dumps(brief, indent=2)}

## Accumulated Findings ({len(findings)} so far)
{findings_text}

## Status
Iteration {iteration + 1} of {max_iterations}.
{"This is the LAST iteration — prioritize the biggest remaining gaps." if iteration == max_iterations - 1 else ""}

What should we research next?"""

    try:
        response = await invoke_via_cli(
            tier=ModelTier.OPUS,
            system=PLAN_SYSTEM.format(date=get_today()),
            user_prompt=user_prompt,
            effort="high",
        )
        text = response.content
        if "RESEARCH_COMPLETE" in text:
            return "RESEARCH_COMPLETE"
        return _parse_json(text)
    except Exception as e:
        logger.error(f"Planning failed: {e}")
        return None


async def _dispatch_researchers(
    questions: list[dict],
    brief: dict,
    recency_info: str,
    max_researcher_turns: int = 20,
) -> tuple[list[ResearchFinding], list[dict]]:
    """Dispatch parallel researcher subagents."""
    tasks = []
    for q in questions:
        question = q.get("question", str(q))
        context = q.get("context", "")
        hints = q.get("search_hints", [])

        full_context = f"Research topic: {brief.get('topic', '')}\n{context}"

        tasks.append(
            run_researcher(
                question=question,
                context=full_context,
                search_hints=hints,
                recency_info=recency_info,
                max_turns=max_researcher_turns,
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    findings = []
    errors = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Researcher {i} failed: {result}")
            errors.append({"node": f"researcher_{i}", "error": str(result)})
        elif result:
            findings.append(result)

    return findings, errors


async def _generate_report(
    query: str,
    brief: dict,
    findings: list[ResearchFinding],
) -> str | None:
    """Generate the final comprehensive report."""
    findings_text = _format_findings(findings)

    brief_text = (
        f"**Topic:** {brief.get('topic', query)}\n"
        f"**Objectives:** {', '.join(brief.get('objectives', []))}\n"
        f"**Key Questions:** {', '.join(brief.get('key_questions', []))}"
    )

    user_prompt = f"""## Research Brief
{brief_text}

## All Research Findings
{findings_text}

Generate the final research report. Synthesize across all findings into a coherent, well-cited report."""

    try:
        response = await invoke_via_cli(
            tier=ModelTier.OPUS,
            system=REPORT_SYSTEM.format(date=get_today()),
            user_prompt=user_prompt,
            effort="high",
        )
        return response.content
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return None


def _format_findings(findings: list[ResearchFinding]) -> str:
    """Format findings for inclusion in prompts."""
    parts = []
    for i, f in enumerate(findings):
        sources_text = "\n".join(
            f"  [{j+1}] {s.get('title', 'Unknown')}: {s.get('url', 'N/A')}"
            for j, s in enumerate(f.sources)
        )
        parts.append(
            f"### Finding {i + 1} (confidence: {f.confidence})\n"
            f"{f.finding}\n\n"
            f"Sources:\n{sources_text}\n\n"
            f"Gaps: {', '.join(f.gaps) if f.gaps else 'None noted'}"
        )
    return "\n\n---\n\n".join(parts)


def _fallback_report(query: str, findings: list[ResearchFinding]) -> str:
    """Build a fallback report from raw findings."""
    parts = [f"# Research: {query}\n\n*Report generation failed. Raw findings below.*\n"]
    for i, f in enumerate(findings):
        parts.append(f"## Finding {i + 1}\n{f.finding}\n")
    return "\n".join(parts)


def _parse_json(text: str) -> dict | None:
    """Extract JSON from LLM response text."""
    # Try code blocks first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try finding any JSON object (handles preamble text)
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    logger.warning(f"Failed to parse JSON from: {text[:200]}...")
    return None


def _error_result(error: str, started_at) -> dict:
    return {
        "final_report": None,
        "status": "failed",
        "source_count": 0,
        "errors": [{"node": "supervisor", "error": error}],
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc),
    }
