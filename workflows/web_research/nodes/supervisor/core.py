"""
Supervisor node implementing the diffusion algorithm.

Uses OPUS for complex reasoning and strategic decisions.

The diffusion algorithm:
1. Generate research questions to address gaps (diffusion out)
2. Delegate to researcher agents
3. Refine draft report (diffusion in - consolidating)
4. Check completeness
5. Repeat or complete
"""

import json
import logging
import re
from typing import Any

from workflows.web_research.state import (
    DeepResearchState,
    ResearchQuestion,
    DiffusionState,
)
from workflows.web_research.prompts import (
    SUPERVISOR_SYSTEM_CACHED,
    SUPERVISOR_USER_TEMPLATE,
    get_today_str,
)
from workflows.web_research.utils import load_prompts_with_translation
from workflows.shared.llm_utils import ModelTier, get_llm, invoke_with_cache

from .llm_integration import _get_supervisor_decision_structured
from .action_handlers import (
    handle_conduct_research,
    handle_refine_draft,
    handle_check_fact,
    handle_research_complete,
)
from .text_extraction import _extract_questions_from_text, _extract_draft_from_text
from .formatting import _format_findings_summary

logger = logging.getLogger(__name__)

# Maximum concurrent researcher agents
MAX_CONCURRENT_RESEARCHERS = 3

# Completeness threshold for automatic completion
COMPLETENESS_THRESHOLD = 0.85


async def supervisor(state: DeepResearchState) -> dict[str, Any]:
    """Supervisor agent coordinating research via diffusion algorithm.

    Actions:
    - conduct_research: Generate questions, dispatch to researchers
    - refine_draft: Update draft report with new findings
    - research_complete: Signal completion

    Uses OPUS for complex reasoning.
    """
    # Get or initialize diffusion state
    diffusion = state.get("diffusion")
    if not diffusion:
        depth = state["input"].get("depth", "standard")
        max_iters = state["input"].get("max_iterations") or {
            "quick": 2,
            "standard": 4,
            "comprehensive": 8,
        }.get(depth, 4)

        diffusion = DiffusionState(
            iteration=0,
            max_iterations=max_iters,
            completeness_score=0.0,
            areas_explored=[],
            areas_to_explore=[],
            last_decision="",
        )

    iteration = diffusion["iteration"]
    max_iterations = diffusion["max_iterations"]

    # Check if we should force complete due to max iterations
    if iteration >= max_iterations:
        logger.info(f"Max iterations ({max_iterations}) reached, completing")
        return {
            "current_status": "research_complete",
            "diffusion": {**diffusion, "iteration": iteration, "last_decision": "max_iterations_reached"},
        }

    # Check if completeness threshold reached
    if diffusion["completeness_score"] >= COMPLETENESS_THRESHOLD:
        logger.info(f"Completeness threshold ({COMPLETENESS_THRESHOLD:.0%}) reached, completing")
        return {
            "current_status": "research_complete",
            "diffusion": {**diffusion, "last_decision": "completeness_threshold_reached"},
        }

    # Build context
    brief = state.get("research_brief", {})
    research_plan = state.get("research_plan", "")
    memory_context = state.get("memory_context", "")
    findings = state.get("research_findings", [])
    draft = state.get("draft_report")
    language_config = state.get("primary_language_config")

    findings_summary = _format_findings_summary(findings)
    draft_content = draft["content"] if draft else "No draft yet."
    gaps_remaining = draft.get("gaps_remaining", []) if draft else ["All areas need research"]

    system_prompt_cached, user_prompt_template = await load_prompts_with_translation(
        SUPERVISOR_SYSTEM_CACHED,
        SUPERVISOR_USER_TEMPLATE,
        language_config,
        "supervisor_system",
        "supervisor_user",
    )

    # Build dynamic user prompt (changes each iteration)
    user_prompt = user_prompt_template.format(
        date=get_today_str(),
        research_brief=json.dumps(brief, indent=2),
        research_plan=research_plan or "No customized plan.",
        memory_context=memory_context or "No memory context.",
        draft_report=draft_content,
        findings_summary=findings_summary,
        iteration=iteration + 1,
        max_iterations=max_iterations,
        completeness_score=int(diffusion["completeness_score"] * 100),
        areas_explored=", ".join(diffusion["areas_explored"]) or "None yet",
        gaps_remaining=", ".join(gaps_remaining) or "Unknown",
        max_concurrent_research_units=MAX_CONCURRENT_RESEARCHERS,
    )

    llm = get_llm(ModelTier.OPUS)

    # Try structured output first (more reliable), fall back to text parsing
    action = None
    action_data = {}
    decision = None
    use_structured = True

    try:
        action, action_data, decision = await _get_supervisor_decision_structured(
            llm, system_prompt_cached, user_prompt, brief
        )
        logger.debug(f"Supervisor decision: {action}")
    except Exception as structured_error:
        logger.warning(f"Structured output failed, falling back to text parsing: {structured_error}")
        use_structured = False

    # Fallback: text parsing with improved extraction
    if not use_structured:
        try:
            # Use cached system prompt for 90% cost reduction on repeated calls
            response = await invoke_with_cache(
                llm,
                system_prompt=system_prompt_cached,
                user_prompt=user_prompt,
            )
            content = response.content

            # Extract thinking (for logging)
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', content, re.DOTALL)
            if thinking_match:
                logger.debug(f"Supervisor thinking: {thinking_match.group(1)[:200]}...")

            # Determine action based on tool calls or text analysis
            content_lower = content.lower()

            if "conductresearch" in content_lower or "conduct_research" in content_lower:
                action = "conduct_research"
                # Try to extract questions
                questions_match = re.search(r'"questions"\s*:\s*\[(.*?)\]', content, re.DOTALL)
                if questions_match:
                    try:
                        # This is a rough extraction; real implementation would use tool binding
                        questions_text = "[" + questions_match.group(1) + "]"
                        questions_raw = json.loads(questions_text)
                        action_data["questions"] = questions_raw
                    except:
                        # Fallback: extract questions from text
                        action_data["questions"] = _extract_questions_from_text(content, brief)
                else:
                    action_data["questions"] = _extract_questions_from_text(content, brief)

            elif "refinedraftreport" in content_lower or "refine_draft" in content_lower:
                action = "refine_draft"
                # Extract update content
                updates_match = re.search(r'"updates"\s*:\s*"([^"]*)"', content, re.DOTALL)
                if updates_match:
                    action_data["updates"] = updates_match.group(1)
                else:
                    action_data["updates"] = _extract_draft_from_text(content)

                # Extract gaps
                gaps_match = re.search(r'"gaps"\s*:\s*\[(.*?)\]', content, re.DOTALL)
                if gaps_match:
                    try:
                        action_data["gaps"] = json.loads("[" + gaps_match.group(1) + "]")
                    except:
                        action_data["gaps"] = []

            elif "researchcomplete" in content_lower or "research_complete" in content_lower:
                action = "research_complete"

            elif "checkfact" in content_lower or "check_fact" in content_lower or "verify_claim" in content_lower:
                action = "check_fact"
                # Extract the claim to verify
                claim_match = re.search(r'"claim"\s*:\s*"([^"]*)"', content, re.DOTALL)
                if claim_match:
                    action_data["claim"] = claim_match.group(1)
                else:
                    # Try to find quoted text after "verify" or "check"
                    verify_match = re.search(r'(?:verify|check|fact.?check)[:\s]+["\']([^"\']+)["\']', content, re.IGNORECASE)
                    if verify_match:
                        action_data["claim"] = verify_match.group(1)

            else:
                # Default: if we have few findings, conduct research; else complete
                if len(findings) < 2:
                    action = "conduct_research"
                    action_data["questions"] = _extract_questions_from_text(content, brief)
                else:
                    action = "refine_draft"
                    action_data["updates"] = content
        except Exception as text_parse_error:
            logger.error(f"Text parsing failed: {text_parse_error}")
            raise

    # Process action (runs for both structured output and text parsing)
    try:
        if action == "conduct_research":
            return await handle_conduct_research(
                action_data,
                diffusion,
                state,
                iteration,
                max_iterations,
                brief,
                findings,
                gaps_remaining,
            )

        elif action == "refine_draft":
            return await handle_refine_draft(
                action_data,
                diffusion,
                iteration,
                max_iterations,
                brief,
                findings,
                draft,
                draft_content,
            )

        elif action == "check_fact":
            return await handle_check_fact(action_data, diffusion, brief)

        elif action == "research_complete":
            return await handle_research_complete(diffusion)

    except Exception as e:
        logger.error(f"Supervisor failed: {e}")

        # On error, try to continue or complete
        if iteration < 1:
            # First iteration error - try to conduct research
            return {
                "pending_questions": [
                    ResearchQuestion(
                        question_id="q0_fallback",
                        question=brief.get("topic", "Research topic"),
                        context="Fallback due to supervisor error",
                        priority=1,
                    )
                ],
                "diffusion": {**diffusion, "iteration": iteration + 1, "last_decision": "error_fallback"},
                "errors": [{"node": "supervisor", "error": str(e)}],
                "current_status": "conduct_research",
            }
        else:
            # Later iterations - complete
            return {
                "diffusion": {**diffusion, "completeness_score": 0.7, "last_decision": "error_complete"},
                "errors": [{"node": "supervisor", "error": str(e)}],
                "current_status": "research_complete",
            }
