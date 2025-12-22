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
from datetime import datetime
from typing import Any

from workflows.research.state import (
    DeepResearchState,
    ResearchQuestion,
    DiffusionState,
    DraftReport,
)
from workflows.research.prompts import (
    SUPERVISOR_SYSTEM_CACHED,
    SUPERVISOR_USER_TEMPLATE,
    get_today_str,
)
from workflows.shared.llm_utils import ModelTier, get_llm, invoke_with_cache
from langchain_tools.perplexity import check_fact

logger = logging.getLogger(__name__)

# Maximum concurrent researcher agents
MAX_CONCURRENT_RESEARCHERS = 3


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
        )

    iteration = diffusion["iteration"]
    max_iterations = diffusion["max_iterations"]

    # Check if we should force complete
    if iteration >= max_iterations:
        logger.info(f"Max iterations ({max_iterations}) reached - completing")
        return {
            "current_status": "research_complete",
            "diffusion": {**diffusion, "iteration": iteration},
        }

    # Build context
    brief = state.get("research_brief", {})
    research_plan = state.get("research_plan", "")
    memory_context = state.get("memory_context", "")
    findings = state.get("research_findings", [])
    draft = state.get("draft_report")

    findings_summary = _format_findings_summary(findings)
    draft_content = draft["content"] if draft else "No draft yet."
    gaps_remaining = draft.get("gaps_remaining", []) if draft else ["All areas need research"]

    # Build dynamic user prompt (changes each iteration)
    user_prompt = SUPERVISOR_USER_TEMPLATE.format(
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

    llm = get_llm(ModelTier.OPUS)  # OPUS for strategic reasoning

    try:
        # Use cached system prompt for 90% cost reduction on repeated calls
        response = await invoke_with_cache(
            llm,
            system_prompt=SUPERVISOR_SYSTEM_CACHED,  # ~800 tokens, cached
            user_prompt=user_prompt,  # Dynamic content
        )
        content = response.content

        # Extract thinking (for logging)
        thinking_match = re.search(r'<thinking>(.*?)</thinking>', content, re.DOTALL)
        if thinking_match:
            logger.debug(f"Supervisor thinking: {thinking_match.group(1)[:200]}...")

        # Determine action based on tool calls or text analysis
        # Look for action indicators in the response
        action = None
        action_data = {}

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

        # Process action
        if action == "conduct_research":
            questions = []
            for i, q in enumerate(action_data.get("questions", [])[:MAX_CONCURRENT_RESEARCHERS]):
                if isinstance(q, dict):
                    question_text = q.get("question", q.get("research_topic", str(q)))
                    context = q.get("context", "")
                else:
                    question_text = str(q)
                    context = ""

                questions.append(
                    ResearchQuestion(
                        question_id=f"q{iteration}_{i}",
                        question=question_text,
                        context=context,
                        priority=i + 1,
                    )
                )

            if not questions:
                # Fallback questions from brief
                for i, kq in enumerate(brief.get("key_questions", [])[:2]):
                    questions.append(
                        ResearchQuestion(
                            question_id=f"q{iteration}_{i}",
                            question=kq,
                            context=brief.get("topic", ""),
                            priority=i + 1,
                        )
                    )

            logger.info(f"Supervisor: conduct_research with {len(questions)} questions")

            return {
                "pending_questions": questions,
                "diffusion": {
                    **diffusion,
                    "iteration": iteration + 1,
                    "areas_explored": diffusion["areas_explored"] + [q["question"][:50] for q in questions],
                },
                "current_status": "conduct_research",
            }

        elif action == "refine_draft":
            new_draft = DraftReport(
                content=action_data.get("updates", draft_content),
                version=(draft.get("version", 0) if draft else 0) + 1,
                last_updated=datetime.utcnow(),
                gaps_remaining=action_data.get("gaps", []),
            )

            # Update completeness based on gaps
            gap_count = len(new_draft["gaps_remaining"])
            completeness = max(0.5, 1.0 - gap_count * 0.1)

            logger.info(f"Supervisor: refine_draft, version {new_draft['version']}, gaps={gap_count}")

            return {
                "draft_report": new_draft,
                "diffusion": {
                    **diffusion,
                    "completeness_score": completeness,
                },
                "current_status": "refine_draft",
            }

        elif action == "check_fact":
            claim = action_data.get("claim")
            if claim:
                logger.info(f"Supervisor: check_fact for claim: '{claim[:50]}...'")

                try:
                    fact_result = await check_fact.ainvoke({
                        "claim": claim,
                        "context": brief.get("topic", ""),
                    })

                    verdict = fact_result.get("verdict", "unverifiable")
                    confidence = fact_result.get("confidence", 0.0)
                    explanation = fact_result.get("explanation", "")

                    logger.info(
                        f"Fact-check result: {verdict} (confidence: {confidence:.2f})"
                    )

                    # Continue with research, incorporating the fact-check result
                    # The fact-check adds context but doesn't change the flow
                    return {
                        "diffusion": diffusion,
                        "current_status": "fact_checked",
                        "errors": [{
                            "node": "supervisor",
                            "type": "fact_check",
                            "claim": claim,
                            "verdict": verdict,
                            "confidence": confidence,
                            "explanation": explanation,
                        }],
                    }
                except Exception as fact_error:
                    logger.warning(f"Fact-check failed: {fact_error}")
                    # Continue without fact-check result
                    return {
                        "diffusion": diffusion,
                        "current_status": "supervising",
                    }
            else:
                logger.warning("check_fact action without claim - continuing")
                return {
                    "diffusion": diffusion,
                    "current_status": "supervising",
                }

        elif action == "research_complete":
            logger.info("Supervisor: research_complete")

            return {
                "diffusion": {
                    **diffusion,
                    "completeness_score": 1.0,
                },
                "current_status": "research_complete",
            }

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
                "diffusion": {**diffusion, "iteration": iteration + 1},
                "errors": [{"node": "supervisor", "error": str(e)}],
                "current_status": "conduct_research",
            }
        else:
            # Later iterations - complete
            return {
                "diffusion": {**diffusion, "completeness_score": 0.7},
                "errors": [{"node": "supervisor", "error": str(e)}],
                "current_status": "research_complete",
            }


def _extract_questions_from_text(content: str, brief: dict) -> list[dict]:
    """Extract research questions from unstructured text."""
    questions = []

    # Look for numbered questions or bullet points
    lines = content.split("\n")
    for line in lines:
        line = line.strip()
        # Match patterns like "1.", "- ", "* ", etc.
        if re.match(r'^[\d]+[.)]\s*', line) or re.match(r'^[-*]\s*', line):
            question = re.sub(r'^[\d]+[.)]\s*|^[-*]\s*', '', line).strip()
            if len(question) > 10 and "?" in question or len(question) > 20:
                questions.append({"question": question, "context": ""})

    # If no questions found, use key questions from brief
    if not questions and brief.get("key_questions"):
        for kq in brief["key_questions"][:2]:
            questions.append({"question": kq, "context": brief.get("topic", "")})

    return questions[:MAX_CONCURRENT_RESEARCHERS]


def _extract_draft_from_text(content: str) -> str:
    """Extract draft content from unstructured text."""
    # Remove thinking tags and other metadata
    content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
    content = re.sub(r'<action>.*?</action>', '', content, flags=re.DOTALL)
    return content.strip()
