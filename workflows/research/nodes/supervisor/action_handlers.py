"""Action handlers for supervisor decisions."""

import logging
from datetime import datetime

from workflows.research.state import (
    ResearchQuestion,
    DraftReport,
    calculate_completeness,
)
from langchain_tools.perplexity import check_fact

logger = logging.getLogger(__name__)

# Maximum concurrent researcher agents
MAX_CONCURRENT_RESEARCHERS = 3


async def handle_conduct_research(
    action_data: dict,
    diffusion: dict,
    state: dict,
    iteration: int,
    max_iterations: int,
    brief: dict,
    findings: list,
    gaps_remaining: list,
) -> dict:
    """Handle conduct_research action."""
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

    # Calculate completeness based on current state
    new_completeness = calculate_completeness(
        findings=findings,
        key_questions=brief.get("key_questions", []),
        iteration=iteration + 1,
        max_iterations=max_iterations,
        gaps_remaining=gaps_remaining,
    )

    # Determine researcher allocation
    # Priority: user-specified (already in state) > LLM decision > default
    current_allocation = state.get("researcher_allocation")
    if not current_allocation and action_data.get("llm_allocation"):
        # Use LLM's allocation decision
        llm_alloc = action_data["llm_allocation"]
        web_count = llm_alloc["web_count"]
        if 1 <= web_count <= 3:
            current_allocation = llm_alloc
            logger.info(f"Using LLM allocation: web={web_count}")

    result = {
        "pending_questions": questions,
        "diffusion": {
            **diffusion,
            "iteration": iteration + 1,
            "areas_explored": diffusion["areas_explored"] + [q["question"][:50] for q in questions],
            "completeness_score": new_completeness,
            "last_decision": "conduct_research",
        },
        "current_status": "conduct_research",
    }

    # Only update allocation if LLM set it (don't overwrite user-specified)
    if current_allocation and not state.get("researcher_allocation"):
        result["researcher_allocation"] = current_allocation

    return result


async def handle_refine_draft(
    action_data: dict,
    diffusion: dict,
    iteration: int,
    max_iterations: int,
    brief: dict,
    findings: list,
    draft: dict,
    draft_content: str,
) -> dict:
    """Handle refine_draft action."""
    new_draft = DraftReport(
        content=action_data.get("updates", draft_content),
        version=(draft.get("version", 0) if draft else 0) + 1,
        last_updated=datetime.utcnow(),
        gaps_remaining=action_data.get("gaps", []),
    )

    # Use multi-signal completeness calculation
    new_completeness = calculate_completeness(
        findings=findings,
        key_questions=brief.get("key_questions", []),
        iteration=iteration,
        max_iterations=max_iterations,
        gaps_remaining=new_draft["gaps_remaining"],
    )

    logger.info(
        f"Supervisor: refine_draft, version {new_draft['version']}, "
        f"gaps={len(new_draft['gaps_remaining'])}, completeness={new_completeness:.0%}"
    )

    return {
        "draft_report": new_draft,
        "diffusion": {
            **diffusion,
            "completeness_score": new_completeness,
            "last_decision": "refine_draft",
        },
        "current_status": "refine_draft",
    }


async def handle_check_fact(action_data: dict, diffusion: dict, brief: dict) -> dict:
    """Handle check_fact action."""
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
                "diffusion": {**diffusion, "last_decision": "check_fact"},
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
                "diffusion": {**diffusion, "last_decision": "check_fact_failed"},
                "current_status": "supervising",
            }
    else:
        logger.warning("check_fact action without claim - continuing")
        return {
            "diffusion": {**diffusion, "last_decision": "check_fact_no_claim"},
            "current_status": "supervising",
        }


async def handle_research_complete(diffusion: dict) -> dict:
    """Handle research_complete action."""
    logger.info("Supervisor: research_complete")

    return {
        "diffusion": {
            **diffusion,
            "last_decision": "research_complete",
        },
        "current_status": "research_complete",
    }
