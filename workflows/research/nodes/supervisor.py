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
    SupervisorDecision,
    calculate_completeness,
)
from workflows.research.prompts import (
    SUPERVISOR_SYSTEM_CACHED,
    SUPERVISOR_USER_TEMPLATE,
    get_today_str,
)
from workflows.research.prompts.translator import get_translated_prompt
from workflows.shared.llm_utils import ModelTier, get_llm, invoke_with_cache
from langchain_tools.perplexity import check_fact

logger = logging.getLogger(__name__)

# Maximum concurrent researcher agents
MAX_CONCURRENT_RESEARCHERS = 3

# Completeness threshold for automatic completion
COMPLETENESS_THRESHOLD = 0.85


async def _get_supervisor_decision_structured(
    llm, system_prompt: str, user_prompt: str, brief: dict
) -> tuple[str, dict]:
    """Try to get supervisor decision using structured output.

    Returns: (action, action_data) tuple

    Raises: Exception if structured output fails
    """
    structured_llm = llm.with_structured_output(SupervisorDecision)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    decision: SupervisorDecision = await structured_llm.ainvoke(messages)

    action = decision.action
    action_data = {}

    if action == "conduct_research":
        # Convert research_questions to the expected format
        action_data["questions"] = [
            {"question": q, "context": brief.get("topic", "")}
            for q in decision.research_questions
        ]
    elif action == "refine_draft":
        action_data["updates"] = decision.draft_updates or ""
        action_data["gaps"] = decision.remaining_gaps

    logger.info(
        f"Supervisor (structured): {action}, reasoning: {decision.reasoning[:100]}..."
    )

    return action, action_data


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
            last_decision="",
        )

    iteration = diffusion["iteration"]
    max_iterations = diffusion["max_iterations"]

    # Check if we should force complete due to max iterations
    if iteration >= max_iterations:
        logger.info(f"Max iterations ({max_iterations}) reached - completing")
        return {
            "current_status": "research_complete",
            "diffusion": {**diffusion, "iteration": iteration, "last_decision": "max_iterations_reached"},
        }

    # Check if completeness threshold reached
    if diffusion["completeness_score"] >= COMPLETENESS_THRESHOLD:
        logger.info(f"Completeness threshold ({COMPLETENESS_THRESHOLD:.0%}) reached - completing")
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

    # Get language-appropriate prompts
    if language_config and language_config["code"] != "en":
        system_prompt_cached = await get_translated_prompt(
            SUPERVISOR_SYSTEM_CACHED,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="supervisor_system",
        )
        user_prompt_template = await get_translated_prompt(
            SUPERVISOR_USER_TEMPLATE,
            language_code=language_config["code"],
            language_name=language_config["name"],
            prompt_name="supervisor_user",
        )
    else:
        system_prompt_cached = SUPERVISOR_SYSTEM_CACHED
        user_prompt_template = SUPERVISOR_USER_TEMPLATE

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

    llm = get_llm(ModelTier.OPUS)  # OPUS for strategic reasoning

    # Try structured output first (more reliable), fall back to text parsing
    action = None
    action_data = {}
    use_structured = True

    try:
        action, action_data = await _get_supervisor_decision_structured(
            llm, system_prompt_cached, user_prompt, brief
        )
        logger.info(f"Supervisor using structured output: {action}")
    except Exception as structured_error:
        logger.warning(f"Structured output failed, falling back to text parsing: {structured_error}")
        use_structured = False

    # Fallback: text parsing with improved extraction
    if not use_structured:
        try:
            # Use cached system prompt for 90% cost reduction on repeated calls
            response = await invoke_with_cache(
                llm,
                system_prompt=system_prompt_cached,  # ~800 tokens, cached
                user_prompt=user_prompt,  # Dynamic content
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
            logger.error(f"Text parsing also failed: {text_parse_error}")
            raise

    # Process action (runs for both structured output and text parsing)
    try:
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

            # Calculate completeness based on current state
            new_completeness = calculate_completeness(
                findings=findings,
                key_questions=brief.get("key_questions", []),
                iteration=iteration + 1,
                max_iterations=max_iterations,
                gaps_remaining=gaps_remaining,
            )

            return {
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

        elif action == "refine_draft":
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

        elif action == "research_complete":
            logger.info("Supervisor: research_complete")

            return {
                "diffusion": {
                    **diffusion,
                    "last_decision": "research_complete",
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


def _extract_questions_from_text(content: str, brief: dict) -> list[dict]:
    """Extract research questions from unstructured text.

    Improved to filter out analysis notes, thinking content, and metadata.
    """
    # 1. Remove thinking blocks entirely - they contain analysis, not questions
    content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
    content = re.sub(r'<action>.*?</action>', '', content, flags=re.DOTALL)

    # 2. Try to find question section markers to focus extraction
    question_section = content
    markers = [
        r'(?:research\s+questions?|questions?\s+to\s+(?:investigate|research|explore))[:\s]+',
        r'(?:I\'ll\s+(?:investigate|research|explore)|conduct\s*_?research\s+on)[:\s]+',
        r'(?:conductresearch|conduct_research)',
    ]
    for marker in markers:
        match = re.search(marker, content, re.IGNORECASE)
        if match:
            question_section = content[match.end():]
            break

    # 3. Extract with stricter validation
    questions = []

    # Patterns that indicate metadata/analysis rather than questions
    metadata_patterns = [
        r'iteration\s+\d+',           # "iteration 1 of 4"
        r'\d+\s*%',                   # percentages
        r'completeness',
        r'areas?\s+explored',
        r'gaps?\s+remaining',
        r'^q\d+[_-]\d+',              # Question IDs like q0_1
        r'not\s+relevant',
        r'too\s+generic',
        r'\*\*:?\s*$',                # Markdown bold endings
        r'^\*+',                      # Markdown headers
        r'current\s+findings',
        r'previous\s+(?:findings|research)',
        r'already\s+(?:covered|explored|researched)',
    ]

    for line in question_section.split("\n"):
        line = line.strip()

        # Must match numbered or bulleted pattern
        if not re.match(r'^[\d]+[.)]\s*', line) and not re.match(r'^[-*]\s*', line):
            continue

        # Extract the question text
        question = re.sub(r'^[\d]+[.)]\s*|^[-*]\s*', '', line).strip()

        # Must be substantial
        if len(question) < 20:
            continue

        # Reject lines that look like metadata or analysis
        if any(re.search(p, question, re.IGNORECASE) for p in metadata_patterns):
            logger.debug(f"Rejected metadata-like question: {question[:50]}...")
            continue

        questions.append({"question": question, "context": ""})

    # Fallback to key questions from brief if nothing extracted
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
