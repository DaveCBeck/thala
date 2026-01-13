"""Text extraction utilities for parsing LLM responses."""

import logging
import re

logger = logging.getLogger(__name__)

# Maximum concurrent researcher agents
MAX_CONCURRENT_RESEARCHERS = 3


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
