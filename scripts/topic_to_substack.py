#!/usr/bin/env python3
"""
Generate a Substack article from a research topic.

Runs an academic literature review (quick quality) and transforms it into
a polished Substack-style essay.

Usage (from project root):
    python scripts/topic_to_substack.py -m "your research topic"
    python scripts/topic_to_substack.py -m "transformer architectures in vision"

Output:
    Saves literature review and essay to .outputs/
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Enable dev mode for LangSmith tracing before any imports
os.environ["THALA_MODE"] = "dev"

from langchain_core.tracers.langchain import wait_for_all_tracers  # noqa: E402

from workflows.shared.llm_utils import ModelTier, get_llm  # noqa: E402
from workflows.research.academic_lit_review import academic_lit_review  # noqa: E402
from workflows.output.substack_review import substack_review_graph  # noqa: E402

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / ".outputs"


async def generate_research_questions(topic: str) -> list[str]:
    """Generate research questions from a topic using an LLM."""
    llm = get_llm(ModelTier.FLASH)

    prompt = f"""Given the research topic: "{topic}"

Generate 3-4 specific, focused research questions that would guide a comprehensive
literature review. Questions should:
- Be specific enough to guide paper discovery
- Cover different aspects of the topic
- Be answerable through academic literature

Return ONLY the questions, one per line, without numbering or bullets."""

    response = await llm.ainvoke(prompt)
    questions = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    return questions[:4]  # Limit to 4 questions


async def run_workflow(topic: str) -> dict:
    """Run the full workflow: lit review -> substack essay."""
    print("\n" + "=" * 60)
    print(f"Topic: {topic}")
    print("=" * 60 + "\n")

    # Step 1: Generate research questions
    print("Generating research questions...")
    questions = await generate_research_questions(topic)
    print("  Questions:")
    for q in questions:
        print(f"    - {q}")
    print()

    # Step 2: Run literature review
    print("Running academic literature review (quick quality)...")
    lit_result = await academic_lit_review(
        topic=topic,
        research_questions=questions,
        quality="quick",
        language="en",
    )

    if not lit_result.get("final_review"):
        raise RuntimeError(f"Literature review failed: {lit_result.get('errors', 'Unknown error')}")

    print(f"  Papers analyzed: {len(lit_result.get('paper_corpus', {}))}")
    print(f"  Review length: {len(lit_result['final_review'].split())} words")
    print()

    # Step 3: Generate Substack essay
    print("Generating Substack essay...")
    essay_result = await substack_review_graph.ainvoke({
        "input": {"literature_review": lit_result["final_review"]}
    })

    if not essay_result.get("final_essay"):
        raise RuntimeError(f"Essay generation failed: {essay_result.get('errors', 'Unknown error')}")

    print(f"  Selected angle: {essay_result.get('selected_angle', 'unknown')}")
    print(f"  Essay length: {len(essay_result['final_essay'].split())} words")
    print(f"  Status: {essay_result.get('status', 'unknown')}")
    print()

    return {
        "topic": topic,
        "research_questions": questions,
        "lit_review": lit_result,
        "essay": essay_result,
    }


def save_outputs(result: dict) -> tuple[Path, Path]:
    """Save literature review and essay to .outputs/"""
    OUTPUT_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_slug = result["topic"][:50].replace(" ", "_").replace("/", "-")

    # Save literature review
    lit_review_path = OUTPUT_DIR / f"lit_review_{topic_slug}_{timestamp}.md"
    with open(lit_review_path, "w") as f:
        f.write(f"# Literature Review: {result['topic']}\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write("*Quality: quick*\n\n")
        f.write("## Research Questions\n\n")
        for q in result["research_questions"]:
            f.write(f"- {q}\n")
        f.write("\n---\n\n")
        f.write(result["lit_review"]["final_review"])

    # Save Substack essay
    essay_path = OUTPUT_DIR / f"substack_{topic_slug}_{timestamp}.md"
    with open(essay_path, "w") as f:
        f.write(f"# {result['topic']}\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write(f"*Angle: {result['essay'].get('selected_angle', 'unknown')}*\n\n")
        f.write("---\n\n")
        f.write(result["essay"]["final_essay"])

    return lit_review_path, essay_path


async def main():
    parser = argparse.ArgumentParser(
        description="Generate a Substack article from a research topic"
    )
    parser.add_argument(
        "-m", "--topic",
        required=True,
        help="Research topic to explore"
    )
    args = parser.parse_args()

    result = await run_workflow(args.topic)
    lit_path, essay_path = save_outputs(result)

    print("=" * 60)
    print("Done!")
    print(f"  Literature review: {lit_path}")
    print(f"  Substack essay: {essay_path}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        wait_for_all_tracers()
