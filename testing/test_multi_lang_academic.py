#!/usr/bin/env python3
"""
Test script for multi-language academic literature review.

Uses the multi_lang workflow with academic-only configuration, producing:
- Per-language literature reviews
- Cross-language comparative analysis
- Final integrated synthesis report

Usage:
    python test_multi_lang_academic.py "topic" [quality] --languages <langs>
    python test_multi_lang_academic.py "topic" test --languages en,es,de
    python test_multi_lang_academic.py "topic" quick --languages major

Language options:
    --languages en,es,de,zh   Specific language codes (comma-separated)
    --languages major         Use MAJOR_10_LANGUAGES (en, zh, es, de, fr, ja, pt, ru, ar, ko)

Valid quality levels: quick, standard, comprehensive (default: quick)

Outputs saved to testing/test_data/:
    multilang-{lang}-{datetime}.md     Per-language reports
    multilang-comparative-{datetime}.md   Cross-language analysis
    multilang-final-{datetime}.md         Final integrated synthesis

Environment:
    Set THALA_MODE=dev in .env to enable LangSmith tracing
"""

import asyncio
import os
from datetime import datetime

# Enable dev mode for LangSmith tracing before any imports
os.environ["THALA_MODE"] = "dev"

import logging

from testing.utils import (
    configure_logging,
    get_output_dir,
    print_section_header,
    format_duration,
    create_test_parser,
    add_quality_argument,
    add_research_questions_argument,
)
from workflows.shared.workflow_state_store import load_workflow_state

configure_logging("multi_lang_academic")
logger = logging.getLogger(__name__)

# Output directory for results
OUTPUT_DIR = get_output_dir()

# Major 10 languages for comprehensive coverage
MAJOR_10_LANGUAGES = ["en", "zh", "es", "de", "fr", "ja", "pt", "ru", "ar", "ko"]

VALID_QUALITIES = ["test", "quick", "standard", "comprehensive"]
DEFAULT_QUALITY = "quick"
DEFAULT_LANGUAGES = ["en", "es"]


# =============================================================================
# Translation
# =============================================================================


async def translate_to_english(text: str, source_language: str) -> str:
    """Translate text to English using LLM.

    Args:
        text: Text to translate
        source_language: Source language name (e.g., "Spanish", "Chinese")

    Returns:
        English translation of the text
    """
    from workflows.shared.llm_utils.models import get_llm, ModelTier

    llm = get_llm(ModelTier.SONNET, max_tokens=16384)

    prompt = f"""Translate the following {source_language} academic literature review to English.

Preserve:
- All formatting (headers, lists, citations)
- Technical terminology (translate but keep original term in parentheses for key concepts)
- Academic tone and structure
- All citations and references

Text to translate:

{text}

Provide only the English translation, no commentary."""

    try:
        response = await llm.ainvoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return f"[Translation failed: {e}]\n\n{text}"


# =============================================================================
# Output Saving
# =============================================================================


async def save_markdown_outputs(result: dict, timestamp: str) -> dict[str, str]:
    """Save all markdown outputs from the multi_lang result.

    Non-English reports are translated to English before saving.

    Args:
        result: Combined result dict with state store data
        timestamp: Timestamp string for filenames

    Returns:
        Dict mapping output type to file path
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    saved_files = {}

    # Save per-language reports (translated to English if needed)
    for lang_result in result.get("language_results", []):
        lang_code = lang_result["language_code"]
        lang_name = lang_result["language_name"]
        full_report = lang_result.get("full_report")

        if full_report:
            # Translate non-English reports
            if lang_code != "en":
                logger.info(f"Translating {lang_name} report to English...")
                translated_report = await translate_to_english(full_report, lang_name)
            else:
                translated_report = full_report

            filename = f"multilang-{lang_code}-{timestamp}.md"
            filepath = OUTPUT_DIR / filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# {lang_name} Literature Review\n\n")
                if lang_code != "en":
                    f.write(f"*Translated from {lang_name} to English*\n\n")
                f.write(f"**Quality:** {lang_result.get('quality_used', 'unknown')}\n")
                f.write(f"**Sources:** {lang_result.get('source_count', 0)}\n")
                f.write(f"**Workflows:** {', '.join(lang_result.get('workflows_run', []))}\n\n")
                f.write("---\n\n")
                f.write(translated_report)
            saved_files[f"lang_{lang_code}"] = str(filepath)
            logger.info(f"Saved {lang_code} report: {filepath}")

    # Save comparative analysis
    comparative = result.get("comparative")
    if comparative:
        filename = f"multilang-comparative-{timestamp}.md"
        filepath = OUTPUT_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# Cross-Language Comparative Analysis\n\n")
            f.write(comparative)
        saved_files["comparative"] = str(filepath)
        logger.info(f"Saved comparative analysis: {filepath}")

    # Save final synthesis
    synthesis = result.get("final_synthesis") or result.get("final_report")
    if synthesis:
        filename = f"multilang-final-{timestamp}.md"
        filepath = OUTPUT_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# Integrated Multi-Language Literature Review\n\n")
            f.write(synthesis)
        saved_files["final"] = str(filepath)
        logger.info(f"Saved final synthesis: {filepath}")

    return saved_files


def print_result_summary(result: dict) -> None:
    """Print a summary of multi-lang results."""
    print_section_header("MULTI-LANGUAGE ACADEMIC LITERATURE REVIEW RESULTS")

    # Language results
    lang_results = result.get("language_results", [])
    print(f"\nLanguages processed: {len(lang_results)}")

    total_sources = 0
    for lang_result in lang_results:
        lang = lang_result["language_code"]
        name = lang_result["language_name"]
        sources = lang_result.get("source_count", 0)
        workflows = lang_result.get("workflows_run", [])
        total_sources += sources
        print(f"  {lang} ({name}): {sources} sources via {', '.join(workflows)}")

    print(f"\nTotal sources: {total_sources}")

    # Synthesis status
    comparative = result.get("comparative")
    synthesis = result.get("final_synthesis") or result.get("final_report")
    print("\n--- Synthesis Status ---")
    print(f"Comparative analysis: {'Yes' if comparative else 'No'}")
    print(f"Final synthesis: {'Yes' if synthesis else 'No'}")

    sonnet_analysis = result.get("sonnet_analysis")
    if sonnet_analysis:
        if sonnet_analysis.get("universal_themes"):
            print(f"\nUniversal themes: {len(sonnet_analysis['universal_themes'])}")
        if sonnet_analysis.get("unique_contributions"):
            print(f"Languages with unique contributions: {len(sonnet_analysis['unique_contributions'])}")

    # Errors
    errors = result.get("errors", [])
    if errors:
        print(f"\n--- Errors ({len(errors)}) ---")
        for error in errors[:5]:
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")


# =============================================================================
# CLI
# =============================================================================


def parse_languages(languages_str: str) -> list[str]:
    """Parse language argument into list of language codes."""
    if languages_str.lower() == "major":
        return MAJOR_10_LANGUAGES.copy()

    # Parse comma-separated list
    langs = [lang.strip().lower() for lang in languages_str.split(",")]

    # Validate language codes (basic check)
    valid_codes = {
        "en", "es", "zh", "ja", "de", "fr", "pt", "ko", "ru", "ar",
        "it", "nl", "pl", "tr", "vi", "th", "id", "hi", "bn", "sv",
        "no", "da", "fi", "cs", "el", "he", "uk", "ro", "hu"
    }
    invalid = [lang for lang in langs if lang not in valid_codes]
    if invalid:
        logger.warning(f"Unknown language codes (proceeding anyway): {invalid}")

    return langs


def parse_args():
    """Parse command line arguments."""
    parser = create_test_parser(
        description="Run multi-language academic literature review",
        default_topic="The impact of large language models on software engineering practices",
        topic_help="Research topic for literature review",
        epilog_examples="""
Examples:
  %(prog)s "transformer architectures" quick --languages en,es,de
  %(prog)s "AI in healthcare" standard --languages major
  %(prog)s "climate change policy" comprehensive --languages en,zh,de,fr
        """
    )

    add_quality_argument(parser, choices=VALID_QUALITIES, default=DEFAULT_QUALITY)

    parser.add_argument(
        "--languages", "-L",
        type=str,
        default=",".join(DEFAULT_LANGUAGES),
        help="Languages: comma-separated codes (en,es,de) or 'major' for top 10"
    )

    add_research_questions_argument(parser)

    return parser.parse_args()


async def main():
    """Run multi-language academic literature review."""
    from workflows.multi_lang import multi_lang_research

    args = parse_args()

    topic = args.topic
    quality = args.quality
    languages = parse_languages(args.languages)

    # Default research questions if not provided
    if args.questions:
        research_questions = args.questions
    else:
        research_questions = [
            f"What are the main research themes in {topic}?",
            f"What methodological approaches are used to study {topic}?",
            f"What are the key findings and debates in {topic}?",
        ]

    print_section_header("MULTI-LANGUAGE ACADEMIC LITERATURE REVIEW")
    print(f"\nTopic: {topic}")
    print(f"Quality: {quality}")
    print(f"Languages: {', '.join(languages)} ({len(languages)} total)")
    print(f"Workflow: academic (literature review only)")
    print(f"Research Questions:")
    for q in research_questions:
        print(f"  - {q}")
    print(f"LangSmith Project: {os.environ.get('LANGSMITH_PROJECT', 'thala-dev')}")
    print(f"\nStarting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    try:
        # Run multi_lang workflow with academic-only config
        workflow_result = await multi_lang_research(
            topic=topic,
            mode="set_languages",
            languages=languages,
            research_questions=research_questions,
            workflows={"academic": True},
            quality=quality,
        )

        # Convert result to dict and load full state from state store
        result = workflow_result.to_dict()
        run_id = result.get("langsmith_run_id")
        if run_id:
            full_state = load_workflow_state("multi_lang", run_id)
            if full_state:
                result = {**full_state, **result}
                logger.info(f"Loaded full state from state store for run {run_id}")
            else:
                logger.warning(f"Could not load state for run {run_id} - detailed metrics unavailable")

        # Print summary
        print_result_summary(result)

        # Save all outputs
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        saved_files = await save_markdown_outputs(result, timestamp)

        print("\n--- Saved Files ---")
        for output_type, filepath in saved_files.items():
            print(f"  {output_type}: {filepath}")

        # Duration
        started_at = result.get("started_at")
        completed_at = result.get("completed_at")
        if started_at and completed_at:
            duration_str = format_duration(started_at, completed_at)
            print(f"\nTotal duration: {duration_str}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        logger.exception(f"Error during multi-lang review: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
