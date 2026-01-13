"""
CLI argument parser utilities for test scripts.

Provides factory functions and helpers for consistent argument parsing
across all test scripts, while preserving CLI-runnable behavior.
"""

import argparse
from argparse import ArgumentParser


def create_test_parser(
    description: str,
    default_topic: str,
    topic_help: str = "Research topic or question",
    epilog_examples: str | None = None,
) -> ArgumentParser:
    """Create argument parser with standard test configuration.

    Creates a parser with a positional topic/theme argument that has a default,
    using RawDescriptionHelpFormatter for nice epilog formatting.

    Args:
        description: Parser description
        default_topic: Default value for the topic argument
        topic_help: Help text for the topic argument
        epilog_examples: Optional examples text for epilog

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog_examples,
    )

    parser.add_argument(
        "topic",
        nargs="?",
        default=default_topic,
        help=topic_help,
    )

    return parser


def add_quality_argument(
    parser: ArgumentParser,
    choices: list[str] | None = None,
    default: str = "quick",
) -> None:
    """Add quality level argument to parser.

    Args:
        parser: ArgumentParser to modify
        choices: Valid quality choices (default: quick, standard, comprehensive)
        default: Default quality level
    """
    if choices is None:
        choices = ["quick", "standard", "comprehensive"]

    parser.add_argument(
        "quality",
        nargs="?",
        default=default,
        choices=choices,
        help=f"Quality level (default: {default})",
    )


def add_language_argument(
    parser: ArgumentParser,
    default: str = "en",
) -> None:
    """Add language argument to parser.

    Args:
        parser: ArgumentParser to modify
        default: Default language code
    """
    parser.add_argument(
        "--language", "-l",
        type=str,
        default=default,
        help=(
            f"Language code (default: {default}). "
            "Supported: en, es, zh, ja, de, fr, pt, ko, ru, ar, it, nl, pl, "
            "tr, vi, th, id, hi, bn, sv, no, da, fi, cs, el, he, uk, ro, hu"
        ),
    )


def add_date_range_arguments(parser: ArgumentParser) -> None:
    """Add date range arguments to parser.

    Adds --from-year and --to-year optional arguments.

    Args:
        parser: ArgumentParser to modify
    """
    parser.add_argument(
        "--from-year",
        type=int,
        default=None,
        help="Start year for date filter",
    )
    parser.add_argument(
        "--to-year",
        type=int,
        default=None,
        help="End year for date filter",
    )


def add_research_questions_argument(parser: ArgumentParser) -> None:
    """Add research questions argument to parser.

    Args:
        parser: ArgumentParser to modify
    """
    parser.add_argument(
        "--questions", "-q",
        type=str,
        nargs="+",
        default=None,
        help="Research questions (if not provided, will be auto-generated)",
    )


def add_translation_arguments(parser: ArgumentParser) -> None:
    """Add translation-related arguments to parser.

    Args:
        parser: ArgumentParser to modify
    """
    lang_group = parser.add_argument_group("Language Options")
    lang_group.add_argument(
        "--translate-to", "-t",
        type=str,
        default=None,
        help="Translate final report to this language (ISO 639-1 code)",
    )
    lang_group.add_argument(
        "--preserve-quotes",
        action="store_true",
        default=True,
        help="Preserve direct quotes in original language when translating (default: True)",
    )
    lang_group.add_argument(
        "--no-preserve-quotes",
        action="store_false",
        dest="preserve_quotes",
        help="Translate quotes along with the report",
    )
