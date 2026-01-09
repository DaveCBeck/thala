"""
Shared testing utilities for workflow tests.

This module provides common functionality used across test scripts:
- datetime_utils: Duration formatting, ISO parsing
- file_management: Logging setup, JSON/MD saving
- checkpoint: Save/load workflow checkpoints
- cli_parser: Argument parser factory and helpers
- result_display: Generic result summary formatting
- quality_analyzer: Base quality analysis framework
"""

from .datetime_utils import (
    parse_iso_datetime,
    make_naive,
    format_duration,
    calculate_duration_seconds,
)
from .file_management import (
    setup_logging,
    get_output_dir,
    get_log_dir,
    save_json_result,
    save_markdown_report,
)
from .checkpoint import (
    get_checkpoint_dir,
    save_checkpoint,
    load_checkpoint,
    list_checkpoints,
)
from .cli_parser import (
    create_test_parser,
    add_quality_argument,
    add_language_argument,
    add_date_range_arguments,
    add_checkpoint_arguments,
    add_research_questions_argument,
    add_translation_arguments,
)
from .result_display import (
    print_section_header,
    safe_preview,
    print_timing,
    print_key_value,
    print_list_preview,
    print_errors,
)
from .quality_analyzer import (
    QualityMetrics,
    BaseQualityAnalyzer,
    print_quality_analysis,
)

__all__ = [
    # datetime_utils
    "parse_iso_datetime",
    "make_naive",
    "format_duration",
    "calculate_duration_seconds",
    # file_management
    "setup_logging",
    "get_output_dir",
    "get_log_dir",
    "save_json_result",
    "save_markdown_report",
    # checkpoint
    "get_checkpoint_dir",
    "save_checkpoint",
    "load_checkpoint",
    "list_checkpoints",
    # cli_parser
    "create_test_parser",
    "add_quality_argument",
    "add_language_argument",
    "add_date_range_arguments",
    "add_checkpoint_arguments",
    "add_research_questions_argument",
    "add_translation_arguments",
    # result_display
    "print_section_header",
    "safe_preview",
    "print_timing",
    "print_key_value",
    "print_list_preview",
    "print_errors",
    # quality_analyzer
    "QualityMetrics",
    "BaseQualityAnalyzer",
    "print_quality_analysis",
]
