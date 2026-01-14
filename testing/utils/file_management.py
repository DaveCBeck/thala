"""
File management utilities for test scripts.

Provides consistent handling of output directories and saving results/reports.
Logging configuration is now handled centrally via core.config.configure_logging().
"""

import json
from datetime import datetime
from pathlib import Path

# Base testing directory
TESTING_DIR = Path(__file__).parent.parent


def get_log_dir() -> Path:
    """Get the project log directory.

    Note: Logs are now stored in ./logs/ at project root via configure_logging().
    This function returns that path for reference.

    Returns:
        Path to logs directory
    """
    log_dir = TESTING_DIR.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    return log_dir


def get_output_dir(subdir: str = "test_data") -> Path:
    """Get/create output directory.

    Args:
        subdir: Subdirectory name under testing/

    Returns:
        Path to output directory
    """
    output_dir = TESTING_DIR / subdir
    output_dir.mkdir(exist_ok=True)
    return output_dir


def save_json_result(
    data: dict,
    prefix: str,
    output_dir: Path | None = None,
) -> Path:
    """Save data as timestamped JSON file.

    Args:
        data: Dictionary to save
        prefix: Filename prefix (e.g., "research_result")
        output_dir: Output directory (default: testing/test_data)

    Returns:
        Path to saved file
    """
    if output_dir is None:
        output_dir = get_output_dir()

    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"{prefix}_{timestamp}.json"

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return filepath


def save_markdown_report(
    content: str,
    prefix: str,
    title: str | None = None,
    metadata: dict | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Save content as timestamped markdown file.

    Args:
        content: Markdown content to save
        prefix: Filename prefix (e.g., "research_report")
        title: Optional title for the document header
        metadata: Optional metadata dict to include (e.g., {"quality": "standard"})
        output_dir: Output directory (default: testing/test_data)

    Returns:
        Path to saved file
    """
    if output_dir is None:
        output_dir = get_output_dir()

    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"{prefix}_{timestamp}.md"

    with open(filepath, "w") as f:
        if title:
            f.write(f"# {title}\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        if metadata:
            meta_str = " | ".join(f"{k}: {v}" for k, v in metadata.items())
            f.write(f"*{meta_str}*\n\n")
        if title or metadata:
            f.write("---\n\n")
        f.write(content)

    return filepath
