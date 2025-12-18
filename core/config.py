"""Thala configuration and environment setup.

This module provides centralized configuration for the Thala system,
including development mode detection and LangSmith tracing setup.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def is_dev_mode() -> bool:
    """Check if running in development mode.

    Returns:
        True if THALA_MODE is set to 'dev', False otherwise.
    """
    return os.getenv("THALA_MODE", "prod").lower() == "dev"


def configure_langsmith() -> None:
    """Configure LangSmith tracing based on THALA_MODE.

    When THALA_MODE=dev:
        - Enables LangSmith tracing
        - Sets project to 'thala-dev'

    When THALA_MODE=prod (or unset):
        - Disables LangSmith tracing

    This function is idempotent and safe to call multiple times.
    """
    if is_dev_mode():
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "thala-dev")
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
