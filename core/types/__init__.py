"""Core type definitions.

This module provides foundational types used across core/ and workflows/ modules.
"""

from .model_tier import ModelTier, is_deepseek_tier, DEEPSEEK_TIERS

__all__ = [
    "ModelTier",
    "is_deepseek_tier",
    "DEEPSEEK_TIERS",
]
