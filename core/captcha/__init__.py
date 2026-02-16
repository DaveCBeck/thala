"""Captcha solving via CapSolver API."""

from .config import CapsolverConfig
from .solver import CaptchaSolveError, CaptchaSolver
from .types import CaptchaType, DetectedCaptcha

__all__ = [
    "CapsolverConfig",
    "CaptchaSolveError",
    "CaptchaSolver",
    "CaptchaType",
    "DetectedCaptcha",
]
