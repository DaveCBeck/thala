"""Captcha solving via CapSolver API."""

from .config import CapsolverConfig, get_capsolver_config
from .errors import CaptchaSolveError
from .solver import CaptchaSolver
from .types import CaptchaType, DetectedCaptcha

__all__ = [
    "CapsolverConfig",
    "CaptchaSolveError",
    "CaptchaSolver",
    "CaptchaType",
    "DetectedCaptcha",
    "get_capsolver_config",
]
