"""Captcha type definitions shared across solver and detection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CaptchaType(str, Enum):
    RECAPTCHA_V2 = "recaptcha_v2"
    RECAPTCHA_V3 = "recaptcha_v3"
    HCAPTCHA = "hcaptcha"
    TURNSTILE = "turnstile"


@dataclass
class DetectedCaptcha:
    """Boundary object consumed by both solver and detection layers."""

    captcha_type: CaptchaType
    site_key: str
    page_url: str
    action: str | None = None  # For Turnstile/reCAPTCHA v3
