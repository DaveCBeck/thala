"""Playwright-specific captcha detection and token injection.

Lives in core/scraping/ (not core/captcha/) to keep the captcha package
free of Playwright dependencies.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.captcha.types import CaptchaType, DetectedCaptcha

if TYPE_CHECKING:
    from playwright.async_api import Page

logger = logging.getLogger(__name__)

# JavaScript to detect captcha type and extract site key from the page DOM.
_DETECT_JS = r"""
() => {
    // reCAPTCHA v2/v3
    const recaptchaEl = document.querySelector('[data-sitekey]');
    if (recaptchaEl && (
        document.querySelector('.g-recaptcha') ||
        document.querySelector('script[src*="recaptcha"]') ||
        typeof window.grecaptcha !== 'undefined'
    )) {
        const siteKey = recaptchaEl.getAttribute('data-sitekey');
        const size = recaptchaEl.getAttribute('data-size');
        return {
            type: size === 'invisible' ? 'recaptcha_v2' : 'recaptcha_v2',
            siteKey: siteKey
        };
    }

    // Check for reCAPTCHA via script tags (no data-sitekey element)
    const recaptchaScript = document.querySelector('script[src*="recaptcha/api.js"]') ||
                            document.querySelector('script[src*="recaptcha/enterprise.js"]');
    if (recaptchaScript || typeof window.grecaptcha !== 'undefined') {
        // Try to find site key from grecaptcha.render calls or inline scripts
        const scripts = document.querySelectorAll('script:not([src])');
        for (const script of scripts) {
            const match = script.textContent.match(/['"]sitekey['"]\s*:\s*['"]([^'"]+)['"]/);
            if (match) return { type: 'recaptcha_v2', siteKey: match[1] };
        }
        // Check for reCAPTCHA v3 render call
        for (const script of scripts) {
            const match = script.textContent.match(/grecaptcha\\.execute\\s*\\(['"]([^'"]+)['"]/);
            if (match) return { type: 'recaptcha_v3', siteKey: match[1] };
        }
    }

    // hCaptcha
    const hcaptchaEl = document.querySelector('.h-captcha[data-sitekey]');
    if (hcaptchaEl) {
        return { type: 'hcaptcha', siteKey: hcaptchaEl.getAttribute('data-sitekey') };
    }
    if (document.querySelector('script[src*="hcaptcha"]') ||
        typeof window.hcaptcha !== 'undefined') {
        const el = document.querySelector('[data-sitekey]');
        if (el) return { type: 'hcaptcha', siteKey: el.getAttribute('data-sitekey') };
    }

    // Cloudflare Turnstile
    const turnstileEl = document.querySelector('.cf-turnstile[data-sitekey]');
    if (turnstileEl) {
        return { type: 'turnstile', siteKey: turnstileEl.getAttribute('data-sitekey') };
    }
    if (document.querySelector('script[src*="turnstile"]') ||
        typeof window.turnstile !== 'undefined') {
        const el = document.querySelector('[data-sitekey]');
        if (el) return { type: 'turnstile', siteKey: el.getAttribute('data-sitekey') };
    }

    return null;
}
"""

# JavaScript templates for injecting solved tokens.
_INJECT_RECAPTCHA = """
(token) => {
    // Set the response textarea
    const textarea = document.querySelector('#g-recaptcha-response') ||
                     document.querySelector('[name="g-recaptcha-response"]');
    if (textarea) {
        textarea.style.display = 'block';
        textarea.value = token;
        textarea.style.display = 'none';
    }
    // Trigger the callback
    if (typeof window.grecaptcha !== 'undefined') {
        try {
            // v2 callback
            const widgetId = window.grecaptcha.getResponse ? 0 : null;
            if (window.___grecaptcha_cfg && window.___grecaptcha_cfg.clients) {
                for (const client of Object.values(window.___grecaptcha_cfg.clients)) {
                    for (const key of Object.keys(client)) {
                        const val = client[key];
                        if (val && typeof val === 'object') {
                            for (const prop of Object.keys(val)) {
                                if (typeof val[prop] === 'function') {
                                    try { val[prop](token); } catch(e) {}
                                }
                            }
                        }
                    }
                }
            }
        } catch(e) {}
    }
    // Also try direct callback attribute
    const el = document.querySelector('[data-callback]');
    if (el) {
        const cbName = el.getAttribute('data-callback');
        if (typeof window[cbName] === 'function') window[cbName](token);
    }
}
"""

_INJECT_HCAPTCHA = """
(token) => {
    const textarea = document.querySelector('[name="h-captcha-response"]') ||
                     document.querySelector('textarea[name*="hcaptcha"]');
    if (textarea) textarea.value = token;
    // Also set g-recaptcha-response (hCaptcha compatibility)
    const gTextarea = document.querySelector('[name="g-recaptcha-response"]');
    if (gTextarea) gTextarea.value = token;
    // Trigger callback
    if (typeof window.hcaptcha !== 'undefined') {
        try {
            const iframe = document.querySelector('iframe[src*="hcaptcha"]');
            if (iframe) {
                const event = new CustomEvent('hcaptcha-response', { detail: token });
                iframe.dispatchEvent(event);
            }
        } catch(e) {}
    }
    const el = document.querySelector('[data-callback]');
    if (el) {
        const cbName = el.getAttribute('data-callback');
        if (typeof window[cbName] === 'function') window[cbName](token);
    }
}
"""

_INJECT_TURNSTILE = """
(token) => {
    // Set hidden input
    const input = document.querySelector('[name="cf-turnstile-response"]') ||
                  document.querySelector('input[name*="turnstile"]');
    if (input) input.value = token;
    // Trigger callback
    const el = document.querySelector('.cf-turnstile[data-callback]');
    if (el) {
        const cbName = el.getAttribute('data-callback');
        if (typeof window[cbName] === 'function') window[cbName](token);
    }
    if (typeof window.turnstile !== 'undefined') {
        try { window.turnstile.getResponse = () => token; } catch(e) {}
    }
}
"""

_TYPE_MAP = {
    "recaptcha_v2": CaptchaType.RECAPTCHA_V2,
    "recaptcha_v3": CaptchaType.RECAPTCHA_V3,
    "hcaptcha": CaptchaType.HCAPTCHA,
    "turnstile": CaptchaType.TURNSTILE,
}

_INJECT_MAP = {
    CaptchaType.RECAPTCHA_V2: _INJECT_RECAPTCHA,
    CaptchaType.RECAPTCHA_V3: _INJECT_RECAPTCHA,
    CaptchaType.HCAPTCHA: _INJECT_HCAPTCHA,
    CaptchaType.TURNSTILE: _INJECT_TURNSTILE,
}


async def detect_captcha(page: "Page") -> DetectedCaptcha | None:
    """Detect captcha type and extract site key from a Playwright page."""
    try:
        result = await page.evaluate(_DETECT_JS)
    except Exception as e:
        logger.debug(f"Captcha detection JS failed: {e}")
        return None

    if not result or not result.get("siteKey"):
        return None

    captcha_type = _TYPE_MAP.get(result["type"])
    if captcha_type is None:
        logger.debug(f"Unknown captcha type from detection: {result['type']}")
        return None

    detected = DetectedCaptcha(
        captcha_type=captcha_type,
        site_key=result["siteKey"],
        page_url=page.url,
    )
    logger.info(f"Detected {detected.captcha_type.value} captcha on {page.url}")
    return detected


async def inject_captcha_token(page: "Page", detected: DetectedCaptcha, token: str) -> None:
    """Inject solved captcha token into the page and trigger submission."""
    inject_js = _INJECT_MAP.get(detected.captcha_type)
    if inject_js is None:
        logger.warning(f"No injection strategy for {detected.captcha_type}")
        return

    try:
        await page.evaluate(inject_js, token)
        logger.debug(f"Injected {detected.captcha_type.value} token")
    except Exception as e:
        logger.warning(f"Token injection failed: {e}")
        raise
