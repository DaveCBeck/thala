"""Tests for core.scraping.captcha_detection."""

from unittest.mock import AsyncMock

import pytest

from core.captcha.types import CaptchaType
from core.scraping.captcha_detection import detect_captcha, inject_captcha_token


def _mock_page(evaluate_result, url="https://example.com"):
    """Create a mock Playwright page with a given evaluate() result."""
    page = AsyncMock()
    page.evaluate = AsyncMock(return_value=evaluate_result)
    page.url = url
    return page


class TestDetectCaptcha:
    @pytest.mark.asyncio
    async def test_detects_recaptcha_v2(self):
        page = _mock_page({"type": "recaptcha_v2", "siteKey": "6Le-abc"})
        result = await detect_captcha(page)
        assert result is not None
        assert result.captcha_type == CaptchaType.RECAPTCHA_V2
        assert result.site_key == "6Le-abc"
        assert result.page_url == "https://example.com"

    @pytest.mark.asyncio
    async def test_detects_recaptcha_v3(self):
        page = _mock_page({"type": "recaptcha_v3", "siteKey": "v3-key"})
        result = await detect_captcha(page)
        assert result is not None
        assert result.captcha_type == CaptchaType.RECAPTCHA_V3
        assert result.site_key == "v3-key"

    @pytest.mark.asyncio
    async def test_detects_hcaptcha(self):
        page = _mock_page({"type": "hcaptcha", "siteKey": "hc-key"})
        result = await detect_captcha(page)
        assert result is not None
        assert result.captcha_type == CaptchaType.HCAPTCHA

    @pytest.mark.asyncio
    async def test_detects_turnstile(self):
        page = _mock_page({"type": "turnstile", "siteKey": "ts-key"})
        result = await detect_captcha(page)
        assert result is not None
        assert result.captcha_type == CaptchaType.TURNSTILE

    @pytest.mark.asyncio
    async def test_returns_none_for_no_captcha(self):
        page = _mock_page(None)
        result = await detect_captcha(page)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_result(self):
        page = _mock_page({})
        result = await detect_captcha(page)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_missing_site_key(self):
        page = _mock_page({"type": "recaptcha_v2", "siteKey": ""})
        result = await detect_captcha(page)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_for_unknown_type(self):
        page = _mock_page({"type": "unknown_captcha", "siteKey": "key-123"})
        result = await detect_captcha(page)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_js_error(self):
        page = AsyncMock()
        page.evaluate = AsyncMock(side_effect=Exception("JS crashed"))
        page.url = "https://example.com"
        result = await detect_captcha(page)
        assert result is None


class TestInjectCaptchaToken:
    @pytest.mark.asyncio
    async def test_injects_recaptcha_token(self):
        page = AsyncMock()
        from core.captcha.types import DetectedCaptcha

        detected = DetectedCaptcha(
            captcha_type=CaptchaType.RECAPTCHA_V2,
            site_key="key",
            page_url="https://example.com",
        )
        await inject_captcha_token(page, detected, "solved-token")
        page.evaluate.assert_called_once()
        # Verify the token was passed as the second argument
        assert page.evaluate.call_args[0][1] == "solved-token"

    @pytest.mark.asyncio
    async def test_injects_hcaptcha_token(self):
        page = AsyncMock()
        from core.captcha.types import DetectedCaptcha

        detected = DetectedCaptcha(
            captcha_type=CaptchaType.HCAPTCHA,
            site_key="key",
            page_url="https://example.com",
        )
        await inject_captcha_token(page, detected, "hc-solved")
        page.evaluate.assert_called_once()
        assert page.evaluate.call_args[0][1] == "hc-solved"

    @pytest.mark.asyncio
    async def test_injects_turnstile_token(self):
        page = AsyncMock()
        from core.captcha.types import DetectedCaptcha

        detected = DetectedCaptcha(
            captcha_type=CaptchaType.TURNSTILE,
            site_key="key",
            page_url="https://example.com",
        )
        await inject_captcha_token(page, detected, "ts-solved")
        page.evaluate.assert_called_once()
        assert page.evaluate.call_args[0][1] == "ts-solved"

    @pytest.mark.asyncio
    async def test_raises_on_injection_failure(self):
        page = AsyncMock()
        page.evaluate = AsyncMock(side_effect=Exception("Injection failed"))
        from core.captcha.types import DetectedCaptcha

        detected = DetectedCaptcha(
            captcha_type=CaptchaType.RECAPTCHA_V2,
            site_key="key",
            page_url="https://example.com",
        )
        with pytest.raises(Exception, match="Injection failed"):
            await inject_captcha_token(page, detected, "token")
