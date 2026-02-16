"""Tests for core.captcha.solver."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.captcha.config import CapsolverConfig
from core.captcha.solver import CaptchaSolveError, CaptchaSolver
from core.captcha.types import CaptchaType, DetectedCaptcha

# Patch targets — the solver imports these lazily inside each method,
# so we patch at the source module level.
_RECAPTCHA = "python3_capsolver.recaptcha.ReCaptcha"
_CLOUDFLARE = "python3_capsolver.cloudflare.Cloudflare"


@pytest.fixture
def config():
    return CapsolverConfig(api_key="test-key", timeout=10.0, max_concurrent=2)


@pytest.fixture
def solver(config):
    return CaptchaSolver(config)


def _success_result(token="test-token-123"):
    return {"errorId": 0, "solution": {"gRecaptchaResponse": token, "token": token}}


def _error_result(code="ERROR_CAPTCHA_UNSOLVABLE", desc="Not solvable"):
    return {"errorId": 1, "errorCode": code, "errorDescription": desc}


class TestCaptchaSolverInit:
    def test_raises_without_api_key(self):
        config = CapsolverConfig(api_key=None)
        with pytest.raises(ValueError, match="CAPSOLVER_API_KEY"):
            CaptchaSolver(config)

    def test_creates_with_valid_config(self, config):
        solver = CaptchaSolver(config)
        assert solver._semaphore._value == 2


class TestSolveRecaptchaV2:
    @pytest.mark.asyncio
    async def test_returns_token(self, solver):
        mock_handler = AsyncMock(return_value=_success_result("rv2-token"))
        with patch(_RECAPTCHA) as MockClass:
            MockClass.return_value.aio_captcha_handler = mock_handler
            token = await solver.solve_recaptcha_v2("https://example.com", "site-key-123")
        assert token == "rv2-token"
        mock_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_retries_on_unsolvable(self, solver):
        mock_handler = AsyncMock(side_effect=[_error_result(), _success_result("retry-token")])
        with patch(_RECAPTCHA) as MockClass:
            MockClass.return_value.aio_captcha_handler = mock_handler
            token = await solver.solve_recaptcha_v2("https://example.com", "site-key-123")
        assert token == "retry-token"
        assert mock_handler.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_on_permanent_error(self, solver):
        mock_handler = AsyncMock(return_value=_error_result("ERROR_ZERO_BALANCE", "No balance"))
        with patch(_RECAPTCHA) as MockClass:
            MockClass.return_value.aio_captcha_handler = mock_handler
            with pytest.raises(CaptchaSolveError, match="ERROR_ZERO_BALANCE"):
                await solver.solve_recaptcha_v2("https://example.com", "site-key-123")

    @pytest.mark.asyncio
    async def test_raises_after_two_unsolvable(self, solver):
        mock_handler = AsyncMock(side_effect=[_error_result(), _error_result()])
        with patch(_RECAPTCHA) as MockClass:
            MockClass.return_value.aio_captcha_handler = mock_handler
            with pytest.raises(CaptchaSolveError, match="ERROR_CAPTCHA_UNSOLVABLE"):
                await solver.solve_recaptcha_v2("https://example.com", "site-key-123")
        assert mock_handler.call_count == 2


class TestSolveRecaptchaV3:
    @pytest.mark.asyncio
    async def test_returns_token(self, solver):
        mock_handler = AsyncMock(return_value=_success_result("rv3-token"))
        with patch(_RECAPTCHA) as MockClass:
            MockClass.return_value.aio_captcha_handler = mock_handler
            token = await solver.solve_recaptcha_v3("https://example.com", "site-key-123", page_action="login")
        assert token == "rv3-token"


class TestSolveHCaptcha:
    @pytest.mark.asyncio
    async def test_returns_token(self, solver):
        # hCaptcha uses raw API calls — mock _solve_via_api
        with patch.object(
            solver,
            "_solve_via_api",
            new_callable=AsyncMock,
            return_value=_success_result("hc-token"),
        ):
            token = await solver.solve_hcaptcha("https://example.com", "hc-key")
        assert token == "hc-token"


class TestSolveTurnstile:
    @pytest.mark.asyncio
    async def test_returns_token(self, solver):
        mock_handler = AsyncMock(return_value=_success_result("ts-token"))
        with patch(_CLOUDFLARE) as MockClass:
            MockClass.return_value.aio_captcha_handler = mock_handler
            token = await solver.solve_turnstile("https://example.com", "ts-key")
        assert token == "ts-token"


class TestSolveDispatcher:
    @pytest.mark.asyncio
    async def test_dispatches_recaptcha_v2(self, solver):
        detected = DetectedCaptcha(
            captcha_type=CaptchaType.RECAPTCHA_V2,
            site_key="key-123",
            page_url="https://example.com",
        )
        mock_handler = AsyncMock(return_value=_success_result("dispatched"))
        with patch(_RECAPTCHA) as MockClass:
            MockClass.return_value.aio_captcha_handler = mock_handler
            token = await solver.solve(detected)
        assert token == "dispatched"

    @pytest.mark.asyncio
    async def test_dispatches_turnstile(self, solver):
        detected = DetectedCaptcha(
            captcha_type=CaptchaType.TURNSTILE,
            site_key="ts-key",
            page_url="https://example.com",
        )
        mock_handler = AsyncMock(return_value=_success_result("ts-dispatched"))
        with patch(_CLOUDFLARE) as MockClass:
            MockClass.return_value.aio_captcha_handler = mock_handler
            token = await solver.solve(detected)
        assert token == "ts-dispatched"

    @pytest.mark.asyncio
    async def test_dispatches_hcaptcha(self, solver):
        detected = DetectedCaptcha(
            captcha_type=CaptchaType.HCAPTCHA,
            site_key="hc-key",
            page_url="https://example.com",
        )
        with patch.object(
            solver,
            "_solve_via_api",
            new_callable=AsyncMock,
            return_value=_success_result("hc-dispatched"),
        ):
            token = await solver.solve(detected)
        assert token == "hc-dispatched"

    @pytest.mark.asyncio
    async def test_dispatches_recaptcha_v3_with_action(self, solver):
        detected = DetectedCaptcha(
            captcha_type=CaptchaType.RECAPTCHA_V3,
            site_key="v3-key",
            page_url="https://example.com",
            action="submit",
        )
        mock_handler = AsyncMock(return_value=_success_result("v3-dispatched"))
        with patch(_RECAPTCHA) as MockClass:
            MockClass.return_value.aio_captcha_handler = mock_handler
            token = await solver.solve(detected)
        assert token == "v3-dispatched"


class TestSemaphore:
    @pytest.mark.asyncio
    async def test_limits_concurrent_solves(self):
        config = CapsolverConfig(api_key="test-key", max_concurrent=1)
        solver = CaptchaSolver(config)

        call_order = []

        async def slow_solve(**kwargs):
            call_order.append("start")
            await asyncio.sleep(0.05)
            call_order.append("end")
            return _success_result()

        with patch(_RECAPTCHA) as MockClass:
            MockClass.return_value.aio_captcha_handler = AsyncMock(side_effect=slow_solve)

            # Launch two concurrent solves with max_concurrent=1
            results = await asyncio.gather(
                solver.solve_recaptcha_v2("https://a.com", "key"),
                solver.solve_recaptcha_v2("https://b.com", "key"),
            )

        # With semaphore=1, they should serialize: start, end, start, end
        assert call_order == ["start", "end", "start", "end"]
        assert all(r == "test-token-123" for r in results)


class TestSolveViaApi:
    """Direct tests for _solve_via_api polling logic."""

    @pytest.mark.asyncio
    async def test_successful_solve(self, solver):
        """Task created and result returned on first poll."""
        create_resp = MagicMock()
        create_resp.json.return_value = {"errorId": 0, "taskId": "task-123"}
        create_resp.raise_for_status = MagicMock()

        result_resp = MagicMock()
        result_resp.json.return_value = {
            "errorId": 0,
            "status": "ready",
            "solution": {"token": "solved-token"},
        }
        result_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[create_resp, result_resp])
        mock_client.is_closed = False

        with patch.object(solver, "_get_http_client", return_value=mock_client):
            result = await solver._solve_via_api("HCaptchaTaskProxyLess", {"websiteURL": "https://example.com", "websiteKey": "key"})

        assert result["status"] == "ready"
        assert result["solution"]["token"] == "solved-token"
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_polls_until_ready(self, solver):
        """Polls multiple times when status is 'processing'."""
        create_resp = MagicMock()
        create_resp.json.return_value = {"errorId": 0, "taskId": "task-456"}
        create_resp.raise_for_status = MagicMock()

        processing_resp = MagicMock()
        processing_resp.json.return_value = {"errorId": 0, "status": "processing"}
        processing_resp.raise_for_status = MagicMock()

        ready_resp = MagicMock()
        ready_resp.json.return_value = {"errorId": 0, "status": "ready", "solution": {"token": "t"}}
        ready_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[create_resp, processing_resp, ready_resp])
        mock_client.is_closed = False

        with patch.object(solver, "_get_http_client", return_value=mock_client), \
             patch("core.captcha.solver.asyncio.sleep", new_callable=AsyncMock):
            result = await solver._solve_via_api("HCaptchaTaskProxyLess", {"websiteURL": "https://example.com", "websiteKey": "key"})

        assert result["status"] == "ready"
        assert mock_client.post.call_count == 3  # create + 2 polls

    @pytest.mark.asyncio
    async def test_returns_error_on_create_failure(self, solver):
        """Returns error dict when createTask fails."""
        create_resp = MagicMock()
        create_resp.json.return_value = {"errorId": 1, "errorCode": "ERROR_KEY_DENIED", "errorDescription": "Bad key"}
        create_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=create_resp)
        mock_client.is_closed = False

        with patch.object(solver, "_get_http_client", return_value=mock_client):
            result = await solver._solve_via_api("HCaptchaTaskProxyLess", {"websiteURL": "https://example.com", "websiteKey": "key"})

        assert result["errorCode"] == "ERROR_KEY_DENIED"
        assert mock_client.post.call_count == 1

    @pytest.mark.asyncio
    async def test_returns_error_on_missing_task_id(self, solver):
        """Returns error when createTask succeeds but no taskId is returned."""
        create_resp = MagicMock()
        create_resp.json.return_value = {"errorId": 0}  # No taskId
        create_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=create_resp)
        mock_client.is_closed = False

        with patch.object(solver, "_get_http_client", return_value=mock_client):
            result = await solver._solve_via_api("HCaptchaTaskProxyLess", {"websiteURL": "https://example.com", "websiteKey": "key"})

        assert result["errorCode"] == "ERROR_NO_TASK_ID"

    @pytest.mark.asyncio
    async def test_timeout_returns_error(self, solver):
        """Returns timeout error when polling exceeds config timeout."""
        # Config has timeout=10.0, poll_interval=3.0, so 3 polls max (3+3+3=9 < 10, 4th would be 12 > 10)
        create_resp = MagicMock()
        create_resp.json.return_value = {"errorId": 0, "taskId": "task-789"}
        create_resp.raise_for_status = MagicMock()

        processing_resp = MagicMock()
        processing_resp.json.return_value = {"errorId": 0, "status": "processing"}
        processing_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=[create_resp] + [processing_resp] * 10)
        mock_client.is_closed = False

        with patch.object(solver, "_get_http_client", return_value=mock_client), \
             patch("core.captcha.solver.asyncio.sleep", new_callable=AsyncMock):
            result = await solver._solve_via_api("HCaptchaTaskProxyLess", {"websiteURL": "https://example.com", "websiteKey": "key"})

        assert result["errorCode"] == "ERROR_TASK_TIMEOUT"

    @pytest.mark.asyncio
    async def test_http_error_raises(self, solver):
        """HTTP errors (500, 502) propagate as httpx.HTTPStatusError."""
        import httpx as httpx_mod

        create_resp = MagicMock()
        create_resp.raise_for_status.side_effect = httpx_mod.HTTPStatusError(
            "Server Error", request=MagicMock(), response=MagicMock(status_code=500)
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=create_resp)
        mock_client.is_closed = False

        with patch.object(solver, "_get_http_client", return_value=mock_client):
            with pytest.raises(httpx_mod.HTTPStatusError):
                await solver._solve_via_api("HCaptchaTaskProxyLess", {"websiteURL": "https://example.com", "websiteKey": "key"})
