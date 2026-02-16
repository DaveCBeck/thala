"""Async captcha solver wrapping python3-capsolver."""

import asyncio
import logging

import httpx

from .config import CapsolverConfig
from .errors import CaptchaSolveError
from .types import CaptchaType, DetectedCaptcha

from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


class CaptchaSolver:
    """Async captcha solver using CapSolver API.

    Supports reCAPTCHA v2/v3, hCaptcha, and Cloudflare Turnstile.
    Uses ProxyLess task types (CapSolver provides its own proxies).
    """

    def __init__(self, config: CapsolverConfig | None = None):
        self._config = config or CapsolverConfig()
        if not self._config.available:
            raise ValueError("CAPSOLVER_API_KEY is not set")
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent)
        self._http_client: httpx.AsyncClient | None = None

    def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create a shared httpx client for CapSolver API calls."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=self._config.timeout)
        return self._http_client

    async def _solve_with_retry(self, solve_fn: Callable[[], Awaitable[dict]]) -> str:
        """Execute a solve function with semaphore and one retry on UNSOLVABLE."""
        async with self._semaphore:
            for attempt in range(2):
                result = await solve_fn()
                if result.get("errorId", 0) == 0:
                    # Extract token — reCAPTCHA uses gRecaptchaResponse, others use token
                    solution = result.get("solution", {})
                    token = solution.get("gRecaptchaResponse") or solution.get("token", "")
                    if not token:
                        raise CaptchaSolveError("ERROR_NO_TOKEN", "Solution contained no token")
                    return token

                error_code = result.get("errorCode", "UNKNOWN")
                error_desc = result.get("errorDescription", "Unknown error")

                if error_code == "ERROR_CAPTCHA_UNSOLVABLE" and attempt == 0:
                    logger.debug("Captcha unsolvable, retrying (no charge)")
                    continue

                raise CaptchaSolveError(error_code, error_desc)

        # Unreachable, but satisfies type checker
        raise CaptchaSolveError("ERROR_RETRY_EXHAUSTED", "All attempts failed")

    async def _solve_via_api(self, task_type: str, payload: dict[str, str]) -> dict:
        """Solve captcha via raw CapSolver API (createTask + poll getTaskResult).

        Used for captcha types not wrapped by python3-capsolver (e.g. hCaptcha).
        Returns the same dict format as python3-capsolver SDK results.
        """
        api_base = "https://api.capsolver.com"
        task = {"type": task_type, **payload}
        client = self._get_http_client()

        # Create task
        resp = await client.post(
            f"{api_base}/createTask",
            json={"clientKey": self._config.api_key, "task": task},
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("errorId", 0) != 0:
            return data  # Let _solve_with_retry handle the error

        task_id = data.get("taskId")
        if not task_id:
            return {"errorId": 1, "errorCode": "ERROR_NO_TASK_ID", "errorDescription": "No taskId returned"}

        # Poll for result
        poll_interval = 3.0
        elapsed = 0.0
        while elapsed < self._config.timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            resp = await client.post(
                f"{api_base}/getTaskResult",
                json={"clientKey": self._config.api_key, "taskId": task_id},
            )
            resp.raise_for_status()
            result = resp.json()

            if result.get("errorId", 0) != 0:
                return result

            if result.get("status") == "ready":
                return result

        return {"errorId": 1, "errorCode": "ERROR_TASK_TIMEOUT", "errorDescription": "Polling timed out"}

    async def solve_recaptcha_v2(self, website_url: str, website_key: str) -> str:
        """Solve reCAPTCHA v2 and return the response token."""
        from python3_capsolver.recaptcha import ReCaptcha
        from python3_capsolver.core.enum import CaptchaTypeEnm

        logger.debug(f"Solving reCAPTCHA v2 for {website_url}")

        async def _solve():
            solver = ReCaptcha(
                api_key=self._config.api_key,
                captcha_type=CaptchaTypeEnm.ReCaptchaV2TaskProxyLess,
            )
            return await solver.aio_captcha_handler(
                task_payload={
                    "websiteURL": website_url,
                    "websiteKey": website_key,
                }
            )

        return await self._solve_with_retry(_solve)

    async def solve_recaptcha_v3(self, website_url: str, website_key: str, page_action: str | None = None) -> str:
        """Solve reCAPTCHA v3 and return the response token."""
        from python3_capsolver.recaptcha import ReCaptcha
        from python3_capsolver.core.enum import CaptchaTypeEnm

        logger.debug(f"Solving reCAPTCHA v3 for {website_url}")

        async def _solve():
            solver = ReCaptcha(
                api_key=self._config.api_key,
                captcha_type=CaptchaTypeEnm.ReCaptchaV3TaskProxyLess,
            )
            payload = {
                "websiteURL": website_url,
                "websiteKey": website_key,
            }
            if page_action:
                payload["pageAction"] = page_action
            return await solver.aio_captcha_handler(task_payload=payload)

        return await self._solve_with_retry(_solve)

    async def solve_hcaptcha(self, website_url: str, website_key: str) -> str:
        """Solve hCaptcha and return the response token.

        Uses raw CapSolver API calls since python3-capsolver doesn't wrap hCaptcha.
        """
        logger.debug(f"Solving hCaptcha for {website_url}")

        async def _solve():
            return await self._solve_via_api(
                task_type="HCaptchaTaskProxyLess",
                payload={"websiteURL": website_url, "websiteKey": website_key},
            )

        return await self._solve_with_retry(_solve)

    async def solve_turnstile(self, website_url: str, website_key: str) -> str:
        """Solve Cloudflare Turnstile and return the response token."""
        from python3_capsolver.cloudflare import Cloudflare
        from python3_capsolver.core.enum import CaptchaTypeEnm

        logger.debug(f"Solving Turnstile for {website_url}")

        async def _solve():
            solver = Cloudflare(
                api_key=self._config.api_key,
                captcha_type=CaptchaTypeEnm.AntiTurnstileTaskProxyLess,
            )
            return await solver.aio_captcha_handler(
                task_payload={
                    "websiteURL": website_url,
                    "websiteKey": website_key,
                }
            )

        return await self._solve_with_retry(_solve)

    async def solve(self, detected: DetectedCaptcha) -> str:
        """Dispatch to the correct solver based on detected captcha type."""
        url, key = detected.page_url, detected.site_key
        if detected.captcha_type == CaptchaType.RECAPTCHA_V2:
            return await self.solve_recaptcha_v2(url, key)
        if detected.captcha_type == CaptchaType.RECAPTCHA_V3:
            return await self.solve_recaptcha_v3(url, key, detected.action)
        if detected.captcha_type == CaptchaType.HCAPTCHA:
            return await self.solve_hcaptcha(url, key)
        if detected.captcha_type == CaptchaType.TURNSTILE:
            return await self.solve_turnstile(url, key)
        raise CaptchaSolveError(
            "ERROR_UNSUPPORTED_TYPE",
            f"Unsupported captcha type: {detected.captcha_type}",
        )
