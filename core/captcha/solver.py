"""Async captcha solver wrapping python3-capsolver."""

import asyncio
import logging

import httpx

from .config import CapsolverConfig
from .types import CaptchaType, DetectedCaptcha

logger = logging.getLogger(__name__)


class CaptchaSolveError(Exception):
    """Captcha solve failed permanently."""

    def __init__(self, error_code: str, error_description: str):
        self.error_code = error_code
        self.error_description = error_description
        super().__init__(f"{error_code}: {error_description}")


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

    async def _solve_with_retry(self, solve_fn) -> str:
        """Execute a solve function with semaphore and one retry on UNSOLVABLE."""
        async with self._semaphore:
            for attempt in range(2):
                result = await solve_fn()
                if result.get("errorId", 0) == 0:
                    # Extract token — reCAPTCHA uses gRecaptchaResponse, others use token
                    solution = result.get("solution", {})
                    return solution.get("gRecaptchaResponse") or solution.get("token") or ""

                error_code = result.get("errorCode", "UNKNOWN")
                error_desc = result.get("errorDescription", "Unknown error")

                if error_code == "ERROR_CAPTCHA_UNSOLVABLE" and attempt == 0:
                    logger.debug("Captcha unsolvable, retrying (no charge)")
                    continue

                raise CaptchaSolveError(error_code, error_desc)

        # Unreachable, but satisfies type checker
        raise CaptchaSolveError("ERROR_RETRY_EXHAUSTED", "All attempts failed")

    async def _solve_via_api(self, task_type: str, payload: dict) -> dict:
        """Solve captcha via raw CapSolver API (createTask + poll getTaskResult).

        Used for captcha types not wrapped by python3-capsolver (e.g. hCaptcha).
        Returns the same dict format as python3-capsolver SDK results.
        """
        api_base = "https://api.capsolver.com"
        task = {"type": task_type, **payload}

        async with httpx.AsyncClient(timeout=self._config.timeout) as client:
            # Create task
            resp = await client.post(
                f"{api_base}/createTask",
                json={"clientKey": self._config.api_key, "task": task},
            )
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
        dispatch = {
            CaptchaType.RECAPTCHA_V2: lambda: self.solve_recaptcha_v2(detected.page_url, detected.site_key),
            CaptchaType.RECAPTCHA_V3: lambda: self.solve_recaptcha_v3(
                detected.page_url, detected.site_key, detected.action
            ),
            CaptchaType.HCAPTCHA: lambda: self.solve_hcaptcha(detected.page_url, detected.site_key),
            CaptchaType.TURNSTILE: lambda: self.solve_turnstile(detected.page_url, detected.site_key),
        }
        solver_fn = dispatch.get(detected.captcha_type)
        if solver_fn is None:
            raise CaptchaSolveError(
                "ERROR_UNSUPPORTED_TYPE",
                f"Unsupported captcha type: {detected.captcha_type}",
            )
        return await solver_fn()
