---
name: captcha-solving-integration
title: "CapSolver Captcha Solving Integration"
date: 2026-02-16
category: data-pipeline
applicability:
  - "When Playwright scraping encounters captcha challenges (reCAPTCHA v2/v3, hCaptcha, Turnstile)"
  - "When automated authentication requires solving captchas programmatically"
  - "When captcha failures must be distinguished from permanent site blocks"
components: [capsolver, playwright, javascript, asyncio, httpx, substack]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [captcha, capsolver, playwright, scraping, authentication, anti-bot, recaptcha, hcaptcha, turnstile]
---

# CapSolver Captcha Solving Integration

## Intent

Provide a gracefully degrading captcha solving system that separates detection/injection from solving logic, enabling automated navigation of captcha-protected sites while maintaining clean dependency boundaries between the captcha solver (pure async) and browser automation (Playwright-specific).

## Motivation

1. **Substack authentication**: Substack's sign-in flow intermittently requires reCAPTCHA v2, blocking automated publishing workflows that use email/password login.

2. **Playwright scraping**: Content platforms deploy captchas (reCAPTCHA, hCaptcha, Turnstile) that halt the Playwright fallback tier of the scraping service.

3. **Error classification**: Captcha failures are transient (CapSolver downtime, balance issues) and should not trigger the same domain blocklisting as permanent site blocks.

4. **Dependency isolation**: Captcha solving logic should not depend on Playwright, allowing reuse in direct HTTP authentication flows.

5. **Cost control**: CapSolver charges per solve, requiring concurrency limits and smart retry logic (free retries on UNSOLVABLE).

## Applicability

**Use this pattern when:**

- Automating interactions with sites that present captchas intermittently
- Multiple captcha types (reCAPTCHA v2/v3, hCaptcha, Turnstile) must be handled with a unified interface
- Concurrency control and retry logic are needed for a paid solving service
- Systems must degrade gracefully when captcha solving is unavailable

**Do NOT use this pattern when:**

- Authentication is available via API tokens or cookies
- Captcha solving latency (15-45 seconds) is unacceptable
- Only a single known captcha type is encountered (direct SDK call is simpler)

## Structure

```
┌─────────────────────────────────────────────────────────────┐
│                     Application Layer                        │
│  SubstackPublisher._login_with_captcha()                    │
│  PlaywrightScraper._handle_captcha()                        │
│  ScraperService.scrape() [CaptchaSolveFailedError handling] │
└───────────────┬─────────────────────────────┬───────────────┘
                │                             │
    ┌───────────▼───────────┐     ┌───────────▼───────────┐
    │    core/captcha/      │     │  core/scraping/       │
    │                       │     │  captcha_detection.py  │
    │  CaptchaSolver        │     │                       │
    │  CaptchaType          │     │  detect_captcha()     │
    │  DetectedCaptcha      │     │  inject_captcha_token()│
    │  CapsolverConfig      │     │                       │
    │  CaptchaSolveError    │     │  (Playwright-specific) │
    │                       │     └───────────────────────┘
    │  (NO Playwright dep)  │
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │   CapSolver API       │
    │   python3-capsolver   │
    │   + raw HTTP API      │
    └───────────────────────┘
```

**Integration flows:**

```
Playwright scraping:
  page.goto() → networkidle → detect_captcha(page) → solver.solve(detected)
  → inject_captcha_token(page, detected, token) → re-check → extract content

Substack auth:
  Api(email, password) fails with "captcha" → _login_with_captcha()
  → extract site_key from sign-in HTML → solver.solve_recaptcha_v2()
  → POST /api/v1/login with captcha_response → authenticated Api
```

## Implementation

### Step 1: Core Captcha Package (No Playwright Dependency)

Types and configuration shared across all consumers:

```python
# core/captcha/types.py
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
```

```python
# core/captcha/config.py
@dataclass
class CapsolverConfig:
    api_key: str | None = field(
        default_factory=lambda: os.environ.get("CAPSOLVER_API_KEY")
    )
    timeout: float = field(
        default_factory=lambda: float(os.environ.get("CAPSOLVER_TIMEOUT", "120.0"))
    )
    max_concurrent: int = field(
        default_factory=lambda: int(os.environ.get("CAPSOLVER_MAX_CONCURRENT", "3"))
    )

    @property
    def available(self) -> bool:
        return bool(self.api_key)
```

```python
# core/captcha/errors.py
class CaptchaSolveError(Exception):
    def __init__(self, error_code: str, error_description: str):
        self.error_code = error_code
        self.error_description = error_description
        super().__init__(f"{error_code}: {error_description}")
```

### Step 2: CaptchaSolver with Semaphore and Retry

Uses python3-capsolver SDK for reCAPTCHA and Turnstile, raw API for hCaptcha (SDK doesn't wrap it):

```python
# core/captcha/solver.py
class CaptchaSolver:
    def __init__(self, config: CapsolverConfig | None = None):
        self._config = config or CapsolverConfig()
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent)
        self._http_client: httpx.AsyncClient | None = None

    async def _solve_with_retry(self, solve_fn: Callable[[], Awaitable[dict]]) -> str:
        """Execute with semaphore and one retry on UNSOLVABLE."""
        async with self._semaphore:
            for attempt in range(2):
                result = await solve_fn()
                if result.get("errorId", 0) == 0:
                    solution = result.get("solution", {})
                    token = solution.get("gRecaptchaResponse") or solution.get("token", "")
                    if not token:
                        raise CaptchaSolveError("ERROR_NO_TOKEN", "Solution contained no token")
                    return token

                error_code = result.get("errorCode", "UNKNOWN")
                if error_code == "ERROR_CAPTCHA_UNSOLVABLE" and attempt == 0:
                    continue  # Free retry — CapSolver doesn't charge

                raise CaptchaSolveError(error_code, result.get("errorDescription", ""))
        raise CaptchaSolveError("ERROR_RETRY_EXHAUSTED", "All attempts failed")

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
        raise CaptchaSolveError("ERROR_UNSUPPORTED_TYPE", f"Unsupported: {detected.captcha_type}")
```

### Step 3: Playwright-Specific Detection and Injection

Lives in `core/scraping/` (not `core/captcha/`) to keep the captcha package free of Playwright dependencies:

```python
# core/scraping/captcha_detection.py
_DETECT_JS = r"""
() => {
    // reCAPTCHA v2/v3 — check data-sitekey, script tags, window.grecaptcha
    const recaptchaEl = document.querySelector('[data-sitekey]');
    if (recaptchaEl && (
        document.querySelector('.g-recaptcha') ||
        document.querySelector('script[src*="recaptcha"]') ||
        typeof window.grecaptcha !== 'undefined'
    )) {
        const size = recaptchaEl.getAttribute('data-size');
        return {
            type: size === 'invisible' ? 'recaptcha_v3' : 'recaptcha_v2',
            siteKey: recaptchaEl.getAttribute('data-sitekey')
        };
    }
    // hCaptcha — check .h-captcha[data-sitekey], script tags
    // Turnstile — check .cf-turnstile[data-sitekey], script tags
    return null;
}
"""

async def detect_captcha(page: "Page") -> DetectedCaptcha | None:
    result = await page.evaluate(_DETECT_JS)
    if not result or not result.get("siteKey"):
        return None
    return DetectedCaptcha(
        captcha_type=_TYPE_MAP[result["type"]],
        site_key=result["siteKey"],
        page_url=page.url,
    )

async def inject_captcha_token(page: "Page", detected: DetectedCaptcha, token: str) -> None:
    inject_js = _INJECT_MAP[detected.captcha_type]
    await page.evaluate(inject_js, token)
```

### Step 4: Integrate into PlaywrightScraper

```python
# core/scraping/playwright_scraper.py
async def _handle_captcha(self, page: "Page", url: str) -> None:
    """Detect and solve captcha on the current page if CapSolver is configured."""
    solver = self._get_solver()
    if solver is None:
        return  # Graceful degradation — no CAPSOLVER_API_KEY

    detected = await detect_captcha(page)
    if not detected:
        return

    try:
        async with asyncio.timeout(150):
            token = await solver.solve(detected)
            await inject_captcha_token(page, detected, token)
            await page.wait_for_load_state("networkidle", timeout=15000)

            # Re-check — some sites redirect after captcha
            if await detect_captcha(page):
                raise CaptchaSolveFailedError(
                    message="Still blocked after captcha injection",
                    url=url, provider="playwright",
                )
        logger.info("Captcha solved and page loaded")
    except TimeoutError:
        raise CaptchaSolveFailedError(
            message="Captcha solve+inject sequence timed out (150s)",
            url=url, provider="playwright",
        )
```

### Step 5: Integrate into Substack Authentication

```python
# utils/substack_publish/publisher.py
async def _login_with_captcha(self, email: str, password: str) -> Api | None:
    """Solve reCAPTCHA and login to Substack directly."""
    config = CapsolverConfig()
    if not config.available:
        return None  # Graceful degradation

    site_key = (
        os.getenv("SUBSTACK_RECAPTCHA_SITE_KEY")
        or _extract_substack_site_key()  # Fetches HTML, parses captcha_site_key
        or "6LdYbsYZAAAAAIFIRh8X_16GoFRLIReh-e-q6qSa"  # Hardcoded fallback
    )

    solver = CaptchaSolver(config)
    token = await solver.solve_recaptcha_v2(
        website_url="https://substack.com/sign-in",
        website_key=site_key,
    )

    # FRAGILE: Bypasses Api.login() which hardcodes captcha_response: None
    api = Api()
    resp = api._session.post(
        f"{api.base_url}/api/v1/login",
        json={
            "captcha_response": token,
            "email": email, "password": password,
            "for_pub": "", "redirect": "/",
        },
    )
    resp.raise_for_status()
    self._set_publication(api)
    return api
```

### Step 6: Distinguish Captcha Errors from Site Blocks

```python
# core/scraping/errors.py
class CaptchaSolveFailedError(ScrapingError):
    """Captcha was detected but solving/injection failed.

    Distinct from SiteBlockedError to prevent incorrect domain blocklisting --
    captcha solve failures may be transient (CapSolver downtime, balance).
    """
    pass

# In ScraperService.scrape():
except CaptchaSolveFailedError as e:
    # Don't blocklist — captcha solve failures may be transient
    logger.warning(f"Captcha solve failed for {url}: {e}")
    raise ScrapingError(str(e), url=url, provider="playwright")
```

## Consequences

### Benefits

1. **Architectural separation**: `core/captcha/` has zero Playwright dependency, enabling reuse in direct HTTP authentication flows (Substack login).

2. **Graceful degradation**: When `CAPSOLVER_API_KEY` is unset, captcha solving is silently skipped; systems continue with reduced capability.

3. **Error granularity**: `CaptchaSolveFailedError` vs `SiteBlockedError` prevents transient captcha failures from triggering permanent domain blocklisting.

4. **Cost control**: Semaphore limits concurrent solves (default 3), and free retry on `ERROR_CAPTCHA_UNSOLVABLE` reduces charges.

5. **Testability**: Pure functions (`detect_captcha`, `inject_captcha_token`) and lazy initialization (`_get_solver()`) enable straightforward mocking.

### Trade-offs

1. **Latency**: Captcha solving adds 15-45 seconds per solve.

2. **Cost**: CapSolver charges per solve; high-volume scraping requires budget monitoring.

3. **SDK gaps**: hCaptcha requires raw API calls (python3-capsolver doesn't wrap it), increasing maintenance surface.

4. **Fragile Substack integration**: `_login_with_captcha` bypasses `Api.login()` and accesses `api._session` directly; coupled to python-substack internals.

5. **Hardcoded fallback site key**: The Substack reCAPTCHA site key has a hardcoded fallback that may become stale.

### Alternatives considered

- **2Captcha / Anti-Captcha**: Similar pricing; CapSolver had better async SDK support at time of evaluation.
- **Cookie-only auth**: Works but cookies expire; email/password with captcha solving is more durable.
- **Human-in-the-loop solving**: Better success rate but requires operators and delays automation.

## Related Patterns

- [Unified Scraping Service with Fallback Chain](./unified-scraping-service-fallback-chain.md) - Captcha solving extends the Playwright tier of this fallback chain
- [Substack Publishing Integration](./substack-publishing-integration.md) - Captcha solving enables email/password auth as an alternative to cookie-based auth
- [Concurrent Scraping with TTL Cache](../async-python/concurrent-scraping-with-ttl-cache.md) - Concurrent scraping that may trigger captchas in parallel
- [Centralized Environment Configuration](../stores/centralized-env-config.md) - CAPSOLVER_API_KEY configuration follows this env-based pattern
- [HTTP Client Cleanup Registry](../../solutions/async-issues/http-client-cleanup-registry.md) - Cleanup registration for the httpx client used in raw API calls

## Known Uses in Thala

- `core/captcha/solver.py`: CaptchaSolver with semaphore, retry, and dispatch
- `core/captcha/config.py`: CapsolverConfig with env-based singleton
- `core/captcha/types.py`: CaptchaType enum and DetectedCaptcha boundary object
- `core/captcha/errors.py`: CaptchaSolveError with structured error codes
- `core/scraping/captcha_detection.py`: Playwright DOM detection and token injection JS
- `core/scraping/playwright_scraper.py`: `_handle_captcha()` integration point
- `core/scraping/errors.py`: CaptchaSolveFailedError (distinct from SiteBlockedError)
- `utils/substack_publish/publisher.py`: `_login_with_captcha()` for Substack auth
- `tests/unit/core/captcha/test_solver.py`: Solver unit tests (330 lines)
- `tests/unit/core/scraping/test_captcha_detection.py`: Detection unit tests (140 lines)
- `tests/unit/utils/substack_publish/test_publisher_captcha.py`: Substack captcha auth tests (171 lines)

## References

- [CapSolver API Documentation](https://docs.capsolver.com/)
- [python3-capsolver SDK](https://pypi.org/project/python3-capsolver/)
- [Playwright Page.evaluate()](https://playwright.dev/python/docs/api/class-page#page-evaluate)
- [python-substack library](https://github.com/ma2za/python-substack)
