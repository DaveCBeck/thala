---
name: unified-scraping-service-fallback-chain
title: "Unified Scraping Service with Fallback Chain"
date: 2026-01-13
category: data-pipeline
applicability:
  - "When scraping web content that may be blocked by anti-bot measures"
  - "When multiple scraping providers are available with different capabilities"
  - "When graceful degradation through provider tiers is required"
  - "When learning from failures to optimize future requests"
  - "When self-hosted scraping infrastructure needs cloud fallback"
components: [firecrawl, firecrawl_clients, playwright, asyncio, pydantic, docker]
complexity: moderate
verified_in_production: true
related_solutions: []
tags: [scraping, fallback-chain, resilience, playwright, firecrawl, rate-limiting, lazy-initialization, self-hosted, dual-client]
---

# Unified Scraping Service with Fallback Chain

## Intent

Provide resilient web scraping through a tiered fallback chain that automatically escalates through providers of increasing capability, with self-hosted Firecrawl as the primary tier and cloud stealth proxy for anti-bot scenarios.

## Motivation

Web scraping faces several challenges that a single provider cannot address:

1. **Anti-bot measures**: Sites employ captchas, JavaScript challenges, and IP blocking
2. **Provider limitations**: Cloud services may not support all sites (e.g., Reddit)
3. **Cost/speed tradeoffs**: Cloud services have rate limits and costs
4. **Transient failures**: Network issues and rate limits cause intermittent failures
5. **Privacy concerns**: Some content shouldn't traverse third-party services

This pattern addresses these challenges through:
- A three-tier fallback chain (local → cloud stealth → browser)
- Self-hosted Firecrawl for fast, unlimited local scraping
- Cloud stealth proxy for anti-bot bypass
- Adaptive learning via domain blocklist
- Content-based blocking detection

## Applicability

Use this pattern when:
- Scraping diverse sites with varying anti-bot measures
- You want to minimize cloud API costs with local infrastructure
- Multiple providers are available with different cost/capability profiles
- Transparent fallback without caller awareness is needed
- Learning from failures to optimize future requests is valuable

Do NOT use this pattern when:
- Only scraping a single, well-known site (direct integration is simpler)
- All sources are APIs with consistent behavior
- Real-time scraping with strict latency requirements
- Cannot host self-hosted services (cloud-only mode still works)

## Structure

```
                    +-----------------+
                    |  scrape(url)    |
                    +--------+--------+
                             |
                    +--------v--------+
                    | Domain in       |  YES   +------------------+
                    | blocklist?      +------->| Playwright       |
                    +--------+--------+        +------------------+
                             | NO
                    +--------v--------+
                    | Tier 1: Local   |  SUCCESS  (return)
                    | Firecrawl       +---------->
                    | (self-hosted)   |
                    +--------+--------+
                             | FAIL / LocalServiceUnavailableError
                    +--------v--------+
                    | Tier 2: Cloud   |  SUCCESS  (return)
                    | Firecrawl       +---------->
                    | (stealth proxy) |
                    +--------+--------+
                             | FAIL (SiteBlockedError)
                             | --> Add domain to blocklist
                    +--------v--------+
                    | Tier 3:         |  SUCCESS  (return)
                    | Playwright      +---------->
                    | (local browser) |
                    +--------+--------+
                             | FAIL
                    +--------v--------+
                    | Raise           |
                    | ScrapingError   |
                    +-----------------+
```

**Tier Summary:**

| Tier | Provider | Cost | Speed | Anti-bot Bypass |
|------|----------|------|-------|-----------------|
| 1 | Local Firecrawl | Free | Fast | None |
| 2 | Cloud Firecrawl Stealth | $$ | Medium | High |
| 3 | Playwright | Free | Slow | Medium |

## Implementation

### Step 1: Configuration Management

Environment-based configuration for both local and cloud clients:

```python
# core/scraping/config.py
from dataclasses import dataclass, field
import os
from typing import Optional


@dataclass
class FirecrawlConfig:
    """Configuration for Firecrawl clients (local and cloud).

    Environment Variables:
        FIRECRAWL_LOCAL_URL: URL of self-hosted Firecrawl (e.g., http://localhost:3002)
        FIRECRAWL_API_KEY: API key for cloud Firecrawl (required for stealth fallback)
        FIRECRAWL_TIMEOUT: Request timeout in seconds (default: 45)
        FIRECRAWL_SKIP_LOCAL: Set to 'true' to skip local and use cloud only
    """

    local_url: Optional[str] = field(
        default_factory=lambda: os.environ.get("FIRECRAWL_LOCAL_URL")
    )
    cloud_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("FIRECRAWL_API_KEY")
    )
    cloud_url: str = "https://api.firecrawl.dev"
    timeout: int = field(
        default_factory=lambda: int(os.environ.get("FIRECRAWL_TIMEOUT", "45"))
    )
    skip_local: bool = field(
        default_factory=lambda: os.environ.get("FIRECRAWL_SKIP_LOCAL", "").lower() == "true"
    )

    @property
    def local_available(self) -> bool:
        """Check if local Firecrawl is configured and not skipped."""
        return bool(self.local_url) and not self.skip_local

    @property
    def cloud_available(self) -> bool:
        """Check if cloud Firecrawl is configured."""
        return bool(self.cloud_api_key)
```

### Step 2: Define the Error Hierarchy

Custom exceptions carry context and drive fallback behavior:

```python
class ScrapingError(Exception):
    """Base scraping exception."""

    def __init__(self, message: str, url: str, provider: str | None = None):
        self.message = message
        self.url = url
        self.provider = provider
        super().__init__(message)


class SiteBlockedError(ScrapingError):
    """Site is blocked by the scraping provider."""
    pass


class LocalServiceUnavailableError(ScrapingError):
    """Local service is unavailable (transient, don't blocklist)."""
    pass
```

### Step 3: Define the Result Model

Unified return type tracks which provider succeeded:

```python
from pydantic import BaseModel, Field

class ScrapeResult(BaseModel):
    """Output schema for scrape operations."""

    url: str
    markdown: str
    links: list[str] = Field(default_factory=list)
    provider: str = "unknown"  # Which provider succeeded
```

### Step 4: Dual Client Manager

Manages separate clients for local and cloud with lazy initialization:

```python
# core/scraping/firecrawl_clients.py
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from firecrawl import AsyncFirecrawl


class FirecrawlClients:
    """Manager for local and cloud Firecrawl clients.

    Provides lazy initialization and maintains separate clients for:
    - Local (self-hosted): No API key needed
    - Cloud: Requires FIRECRAWL_API_KEY for stealth proxy access
    """

    def __init__(self, config: Optional[FirecrawlConfig] = None):
        self._config = config or get_firecrawl_config()
        self._local: "AsyncFirecrawl | None" = None
        self._cloud: "AsyncFirecrawl | None" = None

    def _get_local(self) -> "AsyncFirecrawl | None":
        """Get local Firecrawl client (lazy init)."""
        if not self._config.local_available:
            return None

        if self._local is None:
            from firecrawl import AsyncFirecrawl

            # Local instance - no API key validation by self-hosted server
            self._local = AsyncFirecrawl(
                api_key="local",  # Placeholder - not validated by self-hosted
                api_url=self._config.local_url,
            )
        return self._local

    def _get_cloud(self) -> "AsyncFirecrawl | None":
        """Get cloud Firecrawl client (lazy init)."""
        if not self._config.cloud_available:
            return None

        if self._cloud is None:
            from firecrawl import AsyncFirecrawl

            self._cloud = AsyncFirecrawl(
                api_key=self._config.cloud_api_key,
                api_url=self._config.cloud_url,
            )
        return self._cloud

    @property
    def local(self) -> "AsyncFirecrawl | None":
        return self._get_local()

    @property
    def cloud(self) -> "AsyncFirecrawl | None":
        return self._get_cloud()

    @property
    def config(self) -> FirecrawlConfig:
        return self._config

    async def close(self) -> None:
        """Close all client connections."""
        for client, name in [(self._local, "local"), (self._cloud, "cloud")]:
            if client is not None:
                try:
                    if hasattr(client, "_session") and client._session:
                        await client._session.close()
                except Exception as e:
                    logger.debug(f"Error closing {name} Firecrawl client: {e}")
        self._local = None
        self._cloud = None
```

### Step 5: Implement Blocking Detection

Content-based detection catches soft blocks (captcha pages returned as 200 OK):

```python
def _is_blocked_response(self, result) -> bool:
    """Check if response indicates site blocking."""
    if not hasattr(result, "markdown"):
        return True

    markdown = result.markdown or ""

    # Short content is suspicious
    if len(markdown) < 100:
        return True

    blocking_indicators = [
        "captcha",
        "access denied",
        "please verify",
        "bot detection",
        "enable javascript",
    ]
    markdown_lower = markdown.lower()
    return any(indicator in markdown_lower for indicator in blocking_indicators)
```

### Step 6: Implement the Fallback Chain

The core scrape method with tiered fallback using dual clients:

```python
class ScraperService:
    def __init__(self):
        self._firecrawl_clients: FirecrawlClients | None = None
        self._playwright: PlaywrightScraper | None = None
        self._blocklist: set[str] = set()  # Domains requiring Playwright

    def _get_firecrawl_clients(self) -> FirecrawlClients:
        """Get Firecrawl client manager (lazy initialization)."""
        if self._firecrawl_clients is None:
            from .firecrawl_clients import get_firecrawl_clients
            self._firecrawl_clients = get_firecrawl_clients()
        return self._firecrawl_clients

    async def scrape(self, url: str, include_links: bool = False) -> ScrapeResult:
        """Scrape URL with automatic fallback chain."""
        domain = _extract_domain(url)
        clients = self._get_firecrawl_clients()

        # Skip straight to Playwright for known-blocked domains
        if domain in self._blocklist:
            logger.debug(f"Domain {domain} in blocklist, using Playwright")
            return await self._scrape_playwright(url, include_links)

        # === Tier 1: Local Firecrawl (self-hosted) ===
        if clients.config.local_available:
            try:
                logger.debug("Trying local Firecrawl")
                return await _with_retry(
                    self._scrape_local, url, include_links=include_links
                )

            except LocalServiceUnavailableError as e:
                # Local service down - proceed to cloud (don't add to blocklist)
                logger.warning(f"Local Firecrawl unavailable: {e}")

            except SiteBlockedError:
                # Site blocked locally - try cloud stealth
                logger.debug("Local Firecrawl got blocked, trying cloud stealth")

            except Exception as e:
                logger.debug(f"Local Firecrawl failed: {e}")

        # === Tier 2: Cloud Firecrawl Stealth ===
        if clients.config.cloud_available:
            try:
                logger.debug("Trying cloud Firecrawl stealth")
                return await _with_retry(
                    self._scrape_cloud_stealth, url, include_links=include_links
                )

            except SiteBlockedError:
                # Site blocked even with stealth - add to blocklist
                logger.info(f"Site {domain} blocked by cloud stealth, adding to blocklist")
                self._blocklist.add(domain)

            except Exception as e:
                logger.debug(f"Cloud stealth failed: {e}")

        # === Tier 3: Playwright Fallback ===
        logger.debug("Falling back to Playwright")
        try:
            return await _with_retry(self._scrape_playwright, url, include_links)
        except Exception as e:
            raise ScrapingError(
                f"All scraping methods failed: {e}",
                url=url,
                provider="all",
            )
```

### Step 7: Implement Lazy Initialization

Heavy resources are created only when needed. The `FirecrawlClients` manager is accessed through a module-level singleton:

```python
# Module-level singleton with cleanup registration
_clients: FirecrawlClients | None = None


def get_firecrawl_clients() -> FirecrawlClients:
    """Get the global FirecrawlClients instance."""
    global _clients
    if _clients is None:
        from core.utils.async_http_client import register_cleanup

        _clients = FirecrawlClients()
        register_cleanup("FirecrawlClients", close_firecrawl_clients)
    return _clients


async def close_firecrawl_clients() -> None:
    """Close the global FirecrawlClients instance."""
    global _clients
    if _clients is not None:
        await _clients.close()
        _clients = None
```

Playwright is lazy-initialized within the service:

```python
def _get_playwright(self) -> PlaywrightScraper:
    """Get or create Playwright scraper (lazy initialization)."""
    if self._playwright is None:
        self._playwright = PlaywrightScraper()
    return self._playwright
```

### Step 8: Implement Playwright Scraper with Rate Limiting

Browser-based fallback with stealth fingerprint and rate limiting:

```python
class PlaywrightScraper:
    def __init__(self, timeout: int = 30000, delay: float = 1.5):
        self._browser: Browser | None = None
        self._playwright: Playwright | None = None
        self._timeout = timeout
        self._delay = delay
        self._last_request: float = 0

        # Configure html2text
        self._html2text = html2text.HTML2Text()
        self._html2text.ignore_links = False
        self._html2text.body_width = 0

    async def _rate_limit(self) -> None:
        """Enforce delay between requests."""
        now = time.monotonic()
        elapsed = now - self._last_request
        if self._last_request > 0 and elapsed < self._delay:
            wait_time = self._delay - elapsed
            await asyncio.sleep(wait_time)
        self._last_request = time.monotonic()

    async def _get_browser(self) -> Browser:
        """Get or create browser instance (lazy initialization)."""
        if self._browser is None:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                ],
            )
        return self._browser

    async def scrape(self, url: str) -> str:
        """Scrape URL and return content as markdown."""
        await self._rate_limit()

        browser = await self._get_browser()

        # Create context with realistic browser fingerprint
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
        )
        page = await context.new_page()

        try:
            await page.goto(url, timeout=self._timeout, wait_until="networkidle")
            html = await page.content()
            return self._html2text.handle(html).strip()
        finally:
            await page.close()
            await context.close()

    async def close(self) -> None:
        """Clean up browser resources."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
```

### Step 9: Module-Level Singleton with Cleanup

Global service instance with cleanup registration:

```python
_scraper_service: ScraperService | None = None


def get_scraper_service() -> ScraperService:
    """Get the global scraper service instance."""
    global _scraper_service
    if _scraper_service is None:
        from core.utils.async_http_client import register_cleanup

        _scraper_service = ScraperService()
        register_cleanup("Scraper", close_scraper_service)
    return _scraper_service


async def close_scraper_service() -> None:
    """Close the global scraper service and release resources."""
    global _scraper_service
    if _scraper_service is not None:
        await _scraper_service.close()
        _scraper_service = None
```

## Self-Hosted Firecrawl Setup

For local Firecrawl, use Docker Compose:

```yaml
# services/firecrawl/docker-compose.yml
services:
  firecrawl:
    image: mendableai/firecrawl:latest
    ports:
      - "3002:3002"
    environment:
      - USE_DB_AUTHENTICATION=false
      - NUM_WORKERS_PER_QUEUE=2
      - REDIS_URL=redis://redis:6379
      - PLAYWRIGHT_MICROSERVICE_URL=http://playwright:3000
    depends_on:
      - redis
      - playwright

  playwright:
    image: mendableai/firecrawl-playwright:latest
    ports:
      - "3000:3000"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

Environment configuration:

```bash
# .env
FIRECRAWL_LOCAL_URL=http://localhost:3002    # Self-hosted
FIRECRAWL_API_KEY=fc-xxxxx                    # Cloud fallback (optional)
FIRECRAWL_TIMEOUT=45                          # Request timeout
FIRECRAWL_SKIP_LOCAL=false                    # Set true to skip local tier
```

## Consequences

### Benefits

- **Resilient scraping**: Automatically handles blocked sites and transient failures
- **Cost optimization**: Local tier is free with no rate limits
- **Anti-bot bypass**: Cloud stealth proxy handles sites blocking local requests
- **Adaptive learning**: Blocklist prevents repeated failures for known-problematic domains
- **Transparent integration**: Callers get consistent `ScrapeResult` regardless of provider
- **Resource efficiency**: Lazy initialization means browser only starts if actually needed
- **Cloud independence**: Works entirely locally if cloud API not configured

### Trade-offs

- **In-memory blocklist**: Resets on restart (fresh start vs. persistent learning)
- **Content-based detection**: May false-positive on legitimate short pages
- **First Playwright request slower**: Browser launch overhead on first use
- **Singleton pattern**: Shared state aids learning but complicates testing
- **Docker dependency**: Local tier requires running Docker services

### Async Considerations

- **Rate limiting uses `time.monotonic()`**: Immune to system clock changes
- **Lazy init race condition**: Low risk with singleton pattern, add lock if concurrent init needed
- **No overall timeout wrapper**: Individual operations have timeouts, but no guaranteed upper bound

## Related Patterns

- [Citation Processing with Zotero Integration](./citation-processing-zotero-integration.md) - Uses this service for page content
- [Centralized Environment Configuration](../stores/centralized-env-config.md) - Firecrawl API key configuration
- [Deep Research Workflow Architecture](../langgraph/deep-research-workflow-architecture.md) - Uses scraping in researcher agents
- [HTTP Client Cleanup Registry](../../solutions/async-issues/http-client-cleanup-registry.md) - Cleanup registration pattern

## Known Uses in Thala

- `core/scraping/service.py`: Main ScraperService implementation
- `core/scraping/config.py`: FirecrawlConfig configuration
- `core/scraping/firecrawl_clients.py`: Dual-client manager
- `core/scraping/playwright_scraper.py`: Browser-based fallback
- `core/scraping/errors.py`: Custom exception hierarchy
- `langchain_tools/firecrawl.py`: LangChain tool wrapping the service
- `services/firecrawl/docker-compose.yml`: Self-hosted Firecrawl setup

## References

- [Firecrawl Documentation](https://docs.firecrawl.dev/)
- [Firecrawl Self-Hosting](https://docs.firecrawl.dev/self-hosting)
- [Playwright Python](https://playwright.dev/python/)
- [html2text](https://github.com/Alir3z4/html2text/)
