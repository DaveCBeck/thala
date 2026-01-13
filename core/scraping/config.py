"""Configuration for scraping services."""

import os
from dataclasses import dataclass, field
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

    # Local self-hosted instance
    local_url: Optional[str] = field(
        default_factory=lambda: os.environ.get("FIRECRAWL_LOCAL_URL")
    )

    # Cloud API
    cloud_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("FIRECRAWL_API_KEY")
    )
    cloud_url: str = "https://api.firecrawl.dev"

    # Timeouts
    timeout: int = field(
        default_factory=lambda: int(os.environ.get("FIRECRAWL_TIMEOUT", "45"))
    )

    # Feature flags
    skip_local: bool = field(
        default_factory=lambda: os.environ.get("FIRECRAWL_SKIP_LOCAL", "").lower()
        == "true"
    )

    @property
    def local_available(self) -> bool:
        """Check if local Firecrawl is configured and not skipped."""
        return bool(self.local_url) and not self.skip_local

    @property
    def cloud_available(self) -> bool:
        """Check if cloud Firecrawl is configured."""
        return bool(self.cloud_api_key)


def get_firecrawl_config() -> FirecrawlConfig:
    """Get Firecrawl configuration from environment."""
    return FirecrawlConfig()
