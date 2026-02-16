"""Configuration for CapSolver captcha solving service."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class CapsolverConfig:
    """Configuration for CapSolver API.

    Environment Variables:
        CAPSOLVER_API_KEY: API key for CapSolver service
        CAPSOLVER_TIMEOUT: Solve timeout in seconds (default: 120)
        CAPSOLVER_MAX_CONCURRENT: Max concurrent solves (default: 3)
    """

    api_key: str | None = field(default_factory=lambda: os.environ.get("CAPSOLVER_API_KEY"))
    timeout: float = field(default_factory=lambda: float(os.environ.get("CAPSOLVER_TIMEOUT", "120.0")))
    max_concurrent: int = field(default_factory=lambda: int(os.environ.get("CAPSOLVER_MAX_CONCURRENT", "3")))

    def __post_init__(self):
        if self.timeout <= 0:
            raise ValueError(f"CAPSOLVER_TIMEOUT must be positive, got {self.timeout}")
        if not 1 <= self.max_concurrent <= 20:
            raise ValueError(f"CAPSOLVER_MAX_CONCURRENT must be 1-20, got {self.max_concurrent}")

    @property
    def available(self) -> bool:
        """Check if CapSolver is configured."""
        return bool(self.api_key)


_capsolver_config: CapsolverConfig | None = None


def get_capsolver_config() -> CapsolverConfig:
    """Return a shared CapsolverConfig singleton (created on first call)."""
    global _capsolver_config
    if _capsolver_config is None:
        _capsolver_config = CapsolverConfig()
    return _capsolver_config
