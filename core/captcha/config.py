"""Configuration for CapSolver captcha solving service."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CapsolverConfig:
    """Configuration for CapSolver API.

    Environment Variables:
        CAPSOLVER_API_KEY: API key for CapSolver service
        CAPSOLVER_TIMEOUT: Solve timeout in seconds (default: 120)
        CAPSOLVER_MAX_CONCURRENT: Max concurrent solves (default: 3)
    """

    api_key: Optional[str] = field(default_factory=lambda: os.environ.get("CAPSOLVER_API_KEY"))
    timeout: float = field(default_factory=lambda: float(os.environ.get("CAPSOLVER_TIMEOUT", "120.0")))
    max_concurrent: int = field(default_factory=lambda: int(os.environ.get("CAPSOLVER_MAX_CONCURRENT", "3")))

    @property
    def available(self) -> bool:
        """Check if CapSolver is configured."""
        return bool(self.api_key)
