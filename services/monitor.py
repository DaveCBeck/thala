#!/usr/bin/env python3
"""Lightweight performance monitor for Thala Docker services.

Collects container resource metrics (CPU%, memory) and health endpoint
response times, outputting readable periodic summaries with rolling
statistics (avg/min/max/p90).

Usage:
    ./services.sh monitor              # Default: 30s interval, table output
    ./services.sh monitor --json       # JSON to console
    ./services.sh monitor --once       # Single collection
    ./services.sh monitor --no-save    # Don't save to file
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

# === Configuration ===

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.resolve()
METRICS_DIR = PROJECT_ROOT / "logs" / "services" / "metrics"


@dataclass
class ServiceConfig:
    """Configuration for a monitored service."""

    name: str
    container_names: list[str]
    health_url: Optional[str] = None
    requires_gpu: bool = False
    requires_vpn: bool = False
    memory_limit: Optional[str] = None


SERVICES = [
    ServiceConfig(
        name="es-coherence",
        container_names=["es-coherence"],
        health_url="http://localhost:9201/_cluster/health",
        memory_limit="1G",
    ),
    ServiceConfig(
        name="es-forgotten",
        container_names=["es-forgotten"],
        health_url="http://localhost:9200/_cluster/health",
        memory_limit="1G",
    ),
    ServiceConfig(
        name="chroma",
        container_names=["chroma"],
        health_url="http://localhost:8000/api/v2/heartbeat",
        memory_limit="4G",
    ),
    ServiceConfig(
        name="zotero",
        container_names=["zotero"],
        health_url="http://localhost:23119/local-crud/ping",
        memory_limit="2G",
    ),
    ServiceConfig(
        name="translation-server",
        container_names=["thala-translation-server"],
        health_url="http://localhost:1969/",
    ),
    ServiceConfig(
        name="marker",
        container_names=[
            "marker-marker-api-1",
            "marker-marker-worker-1",
            "marker-redis-1",
            "marker-flower-1",
        ],
        health_url="http://localhost:8001/health",
        requires_gpu=True,
    ),
    ServiceConfig(
        name="retrieve-academic",
        container_names=["retrieve-academic-vpn", "retrieve-academic-api"],
        health_url="http://localhost:8002/health",
        requires_vpn=True,
    ),
    ServiceConfig(
        name="firecrawl",
        container_names=[
            "firecrawl-firecrawl-api-1",
            "firecrawl-playwright-service-1",
            "firecrawl-redis-1",
            "firecrawl-rabbitmq-1",
            "firecrawl-nuq-postgres-1",
        ],
        health_url="http://localhost:3002/",
    ),
]


# === Rolling Statistics ===


@dataclass
class RollingStats:
    """Time-windowed rolling statistics."""

    window_seconds: float = 300.0
    _samples: deque = field(default_factory=deque)

    def add(self, value: float, timestamp: Optional[float] = None) -> None:
        """Add a sample, pruning old entries."""
        ts = timestamp or time.time()
        self._samples.append((ts, value))
        self._prune(ts)

    def _prune(self, now: float) -> None:
        """Remove samples outside the window."""
        cutoff = now - self.window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def values(self) -> list[float]:
        """Get current values in window."""
        self._prune(time.time())
        return [v for _, v in self._samples]

    def count(self) -> int:
        """Number of samples in window."""
        return len(self.values())

    def avg(self) -> Optional[float]:
        vals = self.values()
        return sum(vals) / len(vals) if vals else None

    def min(self) -> Optional[float]:
        vals = self.values()
        return min(vals) if vals else None

    def max(self) -> Optional[float]:
        vals = self.values()
        return max(vals) if vals else None

    def percentile(self, p: int) -> Optional[float]:
        """Calculate p-th percentile."""
        vals = sorted(self.values())
        if not vals:
            return None
        idx = int(len(vals) * p / 100)
        return vals[min(idx, len(vals) - 1)]


# === Data Models ===


@dataclass
class ContainerMetrics:
    """Metrics for a single container."""

    name: str
    running: bool
    cpu_percent: Optional[float] = None
    mem_usage_bytes: Optional[int] = None
    mem_limit_bytes: Optional[int] = None
    mem_percent: Optional[float] = None


@dataclass
class ServiceMetrics:
    """Aggregated metrics for a service."""

    config: ServiceConfig
    containers: list[ContainerMetrics] = field(default_factory=list)
    response_time_ms: Optional[float] = None
    health_error: Optional[str] = None
    skip_reason: Optional[str] = None

    # Rolling stats (populated over time)
    cpu_stats: RollingStats = field(default_factory=lambda: RollingStats())
    mem_stats: RollingStats = field(default_factory=lambda: RollingStats())
    latency_stats: RollingStats = field(default_factory=lambda: RollingStats())


# === Docker Stats Collector ===


def collect_docker_stats() -> dict[str, ContainerMetrics]:
    """Collect current Docker container stats via CLI."""
    try:
        result = subprocess.run(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                '{"name":"{{.Name}}","cpu":"{{.CPUPerc}}","mem":"{{.MemUsage}}","mem_perc":"{{.MemPerc}}"}',
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired:
        return {}
    except FileNotFoundError:
        return {}

    metrics = {}
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        try:
            data = json.loads(line)
            name = data["name"]
            metrics[name] = ContainerMetrics(
                name=name,
                running=True,
                cpu_percent=_parse_percent(data["cpu"]),
                mem_usage_bytes=_parse_mem_usage(data["mem"]),
                mem_limit_bytes=_parse_mem_limit(data["mem"]),
                mem_percent=_parse_percent(data["mem_perc"]),
            )
        except (json.JSONDecodeError, KeyError):
            continue

    return metrics


def _parse_percent(s: str) -> Optional[float]:
    """Parse '2.34%' -> 2.34"""
    try:
        return float(s.rstrip("%"))
    except (ValueError, AttributeError):
        return None


def _parse_mem_usage(s: str) -> Optional[int]:
    """Parse '512MiB / 1GiB' -> bytes for usage part."""
    try:
        usage_part = s.split("/")[0].strip()
        return _parse_mem_value(usage_part)
    except Exception:
        return None


def _parse_mem_limit(s: str) -> Optional[int]:
    """Parse '512MiB / 1GiB' -> bytes for limit part."""
    try:
        parts = s.split("/")
        if len(parts) < 2:
            return None
        limit_part = parts[1].strip()
        return _parse_mem_value(limit_part)
    except Exception:
        return None


def _parse_mem_value(s: str) -> int:
    """Parse memory value like '512MiB' or '1.2GiB' to bytes."""
    s = s.strip()
    multipliers = {
        "B": 1,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
    }
    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * mult)
    return int(float(s))


# === Health Check Collector ===


async def check_health(
    url: str, timeout: float = 5.0
) -> tuple[Optional[float], Optional[str]]:
    """Check health endpoint, return (response_time_ms, error)."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            start = time.perf_counter()
            response = await client.get(url)
            elapsed_ms = (time.perf_counter() - start) * 1000

            if response.status_code < 400:
                return (elapsed_ms, None)
            else:
                return (elapsed_ms, f"HTTP {response.status_code}")
        except httpx.ConnectError:
            return (None, "Connection refused")
        except httpx.TimeoutException:
            return (None, "Timeout")
        except Exception as e:
            return (None, str(e)[:50])


async def collect_all_health_checks(
    services: list[ServiceConfig],
) -> dict[str, tuple[Optional[float], Optional[str]]]:
    """Check all service health endpoints concurrently."""
    tasks = {}
    for svc in services:
        if svc.health_url:
            tasks[svc.name] = check_health(svc.health_url)

    results = {}
    if tasks:
        gathered = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for name, result in zip(tasks.keys(), gathered):
            if isinstance(result, Exception):
                results[name] = (None, str(result)[:50])
            else:
                results[name] = result

    return results


# === Main Monitor Class ===


class ServiceMonitor:
    """Main monitoring coordinator."""

    def __init__(
        self,
        services: list[ServiceConfig],
        window_seconds: float = 300.0,
        interval_seconds: float = 30.0,
        json_output: bool = False,
        save_to_file: bool = True,
    ):
        self.services = services
        self.window_seconds = window_seconds
        self.interval_seconds = interval_seconds
        self.json_output = json_output
        self.save_to_file = save_to_file

        # Initialize per-service metrics with rolling stats
        self.service_metrics: dict[str, ServiceMetrics] = {}
        for svc in services:
            self.service_metrics[svc.name] = ServiceMetrics(
                config=svc,
                cpu_stats=RollingStats(window_seconds),
                mem_stats=RollingStats(window_seconds),
                latency_stats=RollingStats(window_seconds),
            )

        # Prerequisite cache
        self._has_gpu: Optional[bool] = None

    def _check_prerequisites(self, svc: ServiceConfig) -> Optional[str]:
        """Check if service prerequisites are met. Returns skip reason or None."""
        if svc.requires_gpu and not self._check_gpu():
            return "No GPU available"
        if svc.requires_vpn and not self._check_vpn_config(svc.name):
            return "VPN not configured"
        return None

    def _check_gpu(self) -> bool:
        """Check if nvidia-smi is available."""
        if self._has_gpu is not None:
            return self._has_gpu
        try:
            result = subprocess.run(
                ["nvidia-smi"], capture_output=True, timeout=5, check=False
            )
            self._has_gpu = result.returncode == 0
        except Exception:
            self._has_gpu = False
        return self._has_gpu

    def _check_vpn_config(self, service_name: str) -> bool:
        """Check if VPN service has .env configured."""
        env_path = SCRIPT_DIR / service_name / ".env"
        return env_path.exists()

    async def collect_once(self) -> dict[str, ServiceMetrics]:
        """Collect all metrics once."""
        # Get Docker stats
        docker_stats = collect_docker_stats()

        # Build list of services to health-check (skip those with unmet prerequisites)
        services_to_check = []
        for svc in self.services:
            skip_reason = self._check_prerequisites(svc)
            self.service_metrics[svc.name].skip_reason = skip_reason
            if not skip_reason:
                services_to_check.append(svc)

        # Get health checks for eligible services
        health_results = await collect_all_health_checks(services_to_check)

        # Update service metrics
        for svc in self.services:
            metrics = self.service_metrics[svc.name]

            if metrics.skip_reason:
                metrics.health_error = metrics.skip_reason
                metrics.containers = []
                continue

            # Collect container metrics
            metrics.containers = []
            total_cpu = 0.0
            total_mem = 0
            for container_name in svc.container_names:
                if container_name in docker_stats:
                    cm = docker_stats[container_name]
                    metrics.containers.append(cm)
                    if cm.cpu_percent:
                        total_cpu += cm.cpu_percent
                    if cm.mem_usage_bytes:
                        total_mem += cm.mem_usage_bytes
                else:
                    metrics.containers.append(
                        ContainerMetrics(name=container_name, running=False)
                    )

            # Update rolling stats if we have data
            if total_cpu > 0:
                metrics.cpu_stats.add(total_cpu)
            if total_mem > 0:
                metrics.mem_stats.add(total_mem)

            # Health check results
            if svc.name in health_results:
                latency, error = health_results[svc.name]
                metrics.response_time_ms = latency
                metrics.health_error = error
                if latency is not None:
                    metrics.latency_stats.add(latency)
            else:
                # No health check performed (maybe all containers are down)
                metrics.response_time_ms = None
                if not any(c.running for c in metrics.containers):
                    metrics.health_error = "All containers down"

        return self.service_metrics

    def format_table(self, metrics: dict[str, ServiceMetrics]) -> str:
        """Format metrics as human-readable table."""
        lines = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines.append("")
        lines.append("=" * 72)
        lines.append(f"Service Health Report [{timestamp}]")
        lines.append("=" * 72)
        lines.append(
            f"{'Service':<18} {'Status':<8} {'CPU%':<8} {'Memory':<14} {'Latency (ms)'}"
        )
        lines.append("-" * 72)

        for svc in self.services:
            m = metrics[svc.name]

            # Determine status
            if m.skip_reason:
                status = "N/A"
            elif any(c.running for c in m.containers):
                status = "UP" if not m.health_error else "WARN"
            else:
                status = "DOWN"

            # Format CPU
            cpu_avg = m.cpu_stats.avg()
            cpu_str = f"{cpu_avg:.1f}%" if cpu_avg else "-"

            # Format memory
            running_containers = [c for c in m.containers if c.running]
            if running_containers:
                total_mem = sum(c.mem_usage_bytes or 0 for c in running_containers)
                mem_str = self._format_bytes(total_mem)
                if svc.memory_limit:
                    mem_str += f" / {svc.memory_limit}"
            else:
                mem_str = "-"

            # Format latency
            lat = m.latency_stats
            if lat.count() > 0:
                lat_str = f"avg:{lat.avg():.0f} p90:{lat.percentile(90):.0f}"
            else:
                lat_str = "-"

            lines.append(
                f"{svc.name:<18} {status:<8} {cpu_str:<8} {mem_str:<14} {lat_str}"
            )

        return "\n".join(lines)

    def format_json(self, metrics: dict[str, ServiceMetrics]) -> str:
        """Format metrics as JSON for file storage."""
        output = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "services": {},
        }
        for name, m in metrics.items():
            running_containers = [c for c in m.containers if c.running]
            output["services"][name] = {
                "status": "up"
                if running_containers
                else ("n/a" if m.skip_reason else "down"),
                "cpu_avg": round(m.cpu_stats.avg(), 2) if m.cpu_stats.avg() else None,
                "mem_bytes": sum(c.mem_usage_bytes or 0 for c in running_containers),
                "latency_avg_ms": round(m.latency_stats.avg(), 1)
                if m.latency_stats.avg()
                else None,
                "latency_p90_ms": round(m.latency_stats.percentile(90), 1)
                if m.latency_stats.percentile(90)
                else None,
                "error": m.health_error,
            }
        return json.dumps(output)

    def _save_metrics(self, json_line: str) -> None:
        """Append metrics to daily JSONL file."""
        METRICS_DIR.mkdir(exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        filepath = METRICS_DIR / f"{today}.jsonl"
        with open(filepath, "a") as f:
            f.write(json_line + "\n")

    @staticmethod
    def _format_bytes(b: int) -> str:
        """Format bytes as human-readable."""
        for unit in ["B", "K", "M", "G", "T"]:
            if abs(b) < 1024:
                return f"{b:.0f}{unit}" if unit == "B" else f"{b:.1f}{unit}"
            b /= 1024
        return f"{b:.1f}P"

    async def run(self) -> None:
        """Run monitoring loop."""
        print(
            f"Starting service monitor (interval={self.interval_seconds}s, "
            f"window={self.window_seconds}s, save={self.save_to_file})"
        )
        if self.save_to_file:
            print(f"Metrics will be saved to: {METRICS_DIR}/")

        try:
            while True:
                metrics = await self.collect_once()

                # Console output
                if self.json_output:
                    print(self.format_json(metrics))
                else:
                    print(self.format_table(metrics))

                # File persistence
                if self.save_to_file:
                    self._save_metrics(self.format_json(metrics))

                await asyncio.sleep(self.interval_seconds)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")

    async def run_once(self) -> None:
        """Collect once and output."""
        metrics = await self.collect_once()

        if self.json_output:
            print(self.format_json(metrics))
        else:
            print(self.format_table(metrics))

        if self.save_to_file:
            self._save_metrics(self.format_json(metrics))


# === CLI Entry Point ===


def main():
    parser = argparse.ArgumentParser(
        description="Monitor Thala Docker services performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      # Default: 30s interval, table output
  %(prog)s --interval 60        # 60s interval
  %(prog)s --json               # JSON to console
  %(prog)s --once               # Single collection
  %(prog)s --no-save            # Don't save to file
        """,
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=float(os.getenv("MONITOR_INTERVAL_SECONDS", "30")),
        help="Collection interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--window",
        "-w",
        type=float,
        default=float(os.getenv("MONITOR_WINDOW_SECONDS", "300")),
        help="Rolling statistics window in seconds (default: 300)",
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output JSON to console instead of table",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Collect once and exit (no loop)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable file persistence (console only)",
    )

    args = parser.parse_args()

    monitor = ServiceMonitor(
        services=SERVICES,
        window_seconds=args.window,
        interval_seconds=args.interval,
        json_output=args.json,
        save_to_file=not args.no_save,
    )

    if args.once:
        asyncio.run(monitor.run_once())
    else:
        asyncio.run(monitor.run())


if __name__ == "__main__":
    main()
