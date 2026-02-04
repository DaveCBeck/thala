#!/usr/bin/env python3
"""GPU/VRAM stress test for marker-worker.

Tests marker worker with complex PDFs to find optimal batch sizes.
Monitors VRAM usage during processing and reports throughput.

Usage:
    python scripts/marker_stress_test.py --pdf /path/to/test.pdf
    python scripts/marker_stress_test.py --pdf /path/to/test.pdf --runs 3
"""

import argparse
import asyncio
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

# Default marker API endpoint
MARKER_API_URL = "http://localhost:8001"
MARKER_INPUT_DIR = Path("services/marker/data/input")

# Default test PDF
DEFAULT_TEST_PDF = Path(
    "services/retrieve-academic/downloads/10.1186_s13643-018-0740-7/"
    "832c1676b1919d94134a6de1ef7277c8.pdf"
)


@dataclass
class VRAMSample:
    """A single VRAM measurement."""

    timestamp: float
    used_gb: float
    total_gb: float


@dataclass
class TestResult:
    """Results from a single stress test run."""

    pages: int
    processing_time_sec: float
    peak_vram_gb: float
    total_vram_gb: float
    vram_samples: list[VRAMSample]
    status: str
    error: str | None = None

    @property
    def pages_per_second(self) -> float:
        if self.processing_time_sec > 0:
            return self.pages / self.processing_time_sec
        return 0.0

    @property
    def vram_utilization(self) -> float:
        if self.total_vram_gb > 0:
            return self.peak_vram_gb / self.total_vram_gb
        return 0.0


def get_gpu_memory() -> tuple[float, float]:
    """Get GPU memory (used_gb, total_gb) via nvidia-smi."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            used, total = map(int, result.stdout.strip().split(", "))
            return used / 1024, total / 1024
    except Exception:
        pass
    return 0.0, 0.0


def get_page_count(pdf_path: Path) -> int:
    """Get page count from PDF using PyMuPDF."""
    try:
        import fitz

        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
    except Exception:
        return 0


async def monitor_vram(
    stop_event: asyncio.Event,
    samples: list[VRAMSample],
    interval: float = 1.0,
) -> None:
    """Monitor VRAM usage until stop_event is set."""
    start_time = time.time()
    while not stop_event.is_set():
        used, total = get_gpu_memory()
        samples.append(
            VRAMSample(
                timestamp=time.time() - start_time,
                used_gb=used,
                total_gb=total,
            )
        )
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


async def submit_and_wait(
    client: httpx.AsyncClient,
    file_name: str,
    quality: str = "balanced",
) -> tuple[str, dict | None, str | None]:
    """Submit job and wait for completion.

    Returns: (status, result_dict, error_message)
    """
    # Submit job
    response = await client.post(
        f"{MARKER_API_URL}/convert",
        json={
            "file_path": file_name,
            "quality": quality,
            "markdown_only": True,
            "langs": ["English"],
        },
        timeout=30.0,
    )
    response.raise_for_status()
    job_data = response.json()
    job_id = job_data["job_id"]

    # Poll for completion
    while True:
        response = await client.get(
            f"{MARKER_API_URL}/jobs/{job_id}",
            timeout=10.0,
        )
        response.raise_for_status()
        status_data = response.json()

        status = status_data["status"]
        if status == "completed":
            return "completed", status_data.get("result"), None
        elif status == "failed":
            return "failed", None, status_data.get("error", "Unknown error")

        await asyncio.sleep(5.0)


async def run_stress_test(
    pdf_path: Path,
    quality: str = "balanced",
) -> TestResult:
    """Run a single stress test iteration."""
    # Ensure input directory exists
    MARKER_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy PDF to marker input directory
    dest_path = MARKER_INPUT_DIR / pdf_path.name
    if not dest_path.exists() or dest_path.stat().st_size != pdf_path.stat().st_size:
        print(f"Copying {pdf_path.name} to marker input directory...")
        shutil.copy2(pdf_path, dest_path)

    # Get page count
    pages = get_page_count(pdf_path)
    print(f"PDF has {pages} pages")

    # Start VRAM monitoring
    vram_samples: list[VRAMSample] = []
    stop_event = asyncio.Event()
    monitor_task = asyncio.create_task(monitor_vram(stop_event, vram_samples))

    # Run conversion
    start_time = time.time()
    async with httpx.AsyncClient() as client:
        try:
            status, _result, error = await submit_and_wait(
                client, pdf_path.name, quality
            )
        except Exception as e:
            status = "error"
            error = str(e)

    processing_time = time.time() - start_time

    # Stop monitoring
    stop_event.set()
    await monitor_task

    # Calculate peak VRAM
    peak_vram = max((s.used_gb for s in vram_samples), default=0.0)
    total_vram = vram_samples[0].total_gb if vram_samples else 0.0

    return TestResult(
        pages=pages,
        processing_time_sec=processing_time,
        peak_vram_gb=peak_vram,
        total_vram_gb=total_vram,
        vram_samples=vram_samples,
        status=status,
        error=error,
    )


def print_result(result: TestResult, run_num: int | None = None) -> None:
    """Print formatted test result."""
    prefix = f"Run {run_num}: " if run_num else ""
    print(f"\n{prefix}{'=' * 50}")
    print(f"Status: {result.status}")
    if result.error:
        print(f"Error: {result.error}")
    print(f"Pages: {result.pages}")
    print(f"Processing time: {result.processing_time_sec:.1f}s")
    print(f"Throughput: {result.pages_per_second:.2f} pages/sec")
    print(f"Peak VRAM: {result.peak_vram_gb:.2f} GB / {result.total_vram_gb:.2f} GB")
    print(f"VRAM utilization: {result.vram_utilization:.1%}")
    print(f"VRAM samples collected: {len(result.vram_samples)}")


def print_summary(results: list[TestResult]) -> None:
    """Print summary of multiple runs."""
    successful = [r for r in results if r.status == "completed"]
    if not successful:
        print("\nNo successful runs to summarize.")
        return

    avg_time = sum(r.processing_time_sec for r in successful) / len(successful)
    avg_throughput = sum(r.pages_per_second for r in successful) / len(successful)
    max_vram = max(r.peak_vram_gb for r in successful)
    total_vram = successful[0].total_vram_gb

    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print(f"{'=' * 50}")
    print(f"Successful runs: {len(successful)}/{len(results)}")
    print(f"Average processing time: {avg_time:.1f}s")
    print(f"Average throughput: {avg_throughput:.2f} pages/sec")
    print(f"Max peak VRAM: {max_vram:.2f} GB / {total_vram:.2f} GB ({max_vram/total_vram:.1%})")

    # Recommendation
    headroom = total_vram - max_vram
    print(f"\nVRAM headroom: {headroom:.2f} GB")
    if headroom > 4.0:
        print("→ Significant headroom available. Batch sizes can likely be increased.")
    elif headroom > 2.0:
        print("→ Moderate headroom. Small batch size increases may be safe.")
    else:
        print("→ Limited headroom. Current batch sizes are near optimal.")


async def check_marker_health() -> bool:
    """Check if marker service is healthy."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{MARKER_API_URL}/health", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                print(f"Marker service: {data['status']}")
                print(f"GPU: {data['gpu_name']} (available: {data['gpu_available']})")
                print(f"Queue depth: {data['queue_depth']}, Active workers: {data['active_workers']}")
                return data["gpu_available"]
    except Exception as e:
        print(f"Failed to connect to marker service: {e}")
    return False


async def main() -> None:
    parser = argparse.ArgumentParser(description="GPU/VRAM stress test for marker-worker")
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_TEST_PDF,
        help="Path to test PDF",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of test runs (default: 1)",
    )
    parser.add_argument(
        "--quality",
        type=str,
        default="balanced",
        choices=["fast", "balanced", "quality"],
        help="Quality preset (default: balanced, no LLM)",
    )
    args = parser.parse_args()

    # Resolve PDF path
    pdf_path = args.pdf
    if not pdf_path.is_absolute():
        pdf_path = Path.cwd() / pdf_path

    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        return

    print(f"Test PDF: {pdf_path}")
    print(f"Quality preset: {args.quality}")
    print(f"Runs: {args.runs}")
    print()

    # Check marker health
    if not await check_marker_health():
        print("\nMarker service not available. Start it with:")
        print("  cd services/marker && docker-compose up -d")
        return

    # Run tests
    results = []
    for i in range(args.runs):
        if args.runs > 1:
            print(f"\n--- Run {i+1}/{args.runs} ---")
        result = await run_stress_test(pdf_path, args.quality)
        results.append(result)
        print_result(result, i + 1 if args.runs > 1 else None)

    # Print summary if multiple runs
    if args.runs > 1:
        print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
