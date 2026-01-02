"""System metrics utilities."""

import subprocess

import psutil


def get_memory_stats() -> dict:
    """Get RAM and GPU memory stats."""
    ram = psutil.Process().memory_info()
    ram_gb = ram.rss / (1024**3)

    gpu_used_gb, gpu_total_gb = get_gpu_memory()

    return {
        "ram_gb": ram_gb,
        "gpu_used_gb": gpu_used_gb,
        "gpu_total_gb": gpu_total_gb,
    }


def get_gpu_memory() -> tuple[float, float]:
    """Get GPU memory (used_gb, total_gb)."""
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
