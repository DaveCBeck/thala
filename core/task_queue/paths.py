"""Centralized path constants for the .thala/ state directory.

All local state is stored under .thala/:
- .thala/queue/    - Task queue data (queue.json, checkpoints, etc.)
- .thala/output/   - Generated reports and article series
- .thala/.substack-cookies.json - Substack authentication cookies
"""

from pathlib import Path

# Project root detection
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Main state directory
THALA_DIR = PROJECT_ROOT / ".thala"

# Queue state directory (was: topic_queue/)
QUEUE_DIR = THALA_DIR / "queue"
QUEUE_FILE = QUEUE_DIR / "queue.json"
LOCK_FILE = QUEUE_DIR / "queue.lock"
PUBLICATIONS_FILE = QUEUE_DIR / "publications.json"
CURRENT_WORK_FILE = QUEUE_DIR / "current_work.json"
COST_CACHE_FILE = QUEUE_DIR / "cost_cache.json"
DAEMON_PID_FILE = QUEUE_DIR / "daemon.pid"
DAEMON_LOG_FILE = QUEUE_DIR / "daemon.log"
INCREMENTAL_DIR = QUEUE_DIR / "incremental"

# Output directory (was: output/ and .outputs/)
OUTPUT_DIR = THALA_DIR / "output"

# Editorial stances for publications
EDITORIAL_STANCES_DIR = THALA_DIR / "editorial_stances"

# Substack cookies (was: .substack-cookies.json at root)
SUBSTACK_COOKIES_FILE = THALA_DIR / ".substack-cookies.json"


def ensure_directories() -> None:
    """Create .thala directory structure if it doesn't exist."""
    THALA_DIR.mkdir(exist_ok=True)
    QUEUE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
