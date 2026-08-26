"""Run-scoped file logging under ./logs/, alongside the CLI's Rich console output.

Rich `console.print` stays the user-facing summary; this logger captures the
step-by-step detail (useful for the parallel/multi-stage commands, where a
background run's only record afterwards is the log file) into a timestamped
file per invocation.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")

_logger = logging.getLogger("talkshow_crawler")


def setup_logging(command: str, level: int = logging.INFO) -> Path:
    """Point the 'talkshow_crawler' logger at a fresh timestamped file under ./logs/.

    Safe to call once per CLI invocation. Returns the log file path.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = LOG_DIR / f"{command}-{stamp}.log"

    _logger.setLevel(level)
    _logger.handlers.clear()
    _logger.propagate = False

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s [%(threadName)s] %(message)s", datefmt="%H:%M:%S")
    )
    _logger.addHandler(handler)
    return log_path


def get_logger() -> logging.Logger:
    return _logger
