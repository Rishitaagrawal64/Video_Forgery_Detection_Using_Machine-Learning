"""Remove leftovers that were never cleaned up (NFR-005: do not keep user videos longer than needed).

Uploads that are never analyzed, or files left behind by a crash, would otherwise stay on disk forever.
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def purge_stale_files(directory: Path, max_age_hours: float, *, now: float | None = None) -> int:
    """Delete files/folders directly inside `directory` not modified for `max_age_hours`. Returns the count."""
    if max_age_hours <= 0 or not directory.is_dir():
        return 0
    cutoff = (time.time() if now is None else now) - max_age_hours * 3600
    removed = 0
    for entry in directory.iterdir():
        if entry.name.startswith("."):  # keep .gitkeep and similar
            continue
        try:
            if entry.stat().st_mtime >= cutoff:
                continue
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
            removed += 1
        except OSError:
            logger.warning("Could not remove stale entry %s", entry, exc_info=True)
    if removed:
        logger.info("Purged %d stale item(s) from %s", removed, directory)
    return removed
