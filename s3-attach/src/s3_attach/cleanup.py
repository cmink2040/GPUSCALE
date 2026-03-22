"""Clean up locally downloaded model files after successful upload."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


def remove_downloaded(local_path: Path) -> None:
    """Remove a downloaded directory or file after successful upload.

    Args:
        local_path: Path to the directory or file to remove.

    Logs a warning and continues if removal fails (non-fatal).
    """
    if not local_path.exists():
        log.debug("Path does not exist, nothing to clean: %s", local_path)
        return

    try:
        if local_path.is_dir():
            file_count = sum(1 for _ in local_path.rglob("*") if _.is_file())
            total_size = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file())
            shutil.rmtree(local_path)
            log.info(
                "Cleaned up %s (%d files, %.1f MB)",
                local_path,
                file_count,
                total_size / (1024 * 1024),
            )
        else:
            size = local_path.stat().st_size
            local_path.unlink()
            log.info("Cleaned up %s (%.1f MB)", local_path, size / (1024 * 1024))
    except OSError as e:
        log.warning("Failed to clean up %s: %s", local_path, e)


def cleanup_download_dir(download_dir: Path) -> None:
    """Remove the entire downloads directory if it exists and is empty or fully processed.

    This is called after all models have been synced. Only removes the directory
    if it is empty (all per-model subdirectories should have been cleaned up
    individually by remove_downloaded).

    Args:
        download_dir: The top-level downloads directory.
    """
    if not download_dir.exists():
        return

    remaining = list(download_dir.rglob("*"))
    remaining_files = [p for p in remaining if p.is_file()]

    if remaining_files:
        log.warning(
            "Download directory %s still contains %d file(s) — not removing. "
            "Some uploads may have failed.",
            download_dir,
            len(remaining_files),
        )
        return

    try:
        shutil.rmtree(download_dir)
        log.info("Removed empty download directory: %s", download_dir)
    except OSError as e:
        log.warning("Failed to remove download directory %s: %s", download_dir, e)
