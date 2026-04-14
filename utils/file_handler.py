"""
FileHandler utilities — safe file I/O, upload validation, and temp file management.

All file processing in DocAgent goes through these helpers for consistent
validation, size checking, and cleanup.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

ALLOWED_EXTENSIONS: set = {".pdf", ".xlsx", ".xls", ".csv"}
DEFAULT_MAX_SIZE_MB: int = 50


def validate_file(
    file_path: Path,
    max_size_mb: int = DEFAULT_MAX_SIZE_MB,
) -> Optional[str]:
    """
    Validate a file before processing.

    Returns:
        None if the file is valid.
        An error message string if invalid.
    """
    if not file_path.exists():
        return f"File not found: {file_path}"

    if not file_path.is_file():
        return f"Path is not a file: {file_path}"

    ext = file_path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return (
            f"Unsupported file type '{ext}'. "
            f"Accepted: {sorted(ALLOWED_EXTENSIONS)}"
        )

    size_mb = file_path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        return (
            f"File is too large ({size_mb:.1f} MB). "
            f"Maximum allowed: {max_size_mb} MB"
        )

    return None


def save_upload(
    file_bytes: bytes,
    original_name: str,
    temp_dir: Optional[Path] = None,
) -> Path:
    """
    Persist uploaded bytes to a temporary file and return its path.

    If `temp_dir` is None, a new temp directory is created automatically.
    The caller is responsible for calling `cleanup_temp_dir()` when done.
    """
    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="docagent_"))
    temp_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(original_name).name         # strip any path traversal
    dest = temp_dir / safe_name
    dest.write_bytes(file_bytes)
    logger.debug(f"Saved upload → {dest}  ({len(file_bytes) / 1024:.1f} KB)")
    return dest


def cleanup_temp_dir(temp_dir: Path) -> None:
    """Remove a temporary directory and all its contents."""
    try:
        if temp_dir.exists():
            shutil.rmtree(str(temp_dir))
            logger.debug(f"Cleaned up temp dir: {temp_dir}")
    except Exception as exc:
        logger.warning(f"Could not clean up {temp_dir}: {exc}")


def get_file_size_mb(file_path: Path) -> float:
    """Return file size in megabytes."""
    return file_path.stat().st_size / (1024 * 1024)


def make_temp_dir() -> Path:
    """Create and return a fresh temporary directory path."""
    return Path(tempfile.mkdtemp(prefix="docagent_"))
