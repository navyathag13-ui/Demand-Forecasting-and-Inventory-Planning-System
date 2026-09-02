"""
utils.py — Shared helpers: logging setup, directory creation, file I/O.
"""

import logging
import sys
from pathlib import Path

import pandas as pd


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger to write to stdout with timestamps."""
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format=fmt,
        datefmt="%H:%M:%S",
        force=True,
    )


def ensure_dirs(*dirs: Path) -> None:
    """Create directories (including parents) if they do not exist."""
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path, index: bool = True) -> None:
    """Save a DataFrame to CSV, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    logging.getLogger(__name__).info("Saved %s (%d rows).", path.name, len(df))


def print_section(title: str, width: int = 60) -> None:
    """Print a formatted section header to stdout."""
    bar = "=" * width
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)
