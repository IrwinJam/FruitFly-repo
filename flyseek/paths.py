"""
Central path configuration, so the code runs on this PC, a fresh clone, or Kaggle.

Data directory: set the FLYSEEK_DATA environment variable (e.g. /kaggle/input/flyseek-data
on Kaggle). Defaults to C:\\flyseek-data on Windows, ~/flyseek-data elsewhere.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"
DOCS_DIR = REPO_ROOT / "docs"
VIEWER_DATA_DIR = REPO_ROOT / "viewer" / "public" / "data"


def _default_data_dir() -> Path:
    if sys.platform.startswith("win"):
        return Path(r"C:\flyseek-data")
    return Path.home() / "flyseek-data"


DATA_DIR = Path(os.environ.get("FLYSEEK_DATA", _default_data_dir()))
RAW_DIR = DATA_DIR / "raw" / "malecns_v1"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = DATA_DIR / "results"  # large run outputs; small summaries go to docs/
