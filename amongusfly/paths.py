"""
Central path configuration. The data directory is the AMONGUSFLY_DATA environment variable,
defaulting to C:\amongusfly-data on Windows and ~/amongusfly-data elsewhere.
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
        return Path(r"C:\amongusfly-data")
    return Path.home() / "amongusfly-data"


DATA_DIR = Path(os.environ.get("AMONGUSFLY_DATA", _default_data_dir()))
RAW_DIR = DATA_DIR / "raw" / "malecns_v1"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = DATA_DIR / "results"  # large run outputs; small summaries go to docs/
