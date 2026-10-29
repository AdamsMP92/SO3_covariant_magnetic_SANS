"""Test configuration for the source-tree layout."""

from __future__ import annotations

from pathlib import Path
import sys


SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(SOURCE_DIRECTORY))
