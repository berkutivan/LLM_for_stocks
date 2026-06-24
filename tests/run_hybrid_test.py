"""Run hybrid pipeline test via `python -m pytest` (works when pytest.exe launcher is broken)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "Pipeline") not in sys.path:
    sys.path.insert(0, str(ROOT / "Pipeline"))

TEST_TARGET = (
    "tests/test_pipeline.py::TestHybridPipeline::test_hybrid_single_news_model_logs"
)


def main() -> int:
    return pytest.main([TEST_TARGET, "-s", "-v", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
