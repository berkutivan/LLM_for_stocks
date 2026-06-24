"""Скачивает ProsusAI/finbert в Pipeline/llm/models/weights/finbert."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from llm.models.connectors import DEFAULT_MODEL_DIR, download_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Download FinBERT weights into the repo")
    parser.add_argument(
        "--target",
        type=str,
        default=str(DEFAULT_MODEL_DIR),
        help="Destination directory for model files",
    )
    args = parser.parse_args()

    path = download_model(args.target)
    print(f"FinBERT saved to: {path}")


if __name__ == "__main__":
    main()
