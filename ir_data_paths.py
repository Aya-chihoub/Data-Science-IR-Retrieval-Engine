"""
Data directory resolution: local repo, env IR_DATA_DIR, or Kaggle /kaggle/input.
"""
from __future__ import annotations

import os
from pathlib import Path


def _resolve_data_dir() -> Path:
    env = os.environ.get("IR_DATA_DIR")
    if env:
        return Path(env).resolve()

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        # Competition data + optional extra datasets; pick first tree that has docs.json
        for child in sorted(kaggle_input.iterdir()):
            if not child.is_dir():
                continue
            if (child / "docs.json").is_file():
                return child.resolve()
            nested = child / "retrieval-engine-competition"
            if nested.is_dir() and (nested / "docs.json").is_file():
                return nested.resolve()

    return Path("data/retrieval-engine-competition")


DATA_DIR = _resolve_data_dir()
