"""Shared configuration/helpers for the PoolDINO paper implementation."""

import fcntl

import json

from pathlib import Path

from typing import Any

EVAL_RESULTS_FILENAME = "eval_results.json"


def load_eval_results(checkpoint_path: Path) -> dict[str, Any]:
    """Load evaluation results from JSON file in checkpoint folder."""
    results_path = checkpoint_path / EVAL_RESULTS_FILENAME
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return {}


def save_eval_results(checkpoint_path: Path, key: str, results: dict[str, Any]) -> None:
    """Save evaluation results to JSON file in checkpoint folder.

    Merges new results under the given key with existing results.
    Uses file locking to handle concurrent writes from multiple workers.
    """
    results_path = checkpoint_path / EVAL_RESULTS_FILENAME
    lock_path = checkpoint_path / f".{EVAL_RESULTS_FILENAME}.lock"

    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            existing = load_eval_results(checkpoint_path)
            existing[key] = results
            with open(results_path, "w") as f:
                json.dump(existing, f, indent=2)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

