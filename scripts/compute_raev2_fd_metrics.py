#!/usr/bin/env python3
"""Compute the distributional metrics used by the released RAEv2 code."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", type=Path)
    parser.add_argument("--metrics", nargs="+", default=["fid"])
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.samples) as archive:
        images = archive["arr_0"]
    if images.dtype != np.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError("fd_evaluator images must be uint8 NHWC RGB.")

    import torch
    from fd_evaluator import compute_metrics

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = dict(
        images=images,
        metrics=args.metrics,
        reference_images=None,
        device=device,
        batch_size=args.batch_size,
        feature_cache_dir=os.environ.get(
            "NANOGEN_EVALS_CACHE_DIR",
            str(Path.home() / ".cache" / "nanogen-evals" / "features"),
        ),
        feature_cache_key=None,
        reference_feature_cache_key="imagenet256_val",
        verbose=True,
    )
    if args.reference is not None:
        kwargs["fid_reference"] = str(args.reference)
    results = {
        name: float(value) for name, value in compute_metrics(**kwargs).items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
