#!/usr/bin/env python3
"""Export the exact TFDS ImageNet validation-label order for RAEv2 sampling."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default="imagenet2012")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    source = tfds.data_source(
        args.dataset,
        split=args.split,
        data_dir=args.data_dir,
        decoders={"image": tfds.decode.SkipDecoding()},
    )
    labels = np.fromiter(
        (int(source[index]["label"]) for index in range(len(source))),
        dtype=np.int32,
        count=len(source),
    )
    if labels.shape != (50_000,):
        raise ValueError(f"Expected 50,000 ImageNet labels, got {labels.shape}.")
    counts = np.bincount(labels, minlength=1000)
    if not np.array_equal(counts, np.full(1000, 50)):
        raise ValueError("Expected exactly 50 validation labels per ImageNet class.")

    labels_sha256 = hashlib.sha256(labels.tobytes()).hexdigest()
    builder = tfds.builder(args.dataset, data_dir=args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    with open(temporary, "wb") as handle:
        np.savez_compressed(
            handle,
            format_version=np.asarray(1, dtype=np.int64),
            dataset=np.asarray(args.dataset),
            split=np.asarray(args.split),
            tfds_version=np.asarray(str(builder.info.version)),
            labels=labels,
            labels_sha256=np.asarray(labels_sha256),
        )
    os.replace(temporary, args.output)
    print(f"Wrote {len(labels)} labels to {args.output}")
    print(f"labels_sha256={labels_sha256}")


if __name__ == "__main__":
    main()
