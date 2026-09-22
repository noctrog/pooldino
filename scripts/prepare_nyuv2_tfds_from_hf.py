#!/usr/bin/env python3
"""Prepare official TFDS NYUv2 records from the FastDepth HF mirror.

The TFDS builder normally downloads one archive from the MIT FastDepth host.
That host is occasionally unavailable.  The public Hugging Face mirror contains
the same HDF5 examples split into uncompressed tar shards; this script changes
only the download/iteration path and writes the standard
``nyu_depth_v2/0.0.1`` TFDS representation.
"""

from __future__ import annotations

import argparse
import io
import tarfile
import urllib.parse  # Ensure tensorflow_datasets sees urllib.parse on Python 3.13.
from collections.abc import Iterator, Sequence
from pathlib import Path

import h5py
import numpy as np
import tensorflow_datasets as tfds
from tensorflow_datasets.datasets.nyu_depth_v2 import (
    nyu_depth_v2_dataset_builder as nyuv2_builder,
)


_REVISION = "50579a0b591445f91bf7269b28611ad1cc6a05d2"
_BASE_URL = f"https://huggingface.co/datasets/sayakpaul/nyu_depth_v2/resolve/{_REVISION}/data"
_URLS = {
    "train": [f"{_BASE_URL}/train-{index:06d}.tar" for index in range(12)],
    "validation": [f"{_BASE_URL}/val-{index:06d}.tar" for index in range(2)],
}
_EXPECTED_COUNTS = {"train": 47_584, "validation": 654}


def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    del self
    archives = dl_manager.download(_URLS)
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={"archives": archives["train"], "split": "train"},
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                "archives": archives["validation"],
                "split": "validation",
            },
        ),
    ]


def _generate_examples(
    self,
    archives: Sequence[str | Path],
    split: str,
) -> Iterator[tuple[str, dict[str, np.ndarray]]]:
    del self
    count = 0
    for archive_path in archives:
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".h5"):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"Could not read {member.name!r} from {archive_path}.")
                with h5py.File(io.BytesIO(extracted.read()), "r") as h5_file:
                    image = np.asarray(h5_file["rgb"])
                    depth = np.asarray(h5_file["depth"], dtype=np.float16)
                yield (
                    f"{split}-{count:06d}",
                    {
                        "image": np.transpose(image, (1, 2, 0)),
                        "depth": depth,
                    },
                )
                count += 1
    expected = _EXPECTED_COUNTS[split]
    if count != expected:
        raise ValueError(f"Expected {expected} {split} examples, found {count}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/data/user/tensorflow_datasets_array_record"),
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("/data/user/tfds_downloads/nyu_depth_v2"),
    )
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.download_dir.mkdir(parents=True, exist_ok=True)

    # Retain the official builder class/name/version and feature schema so the
    # normal training loader can consume the result without a custom backend.
    nyuv2_builder.Builder._split_generators = _split_generators
    nyuv2_builder.Builder._generate_examples = _generate_examples
    builder = nyuv2_builder.Builder(data_dir=str(args.data_dir))
    builder.download_and_prepare(
        download_dir=str(args.download_dir),
        download_config=tfds.download.DownloadConfig(
            register_checksums=False,
        ),
        # Grain's random-access TFDSDataSource requires ArrayRecord. TFRecord
        # can be read sequentially by tf.data but cannot back this loader.
        file_format=tfds.core.FileFormat.ARRAY_RECORD,
    )
    if builder.info.file_format != tfds.core.FileFormat.ARRAY_RECORD:
        raise ValueError(f"Expected ArrayRecord output, got {builder.info.file_format}.")
    for split, expected in _EXPECTED_COUNTS.items():
        actual = builder.info.splits[split].num_examples
        if actual != expected:
            raise ValueError(f"Expected {expected} {split} examples, found {actual}.")
    print(builder.info)


if __name__ == "__main__":
    main()
