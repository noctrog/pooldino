"""Canonical ADE20K semantic-segmentation data source.

TensorFlow Datasets' scene_parse150 builder currently downloads
annotations_instance.tar. Those RGB PNGs are not the 150-class semantic masks
used by the ADE20K benchmark. This source intentionally reads the official
ADEChallengeData2016/annotations grayscale masks instead.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


_SPLIT_DIRS = {
    "train": "training",
    "training": "training",
    "val": "validation",
    "validation": "validation",
    "test": "validation",
}
_EXPECTED_COUNTS = {"training": 20_210, "validation": 2_000}
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg"})


def resolve_ade20k_root(data_dir: str | Path | None = None) -> Path:
    """Resolve and validate an ADEChallengeData2016 dataset root."""

    candidates: list[Path] = []
    if data_dir is not None:
        candidates.append(Path(data_dir).expanduser())
    env_root = os.environ.get("ADE20K_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            Path(".data/ADEChallengeData2016"),
            Path.home() / ".cache" / "pooldino_datasets" / "ADEChallengeData2016",
        ]
    )

    checked: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.name != "ADEChallengeData2016":
            nested = candidate / "ADEChallengeData2016"
            if nested.is_dir():
                candidate = nested
        checked.append(candidate)
        if (candidate / "images").is_dir() and (candidate / "annotations").is_dir():
            return candidate
    locations = ", ".join(str(path) for path in checked)
    raise FileNotFoundError(
        "Official ADE20K semantic data was not found. Expected an "
        "ADEChallengeData2016 root containing images/ and annotations/. "
        f"Checked: {locations}"
    )


class ADE20KSemanticDataSource:
    """Grain-compatible source for official ADE20K semantic masks."""

    def __init__(
        self,
        data_dir: str | Path | None,
        split: str,
        *,
        verify_count: bool = True,
    ):
        if split not in _SPLIT_DIRS:
            raise ValueError(
                f"Unsupported ADE20K split {split!r}; expected one of "
                f"{sorted(_SPLIT_DIRS)}."
            )
        self.root = resolve_ade20k_root(data_dir)
        self.split_dir = _SPLIT_DIRS[split]
        image_dir = self.root / "images" / self.split_dir
        annotation_dir = self.root / "annotations" / self.split_dir
        if not image_dir.is_dir() or not annotation_dir.is_dir():
            raise FileNotFoundError(
                f"ADE20K split {self.split_dir!r} is incomplete under {self.root}."
            )

        images = {
            path.stem: path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        }
        annotations = {
            path.stem: path
            for path in annotation_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        }
        missing_annotations = sorted(images.keys() - annotations.keys())
        missing_images = sorted(annotations.keys() - images.keys())
        if missing_annotations or missing_images:
            raise ValueError(
                "ADE20K image/semantic-mask filenames do not match: "
                f"missing_annotations={missing_annotations[:5]}, "
                f"missing_images={missing_images[:5]}."
            )
        expected_count = _EXPECTED_COUNTS[self.split_dir]
        if verify_count and len(images) != expected_count:
            raise ValueError(
                f"ADE20K {self.split_dir} should contain {expected_count} paired "
                f"examples, found {len(images)}."
            )
        if not images:
            raise ValueError(f"ADE20K {self.split_dir} contains no paired examples.")
        self.samples = tuple(
            (images[stem], annotations[stem]) for stem in sorted(images)
        )

    def __repr__(self) -> str:
        """Return a stable identity so Grain checkpoints survive restarts."""

        return (
            f"{type(self).__module__}.{type(self).__qualname__}("
            f"root={str(self.root)!r}, split={self.split_dir!r}, "
            f"count={len(self.samples)})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        image_path, annotation_path = self.samples[int(index)]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to decode ADE20K image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        annotation = cv2.imread(str(annotation_path), cv2.IMREAD_UNCHANGED)
        if annotation is None:
            raise ValueError(
                f"Failed to decode ADE20K semantic mask: {annotation_path}"
            )
        if annotation.ndim != 2:
            raise ValueError(
                "ADE20K semantic masks must be single-channel PNGs from "
                "ADEChallengeData2016/annotations, but "
                f"{annotation_path} has shape {annotation.shape}. This may be an "
                "annotations_instance mask."
            )
        if image.shape[:2] != annotation.shape:
            raise ValueError(
                f"ADE20K image/mask shape mismatch for {image_path.stem}: "
                f"image={image.shape[:2]}, mask={annotation.shape}."
            )
        minimum = int(annotation.min())
        maximum = int(annotation.max())
        if minimum < 0 or maximum > 150:
            raise ValueError(
                f"ADE20K semantic labels must lie in [0, 150], got "
                f"[{minimum}, {maximum}] in {annotation_path}."
            )
        return {
            "image": np.asarray(image, dtype=np.uint8),
            "annotation": np.asarray(annotation, dtype=np.uint8),
        }


def ade20k_semantic_data_source(
    split: str,
    data_dir: str | Path | None,
) -> ADE20KSemanticDataSource:
    return ADE20KSemanticDataSource(data_dir, split)


def validate_ade20k_semantic_data(
    data_dir: str | Path | None,
) -> dict[str, object]:
    """Decode every example and report canonical counts and observed labels."""

    histogram = np.zeros(151, dtype=np.int64)
    split_counts: dict[str, int] = {}
    for split in ("training", "validation"):
        source = ADE20KSemanticDataSource(data_dir, split)
        split_counts[split] = len(source)
        for index in range(len(source)):
            annotation = source[index]["annotation"]
            histogram += np.bincount(annotation.ravel(), minlength=151)[:151]
    missing_classes = np.flatnonzero(histogram[1:] == 0) + 1
    if missing_classes.size:
        raise ValueError(
            "Canonical ADE20K semantic data is missing labeled classes: "
            f"{missing_classes.tolist()}."
        )
    return {
        "root": str(resolve_ade20k_root(data_dir)),
        "split_counts": split_counts,
        "observed_label_min": int(np.flatnonzero(histogram)[0]),
        "observed_label_max": int(np.flatnonzero(histogram)[-1]),
        "class_pixel_counts": histogram.tolist(),
    }
