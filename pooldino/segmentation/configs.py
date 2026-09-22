"""Shared configuration/helpers for the PoolDINO paper implementation."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Literal

ColorJitterStrength = Literal["mmseg", "strong"]


@dataclass
class SegDataConfig:
    """Seg-specific data config.

    Carries the subset of ``DataConfig`` fields needed for segmentation plus
    mask/encoder resolution and the ignore label.
    """

    dataset_key: str
    tfds_name: str
    train_split: str
    val_split: str
    mask_field: str
    num_classes: int
    ignore_index: int
    raw_ignore_index: int
    mask_resolution: int
    encoder_resolution: int
    rrc_scale: tuple[float, float]
    color_jitter: bool
    data_dir: str | None = None
    num_workers: int = 8
    normalization_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalization_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    color_jitter_strength: ColorJitterStrength = "mmseg"
    val_keep_ratio: bool = False
    """Use variable-size mmseg-style validation. Forces eval batch_size=1.

    The fixed-size default also preserves aspect ratio, but pads each resized
    image to a batchable square rather than returning variable dimensions.
    """
    cat_max_ratio: float = 0.75
    """Reject training crops where a single (non-ignore) class covers more than
    this fraction of pixels. Matches mmseg's ``RandomCrop(cat_max_ratio=0.75)``;
    set to ``1.0`` to disable the retry. Improves tail-class coverage."""
    cat_max_attempts: int = 10
    """Max retry attempts when looking for a crop below ``cat_max_ratio``."""
    patch_size: int = 14
    """PoolDINO encoder patch size. Used by multi-scale TTA to snap scaled
    encoder resolutions to a valid (divisible) size."""

