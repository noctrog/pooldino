"""Data pipeline for PoolDINO semantic-segmentation eval.

ADE20K uses the official ``ADEChallengeData2016/annotations`` semantic masks;
Pascal VOC 2012 uses TFDS. Augmentations use albumentations' joint image +
mask API so geometric ops are applied to both simultaneously (NEAREST for masks).

The training loop wants batches of the form::

    {"image": (B, H, W, 3) float32,    # normalized, RGB
     "mask":  (B, H, W)    int32}      # ignore = -1 after normalization

Ignore-label normalization:
- ADE20K: raw 0 = unlabeled, 1..150 = classes. We shift classes to 0..149 and
  set previously-0 positions to ``ignore_index`` (-1).
- VOC: raw 255 = ignore. We map 255 -> -1. Classes 0..20 are unchanged.
"""

from __future__ import annotations

import os

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("ALBUMENTATIONS_NO_TELEMETRY", "1")

import albumentations as A
import cv2
import grain.python as grain
import numpy as np

from pooldino.data import DataConfig
from pooldino.data.data import DataLoaders, create_dataloaders, seed_albumentations

from pooldino.segmentation.configs import SegDataConfig


def _normalize_mask_labels(
    mask: np.ndarray,
    *,
    dataset: str,
    raw_ignore_index: int,
    ignore_index: int,
) -> np.ndarray:
    """Convert raw mask labels into a canonical ``int32`` class map.

    ADE20K raw labels are 0 (ignore) .. 150 (classes). We shift to 0..149
    and mark the ignore positions. VOC raw labels are 0..20 (classes) and
    255 (ignore); we map 255 -> ignore_index.
    """
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(
            "Semantic segmentation masks must be single-channel; "
            f"got shape {mask.shape}. For ADE20K, use "
            "ADEChallengeData2016/annotations rather than annotations_instance."
        )
    out = mask.astype(np.int32, copy=True)
    if dataset == "ade":
        if out.size and (int(out.min()) < 0 or int(out.max()) > 150):
            raise ValueError(
                "ADE20K semantic labels must lie in [0, 150], got "
                f"[{int(out.min())}, {int(out.max())}]."
            )
        was_bg = out == raw_ignore_index
        out -= 1
        out[was_bg] = ignore_index
    elif dataset == "voc":
        out[out == raw_ignore_index] = ignore_index
    else:
        raise ValueError(f"Unknown seg dataset key: {dataset!r}")
    return out


_COLOR_JITTER_PRESETS: dict[str, tuple[float, float, float, float]] = {
    # Aligned with mmseg's PhotoMetricDistortion: brightness_delta=32/255 ≈ 0.125,
    # contrast/saturation range (0.5, 1.5) ≈ albumentations jitter=0.5,
    # hue_delta=18/180 = 0.1.
    "mmseg": (0.125, 0.5, 0.5, 0.1),
    # Pre-change default used by several SSL recipes; kept as an ablation knob.
    "strong": (0.4, 0.4, 0.4, 0.1),
}


class SegTrainAugmentations(grain.RandomMapTransform):
    """Joint image + mask training augmentations for semantic segmentation.

    Mirrors mmsegmentation's ADE20K / VOC train pipeline
    (``RandomResize(ratio_range) -> RandomCrop(cat_max_ratio) ->
    RandomFlip -> PhotoMetricDistortion``) using the albumentations joint
    image+mask API, split into three albumentations composes to make room
    for the class-balanced crop retry:

    1. **Pre-crop** ``RandomScale + PadIfNeeded`` runs once per sample.
       Padding uses ``raw_ignore_index`` for the mask (so padded regions
       become ``ignore_index`` after :func:`_normalize_mask_labels`) and 0
       for the image.
    2. **Crop** picks a random ``(mask_resolution, mask_resolution)`` window
       with up to ``seg_cfg.cat_max_attempts`` retries, rejecting windows
       where a single non-ignore class covers more than
       ``seg_cfg.cat_max_ratio`` of pixels (mmseg's
       ``RandomCrop(cat_max_ratio=0.75)`` behavior, ``transforms.py:281``).
       The last attempt is accepted even if the check fails.
    3. **Post-crop** horizontal flip + ColorJitter + ImageNet normalize.
    """

    def __init__(self, seg_cfg: SegDataConfig):
        super().__init__()
        self.seg_cfg = seg_cfg
        color_jitter_p = 0.8 if seg_cfg.color_jitter else 0.0
        b, c, s, h = _COLOR_JITTER_PRESETS[seg_cfg.color_jitter_strength]
        mask_res = seg_cfg.mask_resolution
        lo, hi = seg_cfg.rrc_scale
        # Albumentations' RandomScale treats ``scale_limit=(a, b)`` as a
        # +/- delta around 1, i.e. scale in [1+a, 1+b]. Convert our
        # (lo, hi) tuple from "multiplier bounds" into that delta space.
        scale_limit = (lo - 1.0, hi - 1.0)
        self.pre_crop = A.Compose(
            [
                A.RandomScale(
                    scale_limit=scale_limit,
                    interpolation=cv2.INTER_LINEAR,
                    mask_interpolation=cv2.INTER_NEAREST,
                    p=1.0,
                ),
                A.PadIfNeeded(
                    min_height=mask_res,
                    min_width=mask_res,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    fill_mask=seg_cfg.raw_ignore_index,
                    p=1.0,
                ),
            ]
        )
        self.post_crop = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=b,
                    contrast=c,
                    saturation=s,
                    hue=h,
                    p=color_jitter_p,
                ),
                A.Normalize(mean=seg_cfg.normalization_mean, std=seg_cfg.normalization_std),
            ]
        )

    def _accept_crop(self, mask_crop: np.ndarray) -> bool:
        """mmseg-style ``cat_max_ratio`` check on a candidate crop."""
        labels, counts = np.unique(mask_crop, return_counts=True)
        keep = labels != self.seg_cfg.raw_ignore_index
        counts = counts[keep]
        total = counts.sum()
        if total == 0 or len(counts) <= 1:
            return False
        return counts.max() / total < self.seg_cfg.cat_max_ratio

    def random_map(
        self, element: dict[str, np.ndarray], rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        seed_albumentations(rng, self.pre_crop, self.post_crop)
        image = element["image"]
        mask = element[self.seg_cfg.mask_field]
        pre = self.pre_crop(image=image, mask=mask)
        img_pre, msk_pre = pre["image"], pre["mask"]
        if msk_pre.ndim != 2:
            raise ValueError(
                "Expected a single-channel semantic mask after augmentation, "
                f"got {msk_pre.shape}."
            )

        mask_res = self.seg_cfg.mask_resolution
        max_y = img_pre.shape[0] - mask_res
        max_x = img_pre.shape[1] - mask_res
        skip_check = self.seg_cfg.cat_max_ratio >= 1.0
        # Retry up to cat_max_attempts times to find a class-balanced crop;
        # fall through with the last candidate if none passes.
        for _ in range(max(1, self.seg_cfg.cat_max_attempts)):
            y = int(rng.integers(0, max_y + 1))
            x = int(rng.integers(0, max_x + 1))
            img_crop = img_pre[y : y + mask_res, x : x + mask_res]
            msk_crop = msk_pre[y : y + mask_res, x : x + mask_res]
            if skip_check or self._accept_crop(msk_crop):
                break

        post = self.post_crop(image=img_crop, mask=msk_crop)
        normalized_mask = _normalize_mask_labels(
            post["mask"],
            dataset=self.seg_cfg.dataset_key,
            raw_ignore_index=self.seg_cfg.raw_ignore_index,
            ignore_index=self.seg_cfg.ignore_index,
        )
        return {
            "image": post["image"].astype(np.float32),
            "mask": normalized_mask,
        }


class SegValAugmentations(grain.MapTransform):
    """Deterministic val augmentations.

    Two modes:
    - ``val_keep_ratio=False`` (default): preserve the complete image by
      resizing its longest side to ``mask_resolution`` and padding the short
      side with the raw ignore label. Batchable at any batch size.
    - ``val_keep_ratio=True``: aspect-preserving ``LongestMaxSize`` (matches
      mmseg's ``Resize(scale=(2*mask_res, mask_res), keep_ratio=True)``). The
      output size varies per sample, so this mode **requires batch_size=1**
      at the data loader.
    """

    def __init__(self, seg_cfg: SegDataConfig):
        super().__init__()
        self.seg_cfg = seg_cfg
        mask_res = seg_cfg.mask_resolution
        resize_kwargs = dict(
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
        )
        if seg_cfg.val_keep_ratio:
            resize_ops = [
                A.LongestMaxSize(max_size=2 * mask_res, **resize_kwargs),
                A.SmallestMaxSize(max_size=mask_res, **resize_kwargs),
            ]
        else:
            resize_ops = [
                A.LongestMaxSize(max_size=mask_res, **resize_kwargs),
                A.PadIfNeeded(
                    min_height=mask_res,
                    min_width=mask_res,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=0,
                    fill_mask=seg_cfg.raw_ignore_index,
                    p=1.0,
                ),
            ]
        self.transform = A.Compose(
            [
                *resize_ops,
                A.Normalize(mean=seg_cfg.normalization_mean, std=seg_cfg.normalization_std),
            ]
        )

    def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        image = element["image"]
        mask = element[self.seg_cfg.mask_field]
        result = self.transform(image=image, mask=mask)
        normalized_mask = _normalize_mask_labels(
            result["mask"],
            dataset=self.seg_cfg.dataset_key,
            raw_ignore_index=self.seg_cfg.raw_ignore_index,
            ignore_index=self.seg_cfg.ignore_index,
        )
        return {
            "image": result["image"].astype(np.float32),
            "mask": normalized_mask,
        }


def create_seg_dataloaders(
    seg_cfg: SegDataConfig,
    *,
    batch_size: int,
    train_epochs: int | None = None,
    val_epochs: int | None = 1,
    num_workers: int = 8,
    gcs_bucket: str | None = None,
    val_only: bool = False,
) -> DataLoaders:
    """Build train + val loaders for seg.

    When ``val_only=True`` the train loader is still constructed (grain loads
    the data source lazily) but will only be iterated over by the caller as
    much as it needs. Callers that never touch ``train_loader`` pay no cost.
    """
    data_cfg = DataConfig(
        num_workers=num_workers,
        normalization_mean=seg_cfg.normalization_mean,
        normalization_std=seg_cfg.normalization_std,
        dataset=seg_cfg.tfds_name,
        num_classes=seg_cfg.num_classes,
        train_name=seg_cfg.train_split,
        val_name=seg_cfg.val_split,
        source_resolution=seg_cfg.mask_resolution,
        data_dir=seg_cfg.data_dir,
    )
    val_aug = SegValAugmentations(seg_cfg)
    # In val-only mode we bind the train loader to the deterministic val aug
    # so iterating it once is safe; callers are expected to skip it.
    train_aug = val_aug if val_only else SegTrainAugmentations(seg_cfg)
    return create_dataloaders(
        data_cfg,
        batch_size=batch_size,
        train_epochs=train_epochs,
        val_epochs=val_epochs,
        train_aug=train_aug,
        val_aug=val_aug,
        drop_remainder_train=not val_only,
        drop_remainder_val=False,
        val_shuffle=False,
        gcs_bucket=gcs_bucket,
    )
