"""NYUv2 data, losses, and metrics for frozen-PoolDINO depth decoding."""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("ALBUMENTATIONS_NO_TELEMETRY", "1")

import albumentations as A
import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np

from pooldino.data import DataConfig
from pooldino.data.data import (
    DataLoaders,
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    create_dataloaders,
    seed_albumentations,
)


@dataclass
class DepthDataConfig:
    dataset: str = "nyu_depth_v2"
    train_split: str = "train"
    val_split: str = "validation"
    depth_field: str = "depth"
    height: int = 480
    width: int = 640
    min_depth: float = 0.1
    max_depth: float = 10.0
    silog_lambda: float = 0.5
    gradient_weight: float = 0.0
    eval_crop: tuple[int, int, int, int] | None = (45, 471, 41, 601)
    normalization_mean: tuple[float, float, float] = IMAGENET_DEFAULT_MEAN
    normalization_std: tuple[float, float, float] = IMAGENET_DEFAULT_STD
    color_jitter: bool = True
    num_workers: int = 8

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("Depth image dimensions must be positive.")
        if not 0 < self.min_depth < self.max_depth:
            raise ValueError("Depth bounds must satisfy 0 < min_depth < max_depth.")
        if not 0 <= self.silog_lambda <= 1:
            raise ValueError("silog_lambda must lie in [0, 1].")
        if self.gradient_weight < 0:
            raise ValueError("gradient_weight must be non-negative.")
        if self.eval_crop is not None:
            y0, y1, x0, x1 = self.eval_crop
            if not (0 <= y0 < y1 <= self.height and 0 <= x0 < x1 <= self.width):
                raise ValueError(
                    f"eval_crop={self.eval_crop} is outside {(self.height, self.width)}."
                )


class DepthTrainAugmentations(grain.RandomMapTransform):
    """Photometric augmentation plus a paired horizontal RGB/depth flip."""

    def __init__(self, cfg: DepthDataConfig):
        super().__init__()
        self.cfg = cfg
        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                    p=0.8 if cfg.color_jitter else 0.0,
                ),
                A.Normalize(mean=cfg.normalization_mean, std=cfg.normalization_std),
            ]
        )

    def random_map(
        self,
        element: dict[str, np.ndarray],
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        seed_albumentations(rng, self.transform)
        result = self.transform(
            image=element["image"],
            mask=element[self.cfg.depth_field],
        )
        return {
            "image": result["image"].astype(np.float32),
            "depth": np.asarray(result["mask"], dtype=np.float32),
        }


class DepthValAugmentations(grain.MapTransform):
    """Deterministic ImageNet normalization; depth remains in metres."""

    def __init__(self, cfg: DepthDataConfig):
        super().__init__()
        self.cfg = cfg
        self.transform = A.Compose(
            [A.Normalize(mean=cfg.normalization_mean, std=cfg.normalization_std)]
        )

    def map(self, element: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        result = self.transform(image=element["image"])
        return {
            "image": result["image"].astype(np.float32),
            "depth": np.asarray(element[self.cfg.depth_field], dtype=np.float32),
        }


def create_depth_dataloaders(
    cfg: DepthDataConfig,
    *,
    batch_size: int,
    train_epochs: int | None = None,
    val_epochs: int | None = 1,
    num_workers: int | None = None,
    gcs_bucket: str | None = None,
) -> DataLoaders:
    """Create the TFDS NYUv2 train and validation loaders."""

    data_cfg = DataConfig(
        dataset=cfg.dataset,
        train_name=cfg.train_split,
        val_name=cfg.val_split,
        source_resolution=max(cfg.height, cfg.width),
        num_classes=1,
        num_workers=cfg.num_workers if num_workers is None else num_workers,
        normalization_mean=cfg.normalization_mean,
        normalization_std=cfg.normalization_std,
    )
    return create_dataloaders(
        data_cfg,
        batch_size=batch_size,
        train_epochs=train_epochs,
        val_epochs=val_epochs,
        train_aug=DepthTrainAugmentations(cfg),
        val_aug=DepthValAugmentations(cfg),
        drop_remainder_train=True,
        drop_remainder_val=False,
        val_shuffle=False,
        gcs_bucket=gcs_bucket,
    )


def depth_valid_mask(
    target_depth: jax.Array,
    *,
    min_depth: float,
    max_depth: float,
    eval_crop: tuple[int, int, int, int] | None = None,
    valid_batch_size: jax.Array | int | None = None,
) -> jax.Array:
    """Return valid pixels, excluding crop and padded-batch entries."""

    valid = (
        jnp.isfinite(target_depth)
        & (target_depth >= min_depth)
        & (target_depth <= max_depth)
    )
    if eval_crop is not None:
        y0, y1, x0, x1 = eval_crop
        crop_mask = jnp.zeros(target_depth.shape[-2:], dtype=jnp.bool_)
        crop_mask = crop_mask.at[y0:y1, x0:x1].set(True)
        valid = valid & crop_mask
    if valid_batch_size is not None:
        sample_valid = jnp.arange(target_depth.shape[0]) < valid_batch_size
        valid = valid & sample_valid[:, None, None]
    return valid


def resize_log_depth_in_metric_space(
    log_depth: jax.Array,
    *,
    output_hw: tuple[int, int],
    eps: float = 1e-6,
) -> jax.Array:
    """Resize an unpatchified log-depth map after converting it to metres."""

    if log_depth.ndim != 3:
        raise ValueError(f"Expected log depth in BHW format, got {log_depth.shape}.")
    if output_hw[0] <= 0 or output_hw[1] <= 0:
        raise ValueError(f"output_hw must be positive, got {output_hw}.")
    metric_depth = jnp.exp(jnp.clip(log_depth.astype(jnp.float32), -20.0, 20.0))
    if metric_depth.shape[-2:] != output_hw:
        metric_depth = jax.image.resize(
            metric_depth[..., None],
            (metric_depth.shape[0], output_hw[0], output_hw[1], 1),
            method="bilinear",
            antialias=False,
        )[..., 0]
    return jnp.log(jnp.maximum(metric_depth, eps))


def silog_loss(
    log_prediction: jax.Array,
    target_depth: jax.Array,
    valid: jax.Array,
    *,
    coefficient: float = 0.5,
    eps: float = 1e-6,
) -> jax.Array:
    """Scale-invariant log-depth loss over the declared valid pixels."""

    safe_target = jnp.where(valid, target_depth, 1.0)
    error = log_prediction.astype(jnp.float32) - jnp.log(safe_target.astype(jnp.float32))
    count = jnp.sum(valid, dtype=jnp.float32)
    denom = jnp.maximum(count, 1.0)
    mean = jnp.sum(jnp.where(valid, error, 0.0)) / denom
    mean_sq = jnp.sum(jnp.where(valid, jnp.square(error), 0.0)) / denom
    variance = jnp.maximum(mean_sq - coefficient * jnp.square(mean), 0.0)
    return jnp.where(count > 0, jnp.sqrt(variance + eps), 0.0)


def log_depth_gradient_loss(
    log_prediction: jax.Array,
    target_depth: jax.Array,
    valid: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return mean gradient error plus its sum and valid pair count."""

    safe_log_target = jnp.log(jnp.where(valid, target_depth, 1.0).astype(jnp.float32))
    prediction = log_prediction.astype(jnp.float32)

    valid_x = valid[..., :, 1:] & valid[..., :, :-1]
    valid_y = valid[..., 1:, :] & valid[..., :-1, :]
    error_x = jnp.abs(
        (prediction[..., :, 1:] - prediction[..., :, :-1])
        - (safe_log_target[..., :, 1:] - safe_log_target[..., :, :-1])
    )
    error_y = jnp.abs(
        (prediction[..., 1:, :] - prediction[..., :-1, :])
        - (safe_log_target[..., 1:, :] - safe_log_target[..., :-1, :])
    )
    error_sum = jnp.sum(jnp.where(valid_x, error_x, 0.0)) + jnp.sum(
        jnp.where(valid_y, error_y, 0.0)
    )
    pair_count = jnp.sum(valid_x, dtype=jnp.float32) + jnp.sum(
        valid_y, dtype=jnp.float32
    )
    mean_error = error_sum / jnp.maximum(pair_count, 1.0)
    return mean_error, error_sum, pair_count


def depth_sufficient_statistics(
    log_prediction: jax.Array,
    target_depth: jax.Array,
    valid: jax.Array,
    *,
    min_depth: float,
    max_depth: float,
) -> dict[str, jax.Array]:
    """Return additive statistics for exact dataset-level depth metrics."""

    log_prediction = log_prediction.astype(jnp.float32)
    target = target_depth.astype(jnp.float32)
    safe_target = jnp.where(valid, target, 1.0)
    clipped_log_prediction = jnp.clip(
        log_prediction,
        jnp.log(jnp.asarray(min_depth, dtype=jnp.float32)),
        jnp.log(jnp.asarray(max_depth, dtype=jnp.float32)),
    )
    prediction = jnp.exp(clipped_log_prediction)
    error = prediction - safe_target
    log_error = clipped_log_prediction - jnp.log(safe_target)
    ratio = jnp.maximum(
        prediction / safe_target,
        safe_target / jnp.maximum(prediction, min_depth),
    )
    valid_f = valid.astype(jnp.float32)
    gradient_mean, gradient_sum, gradient_count = log_depth_gradient_loss(
        clipped_log_prediction,
        target,
        valid,
    )
    del gradient_mean
    return {
        "count": jnp.sum(valid_f),
        "abs_rel_sum": jnp.sum(valid_f * jnp.abs(error) / safe_target),
        "sq_error_sum": jnp.sum(valid_f * jnp.square(error)),
        "log_sq_error_sum": jnp.sum(valid_f * jnp.square(log_error)),
        "log_error_sum": jnp.sum(valid_f * log_error),
        "delta1_count": jnp.sum(valid_f * (ratio < 1.25)),
        "delta2_count": jnp.sum(valid_f * (ratio < 1.25**2)),
        "delta3_count": jnp.sum(valid_f * (ratio < 1.25**3)),
        "gradient_error_sum": gradient_sum,
        "gradient_pair_count": gradient_count,
    }


def summarize_depth_statistics(
    statistics: dict[str, float],
    *,
    silog_lambda: float,
    silog_eps: float = 1e-6,
) -> dict[str, float]:
    """Convert accumulated sufficient statistics into reported metrics."""

    count = max(float(statistics.get("count", 0.0)), 1.0)
    mean_log_error = float(statistics.get("log_error_sum", 0.0)) / count
    mean_log_sq = float(statistics.get("log_sq_error_sum", 0.0)) / count
    silog_variance = max(
        mean_log_sq - silog_lambda * mean_log_error**2,
        0.0,
    )
    gradient_count = max(float(statistics.get("gradient_pair_count", 0.0)), 1.0)
    return {
        "abs_rel": float(statistics.get("abs_rel_sum", 0.0)) / count,
        "rmse": float(np.sqrt(float(statistics.get("sq_error_sum", 0.0)) / count)),
        "log_rmse": float(np.sqrt(mean_log_sq)),
        "silog": float(np.sqrt(silog_variance + silog_eps)),
        "delta1": float(statistics.get("delta1_count", 0.0)) / count,
        "delta2": float(statistics.get("delta2_count", 0.0)) / count,
        "delta3": float(statistics.get("delta3_count", 0.0)) / count,
        "gradient_error": float(statistics.get("gradient_error_sum", 0.0))
        / gradient_count,
        "valid_pixels": float(statistics.get("count", 0.0)),
    }


__all__ = [
    "DepthDataConfig",
    "DepthTrainAugmentations",
    "DepthValAugmentations",
    "create_depth_dataloaders",
    "depth_sufficient_statistics",
    "depth_valid_mask",
    "log_depth_gradient_loss",
    "silog_loss",
    "summarize_depth_statistics",
]
