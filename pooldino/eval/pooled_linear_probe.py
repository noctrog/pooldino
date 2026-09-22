"""Cached linear probing of clean pre-repeat pooled RAE representations.

Example:
    python -m pooldino.eval.pooled_linear_probe \
      --checkpoint output/decoders/repeatconv4x4-dinol-vitxl-raev2official-tfds
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import jax
import tyro
from absl import logging

from pooldino.metrics import LinearProbeConfig, linear_probe
from pooldino.eval import save_eval_results
from pooldino.eval.pooled_semantics import (
    DatasetBackend,
    DatasetName,
    bind_pooled_embedding_fn,
    create_pooled_probe_data,
    pooled_semantic_metadata,
    pooled_semantic_result_key,
    restore_pooled_semantic_encoder,
)


@dataclass
class Config:
    checkpoint: Path
    """Pooled-decoder directory containing the tokenizer checkpoint."""
    dataset: DatasetName = "imagenet"
    backend: DatasetBackend = "tfds"
    data_dir: Path | None = None
    """TFDS, RAEv2 Arrow, or class-folder root selected by backend."""
    step: int | None = None
    use_ema: bool = True
    seed: int = 0
    dinov3_checkpoint_path: Path | None = None
    """Optional explicit path to the pinned RAEv2 DINOv3-L checkpoint."""
    gpu_collect_batch_size: int = 32
    num_workers: int = 8
    probe: LinearProbeConfig = field(default_factory=LinearProbeConfig)


def main(cfg: Config) -> None:
    if cfg.gpu_collect_batch_size <= 0:
        raise ValueError("gpu_collect_batch_size must be positive.")
    mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"))
    jax.set_mesh(mesh)
    restored = restore_pooled_semantic_encoder(
        cfg.checkpoint,
        mesh=mesh,
        step=cfg.step,
        use_ema=cfg.use_ema,
        seed=cfg.seed,
        dinov3_checkpoint_path=cfg.dinov3_checkpoint_path,
    )
    collect_batch_size = cfg.gpu_collect_batch_size * jax.device_count()
    data_cfg, data = create_pooled_probe_data(
        restored,
        dataset=cfg.dataset,
        backend=cfg.backend,
        data_dir=cfg.data_dir,
        num_workers=cfg.num_workers,
        batch_size=collect_batch_size,
    )
    feat_fn = bind_pooled_embedding_fn(restored)
    csv_path = cfg.checkpoint / (
        f"pooled_linear_probe_{cfg.dataset}_mean_step{restored.step}_"
        f"{'ema' if cfg.use_ema else 'raw'}.csv"
    )
    frame = linear_probe(
        cfg.probe,
        collect_batch_size,
        feat_fn,
        data,
        mesh=mesh,
        cache_features=True,
        num_classes=data_cfg.num_classes,
        output_path=csv_path,
    )
    results = pooled_semantic_metadata(
        restored,
        dataset=cfg.dataset,
        use_ema=cfg.use_ema,
        train_samples=(data.train_ds_size // jax.device_count()) * jax.device_count(),
        validation_samples=(data.val_ds_size // jax.device_count()) * jax.device_count(),
    )
    results["epochs"] = cfg.probe.epochs
    results["optimizer"] = "lars_warmup_cosine"
    results["optimizer_batch_size"] = cfg.probe.batch_size
    results["warmup_fraction"] = cfg.probe.warmup_fraction
    results["feature_normalization"] = "per_sample_l2"
    results["grid"] = [
        {
            "learning_rate": float(row["base_lr"]),
            "weight_decay": float(row["wd"]),
            "top_1_percent": 100.0 * float(row["top_1"]),
        }
        for _, row in frame.iterrows()
    ]
    results["best_top_1_percent"] = 100.0 * float(frame["top_1"].max())
    results["csv_path"] = str(csv_path.absolute())
    key = pooled_semantic_result_key(
        "linear_probe",
        dataset=cfg.dataset,
        step=restored.step,
        use_ema=cfg.use_ema,
    )
    save_eval_results(cfg.checkpoint, key, results)
    logging.info("Saved %s to %s/eval_results.json", key, cfg.checkpoint)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    main(tyro.cli(Config))
