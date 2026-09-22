"""Weighted kNN evaluation of clean pre-repeat pooled RAE representations.

Example:
    python -m pooldino.eval.pooled_knn \
      --checkpoint output/decoders/repeatconv4x4-dinol-vitxl-raev2official-tfds
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import jax
import tyro
from absl import logging

from pooldino.metrics import KNNConfig, knn
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
    smoke_test: bool = False
    """Restore the encoder and extract one real batch, without running kNN."""
    knn: KNNConfig = field(default_factory=KNNConfig)


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
    if cfg.smoke_test:
        batch = next(iter(data.train_loader))
        features = feat_fn(jax.device_put(batch["image"]))
        features.block_until_ready()
        if features.shape != (batch["image"].shape[0], restored.latent_spec.feat):
            raise ValueError(
                "Unexpected pooled smoke-test feature shape: "
                f"{features.shape}; batch={batch['image'].shape}."
            )
        logging.info(
            "Pooled semantic smoke test passed: images=%s features=%s device=%s",
            batch["image"].shape,
            features.shape,
            features.devices(),
        )
        return
    frame = knn(
        cfg.knn,
        collect_batch_size,
        feat_fn,
        data,
        mesh=mesh,
        num_classes=data_cfg.num_classes,
    )
    results = pooled_semantic_metadata(
        restored,
        dataset=cfg.dataset,
        use_ema=cfg.use_ema,
        train_samples=(data.train_ds_size // jax.device_count()) * jax.device_count(),
        validation_samples=(data.val_ds_size // jax.device_count()) * jax.device_count(),
    )
    results["temperature"] = cfg.knn.temperature
    results["scores"] = {
        f"k_{int(row['k'])}": {
            "top_1_percent": float(row["top_1"]),
            "top_5_percent": float(row["top_5"]),
        }
        for _, row in frame.iterrows()
    }
    results["best_top_1_percent"] = float(frame["top_1"].max())
    key = pooled_semantic_result_key(
        "knn",
        dataset=cfg.dataset,
        step=restored.step,
        use_ema=cfg.use_ema,
    )
    save_eval_results(cfg.checkpoint, key, results)
    logging.info("Saved %s to %s/eval_results.json", key, cfg.checkpoint)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    main(tyro.cli(Config))
