"""Joint kNN and full-LR linear probing for matched frozen compressors."""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import jax
import pandas as pd
import tyro
from absl import logging

from pooldino.metrics import KNNConfig, LinearProbeConfig
from pooldino.metrics.knn import knn_classifier
from pooldino.metrics.linear_probe import eval_top1_classifier, train_classifier
from pooldino.metrics.utils import precompute_features
from pooldino.eval.pooled_baseline_semantics import (
    BaselineCompressor,
    baseline_semantic_metadata,
    bind_baseline_embedding_fn,
    load_projection_artifact,
)
from pooldino.eval.pooled_semantics import (
    DatasetBackend,
    DatasetName,
    create_pooled_probe_data,
    restore_pooled_semantic_encoder,
)


@dataclass
class Config:
    source_checkpoint: Path
    """Official decoder checkpoint used only for the shared frozen DINO source config."""
    output_dir: Path
    compressor: BaselineCompressor
    pool_hw: tuple[int, int]
    projection_artifact: Path | None = None
    dataset: DatasetName = "imagenet"
    backend: DatasetBackend = "imagefolder"
    data_dir: Path | None = None
    step: int | None = 40032
    use_ema: bool = True
    seed: int = 0
    dinov3_checkpoint_path: Path | None = None
    gpu_collect_batch_size: int = 32
    num_workers: int = 16
    smoke_test: bool = False
    knn: KNNConfig = field(default_factory=KNNConfig)
    probe: LinearProbeConfig = field(default_factory=LinearProbeConfig)


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w") as file:
            json.dump(value, file, indent=2)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(cfg: Config) -> None:
    if cfg.gpu_collect_batch_size <= 0:
        raise ValueError("gpu_collect_batch_size must be positive.")
    if any(value <= 0 for value in cfg.pool_hw):
        raise ValueError("pool_hw must be positive.")
    mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"))
    jax.set_mesh(mesh)
    restored = restore_pooled_semantic_encoder(
        cfg.source_checkpoint,
        mesh=mesh,
        step=cfg.step,
        use_ema=cfg.use_ema,
        seed=cfg.seed,
        dinov3_checkpoint_path=cfg.dinov3_checkpoint_path,
    )
    artifact = load_projection_artifact(
        cfg.compressor,
        pool_hw=cfg.pool_hw,
        channels=int(restored.latent_spec.feat),
        path=cfg.projection_artifact,
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
    feature_fn = bind_baseline_embedding_fn(restored, artifact)
    if cfg.smoke_test:
        batch = next(iter(data.train_loader))
        features = feature_fn(jax.device_put(batch["image"]))
        features.block_until_ready()
        expected = (batch["image"].shape[0], int(restored.latent_spec.feat))
        if features.shape != expected:
            raise ValueError(
                f"Unexpected baseline feature shape {features.shape}; expected {expected}."
            )
        if not bool(jax.numpy.isfinite(features).all()):
            raise ValueError("Baseline smoke-test features contain non-finite values.")
        logging.info(
            "Baseline smoke test passed: compressor=%s pool=%s images=%s features=%s artifact=%s",
            cfg.compressor,
            cfg.pool_hw,
            batch["image"].shape,
            features.shape,
            artifact.sha256,
        )
        return

    train_features, train_labels, validation_features, validation_labels = precompute_features(
        collect_batch_size,
        feature_fn,
        data,
        mesh=mesh,
    )
    results = baseline_semantic_metadata(
        restored,
        artifact,
        dataset=cfg.dataset,
        train_samples=int(train_features.shape[0]),
        validation_samples=int(validation_features.shape[0]),
    )
    results["source_checkpoint_path"] = str(cfg.source_checkpoint.expanduser().resolve())
    results["feature_normalization"] = "per_sample_l2"
    results["shared_feature_cache_for_knn_and_linear_probe"] = True

    knn_rows: list[dict[str, float | int]] = []
    for k in cfg.knn.nb_knn:
        top_1, top_5 = knn_classifier(
            train_features,
            train_labels,
            validation_features,
            validation_labels,
            k=k,
            temperature=cfg.knn.temperature,
            num_classes=data_cfg.num_classes,
            val_chunk_size=cfg.knn.gpu_batch_size,
        )
        knn_rows.append({"k": int(k), "top_1_percent": top_1, "top_5_percent": top_5})
        logging.info("k=%s top1=%.4f top5=%.4f", k, top_1, top_5)
    results["knn"] = {
        "temperature": cfg.knn.temperature,
        "grid": knn_rows,
        "best_top_1_percent": max(float(row["top_1_percent"]) for row in knn_rows),
    }
    result_path = cfg.output_dir / "eval_results.json"
    _write_json_atomic(result_path, results)

    linear_rows: list[dict[str, float]] = []
    for base_lr, weight_decay in itertools.product(
        cfg.probe.learning_rates,
        cfg.probe.weight_decays,
    ):
        classifier = train_classifier(
            cfg.probe,
            base_lr,
            weight_decay,
            mesh,
            train_features,
            train_labels,
            num_classes=data_cfg.num_classes,
        )
        top_1 = eval_top1_classifier(
            cfg.probe,
            classifier,
            validation_features,
            validation_labels,
            mesh,
        )
        row = {
            "base_lr": float(base_lr),
            "weight_decay": float(weight_decay),
            "top_1": float(top_1),
            "top_1_percent": 100.0 * float(top_1),
        }
        linear_rows.append(row)
        logging.info(
            "base_lr=%s weight_decay=%s top1=%.4f%%",
            base_lr,
            weight_decay,
            row["top_1_percent"],
        )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.output_dir / "linear_probe_imagenet_full_lr.csv"
    pd.DataFrame(linear_rows).to_csv(csv_path, index=False)
    results["linear_probe"] = {
        "epochs": cfg.probe.epochs,
        "optimizer": "lars_warmup_cosine",
        "optimizer_batch_size": cfg.probe.batch_size,
        "warmup_fraction": cfg.probe.warmup_fraction,
        "learning_rates": [float(value) for value in cfg.probe.learning_rates],
        "weight_decays": [float(value) for value in cfg.probe.weight_decays],
        "grid": linear_rows,
        "best_top_1_percent": max(row["top_1_percent"] for row in linear_rows),
        "csv_path": str(csv_path.resolve()),
    }
    _write_json_atomic(result_path, results)
    logging.info("Saved matched baseline results to %s", result_path)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    main(tyro.cli(Config))
