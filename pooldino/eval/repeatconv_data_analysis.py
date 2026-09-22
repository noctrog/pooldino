"""Data-dependent covariance and PCA analysis for RepeatConv tokenizers.

The expensive operation is one frozen DINOv3-L pass over a stratified image
subset.  Local source blocks are sampled once and reused for every tokenizer.
Leading covariance components are estimated with randomized SVD, so the full
16,384 x 16,384 covariance matrix is never materialized.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import jax
import matplotlib
from flax import nnx

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro
from scipy import fft, linalg
from sklearn.utils.extmath import randomized_svd

from pooldino.data.data import _ImageFolderDataSource
from pooldino.eval.pooled_semantics import restore_pooled_semantic_encoder
from pooldino.eval.repeatconv_operator_analysis import (
    entropy_effective_rank,
    flatten_hwio_kernel,
    operator_gram_eigendecomposition,
    participation_rank,
    restore_tokenizer_kernel,
)
from pooldino.train_decoder import extract_pool_source_tokens
from pooldino.augmentations import adm_center_crop


@dataclass
class Config:
    source_checkpoint: Path
    """One official pooled decoder used to restore the shared DINO source."""
    data_dir: Path
    """ImageNet ImageFolder root; supplied explicitly, never a machine-specific path."""
    dino_checkpoint: Path
    """Local DINOv3 weights used for the frozen source encoder."""
    checkpoints: list[Path] = field(default_factory=list)
    """RepeatConv checkpoints whose local operators will be compared."""
    output_dir: Path = Path("output/pooled-operator-analysis/data")
    fit_split: str = "train"
    """ImageFolder split used to estimate the PCA subspace and mean."""
    eval_split: str = "validation"
    """Disjoint ImageFolder split used for every reported variance metric."""
    step: int = 40032
    use_ema: bool = True
    max_images: int = 1000
    """Stratified images used for frozen DINO extraction."""
    batch_size: int = 2
    num_workers: int = 16
    max_blocks: int = 4096
    """Maximum local blocks retained per pooling geometry."""
    pca_components: int = 1024
    pca_oversamples: int = 32
    pca_power_iterations: int = 2
    seed: int = 0
    save_block_samples: bool = False
    """Save sampled flattened source blocks; disabled because they are large."""


@nnx.jit(
    static_argnames=(
        "backbone_resolution",
        "num_prefix_tokens",
        "layer_indices",
        "aggregation",
        "normalize_each",
        "add_final_mean",
        "representation_eps",
    )
)
def source_patch_batch(
    dino,
    images: jax.Array,
    *,
    backbone_resolution: int,
    num_prefix_tokens: int,
    layer_indices: tuple[int, ...],
    aggregation: str,
    normalize_each: bool,
    add_final_mean: bool,
    representation_eps: float,
) -> jax.Array:
    """Extract the exact clean DINO field presented to RepeatConv."""

    patches, final_mean = extract_pool_source_tokens(
        images,
        dino,
        backbone_resolution=backbone_resolution,
        num_prefix_tokens=num_prefix_tokens,
        layer_indices=layer_indices,
        aggregation=aggregation,
        normalize_each=normalize_each,
        add_final_mean=add_final_mean,
        split_final_mean=False,
        representation_eps=representation_eps,
    )
    if final_mean is not None:
        raise ValueError("Official no-post-pool-LN extraction must not split final_mean.")
    return patches


def bind_source_patch_fn(restored):
    """Bind official source-representation geometry into a jitted callable."""

    representation = restored.cfg.representation
    configured = partial(
        source_patch_batch,
        backbone_resolution=restored.cfg.backbone_resolution,
        num_prefix_tokens=restored.cfg.num_prefix_tokens,
        layer_indices=restored.layer_indices,
        aggregation=representation.aggregation,
        normalize_each=representation.normalize_each,
        add_final_mean=representation.add_final_mean,
        representation_eps=representation.eps,
    )
    return nnx.cached_partial(configured, restored.dino)


def stratified_imagefolder_indices(
    source: _ImageFolderDataSource,
    count: int,
    *,
    seed: int,
) -> np.ndarray:
    """Select as evenly across classes as possible, then shuffle deterministically."""

    if count <= 0:
        raise ValueError("Image count must be positive.")
    count = min(count, len(source.samples))
    by_class: list[list[int]] = [[] for _ in source.class_names]
    for index, (_, label) in enumerate(source.samples):
        by_class[int(label)].append(index)
    rng = np.random.default_rng(seed)
    class_order = rng.permutation(len(by_class))
    base, remainder = divmod(count, len(by_class))
    selected: list[int] = []
    for order_index, class_id in enumerate(class_order):
        quota = base + int(order_index < remainder)
        if quota > len(by_class[class_id]):
            raise ValueError(
                f"Class {class_id} has {len(by_class[class_id])} images, "
                f"but stratification requested {quota}."
            )
        if quota:
            selected.extend(
                int(value) for value in rng.choice(by_class[class_id], size=quota, replace=False)
            )
    selected_array = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected_array)
    return selected_array


def extract_nonoverlapping_blocks(
    patches: np.ndarray,
    pool_hw: tuple[int, int],
    *,
    grid_hw: tuple[int, int] = (16, 16),
) -> np.ndarray:
    """Flatten non-overlapping local DINO blocks as [B, cells, spatial*C]."""

    patches = np.asarray(patches)
    if patches.ndim != 3:
        raise ValueError(f"Expected [B, T, C] patches, got {patches.shape}.")
    batch, tokens, channels = patches.shape
    height, width = grid_hw
    pool_h, pool_w = pool_hw
    if tokens != height * width:
        raise ValueError(f"Expected {height * width} tokens, got {tokens}.")
    if height % pool_h or width % pool_w:
        raise ValueError(f"Pool window {pool_hw} does not divide grid {grid_hw}.")
    field = patches.reshape(batch, height, width, channels)
    field = field.reshape(
        batch,
        height // pool_h,
        pool_h,
        width // pool_w,
        pool_w,
        channels,
    )
    field = np.transpose(field, (0, 1, 3, 2, 4, 5))
    return field.reshape(batch, -1, pool_h * pool_w * channels)


class EqualImageBlockSampler:
    """Collect a bounded, approximately equal number of blocks per image."""

    def __init__(
        self,
        pool_hw: tuple[int, int],
        *,
        max_blocks: int,
        max_images: int,
        seed: int,
    ):
        if max_blocks <= 0 or max_images <= 0:
            raise ValueError("max_blocks and max_images must be positive.")
        self.pool_hw = pool_hw
        self.max_blocks = max_blocks
        self.blocks_per_image = max(1, math.ceil(max_blocks / max_images))
        self.rng = np.random.default_rng(seed)
        self.parts: list[np.ndarray] = []

    def update(self, patches: np.ndarray) -> None:
        blocks = extract_nonoverlapping_blocks(patches, self.pool_hw)
        for image_blocks in blocks:
            take = min(self.blocks_per_image, len(image_blocks))
            indices = self.rng.choice(len(image_blocks), size=take, replace=False)
            self.parts.append(np.asarray(image_blocks[indices], dtype=np.float32))

    def finalize(self) -> np.ndarray:
        if not self.parts:
            raise ValueError(f"No blocks accumulated for {self.pool_hw}.")
        values = np.concatenate(self.parts, axis=0)
        if len(values) > self.max_blocks:
            indices = self.rng.choice(len(values), size=self.max_blocks, replace=False)
            values = values[indices]
        return np.asarray(values, dtype=np.float32)


def _orthonormalized_operator_coordinates(
    centered_blocks: np.ndarray,
    operator: np.ndarray,
    *,
    rank_rtol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project blocks into an orthonormal basis for the operator row space."""

    eigenvalues, eigenvectors = operator_gram_eigendecomposition(operator)
    keep = eigenvalues > float(eigenvalues[0]) * rank_rtol
    vectors = eigenvectors[:, keep]
    inverse_sqrt = (vectors / np.sqrt(eigenvalues[keep])) @ vectors.T
    raw_outputs = centered_blocks @ operator.T
    coordinates = raw_outputs @ inverse_sqrt
    return coordinates, eigenvalues, inverse_sqrt


def _principal_cosines_to_pca(
    operator: np.ndarray,
    inverse_sqrt_gram: np.ndarray,
    pca_components: np.ndarray,
) -> np.ndarray:
    cross = inverse_sqrt_gram @ operator @ pca_components.T
    return np.clip(linalg.svdvals(cross, check_finite=False), 0.0, 1.0)


def _average_principal_cosines_to_pca(
    pca_components: np.ndarray,
    pool_hw: tuple[int, int],
    channels: int,
) -> np.ndarray:
    spatial = pool_hw[0] * pool_hw[1]
    components = pca_components.reshape(len(pca_components), spatial, channels)
    cross = np.sum(components, axis=1).T / np.sqrt(spatial)
    return np.clip(linalg.svdvals(cross, check_finite=False), 0.0, 1.0)


def analyze_block_population(
    fit_blocks: np.ndarray,
    eval_blocks: np.ndarray,
    kernel: np.ndarray,
    *,
    pca_components: int,
    pca_oversamples: int,
    pca_power_iterations: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Estimate input covariance/PCA and compare learned and average row spaces."""

    fit_blocks = np.asarray(fit_blocks, dtype=np.float32)
    eval_blocks = np.asarray(eval_blocks, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float64)
    if fit_blocks.ndim != 2 or eval_blocks.ndim != 2:
        raise ValueError(
            "Expected flattened fit/eval block matrices, got "
            f"{fit_blocks.shape} and {eval_blocks.shape}."
        )
    pool_h, pool_w, channels, out_channels = kernel.shape
    if channels != out_channels:
        raise ValueError("Analysis expects channel-preserving RepeatConv kernels.")
    operator = flatten_hwio_kernel(kernel)
    if fit_blocks.shape[1] != operator.shape[1] or eval_blocks.shape[1] != operator.shape[1]:
        raise ValueError(
            f"Block widths {fit_blocks.shape[1]} and {eval_blocks.shape[1]} do not match "
            f"operator {operator.shape}."
        )
    component_count = min(pca_components, fit_blocks.shape[0] - 1, fit_blocks.shape[1])
    if component_count <= 0:
        raise ValueError("At least two block samples are required for PCA.")

    block_mean = np.mean(fit_blocks, axis=0, dtype=np.float64)
    fit_centered = fit_blocks.copy()
    fit_centered -= block_mean.astype(np.float32)
    eval_centered = eval_blocks.copy()
    eval_centered -= block_mean.astype(np.float32)
    fit_total_sum_squares = float(np.square(fit_centered.astype(np.float64)).sum())
    eval_total_sum_squares = float(np.square(eval_centered.astype(np.float64)).sum())
    if fit_total_sum_squares <= 0 or eval_total_sum_squares <= 0:
        raise ValueError("The sampled fit/eval block population has zero variance.")

    pca_start = time.monotonic()
    _, singular_values, components = randomized_svd(
        fit_centered,
        n_components=component_count,
        n_oversamples=min(
            pca_oversamples,
            max(1, min(fit_centered.shape) - component_count),
        ),
        n_iter=pca_power_iterations,
        power_iteration_normalizer="LU",
        transpose="auto",
        random_state=seed,
    )
    pca_seconds = time.monotonic() - pca_start
    singular_values = np.asarray(singular_values, dtype=np.float64)
    components = np.asarray(components, dtype=np.float64)
    covariance_eigenvalues = np.square(singular_values) / max(len(fit_centered) - 1, 1)
    fit_pca_fraction = float(np.square(singular_values).sum() / fit_total_sum_squares)
    eval_pca_coordinates = eval_centered @ components.T
    pca_fraction = float(
        np.square(eval_pca_coordinates.astype(np.float64)).sum() / eval_total_sum_squares
    )

    learned_coordinates, operator_eigenvalues, inverse_sqrt = _orthonormalized_operator_coordinates(
        eval_centered, operator
    )
    learned_fraction = float(
        np.square(learned_coordinates.astype(np.float64)).sum() / eval_total_sum_squares
    )
    spatial = pool_h * pool_w
    average_coordinates = eval_centered.reshape(len(eval_centered), spatial, channels).sum(
        axis=1
    ) / np.sqrt(spatial)
    average_fraction = float(
        np.square(average_coordinates.astype(np.float64)).sum() / eval_total_sum_squares
    )

    learned_outputs = eval_centered @ operator.T
    covariance_outputs = learned_outputs.astype(np.float64)
    covariance_outputs -= covariance_outputs.mean(axis=0, keepdims=True)
    output_covariance = (covariance_outputs.T @ covariance_outputs) / max(len(eval_centered) - 1, 1)
    output_eigenvalues = np.maximum(linalg.eigvalsh(output_covariance), 0.0)[::-1]
    learned_pca_cosines = _principal_cosines_to_pca(
        operator,
        inverse_sqrt,
        components,
    )
    average_pca_cosines = _average_principal_cosines_to_pca(
        components,
        (pool_h, pool_w),
        channels,
    )

    dct_values = fft.dctn(
        eval_blocks.reshape(len(eval_blocks), pool_h, pool_w, channels).astype(np.float64),
        axes=(1, 2),
        norm="ortho",
    )
    input_frequency_energy = np.square(dct_values - dct_values.mean(axis=0)).sum(axis=(0, 3))
    input_frequency_energy /= max(float(input_frequency_energy.sum()), np.finfo(float).tiny)

    metrics = {
        "pool_window": [pool_h, pool_w],
        "fit_block_samples": len(fit_blocks),
        "eval_block_samples": len(eval_blocks),
        "input_dimension": int(fit_blocks.shape[1]),
        "output_dimension": int(out_channels),
        "pca_components": int(component_count),
        "pca_seconds": float(pca_seconds),
        "fit_pca_explained_variance_fraction": fit_pca_fraction,
        "pca_explained_variance_fraction": pca_fraction,
        "learned_rowspace_variance_fraction": learned_fraction,
        "average_rowspace_variance_fraction": average_fraction,
        "learned_to_pca_variance_ratio": (
            float(learned_fraction / pca_fraction) if pca_fraction > 0 else None
        ),
        "average_to_pca_variance_ratio": (
            float(average_fraction / pca_fraction) if pca_fraction > 0 else None
        ),
        "learned_pca_subspace_overlap": float(
            np.square(learned_pca_cosines).sum() / component_count
        ),
        "average_pca_subspace_overlap": float(
            np.square(average_pca_cosines).sum() / component_count
        ),
        "learned_pca_principal_cosine_mean": float(learned_pca_cosines.mean()),
        "average_pca_principal_cosine_mean": float(average_pca_cosines.mean()),
        "leading_input_covariance_entropy_rank": entropy_effective_rank(covariance_eigenvalues),
        "leading_input_covariance_participation_rank": participation_rank(covariance_eigenvalues),
        "output_covariance_entropy_rank": entropy_effective_rank(output_eigenvalues),
        "output_covariance_participation_rank": participation_rank(output_eigenvalues),
        "input_dct_dc_variance_fraction": float(input_frequency_energy[0, 0]),
        "input_dct_non_dc_variance_fraction": float(1.0 - input_frequency_energy[0, 0]),
    }
    arrays = {
        "block_mean": block_mean.astype(np.float32),
        "leading_input_covariance_eigenvalues": covariance_eigenvalues,
        "pca_components": components.astype(np.float32),
        "operator_gram_eigenvalues": operator_eigenvalues,
        "output_covariance_eigenvalues": output_eigenvalues,
        "learned_pca_principal_cosines": learned_pca_cosines,
        "average_pca_principal_cosines": average_pca_cosines,
        "input_dct_frequency_variance": input_frequency_energy,
    }
    return metrics, arrays


def _load_normalized_image(
    source: _ImageFolderDataSource,
    index: int,
    *,
    resolution: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    sample = source[int(index)]
    image = adm_center_crop(sample["image"], resolution)
    image = image.astype(np.float32) / 255.0
    return (image - mean) / std


def extract_patch_population(
    source: _ImageFolderDataSource,
    indices: np.ndarray,
    *,
    source_fn,
    samplers: dict[str, EqualImageBlockSampler],
    batch_size: int,
    num_workers: int,
    resolution: int,
    mean: np.ndarray,
    std: np.ndarray,
    population_name: str,
) -> tuple[int, float]:
    """Run frozen DINO extraction for one split and update all geometry samplers."""

    extraction_start = time.monotonic()
    processed = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            images = list(
                executor.map(
                    lambda value: _load_normalized_image(
                        source,
                        int(value),
                        resolution=resolution,
                        mean=mean,
                        std=std,
                    ),
                    batch_indices,
                )
            )
            image_batch = np.stack(images, axis=0)
            patches = source_fn(jax.device_put(image_batch))
            patches.block_until_ready()
            patches_host = np.asarray(jax.device_get(patches), dtype=np.float32)
            for sampler in samplers.values():
                sampler.update(patches_host)
            processed += len(batch_indices)
            print(
                f"population={population_name} extracted_images={processed}/{len(indices)} "
                f"elapsed_seconds={time.monotonic() - extraction_start:.1f}",
                flush=True,
            )
    return processed, time.monotonic() - extraction_start


def _save_model_plots(
    output_dir: Path, name: str, metrics: dict[str, Any], arrays: dict[str, np.ndarray]
) -> None:
    input_eigenvalues = arrays["leading_input_covariance_eigenvalues"]
    output_eigenvalues = arrays["output_covariance_eigenvalues"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].semilogy(np.arange(1, len(input_eigenvalues) + 1), input_eigenvalues)
    axes[0].set(title="Leading input covariance spectrum", xlabel="Component", ylabel="Eigenvalue")
    axes[1].semilogy(np.arange(1, len(output_eigenvalues) + 1), output_eigenvalues)
    axes[1].set(
        title="RepeatConv output covariance spectrum", xlabel="Component", ylabel="Eigenvalue"
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(output_dir / "covariance_spectra.png", dpi=190)
    plt.close(fig)

    learned = arrays["learned_pca_principal_cosines"]
    average = arrays["average_pca_principal_cosines"]
    fig, axis = plt.subplots(figsize=(6.8, 4.4))
    axis.plot(np.arange(1, len(learned) + 1), learned, label="RepeatConv vs PCA")
    axis.plot(np.arange(1, len(average) + 1), average, label="Average vs PCA")
    axis.set(
        xlabel="Principal-angle index",
        ylabel="cos(angle)",
        ylim=(-0.02, 1.02),
        title=f"{name}: PCA subspace overlap",
    )
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_principal_cosines.png", dpi=190)
    plt.close(fig)

    frequency = arrays["input_dct_frequency_variance"]
    fig, axis = plt.subplots(figsize=(5.2, 4.4))
    image = axis.imshow(frequency, cmap="magma", interpolation="nearest")
    for row in range(frequency.shape[0]):
        for column in range(frequency.shape[1]):
            axis.text(
                column,
                row,
                f"{frequency[row, column]:.3g}",
                ha="center",
                va="center",
                color="white",
            )
    axis.set(
        xlabel="Horizontal frequency",
        ylabel="Vertical frequency",
        title=f"{name}: input DCT variance",
    )
    fig.colorbar(image, ax=axis, label="Variance fraction")
    fig.tight_layout()
    fig.savefig(output_dir / "input_dct_frequency_variance.png", dpi=190)
    plt.close(fig)

    labels = ["PCA", "RepeatConv row space", "Average row space"]
    values = [
        metrics["pca_explained_variance_fraction"],
        metrics["learned_rowspace_variance_fraction"],
        metrics["average_rowspace_variance_fraction"],
    ]
    fig, axis = plt.subplots(figsize=(6.8, 4.2))
    axis.bar(labels, values)
    axis.set(
        ylabel="Input variance fraction",
        ylim=(0, 1),
        title=f"{name}: rate-matched variance capture",
    )
    axis.tick_params(axis="x", rotation=15)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "variance_capture.png", dpi=190)
    plt.close(fig)


def _write_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: int(row["compression"]))
    with (output_dir / "summary.json").open("w") as file:
        json.dump(ordered, file, indent=2)
        file.write("\n")
    fieldnames = list(ordered[0].keys())
    with (output_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)

    lines = [
        "# RepeatConv data-dependent covariance and PCA analysis",
        "",
        "The source population is a stratified ImageNet subset. PCA is a randomized leading-spectrum estimate; the full high-dimensional covariance matrix is never materialized.",
        "",
        "| Checkpoint | Window | Fit/eval blocks | Held-out PCA variance | RepeatConv variance | Average variance | RepeatConv/PCA overlap | Output covariance e-rank | Input DC variance |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            "| {checkpoint_name} | {pool_window} | {fit_block_samples}/{eval_block_samples} | "
            "{pca_explained_variance_fraction:.4f} | "
            "{learned_rowspace_variance_fraction:.4f} | "
            "{average_rowspace_variance_fraction:.4f} | "
            "{learned_pca_subspace_overlap:.4f} | "
            "{output_covariance_entropy_rank:.2f} | "
            "{input_dct_dc_variance_fraction:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- PCA is fit on the training split. PCA and all operator variance fractions are evaluated on held-out validation blocks.",
            "- Held-out PCA variance is the approximate rate-matched variance ceiling learned from the sampled training population.",
            "- RepeatConv/PCA overlap measures whether RGB-trained compression keeps the same input subspace as PCA.",
            "- Average variance and input DC variance quantify how much local information is explained by spatially constant modes.",
            "- Leading input covariance effective rank is computed only from the retained randomized spectrum and is therefore a lower/partial diagnostic, not the exact full effective rank.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines))


def main(cfg: Config) -> None:
    if not cfg.checkpoints:
        raise ValueError("Pass at least one --checkpoints path.")
    if cfg.max_images <= 0 or cfg.batch_size <= 0 or cfg.max_blocks <= 0:
        raise ValueError("max_images, batch_size, and max_blocks must be positive.")
    if cfg.pca_components <= 0 or cfg.pca_power_iterations < 0:
        raise ValueError("PCA settings are invalid.")
    for split in (cfg.fit_split, cfg.eval_split):
        split_dir = cfg.data_dir / ("val" if split in {"val", "validation", "test"} else split)
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing ImageFolder split {split!r} below {cfg.data_dir}.")
    if cfg.fit_split == cfg.eval_split:
        raise ValueError("fit_split and eval_split must be disjoint.")
    if not cfg.dino_checkpoint.is_file():
        raise FileNotFoundError(f"Missing DINO checkpoint: {cfg.dino_checkpoint}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"))
    jax.set_mesh(mesh)
    restore_start = time.monotonic()
    restored = restore_pooled_semantic_encoder(
        cfg.source_checkpoint,
        mesh=mesh,
        step=cfg.step,
        use_ema=cfg.use_ema,
        seed=cfg.seed,
        dinov3_checkpoint_path=cfg.dino_checkpoint,
    )
    restore_seconds = time.monotonic() - restore_start
    if restored.cfg.post_pool_norm:
        raise ValueError("This analysis requires the official no-post-pool-LN affine path.")
    source_fn = bind_source_patch_fn(restored)

    kernels: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for index, checkpoint in enumerate(cfg.checkpoints):
        kernel, bias, raw_cfg = restore_tokenizer_kernel(
            checkpoint,
            step=cfg.step,
            use_ema=cfg.use_ema,
            channels=restored.latent_spec.feat,
        )
        for key in ("dino_name", "backbone_resolution", "representation"):
            source_value = (
                restored.cfg.dino_name
                if key == "dino_name"
                else restored.cfg.backbone_resolution
                if key == "backbone_resolution"
                else {
                    "layers": list(restored.cfg.representation.layers),
                    "aggregation": restored.cfg.representation.aggregation,
                    "normalize_each": restored.cfg.representation.normalize_each,
                    "add_final_mean": restored.cfg.representation.add_final_mean,
                    "eps": restored.cfg.representation.eps,
                }
            )
            checkpoint_value = raw_cfg[key]
            if key == "representation":
                checkpoint_value = {
                    name: checkpoint_value[name]
                    for name in ("layers", "aggregation", "normalize_each", "add_final_mean", "eps")
                }
            if checkpoint_value != source_value:
                raise ValueError(
                    f"{checkpoint.name} has mismatched source config {key}: "
                    f"{checkpoint_value!r} != {source_value!r}."
                )
        name = checkpoint.name
        kernels[name] = (kernel, bias, raw_cfg)

    def make_samplers(seed_offset: int) -> dict[str, EqualImageBlockSampler]:
        return {
            name: EqualImageBlockSampler(
                tuple(int(value) for value in raw_cfg["pool_window"]),
                max_blocks=cfg.max_blocks,
                max_images=cfg.max_images,
                seed=cfg.seed + seed_offset + index,
            )
            for index, (name, (_, _, raw_cfg)) in enumerate(kernels.items())
        }

    fit_source = _ImageFolderDataSource(str(cfg.data_dir), cfg.fit_split)
    eval_source = _ImageFolderDataSource(str(cfg.data_dir), cfg.eval_split)
    fit_indices = stratified_imagefolder_indices(fit_source, cfg.max_images, seed=cfg.seed)
    eval_indices = stratified_imagefolder_indices(eval_source, cfg.max_images, seed=cfg.seed + 1)
    fit_samplers = make_samplers(1000)
    eval_samplers = make_samplers(2000)
    mean = np.asarray(restored.data_cfg.normalization_mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(restored.data_cfg.normalization_std, dtype=np.float32).reshape(1, 1, 3)
    fit_processed, fit_extraction_seconds = extract_patch_population(
        fit_source,
        fit_indices,
        source_fn=source_fn,
        samplers=fit_samplers,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        resolution=restored.cfg.backbone_resolution,
        mean=mean,
        std=std,
        population_name="fit",
    )
    eval_processed, eval_extraction_seconds = extract_patch_population(
        eval_source,
        eval_indices,
        source_fn=source_fn,
        samplers=eval_samplers,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        resolution=restored.cfg.backbone_resolution,
        mean=mean,
        std=std,
        population_name="eval",
    )

    rows: list[dict[str, Any]] = []
    for checkpoint in cfg.checkpoints:
        name = checkpoint.name
        fit_blocks = fit_samplers[name].finalize()
        eval_blocks = eval_samplers[name].finalize()
        kernel, _, raw_cfg = kernels[name]
        metrics, arrays = analyze_block_population(
            fit_blocks,
            eval_blocks,
            kernel,
            pca_components=cfg.pca_components,
            pca_oversamples=cfg.pca_oversamples,
            pca_power_iterations=cfg.pca_power_iterations,
            seed=cfg.seed,
        )
        compression = int(np.prod(raw_cfg["pool_window"]))
        metrics.update(
            {
                "checkpoint_name": name,
                "checkpoint_path": str(checkpoint.expanduser().resolve()),
                "checkpoint_step": cfg.step,
                "use_ema": cfg.use_ema,
                "compression": compression,
                "unique_tokens": 256 // compression,
            }
        )
        model_dir = cfg.output_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        with (model_dir / "data_analysis.json").open("w") as file:
            json.dump(metrics, file, indent=2)
            file.write("\n")
        np.savez_compressed(model_dir / "data_analysis_arrays.npz", **arrays)
        if cfg.save_block_samples:
            np.save(model_dir / "sampled_fit_source_blocks.npy", fit_blocks)
            np.save(model_dir / "sampled_eval_source_blocks.npy", eval_blocks)
        _save_model_plots(model_dir, name, metrics, arrays)
        rows.append(metrics)

    _write_summary(cfg.output_dir, rows)
    provenance = {
        "cwd": os.getcwd(),
        "jax_platforms": [device.platform for device in jax.devices()],
        "source_checkpoint": str(cfg.source_checkpoint.expanduser().resolve()),
        "checkpoints": [str(path.expanduser().resolve()) for path in cfg.checkpoints],
        "data_dir": str(cfg.data_dir.expanduser().resolve()),
        "fit_split": cfg.fit_split,
        "eval_split": cfg.eval_split,
        "step": cfg.step,
        "use_ema": cfg.use_ema,
        "requested_images": cfg.max_images,
        "fit_processed_images": fit_processed,
        "eval_processed_images": eval_processed,
        "fit_selected_class_count": len(
            {int(fit_source.samples[int(index)][1]) for index in fit_indices}
        ),
        "eval_selected_class_count": len(
            {int(eval_source.samples[int(index)][1]) for index in eval_indices}
        ),
        "max_blocks": cfg.max_blocks,
        "pca_components": cfg.pca_components,
        "pca_oversamples": cfg.pca_oversamples,
        "pca_power_iterations": cfg.pca_power_iterations,
        "seed": cfg.seed,
        "restore_seconds": restore_seconds,
        "fit_dino_extraction_seconds": fit_extraction_seconds,
        "eval_dino_extraction_seconds": eval_extraction_seconds,
    }
    with (cfg.output_dir / "provenance.json").open("w") as file:
        json.dump(provenance, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main(tyro.cli(Config))
