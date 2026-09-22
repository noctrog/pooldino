"""Analyze the learned affine operator in RepeatConv pooled DINO tokenizers.

This module is deliberately CPU-friendly.  It restores only the EMA tokenizer
item from a pooled-decoder checkpoint, then computes parameter-space spectra,
distance to average pooling, and spatial-frequency energy.

Example:
    JAX_PLATFORMS=cpu python -m \
      pooldino.eval.repeatconv_operator_analysis \
      --checkpoints output/decoders/repeatconv2x2-dinol-vitxl-raev2official-tfds \
                    output/decoders/repeatconv4x4-dinol-vitxl-raev2official-tfds \
      --output-dir output/pooled-operator-analysis/kernel
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jmp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
import tyro
from scipy import fft, linalg

from pooldino.train_decoder import PatchDownsampler


@dataclass
class Config:
    checkpoints: list[Path] = field(default_factory=list)
    """Pooled-decoder run directories containing RepeatConv checkpoints."""
    output_dir: Path = Path("output/pooled-operator-analysis/kernel")
    step: int = 40032
    use_ema: bool = True
    channels: int = 1024
    """DINO feature width; validated against checkpoint metadata."""
    rank_rtol: float = 1e-10
    """Relative Gram-eigenvalue threshold used for numerical rank."""


def flatten_hwio_kernel(kernel: np.ndarray) -> np.ndarray:
    """Return the block operator A with shape [out, spatial * in]."""

    kernel = np.asarray(kernel)
    if kernel.ndim != 4:
        raise ValueError(f"Expected an HWIO kernel, got {kernel.shape}.")
    pool_h, pool_w, in_channels, out_channels = kernel.shape
    return np.transpose(kernel, (3, 0, 1, 2)).reshape(
        out_channels,
        pool_h * pool_w * in_channels,
    )


def average_operator(
    pool_hw: tuple[int, int],
    in_channels: int,
    out_channels: int,
) -> np.ndarray:
    """Return strict per-channel average pooling in flattened operator form."""

    if in_channels != out_channels:
        raise ValueError("Strict average pooling requires matching channel widths.")
    pool_h, pool_w = pool_hw
    if pool_h <= 0 or pool_w <= 0:
        raise ValueError("Pooling dimensions must be positive.")
    block = np.eye(in_channels, dtype=np.float64) / (pool_h * pool_w)
    return np.concatenate([block] * (pool_h * pool_w), axis=1)


def entropy_effective_rank(eigenvalues: np.ndarray) -> float:
    """Entropy effective rank of a non-negative spectrum."""

    values = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probabilities = values / total
    probabilities = probabilities[probabilities > 0]
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def participation_rank(eigenvalues: np.ndarray) -> float:
    """Participation-ratio effective rank of a non-negative spectrum."""

    values = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    denominator = float(np.square(values).sum())
    if denominator <= 0:
        return 0.0
    return float(values.sum() ** 2 / denominator)


def operator_gram_eigendecomposition(
    operator: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decompose A A^T and return descending eigenpairs."""

    operator = np.asarray(operator, dtype=np.float64)
    gram = operator @ operator.T
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = linalg.eigh(
        gram,
        overwrite_a=True,
        check_finite=False,
        driver="evd",
    )
    order = np.argsort(eigenvalues)[::-1]
    return np.maximum(eigenvalues[order], 0.0), eigenvectors[:, order]


def average_subspace_principal_cosines(
    kernel: np.ndarray,
    gram_eigenvalues: np.ndarray,
    gram_eigenvectors: np.ndarray,
    *,
    rank_rtol: float,
) -> np.ndarray:
    """Principal-angle cosines between learned and average-pooling row spaces."""

    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.ndim != 4:
        raise ValueError(f"Expected an HWIO kernel, got {kernel.shape}.")
    pool_h, pool_w, _, _ = kernel.shape
    threshold = float(gram_eigenvalues[0]) * rank_rtol
    keep = gram_eigenvalues > threshold
    if not np.any(keep):
        return np.empty((0,), dtype=np.float64)

    # Q_A = (A A^T)^(-1/2) A has orthonormal rows. Q_avg consists of
    # channel-wise constants across the spatial block. Compute Q_A Q_avg^T
    # without materializing Q_avg or the full right-space projector.
    vectors = gram_eigenvectors[:, keep]
    inv_sqrt = (vectors / np.sqrt(gram_eigenvalues[keep])) @ vectors.T
    learned_times_average = np.sum(kernel, axis=(0, 1)).T / np.sqrt(pool_h * pool_w)
    cross = inv_sqrt @ learned_times_average
    cosines = linalg.svdvals(cross, check_finite=False)
    # Roundoff can place singular values a few ulps outside [0, 1].
    return np.clip(cosines, 0.0, 1.0)


def position_kernel_deviation(kernel: np.ndarray) -> float:
    """Deviation from average pooling followed by arbitrary channel mixing."""

    kernel = np.asarray(kernel, dtype=np.float64)
    mean_kernel = np.mean(kernel, axis=(0, 1), keepdims=True)
    denominator = float(np.square(kernel).sum())
    if denominator <= 0:
        return 0.0
    return float(np.square(kernel - mean_kernel).sum() / denominator)


def best_scalar_relative_error(operator: np.ndarray, reference: np.ndarray) -> float:
    """Relative error after optimally rescaling a fixed reference operator."""

    operator = np.asarray(operator, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if operator.shape != reference.shape:
        raise ValueError(f"Operator shapes differ: {operator.shape} and {reference.shape}.")
    denominator = float(np.square(reference).sum())
    if denominator <= 0:
        raise ValueError("Reference operator has zero norm.")
    scale = float(np.sum(operator * reference) / denominator)
    operator_norm = float(np.square(operator).sum())
    if operator_norm <= 0:
        return 0.0
    return float(np.sqrt(np.square(operator - scale * reference).sum() / operator_norm))


def dct_frequency_energy(kernel: np.ndarray) -> np.ndarray:
    """Channel-aggregated orthonormal 2D-DCT energy of an HWIO kernel."""

    kernel = np.asarray(kernel, dtype=np.float64)
    transformed = fft.dctn(kernel, axes=(0, 1), norm="ortho")
    energy = np.square(transformed).sum(axis=(2, 3))
    total = float(energy.sum())
    return energy / total if total > 0 else energy


def analyze_kernel(
    kernel: np.ndarray,
    bias: np.ndarray,
    *,
    rank_rtol: float,
) -> tuple[dict[str, float | int | list[int]], dict[str, np.ndarray]]:
    """Compute all parameter-only RepeatConv diagnostics."""

    kernel = np.asarray(kernel, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    if kernel.ndim != 4:
        raise ValueError(f"Expected an HWIO kernel, got {kernel.shape}.")
    pool_h, pool_w, in_channels, out_channels = kernel.shape
    if bias.shape != (out_channels,):
        raise ValueError(f"Expected bias shape {(out_channels,)}, got {bias.shape}.")

    operator = flatten_hwio_kernel(kernel)
    average = average_operator((pool_h, pool_w), in_channels, out_channels)
    gram_eigenvalues, gram_eigenvectors = operator_gram_eigendecomposition(operator)
    singular_values = np.sqrt(gram_eigenvalues)
    threshold = float(gram_eigenvalues[0]) * rank_rtol
    numerical_rank = int(np.count_nonzero(gram_eigenvalues > threshold))
    principal_cosines = average_subspace_principal_cosines(
        kernel,
        gram_eigenvalues,
        gram_eigenvectors,
        rank_rtol=rank_rtol,
    )
    frequency_energy = dct_frequency_energy(kernel)
    position_norms = np.sqrt(np.square(kernel).sum(axis=(2, 3)))
    spectrum_energy = gram_eigenvalues
    total_spectrum_energy = float(spectrum_energy.sum())
    top_half = min(out_channels // 2, len(spectrum_energy))

    metrics: dict[str, float | int | list[int]] = {
        "pool_window": [pool_h, pool_w],
        "input_channels": in_channels,
        "output_channels": out_channels,
        "input_dimension": int(pool_h * pool_w * in_channels),
        "output_dimension": out_channels,
        "numerical_rank": numerical_rank,
        "entropy_effective_rank": entropy_effective_rank(spectrum_energy),
        "participation_rank": participation_rank(spectrum_energy),
        "stable_rank": (
            float(total_spectrum_energy / spectrum_energy[0]) if spectrum_energy[0] > 0 else 0.0
        ),
        "condition_number": (
            float(singular_values[0] / singular_values[numerical_rank - 1])
            if numerical_rank > 0 and singular_values[numerical_rank - 1] > 0
            else float("inf")
        ),
        "top_half_operator_energy_fraction": (
            float(spectrum_energy[:top_half].sum() / total_spectrum_energy)
            if total_spectrum_energy > 0
            else 0.0
        ),
        "position_kernel_deviation": position_kernel_deviation(kernel),
        "strict_average_best_scalar_relative_error": best_scalar_relative_error(
            operator,
            average,
        ),
        "average_subspace_overlap": (
            float(np.square(principal_cosines).sum() / out_channels) if out_channels > 0 else 0.0
        ),
        "average_principal_cosine_mean": (
            float(principal_cosines.mean()) if principal_cosines.size else 0.0
        ),
        "average_principal_cosine_min": (
            float(principal_cosines.min()) if principal_cosines.size else 0.0
        ),
        "dct_dc_energy_fraction": float(frequency_energy[0, 0]),
        "dct_non_dc_energy_fraction": float(1.0 - frequency_energy[0, 0]),
        "bias_l2_norm": float(np.linalg.norm(bias)),
        "kernel_l2_norm": float(np.linalg.norm(kernel)),
    }
    arrays = {
        "singular_values": singular_values,
        "gram_eigenvalues": gram_eigenvalues,
        "average_principal_cosines": principal_cosines,
        "dct_frequency_energy": frequency_energy,
        "position_kernel_norms": position_norms,
    }
    return metrics, arrays


def _read_config_metadata(checkpoint: Path, step: int) -> dict[str, Any]:
    metadata_path = checkpoint / str(step) / "config" / "metadata"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint config metadata: {metadata_path}")
    with metadata_path.open() as file:
        return json.load(file)


def restore_tokenizer_kernel(
    checkpoint: Path,
    *,
    step: int,
    use_ema: bool,
    channels: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Restore only a tokenizer item, avoiding the DINO and RGB decoder weights."""

    checkpoint = checkpoint.expanduser().resolve()
    raw_cfg = _read_config_metadata(checkpoint, step)
    pool_window = tuple(int(value) for value in raw_cfg["pool_window"])
    if raw_cfg.get("downsample_mode") != "conv":
        raise ValueError(f"{checkpoint.name} is not a learned convolutional tokenizer.")
    checkpoint_channels = int(raw_cfg["vit"]["input_dim"])
    if checkpoint_channels != channels:
        raise ValueError(
            f"Configured channels={channels}, but {checkpoint.name} stores "
            f"input_dim={checkpoint_channels}."
        )

    item = "tokenizer_ema" if use_ema else "tokenizer"
    manager = ocp.CheckpointManager(
        checkpoint,
        item_names=(item,),
        options=ocp.CheckpointManagerOptions(read_only=True),
    )
    try:
        available = manager.all_steps()
        if step not in available:
            raise ValueError(f"Checkpoint step {step} is unavailable; choices: {available}.")
        mesh = jax.make_mesh((jax.device_count(), 1), ("data", "model"))
        policy = jmp.Policy(
            param_dtype=jnp.float32,
            compute_dtype=jnp.float32,
            output_dtype=jnp.float32,
        )
        tokenizer = PatchDownsampler.restore(
            manager,
            step,
            item,
            mesh,
            "conv",
            channels,
            pool_window,
            policy,
        )
        if tokenizer.conv is None:
            raise ValueError("Restored tokenizer unexpectedly has no convolution.")
        kernel = np.asarray(jax.device_get(tokenizer.conv.kernel), dtype=np.float64)
        bias = np.asarray(jax.device_get(tokenizer.conv.bias), dtype=np.float64)
    finally:
        manager.close()
    return kernel, bias, raw_cfg


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _safe_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_safe_json_value(item) for item in value]
    return value


def _save_model_plots(
    output_dir: Path,
    name: str,
    metrics: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> None:
    singular_values = arrays["singular_values"]
    normalized = singular_values / max(float(singular_values[0]), np.finfo(float).tiny)
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.semilogy(np.arange(1, len(normalized) + 1), normalized)
    axis.set(
        xlabel="Singular-value index", ylabel="Value / largest", title=f"{name}: kernel spectrum"
    )
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "kernel_singular_spectrum.png", dpi=180)
    plt.close(fig)

    cosines = arrays["average_principal_cosines"]
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(np.arange(1, len(cosines) + 1), cosines)
    axis.set(
        xlabel="Principal-angle index",
        ylabel="cos(angle)",
        ylim=(-0.02, 1.02),
        title=f"{name}: overlap with averaging",
    )
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "average_principal_cosines.png", dpi=180)
    plt.close(fig)

    for filename, key, title, colorbar in (
        (
            "dct_frequency_energy.png",
            "dct_frequency_energy",
            "DCT kernel-energy fraction",
            "Energy fraction",
        ),
        (
            "position_kernel_norms.png",
            "position_kernel_norms",
            "Spatial kernel Frobenius norms",
            "Frobenius norm",
        ),
    ):
        values = arrays[key]
        fig, axis = plt.subplots(figsize=(5.2, 4.4))
        image = axis.imshow(values, cmap="viridis", interpolation="nearest")
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                axis.text(
                    column,
                    row,
                    f"{values[row, column]:.3g}",
                    ha="center",
                    va="center",
                    color="white",
                )
        axis.set(
            xlabel="Horizontal position/frequency",
            ylabel="Vertical position/frequency",
            title=f"{name}: {title}",
        )
        fig.colorbar(image, ax=axis, label=colorbar)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def _write_summary(
    output_dir: Path,
    rows: list[dict[str, Any]],
) -> None:
    ordered = sorted(rows, key=lambda row: (int(row["compression"]), str(row["checkpoint_name"])))
    with (output_dir / "summary.json").open("w") as file:
        json.dump(_safe_json_value(ordered), file, indent=2)
        file.write("\n")

    fieldnames = [
        "checkpoint_name",
        "pool_window",
        "compression",
        "unique_tokens",
        "numerical_rank",
        "entropy_effective_rank",
        "participation_rank",
        "stable_rank",
        "condition_number",
        "top_half_operator_energy_fraction",
        "position_kernel_deviation",
        "strict_average_best_scalar_relative_error",
        "average_subspace_overlap",
        "average_principal_cosine_mean",
        "average_principal_cosine_min",
        "dct_dc_energy_fraction",
        "dct_non_dc_energy_fraction",
        "bias_l2_norm",
        "kernel_l2_norm",
    ]
    with (output_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)

    lines = [
        "# RepeatConv parameter-only operator analysis",
        "",
        "All values use the clean EMA tokenizer. This pass analyzes the learned affine ",
        "operator itself; covariance/PCA results require the separate data-dependent pass.",
        "",
        "| Checkpoint | Window | Compression | Rank | Entropy rank | Avg subspace overlap | Position deviation | DC energy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ordered:
        lines.append(
            "| {checkpoint_name} | {pool_window} | {compression}x | {numerical_rank} | "
            "{entropy_effective_rank:.2f} | {average_subspace_overlap:.4f} | "
            "{position_kernel_deviation:.4f} | {dct_dc_energy_fraction:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Interpretation guide:",
            "",
            "- average-subspace overlap near 1 means the learned row space is equivalent to average pooling up to channel mixing;",
            "- position deviation 0 means every spatial position uses the same channel-mixing matrix;",
            "- DC energy 1 means all parameter energy lies in the spatially constant DCT mode;",
            "- effective rank far below 1024 means the nominal channel capacity is not fully used.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines))


def _save_combined_plots(
    output_dir: Path, results: list[tuple[str, dict[str, Any], dict[str, np.ndarray]]]
) -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for name, _, arrays in results:
        values = arrays["singular_values"]
        values = values / max(float(values[0]), np.finfo(float).tiny)
        axis.semilogy(np.arange(1, len(values) + 1), values, label=name)
    axis.set(
        xlabel="Singular-value index", ylabel="Value / largest", title="RepeatConv kernel spectra"
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "combined_kernel_spectra.png", dpi=200)
    plt.close(fig)

    names = [name for name, _, _ in results]
    overlap = [float(metrics["average_subspace_overlap"]) for _, metrics, _ in results]
    dc = [float(metrics["dct_dc_energy_fraction"]) for _, metrics, _ in results]
    position = [float(metrics["position_kernel_deviation"]) for _, metrics, _ in results]
    x = np.arange(len(names))
    width = 0.26
    fig, axis = plt.subplots(figsize=(9.0, 4.8))
    axis.bar(x - width, overlap, width, label="Average subspace overlap")
    axis.bar(x, dc, width, label="DCT DC energy")
    axis.bar(x + width, position, width, label="Position deviation")
    axis.set_xticks(x, names, rotation=20, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Fraction")
    axis.set_title("How far trained RepeatConv departs from averaging")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "combined_average_comparison.png", dpi=200)
    plt.close(fig)


def main(cfg: Config) -> None:
    if not cfg.checkpoints:
        raise ValueError("Pass at least one --checkpoints path.")
    if cfg.rank_rtol <= 0:
        raise ValueError("rank_rtol must be positive.")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, dict[str, Any], dict[str, np.ndarray]]] = []
    summary_rows: list[dict[str, Any]] = []
    for checkpoint in cfg.checkpoints:
        kernel, bias, raw_cfg = restore_tokenizer_kernel(
            checkpoint,
            step=cfg.step,
            use_ema=cfg.use_ema,
            channels=cfg.channels,
        )
        name = checkpoint.name
        model_dir = cfg.output_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        metrics, arrays = analyze_kernel(kernel, bias, rank_rtol=cfg.rank_rtol)
        pool_h, pool_w = (int(value) for value in metrics["pool_window"])
        compression = pool_h * pool_w
        metrics.update(
            {
                "checkpoint_name": name,
                "checkpoint_path": str(checkpoint.expanduser().resolve()),
                "checkpoint_step": cfg.step,
                "use_ema": cfg.use_ema,
                "stage1_profile": raw_cfg.get("stage1_profile"),
                "compression": compression,
                "unique_tokens": 256 // compression,
            }
        )
        with (model_dir / "kernel_analysis.json").open("w") as file:
            json.dump(_safe_json_value(metrics), file, indent=2)
            file.write("\n")
        np.savez_compressed(model_dir / "kernel_analysis_arrays.npz", **arrays)
        _save_model_plots(model_dir, name, metrics, arrays)
        results.append((name, metrics, arrays))
        summary_rows.append(metrics)

    _write_summary(cfg.output_dir, summary_rows)
    _save_combined_plots(cfg.output_dir, results)
    provenance = {
        "cwd": os.getcwd(),
        "jax_platforms": [device.platform for device in jax.devices()],
        "checkpoints": [str(path.expanduser().resolve()) for path in cfg.checkpoints],
        "step": cfg.step,
        "use_ema": cfg.use_ema,
        "rank_rtol": cfg.rank_rtol,
    }
    with (cfg.output_dir / "provenance.json").open("w") as file:
        json.dump(provenance, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main(tyro.cli(Config))
