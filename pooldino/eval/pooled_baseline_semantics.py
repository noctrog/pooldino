"""Matched average, PCA, and random projections of clean pooled DINO fields."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from pooldino.train_decoder import extract_pool_source_tokens

BaselineCompressor = Literal["average", "pca", "random"]


@dataclass(frozen=True)
class ProjectionArtifact:
    method: BaselineCompressor
    pool_hw: tuple[int, int]
    operator: np.ndarray
    center: np.ndarray
    path: str | None
    sha256: str | None
    metadata: dict[str, Any]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _npz_scalar(values: np.lib.npyio.NpzFile, name: str) -> Any:
    value = np.asarray(values[name])
    if value.shape != ():
        raise ValueError(f"Expected scalar {name!r}, got shape {value.shape}.")
    return value.item()


def _validate_projection(
    operator: np.ndarray,
    center: np.ndarray,
    *,
    pool_hw: tuple[int, int],
    channels: int,
) -> tuple[np.ndarray, np.ndarray]:
    input_dimension = int(np.prod(pool_hw) * channels)
    operator = np.asarray(operator, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    if operator.shape != (channels, input_dimension):
        raise ValueError(
            f"Expected projection shape {(channels, input_dimension)}, got {operator.shape}."
        )
    if center.shape != (input_dimension,):
        raise ValueError(f"Expected center shape {(input_dimension,)}, got {center.shape}.")
    if not np.isfinite(operator).all() or not np.isfinite(center).all():
        raise ValueError("Projection artifacts must contain only finite values.")
    return operator, center


def average_projection(pool_hw: tuple[int, int], channels: int) -> ProjectionArtifact:
    """Build fixed local average pooling in flattened block coordinates."""

    spatial = int(np.prod(pool_hw))
    block = np.eye(channels, dtype=np.float32) / spatial
    operator = np.concatenate([block] * spatial, axis=1)
    center = np.zeros((spatial * channels,), dtype=np.float32)
    return ProjectionArtifact(
        method="average",
        pool_hw=pool_hw,
        operator=operator,
        center=center,
        path=None,
        sha256=None,
        metadata={"definition": "fixed_per_channel_local_average", "centered": False},
    )


def load_projection_artifact(
    method: BaselineCompressor,
    *,
    pool_hw: tuple[int, int],
    channels: int,
    path: Path | None,
) -> ProjectionArtifact:
    """Load and validate a frozen projection used by every downstream metric."""

    if method == "average":
        if path is not None:
            raise ValueError("Average pooling does not accept a projection artifact.")
        return average_projection(pool_hw, channels)
    if path is None:
        raise ValueError(f"{method} requires --projection-artifact.")
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing projection artifact: {path}")

    with np.load(path, allow_pickle=False) as values:
        if method == "pca":
            required = {"pca_components", "block_mean"}
            if not required.issubset(values.files):
                raise ValueError(f"PCA artifact is missing {sorted(required - set(values.files))}.")
            operator = values["pca_components"]
            center = values["block_mean"]
            metadata: dict[str, Any] = {
                "definition": "training_fitted_rate_matched_pca",
                "centered": True,
            }
        else:
            required = {"operator", "center", "pool_window", "seed"}
            if not required.issubset(values.files):
                raise ValueError(
                    f"Random artifact is missing {sorted(required - set(values.files))}."
                )
            stored_pool = tuple(int(value) for value in np.asarray(values["pool_window"]).tolist())
            if stored_pool != pool_hw:
                raise ValueError(
                    f"Random artifact pool window {stored_pool} does not match {pool_hw}."
                )
            operator = values["operator"]
            center = values["center"]
            metadata = {
                "definition": "seeded_orthonormal_gaussian_row_projection",
                "centered": True,
                "seed": int(_npz_scalar(values, "seed")),
            }
            if "center_source_path" in values.files:
                metadata["center_source_path"] = str(_npz_scalar(values, "center_source_path"))
            if "center_source_sha256" in values.files:
                metadata["center_source_sha256"] = str(_npz_scalar(values, "center_source_sha256"))

    operator, center = _validate_projection(
        operator,
        center,
        pool_hw=pool_hw,
        channels=channels,
    )
    if method in {"pca", "random"}:
        gram = operator.astype(np.float64) @ operator.astype(np.float64).T
        orthogonality_error = float(np.max(np.abs(gram - np.eye(channels))))
        if orthogonality_error > 5e-3:
            raise ValueError(
                f"{method} projection rows are not orthonormal: max error "
                f"{orthogonality_error:.3g}."
            )
        metadata["row_orthogonality_max_error"] = orthogonality_error
    return ProjectionArtifact(
        method=method,
        pool_hw=pool_hw,
        operator=operator,
        center=center,
        path=str(path),
        sha256=file_sha256(path),
        metadata=metadata,
    )


def projected_local_block_mean(
    patches: jax.Array,
    operator: jax.Array,
    center: jax.Array,
    *,
    grid_hw: tuple[int, int],
    pool_hw: tuple[int, int],
) -> jax.Array:
    """Project local blocks and mean their outputs without materializing tokens."""

    if patches.ndim != 3:
        raise ValueError(f"Expected [B, T, C] patches, got {patches.shape}.")
    batch, tokens, channels = patches.shape
    height, width = grid_hw
    pool_h, pool_w = pool_hw
    if tokens != height * width:
        raise ValueError(f"Expected {height * width} patch tokens, got {tokens}.")
    if height % pool_h or width % pool_w:
        raise ValueError(f"Pool window {pool_hw} does not divide grid {grid_hw}.")
    input_dimension = pool_h * pool_w * channels
    if operator.shape != (channels, input_dimension):
        raise ValueError(
            f"Expected operator shape {(channels, input_dimension)}, got {operator.shape}."
        )
    if center.shape != (input_dimension,):
        raise ValueError(f"Expected center shape {(input_dimension,)}, got {center.shape}.")

    field = patches.reshape(batch, height, width, channels)
    blocks = field.reshape(
        batch,
        height // pool_h,
        pool_h,
        width // pool_w,
        pool_w,
        channels,
    )
    blocks = jnp.transpose(blocks, (0, 1, 3, 2, 4, 5))
    blocks = blocks.reshape(batch, -1, input_dimension)
    mean_block = jnp.mean(blocks.astype(jnp.float32), axis=1)
    return jnp.einsum(
        "bi,oi->bo",
        mean_block - center.astype(jnp.float32),
        operator.astype(jnp.float32),
    )


@nnx.jit(
    static_argnames=(
        "backbone_resolution",
        "num_prefix_tokens",
        "layer_indices",
        "aggregation",
        "normalize_each",
        "add_final_mean",
        "representation_eps",
        "grid_hw",
        "pool_hw",
    )
)
def baseline_embedding_batch(
    dino,
    operator: jax.Array,
    center: jax.Array,
    images: jax.Array,
    *,
    backbone_resolution: int,
    num_prefix_tokens: int,
    layer_indices: tuple[int, ...],
    aggregation: str,
    normalize_each: bool,
    add_final_mean: bool,
    representation_eps: float,
    grid_hw: tuple[int, int],
    pool_hw: tuple[int, int],
) -> jax.Array:
    """Extract the official source field and apply a frozen matched compressor."""

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
        raise ValueError("Baseline extraction unexpectedly split the final spatial mean.")
    return projected_local_block_mean(
        patches,
        operator,
        center,
        grid_hw=grid_hw,
        pool_hw=pool_hw,
    )


def bind_baseline_embedding_fn(restored, artifact: ProjectionArtifact):
    """Bind source geometry and one immutable baseline projection."""

    if restored.cfg.post_pool_norm:
        raise ValueError("Matched semantic baselines require the official no-post-pool-LN path.")
    channels = int(restored.latent_spec.feat)
    _validate_projection(
        artifact.operator,
        artifact.center,
        pool_hw=artifact.pool_hw,
        channels=channels,
    )
    representation = restored.cfg.representation
    configured = partial(
        baseline_embedding_batch,
        backbone_resolution=restored.cfg.backbone_resolution,
        num_prefix_tokens=restored.cfg.num_prefix_tokens,
        layer_indices=restored.layer_indices,
        aggregation=representation.aggregation,
        normalize_each=representation.normalize_each,
        add_final_mean=representation.add_final_mean,
        representation_eps=representation.eps,
        grid_hw=restored.grid_hw,
        pool_hw=artifact.pool_hw,
    )
    return nnx.cached_partial(
        configured,
        restored.dino,
        jnp.asarray(artifact.operator),
        jnp.asarray(artifact.center),
    )


def baseline_semantic_metadata(
    restored,
    artifact: ProjectionArtifact,
    *,
    dataset: str,
    train_samples: int,
    validation_samples: int,
) -> dict[str, Any]:
    pool_h, pool_w = artifact.pool_hw
    grid_h, grid_w = restored.grid_hw
    metadata: dict[str, Any] = {
        "dataset": dataset,
        "source_checkpoint_step": int(restored.step),
        "stage1_profile": restored.cfg.stage1_profile,
        "dino_name": restored.cfg.dino_name,
        "backbone_resolution": restored.cfg.backbone_resolution,
        "source_layers": list(restored.layer_indices),
        "compressor": artifact.method,
        "pool_window": [pool_h, pool_w],
        "latent_grid": [grid_h // pool_h, grid_w // pool_w],
        "latent_tokens": int((grid_h // pool_h) * (grid_w // pool_w)),
        "feature_dimension": int(restored.latent_spec.feat),
        "feature_reduction": "mean_of_unique_projected_local_blocks",
        "train_samples": int(train_samples),
        "validation_samples": int(validation_samples),
        "projection_artifact_path": artifact.path,
        "projection_artifact_sha256": artifact.sha256,
        "projection": artifact.metadata,
    }
    if artifact.method == "average":
        metadata["equivalent_after_token_mean_for_pool_windows"] = [
            [1, 1],
            [2, 2],
            [2, 4],
            [4, 2],
            [4, 4],
        ]
    return metadata
