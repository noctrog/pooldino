"""Shared feature extraction for semantic evaluation of pooled RAE latents."""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import jmp
from flax import nnx

from pooldino.data import DATASET_PRESETS, DataConfig, DataLoaders, create_dataloaders
from pooldino.pooled_generator import (
    RestoredPooledDecoderComponents,
    encode_pooled_decoder_latents_impl,
    restore_pooled_decoder_components,
)
from pooldino.augmentations import ADMCenterCropAugmentations, ADMCenterCropConfig

DatasetName = Literal[tuple(DATASET_PRESETS.keys())]  # ty: ignore[invalid-type-form]
DatasetBackend = Literal["tfds", "raev2_hf", "imagefolder"]


def mean_pooled_embedding(tokens: jax.Array) -> jax.Array:
    """Reduce clean pre-repeat tokens to a rate-matched image representation."""

    if tokens.ndim != 3:
        raise ValueError(f"Expected pooled tokens with shape [B, T, D], got {tokens.shape}.")
    if tokens.shape[1] == 0:
        raise ValueError("Cannot aggregate an empty pooled-token sequence.")
    return jnp.mean(tokens.astype(jnp.float32), axis=1)


@nnx.jit(
    static_argnames=(
        "backbone_resolution",
        "num_prefix_tokens",
        "layer_indices",
        "aggregation",
        "normalize_each",
        "add_final_mean",
        "post_pool_norm",
        "representation_eps",
        "grid_hw",
        "pool_hw",
    )
)
def pooled_embedding_batch(
    dino,
    tokenizer,
    images: jax.Array,
    *,
    backbone_resolution: int,
    num_prefix_tokens: int,
    layer_indices: tuple[int, ...],
    aggregation: str,
    normalize_each: bool,
    add_final_mean: bool,
    post_pool_norm: bool,
    representation_eps: float,
    grid_hw: tuple[int, int],
    pool_hw: tuple[int, int],
) -> jax.Array:
    """Extract and mean-pool the exact clean latent field used by stage two."""

    tokens = encode_pooled_decoder_latents_impl(
        images,
        dino,
        tokenizer,
        backbone_resolution=backbone_resolution,
        num_prefix_tokens=num_prefix_tokens,
        layer_indices=layer_indices,
        aggregation=aggregation,
        normalize_each=normalize_each,
        add_final_mean=add_final_mean,
        post_pool_norm=post_pool_norm,
        representation_eps=representation_eps,
        grid_hw=grid_hw,
        pool_hw=pool_hw,
    )
    return mean_pooled_embedding(tokens)


def restore_pooled_semantic_encoder(
    checkpoint: Path,
    *,
    mesh: jax.sharding.Mesh,
    step: int | None,
    use_ema: bool,
    seed: int,
    dinov3_checkpoint_path: Path | None,
) -> RestoredPooledDecoderComponents:
    """Restore the frozen source encoder and tokenizer without the RGB decoder."""

    source_mp = jmp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=jnp.float32,
        output_dtype=jnp.float32,
    )
    restored = restore_pooled_decoder_components(
        checkpoint,
        mesh=mesh,
        mp=source_mp,
        seed=seed,
        step=step,
        use_ema=use_ema,
        restore_decoder=False,
        source_encoder_fp32=True,
        dinov3_checkpoint_path=dinov3_checkpoint_path,
    )
    restored.dino.eval()
    restored.tokenizer.eval()
    return restored


def bind_pooled_embedding_fn(restored: RestoredPooledDecoderComponents):
    """Bind checkpoint geometry into a jitted image-to-vector callable."""

    rep = restored.cfg.representation
    embedding_with_geometry = partial(
        pooled_embedding_batch,
        backbone_resolution=restored.cfg.backbone_resolution,
        num_prefix_tokens=restored.cfg.num_prefix_tokens,
        layer_indices=restored.layer_indices,
        aggregation=rep.aggregation,
        normalize_each=rep.normalize_each,
        add_final_mean=rep.add_final_mean,
        post_pool_norm=restored.cfg.post_pool_norm,
        representation_eps=rep.eps,
        grid_hw=restored.grid_hw,
        pool_hw=restored.cfg.pool_window,
    )
    return nnx.cached_partial(
        embedding_with_geometry,
        restored.dino,
        restored.tokenizer,
    )


def create_pooled_probe_data(
    restored: RestoredPooledDecoderComponents,
    *,
    dataset: str,
    backend: DatasetBackend,
    data_dir: Path | None,
    num_workers: int,
    batch_size: int,
) -> tuple[DataConfig, DataLoaders]:
    """Create deterministic train/validation loaders for cached probing."""

    if backend in {"raev2_hf", "imagefolder"} and dataset != "imagenet":
        raise ValueError(f"The {backend} backend is only valid for ImageNet probes.")
    data_cfg = DataConfig.from_preset(
        dataset,
        backend=backend,
        data_dir=None if data_dir is None else str(data_dir.expanduser()),
        num_workers=num_workers,
    )
    augmentation = ADMCenterCropAugmentations(
        ADMCenterCropConfig(
            crop_size=restored.cfg.backbone_resolution,
            output_size=restored.cfg.backbone_resolution,
            horizontal_flip=False,
        ),
        data_cfg,
    )
    loaders = create_dataloaders(
        data_cfg,
        batch_size,
        train_epochs=1,
        val_epochs=1,
        train_aug=augmentation,
        val_aug=augmentation,
        drop_remainder_train=False,
        drop_remainder_val=False,
        val_shuffle=False,
        train_shuffle=False,
    )
    return data_cfg, loaders


def pooled_semantic_metadata(
    restored: RestoredPooledDecoderComponents,
    *,
    dataset: str,
    use_ema: bool,
    train_samples: int,
    validation_samples: int,
) -> dict[str, object]:
    """Return common provenance recorded beside every semantic result."""

    grid = restored.latent_spec.grid
    return {
        "dataset": dataset,
        "checkpoint_step": int(restored.step),
        "use_ema_tokenizer": use_ema,
        "stage1_profile": restored.cfg.stage1_profile,
        "dino_name": restored.cfg.dino_name,
        "backbone_resolution": restored.cfg.backbone_resolution,
        "source_layers": list(restored.layer_indices),
        "pool_window": list(restored.cfg.pool_window),
        "latent_grid": list(grid),
        "latent_tokens": int(grid[0] * grid[1]),
        "feature_dimension": int(restored.latent_spec.feat),
        "feature_reduction": "mean_of_clean_pre_repeat_tokens",
        "train_samples": int(train_samples),
        "validation_samples": int(validation_samples),
    }


def pooled_semantic_result_key(
    evaluation: Literal["knn", "linear_probe"],
    *,
    dataset: str,
    step: int,
    use_ema: bool,
) -> str:
    weights = "ema" if use_ema else "raw"
    return f"pooled_{evaluation}_{dataset}_mean_step{step}_{weights}"
