"""Compute per-token statistics for pooled-decoder latents."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
from jax.experimental import multihost_utils
import jmp
import numpy as np
import tyro
from absl import logging
from tqdm import tqdm

from pooldino.data import create_dataloaders
from pooldino.pooled_generator import (
    encode_pooled_decoder_latents,
    restore_pooled_decoder_components,
)
from pooldino.augmentations import ADMCenterCropAugmentations, ADMCenterCropConfig
from pooldino.augmentations.decoder import RAEv2GithubDecoderAugmentations
from pooldino.utils import (
    init_distributed,
    is_primary_host,
    prefetch_to_mesh,
)

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


@dataclass
class Args:
    checkpoint: Path
    output: Path | None = None
    skip_mean: bool = False
    batch_size: int = 256
    num_workers: int = 8
    distributed: bool = False
    split: Literal["train", "validation"] = "train"
    max_samples: int | None = None
    step: int | None = None
    use_ema: bool = True


def update_host_running_stats(
    count: int,
    mean: np.ndarray,
    m2: np.ndarray,
    batch: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Merge a batch into Welford statistics using real host float64 math."""

    batch = np.asarray(batch, dtype=np.float64)
    if batch.ndim != mean.ndim + 1 or batch.shape[1:] != mean.shape:
        raise ValueError(
            f"Expected stats batch [N, {mean.shape}], got {batch.shape}."
        )
    batch_count = int(batch.shape[0])
    if batch_count == 0:
        return count, mean, m2
    batch_mean = np.mean(batch, axis=0, dtype=np.float64)
    centered = batch - batch_mean
    batch_m2 = np.sum(centered * centered, axis=0, dtype=np.float64)
    if count == 0:
        return batch_count, batch_mean, batch_m2

    combined_count = count + batch_count
    delta = batch_mean - mean
    combined_mean = mean + delta * (batch_count / combined_count)
    combined_m2 = (
        m2
        + batch_m2
        + delta * delta * (count * batch_count / combined_count)
    )
    return combined_count, combined_mean, combined_m2


def trim_host_stats_batch(
    batch: np.ndarray,
    remaining: int,
    *,
    valid_size: int | None = None,
) -> np.ndarray:
    """Remove device padding, then trim to the exact requested sample count."""

    if remaining < 0:
        raise ValueError("remaining must be non-negative.")
    batch = np.asarray(batch)
    if valid_size is not None:
        if not 0 <= valid_size <= len(batch):
            raise ValueError(
                f"valid_size must lie in [0, {len(batch)}], got {valid_size}."
            )
        batch = batch[:valid_size]
    return batch[:remaining]


def gather_latents_to_host(latents: jax.Array) -> np.ndarray:
    """Materialize one global batch on every host for deterministic reduction."""

    if jax.process_count() > 1:
        latents = multihost_utils.process_allgather(latents, tiled=True)
    return np.asarray(jax.device_get(latents), dtype=np.float64)


def raev2_stats_padding_count(sample_count: int, *, world_size: int = 8) -> int:
    """Number of duplicated-prefix samples from DistributedSampler(drop_last=False)."""

    if sample_count <= 0 or world_size <= 0:
        raise ValueError("sample_count and world_size must be positive.")
    return (-sample_count) % world_size


def pad_raev2_stats_population(
    values: np.ndarray,
    *,
    world_size: int = 8,
) -> np.ndarray:
    """Small-array reference for the released stats sampler's padded population."""

    values = np.asarray(values)
    padding = raev2_stats_padding_count(len(values), world_size=world_size)
    if padding == 0:
        return values
    repetitions = (padding + len(values) - 1) // len(values)
    prefix = np.concatenate([values] * repetitions, axis=0)[:padding]
    return np.concatenate((values, prefix), axis=0)


def make_stats_augmentation(restored, data_cfg):
    """Select the source profile's exact encoder-statistics image transform."""

    if getattr(restored.cfg, "stage1_profile", "legacy") == "raev2_github":
        return RAEv2GithubDecoderAugmentations(restored.aug_cfg, data_cfg)
    return ADMCenterCropAugmentations(
        ADMCenterCropConfig(
            crop_size=256,
            output_size=restored.cfg.backbone_resolution,
            horizontal_flip=False,
        ),
        data_cfg,
    )


def main(args: Args) -> None:
    if args.distributed:
        init_distributed()
    num_devices = jax.device_count()
    mesh = jax.make_mesh((num_devices, 1), ("data", "model"))
    jax.set_mesh(mesh)
    mp = jmp.Policy(
        param_dtype=jnp.float32,
        # The released RAEv2 normalization-statistics pass does not use
        # autocast: frozen encoder/tokenizer inference is fp32.
        compute_dtype=jnp.float32,
        output_dtype=jnp.float32,
    )

    restored = restore_pooled_decoder_components(
        args.checkpoint,
        mesh=mesh,
        mp=mp,
        step=args.step,
        use_ema=args.use_ema,
        restore_decoder=False,
        source_encoder_fp32=True,
    )
    cfg = restored.cfg
    if cfg.stage1_profile == "raev2_github" and args.skip_mean:
        raise ValueError("RAEv2 source-profile normalization stats require the mean.")
    source_stats = cfg.stage1_profile == "raev2_github"
    if source_stats and jax.process_count() != 1:
        raise ValueError(
            "Exact RAEv2 statistics currently require one JAX process (it may own "
            "eight GPUs) so the fixed 8-rank DistributedSampler padding can be "
            "reproduced without Grain process-shard truncation."
        )
    output_path = args.output or args.checkpoint / "pooled_latent_stats.npz"

    data_cfg = replace(restored.data_cfg, num_workers=args.num_workers)
    use_train = args.split == "train"
    augmentation = make_stats_augmentation(restored, data_cfg)
    loaders = create_dataloaders(
        data_cfg,
        batch_size=args.batch_size * jax.local_device_count(),
        train_epochs=1 if use_train else None,
        val_epochs=None if use_train else 1,
        train_aug=augmentation if use_train else None,
        val_aug=None if use_train else augmentation,
        drop_remainder_train=False,
        drop_remainder_val=False,
        val_shuffle=False,
        train_shuffle=not source_stats,
    )
    loader = loaders.train_loader if use_train else loaders.val_loader
    dataset_size = loaders.train_ds_size if use_train else loaders.val_ds_size
    source_sample_count = (
        dataset_size
        if args.max_samples is None
        else min(dataset_size, args.max_samples)
    )
    if source_sample_count <= 0:
        raise ValueError("The requested pooled-statistics sample count must be positive.")
    padding_count = (
        raev2_stats_padding_count(source_sample_count, world_size=8)
        if source_stats
        else 0
    )
    effective_count = source_sample_count + padding_count

    total_count = 0
    running_mean = np.zeros(
        (restored.latent_spec.num_latents, restored.latent_spec.feat),
        dtype=np.float64,
    )
    running_m2 = np.zeros_like(running_mean)
    global_batch_size = args.batch_size * num_devices
    num_batches = (source_sample_count + global_batch_size - 1) // global_batch_size
    padding_prefix: list[np.ndarray] = []
    padding_prefix_count = 0
    padding_prefix_required = min(source_sample_count, padding_count)

    for batch in tqdm(
        prefetch_to_mesh(
            iter(loader),
            1,
            mesh,
            pad_to=mesh.size,
        ),
        desc="Computing pooled stats",
        total=num_batches,
        disable=not is_primary_host(),
    ):
        if total_count >= source_sample_count:
            break
        latents = encode_pooled_decoder_latents(
            batch["image"],
            restored.dino,
            restored.tokenizer,
            backbone_resolution=cfg.backbone_resolution,
            num_prefix_tokens=cfg.num_prefix_tokens,
            layer_indices=restored.layer_indices,
            aggregation=cfg.representation.aggregation,
            normalize_each=cfg.representation.normalize_each,
            add_final_mean=cfg.representation.add_final_mean,
            post_pool_norm=cfg.post_pool_norm,
            representation_eps=cfg.representation.eps,
            grid_hw=restored.grid_hw,
            pool_hw=cfg.pool_window,
        )
        # Keep device inference in its configured dtype, then perform the
        # reduction on the host.  Requesting jnp.float64 while JAX x64 is off
        # silently truncates to float32 and is not a stable accumulator.
        latents_host = gather_latents_to_host(latents)
        valid_size = int(np.asarray(jax.device_get(batch["_valid_size"])))
        remaining = source_sample_count - total_count
        latents_host = trim_host_stats_batch(
            latents_host,
            remaining,
            valid_size=valid_size,
        )
        if padding_prefix_count < padding_prefix_required:
            take = min(
                padding_prefix_required - padding_prefix_count,
                len(latents_host),
            )
            padding_prefix.append(np.array(latents_host[:take], copy=True))
            padding_prefix_count += take
        total_count, running_mean, running_m2 = update_host_running_stats(
            total_count,
            running_mean,
            running_m2,
            latents_host,
        )

    if total_count != source_sample_count:
        raise RuntimeError(
            f"Data loader yielded {total_count} samples, expected exactly "
            f"{source_sample_count}."
        )
    if padding_count:
        if padding_prefix_count != padding_prefix_required:
            raise RuntimeError(
                f"Captured {padding_prefix_count} padding-prefix samples, expected "
                f"{padding_prefix_required}."
            )
        prefix = np.concatenate(padding_prefix, axis=0)
        repetitions = (padding_count + len(prefix) - 1) // len(prefix)
        padded_values = np.concatenate([prefix] * repetitions, axis=0)[:padding_count]
        total_count, running_mean, running_m2 = update_host_running_stats(
            total_count,
            running_mean,
            running_m2,
            padded_values,
        )
    if total_count != effective_count:
        raise RuntimeError(
            f"Effective stats count is {total_count}, expected {effective_count}."
        )
    variance = np.maximum(running_m2 / total_count, 0.0)
    if is_primary_host():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "var": np.asarray(variance, dtype=np.float32),
            "count": np.asarray(total_count, dtype=np.int64),
            "source_sample_count": np.asarray(source_sample_count, dtype=np.int64),
            "sampler_world_size": np.asarray(8 if source_stats else 1, dtype=np.int64),
            "shape": np.asarray(running_mean.shape, dtype=np.int64),
            "format_version": np.asarray(1, dtype=np.int64),
            "decoder_path": np.asarray(restored.identity.checkpoint_path),
            "decoder_step": np.asarray(
                restored.identity.checkpoint_step,
                dtype=np.int64,
            ),
            "decoder_use_ema": np.asarray(restored.identity.use_ema),
            "decoder_stage1_profile": np.asarray(
                restored.identity.stage1_profile,
            ),
            "decoder_config_sha256": np.asarray(
                restored.identity.config_sha256,
            ),
        }
        if not args.skip_mean:
            payload["mean"] = np.asarray(running_mean, dtype=np.float32)
        np.savez(output_path, **payload)
        logging.info("Saved %d-sample pooled latent statistics to %s", total_count, output_path)
        logging.info("Shape: %s", running_mean.shape)


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    main(tyro.cli(Args))
