"""Train a direct LPIPS/GAN image decoder from pooled frozen-DINO patches.

The frozen DINO patch field is spatially pooled and decoded to RGB. Learned
pooling uses a local affine map optimized jointly with the decoder; average
pooling provides a fixed baseline. The paper uses the DINOv3/RAEv2 profile.

Experiment grammar::

    {pool,repeatpool,conv,repeatconv}{1x1,2x2,4x4,2x4,4x2}-dino{s,b,l,g}-vit{s,b,l,xl}
        [-mlsK][-strideN][-sum][-raw][-gmean][-poolln][-noiseX][-eN][-fp32]
        [-raev2][-raev2official][-tfds]

Examples::

    pool1x1-dinol-vitb
    pool2x2-dinol-vitb
    pool2x4-dinol-vitb
    pool4x2-dinol-vitb
    pool4x4-dinol-vitb
    repeatpool2x2-dinol-vitb
    repeatpool4x4-dinol-vitb
    repeatconv2x2-dinol-vitl-raev2-gmean-poolln
    pool2x4-dinol-vitb-mls7-stride2-gmean
    pool2x4-dinol-vitl-mls7-stride2-gmean-raev2
    pool1x1-dinol-vitxl-raev2official-tfds

For DINOv2-L at 224 resolution, ``mls7-stride2`` selects blocks
(11, 13, 15, 17, 19, 21, 23), matching the layer spacing used by RAEv2 K7.
The default ``noise0.8`` applies RAEv2-style per-image uniform Gaussian
noising during decoder training.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Literal

import flax.nnx as nnx
import grain.python as grain
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tyro
import wandb
from absl import logging
from dacite import Config as DaciteConfig, from_dict
from einops import rearrange
from lpips_nnx import LPIPS
from tqdm import tqdm

from pooldino.data import (
    DataLoaders,
    RAEv2EpochSemantics,
    create_dataloaders,
    raev2_steps_per_epoch,
    raev2_total_updates,
)
from pooldino.experiment import ExperimentSpec, flag, option, positional
from pooldino.models.transformer import set_attn_implementation
from pooldino.pretrained.dinov3 import DINOV3_VITL16_SHA256, DINOv3ViTL16
from pooldino.pretrained.raev2_dino_discriminator import (
    RAEV2_DINO_S8_SHA256,
    RAEv2DinoS8,
)
from pooldino.features import (
    RepresentationConfig,
    extract_representation_components,
    extract_representation,
    layer_norm_no_affine,
    representation_name,
)
from pooldino.backbone import load_backbone
from pooldino.decoder_config import (
    GeneratorConfig,
    ITEM_NAMES,
    build_schedules,
    get_experiment as get_decoder_experiment,
    train_discriminator_step,
    train_generator_step,
)
from pooldino.gan_training import (
    _decoder_output_tokens,
    _image_stats,
    compute_adaptive_weight,
    RAEV2_SOURCE_WORLD_SIZE,
    _raev2_rank_diffaug,
    raev2_ddp_weighted_loss,
    raev2_github_decoder_transformer_config,
    raev2_rank_means,
    set_lpips_compute_dtype,
)
from pooldino.augmentations.decoder import (
    RAEDecoderTrainAugmentations,
    RAEDecoderValAugmentations,
    RAEv2GithubDecoderAugmentations,
)
from pooldino.diffaug import DiffAug
from pooldino.models.decoder import RAEDecoder
from pooldino.models.discriminator import DinoDisc
from pooldino.training import BaseArgs, init_wandb
from pooldino.utils import (
    Restorable,
    TrainingProfiler,
    determine_save_path,
    init_distributed,
    is_primary_host,
    open_restore_manager,
    prefetch_to_mesh,
    restore_data_loader,
    restore_optimizer_state,
)

jax.config.update("jax_optimization_level", "O1")
_jax_cache_dir = Path(
    os.environ.get(
        "JAX_COMPILATION_CACHE_DIR",
        Path(os.environ.get("TMPDIR", "/tmp")) / "jax_cache",
    )
)
jax.config.update("jax_compilation_cache_dir", str(_jax_cache_dir))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)


DINO_NAMES = {
    "s": "facebook/dinov2-with-registers-small",
    "b": "facebook/dinov2-with-registers-base",
    "l": "facebook/dinov2-with-registers-large",
    "g": "facebook/dinov2-with-registers-giant",
}
RAEV2_GITHUB_DINO_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"

POOL_WINDOWS = {
    "1x1": (1, 1),
    "2x2": (2, 2),
    "4x4": (4, 4),
    "2x4": (2, 4),
    "4x2": (4, 2),
}

POOLED_ITEM_NAMES = [*ITEM_NAMES, "tokenizer", "tokenizer_ema"]
DownsampleMode = Literal["mean", "conv"]
PoolMode = Literal["pool", "repeatpool", "conv", "repeatconv"]


@dataclass
class PooledDecoderConfig(GeneratorConfig):
    """Pixel-decoder config plus the frozen pooled-DINO tokenizer definition."""

    dino_name: str = DINO_NAMES["l"]
    dino_checkpoint_path: str | None = None
    dino_checkpoint_sha256: str | None = None
    discriminator_checkpoint_path: str | None = None
    discriminator_checkpoint_sha256: str | None = None
    backbone_resolution: int = 224
    num_prefix_tokens: int = 5
    pool_window: tuple[int, int] = (2, 4)
    repeat_pool: bool = False
    post_pool_norm: bool = False
    downsample_mode: DownsampleMode = "mean"
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)

    def __post_init__(self):
        ph, pw = self.pool_window
        if ph <= 0 or pw <= 0:
            raise ValueError("pool_window dimensions must be positive.")
        if self.backbone_resolution <= 0:
            raise ValueError("backbone_resolution must be positive.")
        if self.noise_tau < 0:
            raise ValueError("noise_tau must be non-negative.")
        if self.downsample_mode not in ("mean", "conv"):
            raise ValueError(f"Unsupported downsample_mode: {self.downsample_mode!r}.")


@dataclass
class Args(BaseArgs):
    project_name: str = "pooldino-pooled-decoder"
    experiment: str = "repeatconv2x2-dinol-vitxl-raev2official-tfds"
    gpu_batch_size: int = 64
    implementation: Literal["cudnn", "xla"] = "xla"
    dinov3_checkpoint_path: Path | None = None
    """Optional DINOv3 override; otherwise use the environment or shared cache."""
    dino_discriminator_checkpoint_path: Path | None = None
    """Optional DINO-S/8 override; otherwise use the environment or shared cache."""


def pool_patch_tokens(
    patches: jax.Array,
    *,
    grid_hw: tuple[int, int],
    pool_hw: tuple[int, int],
) -> jax.Array:
    """Non-overlapping 2D average pooling of row-major patch tokens.

    Args:
        patches: DINO patch tokens with shape ``(B, H*W, C)``.
        grid_hw: Source patch-grid height and width.
        pool_hw: Pool-window height and width.

    Returns:
        Pooled row-major tokens with shape
        ``(B, (H/ph)*(W/pw), C)``.
    """
    if patches.ndim != 3:
        raise ValueError(f"Expected BTD patches, got shape {patches.shape}.")
    height, width = grid_hw
    pool_h, pool_w = pool_hw
    if height <= 0 or width <= 0 or pool_h <= 0 or pool_w <= 0:
        raise ValueError("Grid and pool dimensions must be positive.")
    if height % pool_h != 0 or width % pool_w != 0:
        raise ValueError(
            f"pool_hw={pool_hw} must evenly divide grid_hw={grid_hw}."
        )
    if patches.shape[1] != height * width:
        raise ValueError(
            f"Expected {height * width} patch tokens for grid {grid_hw}, "
            f"got {patches.shape[1]}."
        )

    batch, _, channels = patches.shape
    field = patches.reshape(batch, height, width, channels)
    field = field.reshape(
        batch,
        height // pool_h,
        pool_h,
        width // pool_w,
        pool_w,
        channels,
    )
    pooled = jnp.mean(field, axis=(2, 4))
    return pooled.reshape(batch, -1, channels)


def _strided_mean_kernel_init(pool_hw: tuple[int, int]):
    def init(key, shape, dtype=jnp.float32):
        del key
        if len(shape) != 4:
            raise ValueError(f"Expected HWIO conv kernel shape, got {shape}.")
        pool_h, pool_w = pool_hw
        if shape[0] != pool_h or shape[1] != pool_w:
            raise ValueError(
                f"Expected kernel spatial shape {pool_hw}, got {shape[:2]}."
            )
        in_channels, out_channels = shape[2], shape[3]
        eye = jnp.eye(in_channels, out_channels, dtype=dtype)
        return jnp.broadcast_to(
            eye[None, None, :, :] / float(pool_h * pool_w),
            shape,
        )

    return init


class PatchDownsampler(nnx.Module, Restorable):
    """Patch-token downsampler used before the pixel decoder.

    ``mode='mean'`` is the old stateless non-overlapping average pool.
    ``mode='conv'`` is a channel-preserving strided convolution initialized as
    per-channel average pooling.
    """

    def __init__(
        self,
        mode: DownsampleMode,
        channels: int,
        pool_hw: tuple[int, int],
        mp: jmp.Policy,
        *,
        rngs: nnx.Rngs,
    ):
        self.mode = mode
        self.channels = channels
        self.pool_hw = pool_hw
        self.mp = mp
        if mode == "conv":
            self.conv = nnx.Conv(
                channels,
                channels,
                kernel_size=pool_hw,
                strides=pool_hw,
                padding="VALID",
                use_bias=True,
                dtype=mp.compute_dtype,
                param_dtype=mp.param_dtype,
                kernel_init=_strided_mean_kernel_init(pool_hw),
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            )
        else:
            self.conv = None

    def __call__(
        self,
        patches: jax.Array,
        *,
        grid_hw: tuple[int, int],
    ) -> jax.Array:
        if self.mode == "mean":
            return pool_patch_tokens(patches, grid_hw=grid_hw, pool_hw=self.pool_hw)
        if self.conv is None:
            raise ValueError("Strided-conv downsampler is not initialized.")
        batch, _, channels = patches.shape
        height, width = grid_hw
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}.")
        field = patches.reshape(batch, height, width, channels)
        field = self.mp.cast_to_compute(field)
        field = self.conv(field)
        field = self.mp.cast_to_output(field)
        return field.reshape(batch, -1, channels)


class PooledGenerator(nnx.Module):
    def __init__(self, decoder: RAEDecoder, tokenizer: PatchDownsampler):
        self.decoder = decoder
        self.tokenizer = tokenizer


def pooled_grid_shape(
    grid_hw: tuple[int, int],
    pool_hw: tuple[int, int],
) -> tuple[int, int]:
    height, width = grid_hw
    pool_h, pool_w = pool_hw
    if height % pool_h != 0 or width % pool_w != 0:
        raise ValueError(
            f"pool_hw={pool_hw} must evenly divide grid_hw={grid_hw}."
        )
    return height // pool_h, width // pool_w


def repeat_pooled_tokens(
    pooled: jax.Array,
    *,
    pooled_grid_hw: tuple[int, int],
    pool_hw: tuple[int, int],
) -> jax.Array:
    """Nearest-repeat pooled tokens back to the source DINO patch grid."""
    if pooled.ndim != 3:
        raise ValueError(f"Expected BTD pooled tokens, got shape {pooled.shape}.")
    pooled_h, pooled_w = pooled_grid_hw
    pool_h, pool_w = pool_hw
    if pooled.shape[1] != pooled_h * pooled_w:
        raise ValueError(
            f"Expected {pooled_h * pooled_w} pooled tokens for grid {pooled_grid_hw}, "
            f"got {pooled.shape[1]}."
        )

    batch, _, channels = pooled.shape
    field = pooled.reshape(batch, pooled_h, pooled_w, channels)
    field = jnp.repeat(field, pool_h, axis=1)
    field = jnp.repeat(field, pool_w, axis=2)
    return field.reshape(batch, -1, channels)


def resolve_representation_layers(
    cfg: RepresentationConfig,
    num_layers: int,
) -> tuple[int, ...]:
    """Use the normal final output unless a nontrivial MLS target is requested."""
    if (
        not cfg.layers
        and cfg.last_k == 1
        and cfg.layer_stride == 1
        and cfg.aggregation == "mean"
        and cfg.normalize_each
        and not cfg.add_final_mean
    ):
        return ()
    return cfg.resolve_layers(num_layers)


def extract_pool_source_tokens(
    images: jax.Array,
    dino,
    *,
    backbone_resolution: int,
    num_prefix_tokens: int,
    layer_indices: tuple[int, ...],
    aggregation: str,
    normalize_each: bool,
    add_final_mean: bool,
    split_final_mean: bool,
    representation_eps: float,
) -> tuple[jax.Array, jax.Array | None]:
    batch, _, _, channels = images.shape
    resized = jax.image.resize(
        images,
        (batch, backbone_resolution, backbone_resolution, channels),
        method="bilinear",
    )
    if split_final_mean and add_final_mean:
        # Keep the global signal out of the local downsampler so the downsampled
        # local field can be normalized before broadcasting final_mean back in.
        patches, _, final_mean = extract_representation_components(
            dino,
            resized,
            layer_indices=layer_indices,
            num_prefix_tokens=num_prefix_tokens,
            aggregation=aggregation,
            normalize_each=normalize_each,
            eps=representation_eps,
        )
        return patches, final_mean
    patches, _ = extract_representation(
        dino,
        resized,
        layer_indices=layer_indices,
        num_prefix_tokens=num_prefix_tokens,
        aggregation=aggregation,
        normalize_each=normalize_each,
        add_final_mean=add_final_mean,
        eps=representation_eps,
    )
    return patches, None


@nnx.jit(
    static_argnames=(
        "backbone_resolution",
        "num_prefix_tokens",
        "layer_indices",
        "aggregation",
        "normalize_each",
        "add_final_mean",
        "split_final_mean",
        "representation_eps",
    )
)
def extract_pool_source_step(
    images: jax.Array,
    dino,
    *,
    backbone_resolution: int,
    num_prefix_tokens: int,
    layer_indices: tuple[int, ...],
    aggregation: str,
    normalize_each: bool,
    add_final_mean: bool,
    split_final_mean: bool,
    representation_eps: float,
) -> tuple[jax.Array, jax.Array | None]:
    return extract_pool_source_tokens(
        images,
        dino,
        backbone_resolution=backbone_resolution,
        num_prefix_tokens=num_prefix_tokens,
        layer_indices=layer_indices,
        aggregation=aggregation,
        normalize_each=normalize_each,
        add_final_mean=add_final_mean,
        split_final_mean=split_final_mean,
        representation_eps=representation_eps,
    )


def finish_pooled_tokens(
    key: jax.Array,
    patches: jax.Array,
    final_mean: jax.Array | None,
    tokenizer: PatchDownsampler,
    *,
    post_pool_norm: bool,
    representation_eps: float,
    grid_hw: tuple[int, int],
    pool_hw: tuple[int, int],
    repeat_pool: bool,
    noise_tau: float,
) -> jax.Array:
    pooled = tokenizer(patches, grid_hw=grid_hw)
    if post_pool_norm:
        pooled = layer_norm_no_affine(pooled, representation_eps)
    if final_mean is not None:
        pooled = pooled + final_mean

    # RAEv2-style decoder noising: each image receives a scalar sigma sampled
    # uniformly from [0, noise_tau], followed by isotropic Gaussian noise.
    if noise_tau > 0:
        sigma_key, noise_key = jax.random.split(key)
        sigma = noise_tau * jax.random.uniform(
            sigma_key,
            (patches.shape[0], 1, 1),
            dtype=pooled.dtype,
        )
        pooled = pooled + sigma * jax.random.normal(
            noise_key,
            pooled.shape,
            dtype=pooled.dtype,
        )
    if repeat_pool:
        pooled = repeat_pooled_tokens(
            pooled,
            pooled_grid_hw=pooled_grid_shape(grid_hw, pool_hw),
            pool_hw=pool_hw,
        )
    return pooled


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
        "repeat_pool",
        "noise_tau",
    )
)
def encode_pooled_tokens(
    key: jax.Array,
    images: jax.Array,
    dino,
    tokenizer: PatchDownsampler,
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
    repeat_pool: bool,
    noise_tau: float,
) -> jax.Array:
    """Extract, spatially pool, optionally noise, and optionally repeat tokens."""
    patches, final_mean = extract_pool_source_tokens(
        images,
        dino,
        backbone_resolution=backbone_resolution,
        num_prefix_tokens=num_prefix_tokens,
        layer_indices=layer_indices,
        aggregation=aggregation,
        normalize_each=normalize_each,
        add_final_mean=add_final_mean,
        split_final_mean=post_pool_norm,
        representation_eps=representation_eps,
    )
    return finish_pooled_tokens(
        key,
        patches,
        final_mean,
        tokenizer,
        post_pool_norm=post_pool_norm,
        representation_eps=representation_eps,
        grid_hw=grid_hw,
        pool_hw=pool_hw,
        repeat_pool=repeat_pool,
        noise_tau=noise_tau,
    )


@nnx.jit(
    static_argnames=(
        "post_pool_norm",
        "representation_eps",
        "grid_hw",
        "pool_hw",
        "repeat_pool",
        "noise_tau",
    )
)
def tokenize_pool_source_step(
    key: jax.Array,
    patches: jax.Array,
    final_mean: jax.Array | None,
    tokenizer: PatchDownsampler,
    *,
    post_pool_norm: bool,
    representation_eps: float,
    grid_hw: tuple[int, int],
    pool_hw: tuple[int, int],
    repeat_pool: bool,
    noise_tau: float,
) -> jax.Array:
    return finish_pooled_tokens(
        key,
        patches,
        final_mean,
        tokenizer,
        post_pool_norm=post_pool_norm,
        representation_eps=representation_eps,
        grid_hw=grid_hw,
        pool_hw=pool_hw,
        repeat_pool=repeat_pool,
        noise_tau=noise_tau,
    )


def pooled_generator_replay_keys(
    token_key: jax.Array,
    generator_key: jax.Array,
    *,
    use_gan: bool,
    exact_raev2: bool,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Resolve adaptive, tokenizer, and DiffAug keys for one generator graph.

    RAEv2 computes the adaptive ratio and weighted backward from one noisy
    reconstruction graph.  The memory-bounded JAX implementation replays that
    graph, so exact parity requires replaying both its latent-noise key and its
    DiffAug key.  Legacy experiments retain their historical key splitting.
    """
    if exact_raev2:
        return generator_key, token_key, generator_key
    if use_gan:
        adaptive_key, update_key = jax.random.split(generator_key)
    else:
        adaptive_key = generator_key
        update_key = generator_key
    update_token_key, update_aug_key = jax.random.split(update_key)
    return adaptive_key, update_token_key, update_aug_key


@nnx.jit(
    static_argnames=(
        "should_update_ema",
        "ema_momentum",
        "use_gan",
        "use_lpips",
        "lpips_weight",
        "post_pool_norm",
        "representation_eps",
        "grid_hw",
        "pool_hw",
        "repeat_pool",
        "noise_tau",
        "source_world_size",
    ),
    donate_argnames=("generator", "generator_ema", "optim_dec"),
)
def generator_update_with_tokenizer_step(
    token_key: jax.Array,
    augmentation_key: jax.Array,
    generator: PooledGenerator,
    generator_ema: PooledGenerator,
    optim_dec: nnx.Optimizer,
    discriminator: DinoDisc,
    images: jax.Array,
    patches: jax.Array,
    final_mean: jax.Array | None,
    adaptive_weight: jax.Array,
    gan_weight: float,
    diffaug: DiffAug,
    lpips: LPIPS,
    *,
    use_gan: bool,
    use_lpips: bool,
    lpips_weight: float,
    ema_momentum: float,
    should_update_ema: bool,
    post_pool_norm: bool,
    representation_eps: float,
    grid_hw: tuple[int, int],
    pool_hw: tuple[int, int],
    repeat_pool: bool,
    noise_tau: float,
    source_world_size: int | None = None,
) -> dict[str, jax.Array]:
    image_mean, image_std = _image_stats(images.dtype)
    target_pixels = images * image_std + image_mean
    target_lpips = 2.0 * target_pixels - 1.0
    if generator.decoder.cfg.clip_generator_inputs:
        target_lpips = jnp.clip(target_lpips, -1.0, 1.0)

    def loss_fn(
        model: PooledGenerator,
        disc: DinoDisc,
        perceptual_model: LPIPS,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        latents = finish_pooled_tokens(
            token_key,
            patches,
            final_mean,
            model.tokenizer,
            post_pool_norm=post_pool_norm,
            representation_eps=representation_eps,
            grid_hw=grid_hw,
            pool_hw=pool_hw,
            repeat_pool=repeat_pool,
            noise_tau=noise_tau,
        )
        reconstruction_tokens = model.decoder(latents, deterministic=True)
        reconstruction_tokens = reconstruction_tokens[
            :, : _decoder_output_tokens(model.decoder)
        ]
        reconstruction = model.decoder.unpatchify(
            reconstruction_tokens,
            denorm_output=False,
        )
        reconstruction_pixels = model.decoder.to_pixels(reconstruction)
        reconstruction_normalized = model.decoder.to_discriminator_input(
            reconstruction
        )

        if source_world_size is None:
            reconstruction_loss = jnp.mean(
                jnp.abs(reconstruction_pixels - target_pixels)
            )
            if use_lpips:
                perceptual_loss = jnp.mean(
                    perceptual_model(reconstruction_normalized, target_lpips)
                )
            else:
                perceptual_loss = jnp.asarray(0.0, dtype=reconstruction_loss.dtype)
            if use_gan:
                fake_input = diffaug(augmentation_key, reconstruction_normalized)
                adversarial_loss = -jnp.mean(disc(fake_input))
            else:
                adversarial_loss = jnp.asarray(0.0, dtype=reconstruction_loss.dtype)
            reconstruction_total = reconstruction_loss + lpips_weight * perceptual_loss
            total_loss = (
                reconstruction_total
                + gan_weight
                * jax.lax.stop_gradient(adaptive_weight)
                * adversarial_loss
            )
            logged_adaptive_weight = adaptive_weight
        else:
            reconstruction_by_rank = raev2_rank_means(
                jnp.abs(reconstruction_pixels - target_pixels),
                source_world_size,
            )
            if use_lpips:
                perceptual_by_rank = raev2_rank_means(
                    perceptual_model(reconstruction_normalized, target_lpips),
                    source_world_size,
                )
            else:
                perceptual_by_rank = jnp.zeros_like(reconstruction_by_rank)
            if use_gan:
                rank_keys = jax.random.split(augmentation_key, source_world_size)
                fake_input = _raev2_rank_diffaug(
                    rank_keys,
                    reconstruction_normalized,
                    diffaug,
                    source_world_size,
                )
                adversarial_by_rank = -raev2_rank_means(
                    disc(fake_input),
                    source_world_size,
                )
            else:
                adversarial_by_rank = jnp.zeros_like(reconstruction_by_rank)
            total_loss = raev2_ddp_weighted_loss(
                reconstruction_by_rank,
                perceptual_by_rank,
                adversarial_by_rank,
                adaptive_weight,
                lpips_weight=lpips_weight,
                gan_weight=gan_weight,
            )
            reconstruction_loss = jnp.mean(reconstruction_by_rank)
            perceptual_loss = jnp.mean(perceptual_by_rank)
            adversarial_loss = jnp.mean(adversarial_by_rank)
            logged_adaptive_weight = jnp.mean(adaptive_weight)
        metrics = {
            "loss": total_loss,
            "recon_loss": reconstruction_loss,
            "lpips_loss": perceptual_loss,
            "gan_loss": adversarial_loss,
            "gan_weight": logged_adaptive_weight,
        }
        return total_loss, metrics

    (_, metrics), gradients = nnx.value_and_grad(
        loss_fn,
        argnums=0,
        has_aux=True,
    )(generator, discriminator, lpips)
    optim_dec.update(generator, gradients)

    if should_update_ema:
        new_ema_state = jax.tree.map(
            lambda target, source: target * ema_momentum + source * (1.0 - ema_momentum),
            nnx.state(generator_ema, nnx.Param),
            nnx.state(generator, nnx.Param),
        )
        nnx.update(generator_ema, new_ema_state)
    return metrics


def item_names_for_config(cfg: PooledDecoderConfig) -> list[str]:
    return POOLED_ITEM_NAMES if cfg.downsample_mode == "conv" else ITEM_NAMES


def save_pooled_checkpoint(
    manager: ocp.CheckpointManager | None,
    step: int,
    decoder: RAEDecoder,
    decoder_ema: RAEDecoder,
    tokenizer: PatchDownsampler,
    tokenizer_ema: PatchDownsampler,
    discriminator: DinoDisc,
    optim_dec: nnx.Optimizer,
    optim_disc: nnx.Optimizer,
    data_iter,
    cfg: PooledDecoderConfig,
) -> bool:
    if manager is None:
        return False

    items = {
        "decoder": ocp.args.PyTreeSave(nnx.to_pure_dict(nnx.state(decoder))),
        "decoder_ema": ocp.args.PyTreeSave(nnx.to_pure_dict(nnx.state(decoder_ema))),
        "discriminator": ocp.args.PyTreeSave(nnx.to_pure_dict(nnx.state(discriminator))),
        "optim_dec": ocp.args.PyTreeSave(nnx.to_pure_dict(nnx.state(optim_dec))),
        "optim_disc": ocp.args.PyTreeSave(nnx.to_pure_dict(nnx.state(optim_disc))),
        "loader": grain.PyGrainCheckpointSave(data_iter),
        "config": ocp.args.JsonSave(asdict(cfg)),
    }
    if cfg.downsample_mode == "conv":
        items["tokenizer"] = ocp.args.PyTreeSave(
            nnx.to_pure_dict(nnx.state(tokenizer))
        )
        items["tokenizer_ema"] = ocp.args.PyTreeSave(
            nnx.to_pure_dict(nnx.state(tokenizer_ema))
        )
    return manager.save(step, args=ocp.args.Composite(**items))


class PooledDecoderExperiment(
    ExperimentSpec,
    examples=(
        "pool1x1-dinol-vitb",
        "pool2x2-dinol-vitb",
        "pool2x4-dinol-vitb",
        "pool4x2-dinol-vitb",
        "pool4x4-dinol-vitb",
        "repeatpool2x2-dinol-vitb",
        "repeatpool4x4-dinol-vitb",
        "repeatconv2x2-dinol-vitl-raev2-gmean-poolln",
        "repeatconv4x4-dinol-vitl-raev2-gmean-poolln",
        "pool2x4-dinol-vitb-mls7-stride2-gmean",
        "pool2x4-dinol-vitl-mls7-stride2-gmean-poolln-raev2",
        "pool1x1-dinol-vitxl-raev2official-tfds",
    ),
):
    pool_mode: PoolMode = positional(group_next=True)
    pool: Literal["1x1", "2x2", "4x4", "2x4", "4x2"] = positional()
    backbone: Literal["s", "b", "l", "g"] = positional(prefix="dino")
    decoder: Literal["s", "b", "l", "xl"] = positional(prefix="vit")

    fp32: bool = flag()
    sum: bool = flag()
    raw: bool = flag()
    gmean: bool = flag()
    poolln: bool = flag()
    raev2: bool = flag()
    raev2official: bool = flag()
    tfds: bool = flag()

    e: int | None = option()
    mls: int | None = option()
    stride: int | None = option()
    noise: float | None = option()


def build_config(spec: PooledDecoderExperiment) -> PooledDecoderConfig:
    if spec.raev2 and spec.raev2official:
        raise ValueError("Choose either legacy -raev2 or exact -raev2official, not both.")
    if spec.tfds and not spec.raev2official:
        raise ValueError("-tfds is only supported with -raev2official.")
    if spec.raev2official:
        if spec.backbone != "l" or spec.decoder != "xl":
            raise ValueError(
                "The released RAEv2 ImageNet K7 recipe is DINOv3-L/16 + ViT-XL."
            )
        if spec.sum or spec.raw or spec.poolln:
            raise ValueError(
                "-raev2official fixes mean aggregation, per-layer normalization, "
                "and no post-pool LayerNorm."
            )
        if spec.mls not in (None, 7) or spec.stride not in (None, 2):
            raise ValueError("-raev2official fixes K7 layers 11,13,...,23.")

    decoder_name = f"vit{spec.decoder}"
    if spec.raev2:
        decoder_name += "-raev2"
    if spec.fp32:
        decoder_name += "-fp32"
    if spec.e is not None:
        decoder_name += f"-e{spec.e}"
    decoder_cfg = get_decoder_experiment(decoder_name)

    if spec.raev2official:
        official_transformer = replace(
            decoder_cfg.vit.transformer,
            **raev2_github_decoder_transformer_config("xl"),
        )
        decoder_cfg = replace(
            decoder_cfg,
            train=replace(
                decoder_cfg.train,
                schedule_profile="raev2_github",
            ),
            vit=replace(
                decoder_cfg.vit,
                transformer=official_transformer,
                use_pos_embeds=True,
                pos_embed_type="sincos",
                use_cls=True,
                cls_token_init="zeros",
                input_proj_bias=True,
                input_proj_init="torch_uniform",
                final_norm_eps=1e-12,
                latent_upsample="bilinear",
                pixel_output="rgb",
                clip_generator_inputs=False,
            ),
            aug=replace(
                decoder_cfg.aug,
                resize=(256, 256),
                crop_size=(256, 256),
            ),
            data=replace(
                decoder_cfg.data,
                backend="tfds" if spec.tfds else "raev2_hf",
                data_dir=None if spec.tfds else "data/imagenet-256",
                num_workers=4,
            ),
            stage1_profile="raev2_github",
        )

    if spec.raev2official:
        representation = RepresentationConfig(
            layers=(11, 13, 15, 17, 19, 21, 23),
            aggregation="mean",
            normalize_each=True,
            add_final_mean=True,
            eps=1e-5,
        )
    else:
        representation = RepresentationConfig(
            last_k=spec.mls or 1,
            layer_stride=spec.stride or 1,
            aggregation="sum" if spec.sum else "mean",
            normalize_each=not spec.raw,
            add_final_mean=spec.gmean,
        )
    return PooledDecoderConfig(
        train=decoder_cfg.train,
        vit=decoder_cfg.vit,
        data=decoder_cfg.data,
        aug=decoder_cfg.aug,
        diffaug=decoder_cfg.diffaug,
        patch_size=decoder_cfg.patch_size,
        out_channels=decoder_cfg.out_channels,
        noise_tau=0.8 if spec.noise is None else spec.noise,
        noise_mode="uniform",
        compute_dtype=decoder_cfg.compute_dtype,
        stage1_profile=decoder_cfg.stage1_profile,
        dino_name=(
            RAEV2_GITHUB_DINO_NAME
            if spec.raev2official
            else DINO_NAMES[spec.backbone]
        ),
        dino_checkpoint_sha256=(
            DINOV3_VITL16_SHA256 if spec.raev2official else None
        ),
        # Resolve the exact source checkpoint lazily so CLI/environment paths
        # and pooldino's shared cache retain their documented precedence.
        discriminator_checkpoint_path=None,
        discriminator_checkpoint_sha256=(
            RAEV2_DINO_S8_SHA256 if spec.raev2official else None
        ),
        backbone_resolution=256 if spec.raev2official else 224,
        num_prefix_tokens=5,
        pool_window=POOL_WINDOWS[spec.pool],
        repeat_pool=spec.pool_mode in ("repeatpool", "repeatconv"),
        post_pool_norm=spec.poolln,
        downsample_mode="conv" if spec.pool_mode in ("conv", "repeatconv") else "mean",
        representation=representation,
    )


def get_experiment(name: str) -> PooledDecoderConfig:
    return build_config(PooledDecoderExperiment.parse_or_raise(name))


def resolve_official_checkpoint_config(
    cfg: PooledDecoderConfig,
    args: Args,
) -> PooledDecoderConfig:
    """Resolve and persist exact frozen-weight identities for checkpoint restore."""
    if cfg.stage1_profile != "raev2_github":
        return cfg
    encoder_checkpoint = DINOv3ViTL16.resolve_checkpoint_path(
        args.dinov3_checkpoint_path,
        persisted_path=cfg.dino_checkpoint_path,
    ).resolve()
    discriminator_checkpoint = RAEv2DinoS8.resolve_checkpoint_path(
        args.dino_discriminator_checkpoint_path,
        persisted_path=cfg.discriminator_checkpoint_path,
    ).resolve()
    return replace(
        cfg,
        dino_checkpoint_path=str(encoder_checkpoint),
        dino_checkpoint_sha256=DINOV3_VITL16_SHA256,
        discriminator_checkpoint_path=str(discriminator_checkpoint),
        discriminator_checkpoint_sha256=RAEV2_DINO_S8_SHA256,
    )


def materialize_pooled_decoder_config(
    cfg: PooledDecoderConfig,
    *,
    hidden_size: int,
    num_prefix_tokens: int,
    grid_hw: tuple[int, int],
) -> PooledDecoderConfig:
    """Bind deterministic frozen-backbone geometry into the saved config."""

    pooled_grid = pooled_grid_shape(grid_hw, cfg.pool_window)
    decoder_input_grid = grid_hw if cfg.repeat_pool else pooled_grid
    if cfg.stage1_profile == "raev2_github":
        decoder_vit = replace(
            cfg.vit,
            patch=None,
            # GeneralDecoder always predicts the 16x16 RGB patch grid and
            # bilinearly resizes a shorter latent grid internally.
            num_patches=256,
            input_dim=hidden_size,
            num_registers=0,
            latent_grid_hw=decoder_input_grid,
            latent_upsample="bilinear",
        )
    else:
        decoder_num_registers = 0 if cfg.repeat_pool else 256
        num_tokens = decoder_input_grid[0] * decoder_input_grid[1]
        decoder_vit = replace(
            cfg.vit,
            patch=None,
            num_patches=num_tokens,
            input_dim=hidden_size,
            num_registers=decoder_num_registers,
        )
    return replace(
        cfg,
        vit=decoder_vit,
        num_prefix_tokens=num_prefix_tokens,
    )


def _config_mismatch_paths(expected: object, actual: object, prefix: str = "") -> list[str]:
    """Return stable dotted paths whose persisted semantic values differ."""

    if isinstance(expected, dict) and isinstance(actual, dict):
        mismatches: list[str] = []
        for key in sorted(expected.keys() | actual.keys()):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in expected or key not in actual:
                mismatches.append(path)
            else:
                mismatches.extend(
                    _config_mismatch_paths(expected[key], actual[key], path)
                )
        return mismatches
    return [] if expected == actual else [prefix]


def _stage1_semantic_config(cfg: PooledDecoderConfig) -> dict[str, object]:
    """Drop runtime locators while retaining every source-critical value."""

    value = asdict(cfg)
    # Worker count is a runtime throughput knob. Frozen-weight paths may move
    # across machines, but their SHA-256 identities below must still match.
    value["data"]["num_workers"] = None
    value["dino_checkpoint_path"] = None
    value["discriminator_checkpoint_path"] = None
    return value


def resolve_pooled_decoder_restore_config(
    target_cfg: PooledDecoderConfig,
    restored_cfg: PooledDecoderConfig,
) -> PooledDecoderConfig:
    """Validate an exact resume and keep its persisted training recipe intact.

    Legacy runs retain their historical behavior. For the GitHub source
    profile, the requested experiment is only an identity check: model, data,
    loss, optimizer, schedule, pooling, representation, and frozen-weight
    hashes must match the checkpoint. Runtime Args remain outside this config.
    """

    exact_resume = (
        target_cfg.stage1_profile == "raev2_github"
        or restored_cfg.stage1_profile == "raev2_github"
    )
    if exact_resume:
        # Checkpoints persist the geometry derived after constructing the
        # frozen official DINOv3-L/16 (1024 channels, 5 prefix tokens, 16x16
        # patch grid). Materialize the fresh experiment spec before comparing
        # it with that saved representation; do not erase these derived fields
        # from the checkpoint, because they are still identity-critical.
        if target_cfg.stage1_profile == "raev2_github":
            target_grid_side = target_cfg.backbone_resolution // 16
            target_cfg = materialize_pooled_decoder_config(
                target_cfg,
                hidden_size=1024,
                num_prefix_tokens=5,
                grid_hw=(target_grid_side, target_grid_side),
            )
        mismatches = _config_mismatch_paths(
            _stage1_semantic_config(target_cfg),
            _stage1_semantic_config(restored_cfg),
        )
        if mismatches:
            preview = ", ".join(mismatches[:16])
            if len(mismatches) > 16:
                preview += f", ... ({len(mismatches)} total)"
            raise ValueError(
                "Exact RAEv2 stage-one restore does not match the requested "
                f"experiment contract: {preview}. Start a deliberate fork "
                "instead of resuming this checkpoint."
            )
        # Never replace TrainConfig on an exact resume: changing epochs also
        # changes the already-running cosine schedule.
        return restored_cfg

    if restored_cfg.train.batch_size != target_cfg.train.batch_size:
        raise ValueError(
            "Restoring with a different global batch size is not supported."
        )
    return replace(restored_cfg, train=target_cfg.train)


def validate_stage1_runtime_batch(
    cfg: PooledDecoderConfig,
    *,
    aggregate_micro_batch: int,
    grad_acc_steps: int,
) -> None:
    """Require the released no-accumulation batch topology for exact stage one."""

    if cfg.stage1_profile != "raev2_github":
        return
    if aggregate_micro_batch != 512 or grad_acc_steps != 1:
        raise ValueError(
            "Exact RAEv2 stage one requires aggregate micro-batch 512 and "
            "grad_acc_steps=1 for adaptive-weight parity; got "
            f"aggregate_micro_batch={aggregate_micro_batch}, "
            f"grad_acc_steps={grad_acc_steps}."
        )


def training_update_counts(
    dataset_size: int,
    batch_size: int,
    epochs: int,
    *,
    stage1_profile: str,
    grad_accum_steps: int = 1,
    source_world_size: int = 8,
) -> tuple[int, int]:
    """Return ``(steps_per_epoch, total_updates)`` with source semantics."""
    if dataset_size <= 0 or batch_size <= 0 or epochs <= 0:
        raise ValueError("dataset_size, batch_size, and epochs must be positive.")
    if stage1_profile == "raev2_github":
        # Reproduce DistributedSampler padding, per-rank DataLoader
        # ``drop_last``, and complete gradient-accumulation groups from the
        # released eight-rank launcher.  This can differ from floor(N / B).
        steps_per_epoch = raev2_steps_per_epoch(
            dataset_size,
            batch_size,
            world_size=source_world_size,
            grad_accum_steps=grad_accum_steps,
        )
        total_updates = raev2_total_updates(
            dataset_size,
            batch_size,
            epochs,
            world_size=source_world_size,
            grad_accum_steps=grad_accum_steps,
        )
    else:
        # Preserve the historical continuous-stream floor for legacy runs.
        total_updates = (dataset_size * epochs) // batch_size
        steps_per_epoch = total_updates // epochs
    if steps_per_epoch == 0:
        raise ValueError("The global batch size exceeds the available training set.")
    return steps_per_epoch, total_updates


def write_success_marker(save_path: Path | str | None) -> None:
    """Create a local marker used by retry wrappers to detect completed runs."""
    if save_path is None:
        return
    if isinstance(save_path, str):
        if save_path.startswith("gs://"):
            logging.warning("Skipping local _SUCCESS marker for GCS checkpoint path.")
            return
        marker_dir = Path(save_path)
    else:
        marker_dir = save_path
    marker_dir.mkdir(parents=True, exist_ok=True)
    (marker_dir / "_SUCCESS").touch()


def main(args: Args, cfg: PooledDecoderConfig) -> None:
    if args.distributed:
        init_distributed()
    if not is_primary_host():
        logging.set_verbosity(logging.WARNING)

    compute_dtype = jnp.bfloat16 if cfg.compute_dtype == "bfloat16" else jnp.float32
    mp = jmp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=compute_dtype,
        output_dtype=jnp.float32,
    )
    target_cfg = cfg
    rngs = nnx.Rngs(args.seed + jax.process_index())

    num_devices = jax.device_count()
    if num_devices % args.fsdp != 0:
        raise ValueError(
            f"Number of devices ({num_devices}) must be divisible by fsdp ({args.fsdp})."
        )
    data_parallel_size = num_devices // args.fsdp
    mesh = jax.make_mesh((data_parallel_size, args.fsdp), ("data", "model"))
    jax.set_mesh(mesh)

    default_output = f"output/decoders/{args.experiment}"
    checkpoint_item_names = item_names_for_config(target_cfg)
    restore_manager, checkpoint_step = open_restore_manager(
        args.restore,
        args.maybe_restore,
        default_output,
        args.gcs_bucket,
        checkpoint_item_names,
    )
    if restore_manager is not None and checkpoint_step > 0:
        raw_cfg = restore_manager.restore(
            checkpoint_step,
            args=ocp.args.Composite(config=ocp.args.JsonRestore()),
        )["config"]
        restored_cfg = from_dict(
            PooledDecoderConfig,
            raw_cfg,
            config=DaciteConfig(cast=[tuple], strict=False),
        )
        cfg = resolve_pooled_decoder_restore_config(target_cfg, restored_cfg)

    if cfg.train.batch_size % (args.gpu_batch_size * data_parallel_size) != 0:
        raise ValueError(
            "Global batch size must be divisible by the aggregate micro batch."
        )
    grad_acc_steps = cfg.train.batch_size // (
        args.gpu_batch_size * data_parallel_size
    )
    micro_batch_size = args.gpu_batch_size * data_parallel_size
    validate_stage1_runtime_batch(
        cfg,
        aggregate_micro_batch=micro_batch_size,
        grad_acc_steps=grad_acc_steps,
    )

    data_cfg = cfg.data
    if args.num_data_workers is not None:
        data_cfg = replace(data_cfg, num_workers=args.num_data_workers)
    cfg = replace(cfg, data=data_cfg)
    # Resolve both source-of-truth checkpoints before constructing models, and
    # save absolute paths plus checksums so later stages do not depend on the
    # launch shell's environment.
    cfg = resolve_official_checkpoint_config(cfg, args)
    if cfg.stage1_profile == "raev2_github":
        train_augment = RAEv2GithubDecoderAugmentations(cfg.aug, data_cfg)
        val_augment = RAEv2GithubDecoderAugmentations(cfg.aug, data_cfg)
    else:
        train_augment = RAEDecoderTrainAugmentations(cfg.aug, data_cfg)
        val_augment = RAEDecoderValAugmentations(cfg.aug, data_cfg)
    data: DataLoaders = create_dataloaders(
        data_cfg,
        micro_batch_size,
        train_aug=train_augment,
        val_aug=val_augment,
        val_epochs=1,
        drop_remainder_train=True,
        drop_remainder_val=False,
        val_shuffle=True,
        gcs_bucket=args.gcs_bucket if args.data_in_bucket else None,
        train_epoch_semantics=(
            RAEv2EpochSemantics(
                global_batch_size=cfg.train.batch_size,
                source_world_size=8,
                grad_accum_steps=grad_acc_steps,
            )
            if cfg.stage1_profile == "raev2_github"
            else None
        ),
    )
    steps_per_epoch, total_updates = training_update_counts(
        data.train_ds_size,
        cfg.train.batch_size,
        cfg.train.epochs,
        stage1_profile=cfg.stage1_profile,
        grad_accum_steps=grad_acc_steps,
        source_world_size=8,
    )
    data_iter = iter(data.train_loader)
    lr_decoder, lr_discriminator = build_schedules(cfg.train, total_updates)

    if cfg.stage1_profile == "raev2_github":
        dino = load_backbone(
            cfg.dino_name,
            resolution=cfg.backbone_resolution,
            checkpoint_path=(
                cfg.dino_checkpoint_path
            ),
            dtype=mp.compute_dtype,
            param_dtype=mp.param_dtype,
        )
    else:
        dino = load_backbone(
            cfg.dino_name,
            resolution=cfg.backbone_resolution,
            dtype=mp.param_dtype,
        )
    dino.eval()
    hidden_size = int(dino.config.hidden_size)
    if cfg.backbone_resolution % dino.patch_size != 0:
        raise ValueError(
            f"Backbone resolution {cfg.backbone_resolution} is not divisible by "
            f"patch size {dino.patch_size}."
        )
    grid_side = cfg.backbone_resolution // dino.patch_size
    grid_hw = (grid_side, grid_side)
    pooled_grid = pooled_grid_shape(grid_hw, cfg.pool_window)
    decoder_input_grid = grid_hw if cfg.repeat_pool else pooled_grid
    num_tokens = decoder_input_grid[0] * decoder_input_grid[1]
    decoder_num_registers = 0 if cfg.repeat_pool else 256
    layer_indices = resolve_representation_layers(
        cfg.representation,
        dino.config.num_hidden_layers,
    )

    if cfg.stage1_profile == "raev2_github":
        decoder_num_registers = 0
    cfg = materialize_pooled_decoder_config(
        cfg,
        hidden_size=hidden_size,
        num_prefix_tokens=dino.num_prefix_tokens,
        grid_hw=grid_hw,
    )
    decoder_vit = cfg.vit
    num_output_tokens = (
        decoder_num_registers
        if decoder_num_registers > 0
        else int(decoder_vit.num_patches)
    )
    output_grid_side = int(num_output_tokens**0.5)
    if output_grid_side * output_grid_side != num_output_tokens:
        raise ValueError(
            f"Decoder output token count {num_output_tokens} must form a square grid."
        )
    if output_grid_side * cfg.patch_size != cfg.aug.crop_size[0]:
        raise ValueError(
            "Decoder output grid and patch size must match the crop size: "
            f"{output_grid_side} * {cfg.patch_size} != {cfg.aug.crop_size[0]}."
        )

    logging.info("DINO: %s", cfg.dino_name)
    logging.info(
        "Representation: %s",
        representation_name(cfg.representation, dino.config.num_hidden_layers),
    )
    logging.info(
        "Post-pool normalization: %s",
        "non-affine LayerNorm" if cfg.post_pool_norm else "none",
    )
    logging.info(
        "Spatial pooling: mode=%s downsample=%s source=%s window=%s pooled=%s "
        "decoder_input=%s tokens=%d channels=%d",
        "repeatpool" if cfg.repeat_pool else "pool",
        cfg.downsample_mode,
        grid_hw,
        cfg.pool_window,
        pooled_grid,
        decoder_input_grid,
        num_tokens,
        hidden_size,
    )
    logging.info(
        "Decoder output tokens: %d registers: %d",
        num_output_tokens,
        decoder_num_registers,
    )
    logging.info("RAEv2-style decoder noise tau: %.3f", cfg.noise_tau)

    use_wandb, wandb_resume_step = init_wandb(
        args,
        cfg,
        project_name=args.project_name,
    )
    save_path = determine_save_path(
        checkpoint_enabled=args.checkpoint,
        checkpoint_dir=args.checkpoint_dir,
        default_path=default_output,
        gcs_bucket=args.gcs_bucket,
    )

    if restore_manager is not None:
        decoder = RAEDecoder.restore(
            restore_manager,
            checkpoint_step,
            "decoder",
            mesh,
            decoder_vit,
            cfg.patch_size,
            cfg.out_channels,
            mp,
        )
        decoder_ema = RAEDecoder.restore(
            restore_manager,
            checkpoint_step,
            "decoder_ema",
            mesh,
            decoder_vit,
            cfg.patch_size,
            cfg.out_channels,
            mp,
        )
        discriminator = DinoDisc.restore(
            restore_manager,
            checkpoint_step,
            "discriminator",
            mesh,
            ks=9,
            mp=mp,
            official_raev2=cfg.stage1_profile == "raev2_github",
            dino_ckpt_path=cfg.discriminator_checkpoint_path,
        )
        if cfg.downsample_mode == "conv":
            tokenizer = PatchDownsampler.restore(
                restore_manager,
                checkpoint_step,
                "tokenizer",
                mesh,
                cfg.downsample_mode,
                hidden_size,
                cfg.pool_window,
                mp,
            )
            tokenizer_ema = PatchDownsampler.restore(
                restore_manager,
                checkpoint_step,
                "tokenizer_ema",
                mesh,
                cfg.downsample_mode,
                hidden_size,
                cfg.pool_window,
                mp,
            )
        else:
            tokenizer = PatchDownsampler(
                cfg.downsample_mode,
                hidden_size,
                cfg.pool_window,
                mp,
                rngs=rngs,
            )
            tokenizer_ema = PatchDownsampler(
                cfg.downsample_mode,
                hidden_size,
                cfg.pool_window,
                mp,
                rngs=rngs,
            )
    else:
        decoder = RAEDecoder(
            decoder_vit,
            cfg.patch_size,
            cfg.out_channels,
            mp,
            rngs=rngs,
        )
        decoder_ema = RAEDecoder(
            decoder_vit,
            cfg.patch_size,
            cfg.out_channels,
            mp,
            rngs=rngs,
        )
        nnx.update(
            decoder_ema,
            jax.tree.map(lambda value: jnp.copy(value), nnx.state(decoder, nnx.Param)),
        )
        discriminator = DinoDisc(
            ks=9,
            mp=mp,
            official_raev2=cfg.stage1_profile == "raev2_github",
            dino_ckpt_path=cfg.discriminator_checkpoint_path,
            rngs=rngs,
        )
        tokenizer = PatchDownsampler(
            cfg.downsample_mode,
            hidden_size,
            cfg.pool_window,
            mp,
            rngs=rngs,
        )
        tokenizer_ema = PatchDownsampler(
            cfg.downsample_mode,
            hidden_size,
            cfg.pool_window,
            mp,
            rngs=rngs,
        )
        nnx.update(
            tokenizer_ema,
            jax.tree.map(lambda value: jnp.copy(value), nnx.state(tokenizer, nnx.Param)),
        )

    set_attn_implementation(decoder, args.implementation)
    set_attn_implementation(decoder_ema, args.implementation)
    diffaug = DiffAug(cfg.diffaug)
    generator = PooledGenerator(decoder, tokenizer)
    generator_ema = PooledGenerator(decoder_ema, tokenizer_ema)

    if save_path is not None:
        path_string = (
            str(save_path.absolute()) if isinstance(save_path, Path) else save_path
        )
        manager = ocp.CheckpointManager(
            path_string,
            item_names=checkpoint_item_names,
            options=ocp.CheckpointManagerOptions(
                save_interval_steps=steps_per_epoch,
                save_on_steps=frozenset([total_updates]),
                max_to_keep=2,
                create=True,
                read_only=False,
                keep_checkpoints_without_metrics=args.keep_checkpoints_without_metrics,
            ),
        )
    else:
        manager = None

    decoder_optimizer_target = generator if cfg.downsample_mode == "conv" else decoder
    decoder_optimizer = nnx.Optimizer(
        decoder_optimizer_target,
        optax.MultiSteps(
            optax.adamw(
                lr_decoder,
                cfg.train.adam_b1,
                cfg.train.adam_b2,
                weight_decay=cfg.train.weight_decay,
            ),
            grad_acc_steps,
        ),
        wrt=nnx.Param,
    )
    discriminator_optimizer = nnx.Optimizer(
        discriminator,
        optax.MultiSteps(
            optax.adamw(
                lr_discriminator,
                cfg.train.adam_b1,
                cfg.train.adam_b2,
                weight_decay=cfg.train.weight_decay,
            ),
            grad_acc_steps,
        ),
        wrt=nnx.Param,
    )
    if restore_manager is not None:
        restore_optimizer_state(
            restore_manager,
            checkpoint_step,
            decoder_optimizer,
            mesh,
            "optim_dec",
        )
        restore_optimizer_state(
            restore_manager,
            checkpoint_step,
            discriminator_optimizer,
            mesh,
            "optim_disc",
        )
        data_iter = restore_data_loader(
            restore_manager,
            checkpoint_step,
            data_iter,
        )

    decoder_params = sum(
        value.size for value in jax.tree.leaves(nnx.state(decoder, nnx.Param))
    )
    discriminator_params = sum(
        value.size
        for value in jax.tree.leaves(nnx.state(discriminator, nnx.Param))
    )
    tokenizer_params = sum(
        value.size for value in jax.tree.leaves(nnx.state(tokenizer, nnx.Param))
    )
    logging.info("Decoder parameters: %.2fM", decoder_params / 1_000_000)
    logging.info("Tokenizer parameters: %.2fM", tokenizer_params / 1_000_000)
    logging.info("Discriminator parameters: %.2fM", discriminator_params / 1_000_000)

    train_stream = prefetch_to_mesh(data_iter, args.prefetch, mesh)
    lpips = LPIPS(rngs=rngs)
    if cfg.stage1_profile == "raev2_github":
        set_lpips_compute_dtype(lpips, mp.compute_dtype)
    discriminator.eval()
    updates_completed = int(
        optax.tree_utils.tree_get(decoder_optimizer.opt_state, "gradient_step")
    )
    base_key = rngs()
    micro_step = updates_completed * grad_acc_steps
    progress = tqdm(
        desc="Update",
        initial=updates_completed,
        total=total_updates,
        bar_format="{desc:<5.5}{percentage:3.0f}%|{bar:10}{r_bar}",
        disable=not is_primary_host(),
    )
    profiler = TrainingProfiler(
        mode=args.profile_mode,
        port=args.profiler_port,
        start_step=args.profiler_start_step,
        stop_step=args.profiler_stop_step,
    )

    validation_images = None
    log_tokens = None
    if use_wandb:
        validation_images = jnp.asarray(next(iter(data.val_loader))["image"][:64])
        if cfg.downsample_mode != "conv":
            log_tokens = encode_pooled_tokens(
                rngs(),
                validation_images,
                dino,
                tokenizer_ema,
                backbone_resolution=cfg.backbone_resolution,
                num_prefix_tokens=cfg.num_prefix_tokens,
                layer_indices=layer_indices,
                aggregation=cfg.representation.aggregation,
                normalize_each=cfg.representation.normalize_each,
                add_final_mean=cfg.representation.add_final_mean,
                post_pool_norm=cfg.post_pool_norm,
                representation_eps=cfg.representation.eps,
                grid_hw=grid_hw,
                pool_hw=cfg.pool_window,
                repeat_pool=cfg.repeat_pool,
                noise_tau=0.0,
            )

    for samples in train_stream:
        use_lpips = updates_completed >= steps_per_epoch * cfg.train.lpips_start
        use_gan = updates_completed >= steps_per_epoch * cfg.train.disc_gan_start
        train_discriminator = (
            updates_completed >= steps_per_epoch * cfg.train.disc_start
        )
        micro_step += 1
        previous_updates = updates_completed
        mini_step = int(
            optax.tree_utils.tree_get(decoder_optimizer.opt_state, "mini_step")
        )
        should_update_ema = (mini_step + 1) % grad_acc_steps == 0
        token_key, generator_key, discriminator_key = jax.random.split(
            jax.random.fold_in(base_key, micro_step),
            3,
        )
        discriminator.eval()
        if cfg.downsample_mode == "conv":
            adaptive_key, update_token_key, update_aug_key = (
                pooled_generator_replay_keys(
                    token_key,
                    generator_key,
                    use_gan=use_gan,
                    exact_raev2=cfg.stage1_profile == "raev2_github",
                )
            )
            patches, final_mean = extract_pool_source_step(
                samples["image"],
                dino,
                backbone_resolution=cfg.backbone_resolution,
                num_prefix_tokens=cfg.num_prefix_tokens,
                layer_indices=layer_indices,
                aggregation=cfg.representation.aggregation,
                normalize_each=cfg.representation.normalize_each,
                add_final_mean=cfg.representation.add_final_mean,
                split_final_mean=cfg.post_pool_norm,
                representation_eps=cfg.representation.eps,
            )
            pooled_tokens = tokenize_pool_source_step(
                token_key,
                patches,
                final_mean,
                tokenizer,
                post_pool_norm=cfg.post_pool_norm,
                representation_eps=cfg.representation.eps,
                grid_hw=grid_hw,
                pool_hw=cfg.pool_window,
                repeat_pool=cfg.repeat_pool,
                noise_tau=cfg.noise_tau,
            )
            if use_gan:
                adaptive_weight, reconstruction_norm, gan_norm = compute_adaptive_weight(
                    adaptive_key,
                    decoder,
                    discriminator,
                    samples["image"],
                    pooled_tokens,
                    diffaug,
                    lpips,
                    use_lpips=use_lpips,
                    lpips_weight=cfg.train.lpips_weight,
                    max_d_weight=cfg.train.max_d_weight,
                    source_world_size=(
                        RAEV2_SOURCE_WORLD_SIZE
                        if cfg.stage1_profile == "raev2_github"
                        else None
                    ),
                )
            else:
                adaptive_weight = jnp.asarray(0.0, dtype=samples["image"].dtype)
                reconstruction_norm = jnp.asarray(0.0, dtype=jnp.float32)
                gan_norm = jnp.asarray(0.0, dtype=jnp.float32)
            generator_metrics = generator_update_with_tokenizer_step(
                update_token_key,
                update_aug_key,
                generator,
                generator_ema,
                decoder_optimizer,
                discriminator,
                samples["image"],
                patches,
                final_mean,
                adaptive_weight,
                cfg.train.gan_weight,
                diffaug,
                lpips,
                use_gan=use_gan,
                use_lpips=use_lpips,
                lpips_weight=cfg.train.lpips_weight,
                ema_momentum=cfg.train.ema_decay,
                should_update_ema=should_update_ema,
                post_pool_norm=cfg.post_pool_norm,
                representation_eps=cfg.representation.eps,
                grid_hw=grid_hw,
                pool_hw=cfg.pool_window,
                repeat_pool=cfg.repeat_pool,
                noise_tau=cfg.noise_tau,
                source_world_size=(
                    RAEV2_SOURCE_WORLD_SIZE
                    if cfg.stage1_profile == "raev2_github"
                    else None
                ),
            )
            generator_metrics["recon_grad_norm"] = jnp.mean(reconstruction_norm)
            generator_metrics["gan_grad_norm"] = jnp.mean(gan_norm)
        else:
            pooled_tokens = encode_pooled_tokens(
                token_key,
                samples["image"],
                dino,
                tokenizer,
                backbone_resolution=cfg.backbone_resolution,
                num_prefix_tokens=cfg.num_prefix_tokens,
                layer_indices=layer_indices,
                aggregation=cfg.representation.aggregation,
                normalize_each=cfg.representation.normalize_each,
                add_final_mean=cfg.representation.add_final_mean,
                post_pool_norm=cfg.post_pool_norm,
                representation_eps=cfg.representation.eps,
                grid_hw=grid_hw,
                pool_hw=cfg.pool_window,
                repeat_pool=cfg.repeat_pool,
                noise_tau=cfg.noise_tau,
            )
            generator_metrics = train_generator_step(
                generator_key,
                decoder,
                decoder_ema,
                decoder_optimizer,
                discriminator,
                samples["image"],
                pooled_tokens,
                use_gan=use_gan,
                gan_weight=cfg.train.gan_weight,
                diffaug=diffaug,
                max_d_weight=cfg.train.max_d_weight,
                use_lpips=use_lpips,
                lpips=lpips,
                lpips_weight=cfg.train.lpips_weight,
                ema_momentum=cfg.train.ema_decay,
                should_update_ema=should_update_ema,
                reuse_adaptive_key=cfg.stage1_profile == "raev2_github",
                source_world_size=(
                    RAEV2_SOURCE_WORLD_SIZE
                    if cfg.stage1_profile == "raev2_github"
                    else None
                ),
            )
        discriminator_metrics = {}
        if train_discriminator:
            discriminator_tokens = pooled_tokens
            if cfg.stage1_profile == "raev2_github":
                # RAEv2 switches the RAE to eval mode and recomputes the
                # discriminator reconstruction, so encoder noise is disabled.
                if cfg.downsample_mode == "conv":
                    discriminator_tokens = tokenize_pool_source_step(
                        token_key,
                        patches,
                        final_mean,
                        tokenizer,
                        post_pool_norm=cfg.post_pool_norm,
                        representation_eps=cfg.representation.eps,
                        grid_hw=grid_hw,
                        pool_hw=cfg.pool_window,
                        repeat_pool=cfg.repeat_pool,
                        noise_tau=0.0,
                    )
                else:
                    discriminator_tokens = encode_pooled_tokens(
                        token_key,
                        samples["image"],
                        dino,
                        tokenizer,
                        backbone_resolution=cfg.backbone_resolution,
                        num_prefix_tokens=cfg.num_prefix_tokens,
                        layer_indices=layer_indices,
                        aggregation=cfg.representation.aggregation,
                        normalize_each=cfg.representation.normalize_each,
                        add_final_mean=cfg.representation.add_final_mean,
                        post_pool_norm=cfg.post_pool_norm,
                        representation_eps=cfg.representation.eps,
                        grid_hw=grid_hw,
                        pool_hw=cfg.pool_window,
                        repeat_pool=cfg.repeat_pool,
                        noise_tau=0.0,
                    )
                decoder.eval()
            elif cfg.downsample_mode == "conv":
                discriminator_tokens = tokenize_pool_source_step(
                    token_key,
                    patches,
                    final_mean,
                    tokenizer,
                    post_pool_norm=cfg.post_pool_norm,
                    representation_eps=cfg.representation.eps,
                    grid_hw=grid_hw,
                    pool_hw=cfg.pool_window,
                    repeat_pool=cfg.repeat_pool,
                    noise_tau=cfg.noise_tau,
                )
            discriminator.train()
            discriminator_metrics = train_discriminator_step(
                discriminator_key,
                discriminator,
                discriminator_optimizer,
                decoder,
                samples["image"],
                discriminator_tokens,
                diffaug,
            )
            if cfg.stage1_profile == "raev2_github":
                decoder.train()

        updates_completed = int(
            optax.tree_utils.tree_get(decoder_optimizer.opt_state, "gradient_step")
        )
        ran_update = updates_completed > previous_updates
        if ran_update:
            progress.update(updates_completed - previous_updates)
            profiler.step(updates_completed)
            if manager and manager.reached_preemption(updates_completed):
                save_pooled_checkpoint(
                    manager,
                    updates_completed,
                    decoder,
                    decoder_ema,
                    tokenizer,
                    tokenizer_ema,
                    discriminator,
                    decoder_optimizer,
                    discriminator_optimizer,
                    data_iter,
                    cfg,
                )
                manager.wait_until_finished()
                break
            if manager and manager.should_save(updates_completed):
                save_pooled_checkpoint(
                    manager,
                    updates_completed,
                    decoder,
                    decoder_ema,
                    tokenizer,
                    tokenizer_ema,
                    discriminator,
                    decoder_optimizer,
                    discriminator_optimizer,
                    data_iter,
                    cfg,
                )

        metrics = {}
        if use_wandb and updates_completed > wandb_resume_step:
            if ran_update and updates_completed % args.wandb_log_every == 0:
                discriminator_updates = int(
                    optax.tree_utils.tree_get(
                        discriminator_optimizer.opt_state,
                        "gradient_step",
                    )
                )
                metrics = {
                    "train/lr_dec": float(lr_decoder(updates_completed)),
                    # The official scheduler advances only on discriminator
                    # optimizer updates, which begin at epoch six.
                    "train/lr_disc": float(lr_discriminator(discriminator_updates)),
                    **{
                        f"train/{name}": float(value)
                        for name, value in generator_metrics.items()
                    },
                    **{
                        f"train/{name}": float(value)
                        for name, value in discriminator_metrics.items()
                    },
                }
            if ran_update and updates_completed % steps_per_epoch == 0:
                if cfg.downsample_mode == "conv":
                    assert validation_images is not None
                    val_tokens = encode_pooled_tokens(
                        rngs(),
                        validation_images,
                        dino,
                        tokenizer_ema,
                        backbone_resolution=cfg.backbone_resolution,
                        num_prefix_tokens=cfg.num_prefix_tokens,
                        layer_indices=layer_indices,
                        aggregation=cfg.representation.aggregation,
                        normalize_each=cfg.representation.normalize_each,
                        add_final_mean=cfg.representation.add_final_mean,
                        post_pool_norm=cfg.post_pool_norm,
                        representation_eps=cfg.representation.eps,
                        grid_hw=grid_hw,
                        pool_hw=cfg.pool_window,
                        repeat_pool=cfg.repeat_pool,
                        noise_tau=0.0,
                    )
                else:
                    assert log_tokens is not None
                    val_tokens = log_tokens
                tokens = decoder_ema(val_tokens, deterministic=True)
                reconstruction = decoder_ema.unpatchify(
                    tokens[:, :num_output_tokens],
                    denorm_output=True,
                )
                reconstruction = np.clip(np.asarray(reconstruction), 0.0, 1.0)
                side = min(8, int(np.sqrt(reconstruction.shape[0])))
                grid = rearrange(
                    reconstruction[: side * side],
                    "(r c) h w d -> (r h) (c w) d",
                    r=side,
                    c=side,
                )
                metrics["val/recon_grid"] = wandb.Image(
                    grid,
                    caption=f"epoch {updates_completed // steps_per_epoch}",
                )
            if metrics:
                metrics["step"] = updates_completed
                wandb.log(metrics)

        if updates_completed >= total_updates:
            break

    completed_training = updates_completed >= total_updates
    progress.close()
    if completed_training and manager is not None:
        save_pooled_checkpoint(
            manager,
            updates_completed,
            decoder,
            decoder_ema,
            tokenizer,
            tokenizer_ema,
            discriminator,
            decoder_optimizer,
            discriminator_optimizer,
            data_iter,
            cfg,
        )
        manager.wait_until_finished()
    if manager is not None:
        manager.close()
    if completed_training:
        write_success_marker(save_path)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    cli_args = tyro.cli(Args)
    main(cli_args, get_experiment(cli_args.experiment))
