"""Frozen PoolDINO source helpers and the task-specific dense ViT decoder.

The original pooled RGB checkpoints store ``decoder_ema`` and
``tokenizer_ema`` as separate Orbax items.  Dense-task training restores only
the latter: DINO and RepeatConv are frozen, the compressed tokens are nearest
repeated to the original 16x16 patch grid, and a newly initialized ViT-XL is
trained for one downstream task.

This module is shared implementation code.  Segmentation and depth always
instantiate, optimize, and checkpoint independent :class:`PooledTaskDecoder`
objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jmp

from pooldino.models.transformer import set_attn_implementation
from pooldino.models.vit import ViTConfig, ViTEncoder
from pooldino.pooled_generator import (
    PooledDecoderIdentity,
    RestoredPooledDecoderComponents,
    _canonical_artifact_path,
    _open_pooled_manager,
    encode_pooled_decoder_latents_impl,
    restore_pooled_decoder_components,
)
from pooldino.train_decoder import (
    PatchDownsampler,
    pooled_grid_shape,
    repeat_pooled_tokens,
)
from pooldino.models.decoder import RAEDecoder
from pooldino.utils import Restorable


@dataclass
class DenseOptimConfig:
    """Optimizer schedule shared structurally, but configured per task."""

    epochs: int = 80
    batch_size: int = 32
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    lr_start: float = 0.0
    lr_peak: float = 2e-4
    lr_final: float = 1e-6
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    lr_schedule: Literal["warmup_cosine"] = "warmup_cosine"
    grad_clip_norm: float = 3.0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive.")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("warmup_epochs must lie in [0, epochs).")
        if self.lr_start < 0 or self.lr_peak <= 0 or self.lr_final < 0:
            raise ValueError("Learning rates must be non-negative and lr_peak positive.")
        if self.weight_decay < 0 or self.grad_clip_norm <= 0:
            raise ValueError("weight_decay must be non-negative and grad_clip_norm positive.")


@dataclass
class PooledTaskDecoderConfig:
    """A 256-position RAEv2 ViT followed by a patchwise task projection."""

    vit: ViTConfig = field(default_factory=ViTConfig)
    output_channels: int = 1
    output_grid: tuple[int, int] = (16, 16)
    output_patch_size: int = 1
    head_bias: float = 0.0

    def __post_init__(self) -> None:
        if self.output_channels <= 0:
            raise ValueError("output_channels must be positive.")
        if self.output_grid[0] <= 0 or self.output_grid[1] <= 0:
            raise ValueError("output_grid dimensions must be positive.")
        if self.output_patch_size <= 0:
            raise ValueError("output_patch_size must be positive.")
        expected_tokens = self.output_grid[0] * self.output_grid[1]
        if self.vit.patch is not None:
            raise ValueError("PooledTaskDecoder consumes tokens, so vit.patch must be None.")
        if self.vit.num_patches != expected_tokens:
            raise ValueError(
                f"vit.num_patches must be {expected_tokens}, got {self.vit.num_patches}."
            )
        if self.vit.latent_upsample != "none":
            raise ValueError(
                "Task decoders consume nearest-repeated 16x16 tokens; latent interpolation "
                "must be disabled."
            )


def task_decoder_config_from_source(
    restored: RestoredPooledDecoderComponents,
    *,
    output_channels: int,
    output_patch_size: int = 1,
    head_bias: float = 0.0,
) -> PooledTaskDecoderConfig:
    """Construct the exact RAEv2 ViT-XL geometry for a new task decoder."""

    hidden_size = restored.latent_spec.feat
    output_grid = restored.grid_hw
    num_tokens = output_grid[0] * output_grid[1]
    vit = replace(
        restored.cfg.vit,
        patch=None,
        num_patches=num_tokens,
        input_dim=hidden_size,
        output_dim=None,
        num_registers=0,
        latent_grid_hw=output_grid,
        latent_upsample="none",
    )
    return PooledTaskDecoderConfig(
        vit=vit,
        output_channels=output_channels,
        output_grid=output_grid,
        output_patch_size=output_patch_size,
        head_bias=head_bias,
    )


def unpatchify_task_values(
    values: jax.Array,
    *,
    grid_hw: tuple[int, int],
    patch_size: int,
    output_channels: int,
) -> jax.Array:
    """Turn one flattened output patch per token into a dense output map."""

    if values.ndim != 3:
        raise ValueError(f"Expected task values in BTD format, got {values.shape}.")
    grid_h, grid_w = grid_hw
    expected_tokens = grid_h * grid_w
    expected_features = patch_size**2 * output_channels
    if values.shape[1:] != (expected_tokens, expected_features):
        raise ValueError(
            "Task value shape does not match its output geometry: "
            f"got {values.shape[1:]}, expected "
            f"{(expected_tokens, expected_features)}."
        )
    batch = values.shape[0]
    if patch_size == 1:
        # Segmentation predicts one class-logit vector per ViT position. Keep
        # this path as the original direct reshape rather than routing it
        # through the depth-specific patch transpose. Besides being simpler,
        # this avoids needlessly changing the large 512x512x150 XLA graph.
        return values.reshape(batch, grid_h, grid_w, output_channels)
    values = values.reshape(
        batch,
        grid_h,
        grid_w,
        patch_size,
        patch_size,
        output_channels,
    )
    values = jnp.transpose(values, (0, 1, 3, 2, 4, 5))
    return values.reshape(
        batch,
        grid_h * patch_size,
        grid_w * patch_size,
        output_channels,
    )


class PooledTaskDecoder(nnx.Module, Restorable):
    """Independent task ViT operating on the repeated 16x16 token field."""

    def __init__(
        self,
        cfg: PooledTaskDecoderConfig,
        mp: jmp.Policy,
        *,
        rngs: nnx.Rngs,
    ):
        self.cfg = cfg
        self.mp = mp
        self.vit = ViTEncoder(cfg.vit, mp, rngs=rngs)
        self.task_head = nnx.Linear(
            cfg.vit.transformer.residual_dim,
            cfg.output_patch_size**2 * cfg.output_channels,
            use_bias=True,
            dtype=mp.compute_dtype,
            param_dtype=mp.param_dtype,
            kernel_init=nnx.initializers.truncated_normal(0.02),
            bias_init=nnx.initializers.constant(cfg.head_bias),
            rngs=rngs,
        )

    def __call__(
        self,
        repeated_tokens: jax.Array,
        *,
        output_hw: tuple[int, int],
        deterministic: bool | None = None,
    ) -> jax.Array:
        """Return dense logits/values with shape ``(B, H, W, C_out)``."""

        if repeated_tokens.ndim != 3:
            raise ValueError(
                f"Expected repeated tokens in BTD format, got {repeated_tokens.shape}."
            )
        expected_tokens = self.cfg.output_grid[0] * self.cfg.output_grid[1]
        if repeated_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"Expected {expected_tokens} repeated tokens, got "
                f"{repeated_tokens.shape[1]}."
            )
        if output_hw[0] <= 0 or output_hw[1] <= 0:
            raise ValueError(f"output_hw must be positive, got {output_hw}.")

        features = self.vit(repeated_tokens, deterministic=deterministic)
        if self.vit.use_cls:
            features = features[:, 1:]
        if self.vit.num_reg:
            features = features[:, self.vit.num_reg :]
        values = self.task_head(self.mp.cast_to_compute(features))
        values = unpatchify_task_values(
            self.mp.cast_to_output(values),
            grid_hw=self.cfg.output_grid,
            patch_size=self.cfg.output_patch_size,
            output_channels=self.cfg.output_channels,
        )
        native_output_hw = (
            self.cfg.output_grid[0] * self.cfg.output_patch_size,
            self.cfg.output_grid[1] * self.cfg.output_patch_size,
        )
        if output_hw != native_output_hw:
            batch = values.shape[0]
            values = jax.image.resize(
                values,
                (batch, output_hw[0], output_hw[1], self.cfg.output_channels),
                method="bilinear",
                antialias=False,
            )
        return values

    def get_state(self) -> dict[str, dict]:
        return {"decoder": nnx.to_pure_dict(nnx.state(self))}


def _copy_rgb_decoder_vit_parameters(
    task_decoder: PooledTaskDecoder,
    rgb_decoder: RAEDecoder,
) -> None:
    """Copy only transformer parameters, leaving the task head untouched."""

    source_state = nnx.state(rgb_decoder, nnx.Param)
    if "decoder_proj" not in source_state:
        raise ValueError("RGB decoder checkpoint has no decoder_proj parameters.")
    source_state.pop("decoder_proj")
    target_state = nnx.state(task_decoder.vit, nnx.Param)

    source_pure = nnx.to_pure_dict(source_state)
    target_pure = nnx.to_pure_dict(target_state)
    if jax.tree.structure(source_pure) != jax.tree.structure(target_pure):
        raise ValueError(
            "RGB decoder transformer structure does not match the task ViT."
        )
    source_shapes = [tuple(value.shape) for value in jax.tree.leaves(source_pure)]
    target_shapes = [tuple(value.shape) for value in jax.tree.leaves(target_pure)]
    if source_shapes != target_shapes:
        raise ValueError(
            "RGB decoder transformer parameter shapes do not match the task ViT: "
            f"source={source_shapes}, target={target_shapes}."
        )
    nnx.update(task_decoder.vit, source_state)


def initialize_task_vit_from_rgb_decoder(
    task_decoder: PooledTaskDecoder,
    checkpoint: Path | str,
    *,
    mesh: jax.sharding.Mesh,
    mp: jmp.Policy,
    step: int = 40032,
    use_ema: bool = True,
) -> PooledDecoderIdentity:
    """Initialize a task ViT from a pooled RGB decoder checkpoint.

    The temporary :class:`RAEDecoder` is restored only to obtain its ViT and
    input-projection parameters.  Its RGB projection and image statistics are
    discarded, while the task-specific output head keeps its random
    initialization.
    """

    manager, restore_step, raw_cfg, config_sha256 = _open_pooled_manager(
        checkpoint,
        step=step,
    )
    try:
        item = "decoder_ema" if use_ema else "decoder"
        rgb_decoder = RAEDecoder.restore(
            manager,
            restore_step,
            item,
            mesh,
            cfg=task_decoder.cfg.vit,
            patch_size=raw_cfg.patch_size,
            num_channels=raw_cfg.out_channels,
            mp=mp,
        )
        _copy_rgb_decoder_vit_parameters(task_decoder, rgb_decoder)
        del rgb_decoder
    finally:
        manager.close()

    return PooledDecoderIdentity(
        checkpoint_path=_canonical_artifact_path(checkpoint),
        checkpoint_step=restore_step,
        use_ema=use_ema,
        stage1_profile=getattr(raw_cfg, "stage1_profile", "legacy"),
        config_sha256=config_sha256,
    )


def restore_frozen_pooled_source(
    checkpoint: Path | str,
    *,
    mesh: jax.sharding.Mesh,
    mp: jmp.Policy,
    step: int = 40032,
    implementation: Literal["cudnn", "xla"] = "xla",
    dinov3_checkpoint_path: Path | str | None = None,
) -> RestoredPooledDecoderComponents:
    """Restore DINO and ``tokenizer_ema`` without loading the RGB ViT-XL."""

    restored = restore_pooled_decoder_components(
        checkpoint,
        mesh=mesh,
        mp=mp,
        step=step,
        use_ema=True,
        restore_decoder=False,
        implementation=implementation,
        dinov3_checkpoint_path=dinov3_checkpoint_path,
    )
    if restored.decoder is not None:
        raise AssertionError("restore_decoder=False unexpectedly returned a decoder.")
    if restored.step != step:
        raise ValueError(
            f"Requested pooled decoder step {step}, restored step {restored.step}."
        )
    if restored.grid_hw != (16, 16):
        raise ValueError(
            f"Dense task decoders require the official 16x16 source grid, got "
            f"{restored.grid_hw}."
        )
    if restored.cfg.stage1_profile != "raev2_github":
        raise ValueError(
            "Dense task training currently requires a raev2_github pooled decoder."
        )
    restored.dino.eval()
    restored.tokenizer.eval()
    return restored


def source_encoding_kwargs(
    restored: RestoredPooledDecoderComponents,
) -> dict[str, object]:
    """Return static arguments for :func:`encode_repeated_pooled_tokens`."""

    representation = restored.cfg.representation
    return {
        "backbone_resolution": restored.cfg.backbone_resolution,
        "num_prefix_tokens": restored.cfg.num_prefix_tokens,
        "layer_indices": restored.layer_indices,
        "aggregation": representation.aggregation,
        "normalize_each": representation.normalize_each,
        "add_final_mean": representation.add_final_mean,
        "post_pool_norm": restored.cfg.post_pool_norm,
        "representation_eps": representation.eps,
        "grid_hw": restored.grid_hw,
        "pool_hw": restored.cfg.pool_window,
    }


def encode_repeated_pooled_tokens(
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
) -> jax.Array:
    """Encode clean unique tokens, then nearest-repeat to the DINO grid."""

    unique = encode_pooled_decoder_latents_impl(
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
    repeated = repeat_pooled_tokens(
        unique,
        pooled_grid_hw=pooled_grid_shape(grid_hw, pool_hw),
        pool_hw=pool_hw,
    )
    expected_tokens = grid_hw[0] * grid_hw[1]
    if repeated.shape[1] != expected_tokens:
        raise ValueError(
            f"Nearest repeat produced {repeated.shape[1]} tokens; expected "
            f"{expected_tokens}."
        )
    return jax.lax.stop_gradient(repeated)


def configure_task_attention(
    decoder: PooledTaskDecoder,
    implementation: Literal["cudnn", "xla"],
) -> None:
    """Set the attention backend on the newly trained task ViT only."""

    set_attn_implementation(decoder, implementation)


__all__ = [
    "DenseOptimConfig",
    "PooledTaskDecoder",
    "PooledTaskDecoderConfig",
    "configure_task_attention",
    "encode_repeated_pooled_tokens",
    "initialize_task_vit_from_rgb_decoder",
    "restore_frozen_pooled_source",
    "source_encoding_kwargs",
    "task_decoder_config_from_source",
]
