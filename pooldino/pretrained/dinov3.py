"""Frozen DINOv3 ViT-L/16 adapter for the released RAEv2 stage-one recipe.

The architecture and checkpoint key mapping follow the public DINOv3 model
loaded by RAEv2's pinned GitHub implementation.  The module resolves the exact
checkpoint through an explicit path, an environment override, or pooldino's
checksum-verified shared cache; it never falls back to DINOv2 weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jmp
import numpy as np

from pooldino.models.transformer import LayerNorm
from pooldino.pretrained.raev2_assets import (
    DINOV3_VITL16_ASSET,
    raev2_asset_cache_path,
    resolve_raev2_asset,
    sha256_file,
)


DINOV3_VITL16_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_VITL16_FILENAME = DINOV3_VITL16_ASSET.filename
DINOV3_VITL16_SHA256 = DINOV3_VITL16_ASSET.sha256
DEFAULT_DINOV3_VITL16_PATH = raev2_asset_cache_path(DINOV3_VITL16_ASSET)


@dataclass(frozen=True)
class DINOv3ViTL16Config:
    image_size: int = 256
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_register_tokens: int = 4
    layer_norm_eps: float = 1e-5
    layerscale_value: float = 1e-5
    rope_theta: float = 100.0


def _trunc_normal(std: float = 0.02):
    return nnx.initializers.truncated_normal(std)


def checkpoint_sha256(path: str | Path) -> str:
    return sha256_file(path)


class DINOv3PatchEmbed(nnx.Module):
    def __init__(self, cfg: DINOv3ViTL16Config, mp: jmp.Policy, *, rngs: nnx.Rngs):
        self.proj = nnx.Conv(
            cfg.num_channels,
            cfg.hidden_size,
            kernel_size=(cfg.patch_size, cfg.patch_size),
            strides=(cfg.patch_size, cfg.patch_size),
            padding="VALID",
            use_bias=True,
            dtype=mp.compute_dtype,
            param_dtype=mp.param_dtype,
            kernel_init=_trunc_normal(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def __call__(self, images: jax.Array) -> jax.Array:
        field = self.proj(images)
        return field.reshape(field.shape[0], -1, field.shape[-1])


def _rope_sincos(
    height: int,
    width: int,
    head_dim: int,
    *,
    base: float,
    periods: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """DINOv3 axial RoPE with separately normalized H/W coordinates."""
    if head_dim % 4:
        raise ValueError("DINOv3 RoPE head_dim must be divisible by four.")
    coords_h = jnp.arange(0.5, height, dtype=jnp.float32) / height
    coords_w = jnp.arange(0.5, width, dtype=jnp.float32) / width
    grid_h, grid_w = jnp.meshgrid(coords_h, coords_w, indexing="ij")
    coords = jnp.stack((grid_h, grid_w), axis=-1).reshape(-1, 2)
    coords = 2.0 * coords - 1.0
    if periods is None:
        periods = base ** (
            2.0 * jnp.arange(head_dim // 4, dtype=jnp.float32) / (head_dim // 2)
        )
    else:
        periods = jnp.asarray(periods, dtype=jnp.float32)
        if periods.shape != (head_dim // 4,):
            raise ValueError(
                f"Expected {head_dim // 4} RoPE periods, got {periods.shape}."
            )
    angles = 2.0 * jnp.pi * coords[:, :, None] / periods[None, None, :]
    angles = jnp.tile(angles.reshape(height * width, head_dim // 2), (1, 2))
    return jnp.sin(angles), jnp.cos(angles)


def _rotate_half(x: jax.Array) -> jax.Array:
    first, second = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-second, first), axis=-1)


class DINOv3Attention(nnx.Module):
    def __init__(self, cfg: DINOv3ViTL16Config, mp: jmp.Policy, *, rngs: nnx.Rngs):
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.qkv = nnx.Linear(
            cfg.hidden_size,
            3 * cfg.hidden_size,
            use_bias=True,
            dtype=mp.compute_dtype,
            param_dtype=mp.param_dtype,
            kernel_init=_trunc_normal(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        self.proj = nnx.Linear(
            cfg.hidden_size,
            cfg.hidden_size,
            use_bias=True,
            dtype=mp.compute_dtype,
            param_dtype=mp.param_dtype,
            kernel_init=_trunc_normal(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        rope: tuple[jax.Array, jax.Array],
    ) -> jax.Array:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch,
            tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        q, k, v = (qkv[:, :, i] for i in range(3))

        sin, cos = rope
        num_patches = sin.shape[0]
        prefix = tokens - num_patches
        if prefix < 0:
            raise ValueError("RoPE table has more patches than the input sequence.")

        q_dtype, k_dtype = q.dtype, k.dtype
        sin = sin[None, :, None, :]
        cos = cos[None, :, None, :]
        q_patches = q[:, prefix:].astype(jnp.float32)
        k_patches = k[:, prefix:].astype(jnp.float32)
        q_patches = q_patches * cos + _rotate_half(q_patches) * sin
        k_patches = k_patches * cos + _rotate_half(k_patches) * sin
        q = jnp.concatenate((q[:, :prefix], q_patches.astype(q_dtype)), axis=1)
        k = jnp.concatenate((k[:, :prefix], k_patches.astype(k_dtype)), axis=1)

        attended = jax.nn.dot_product_attention(q, k, v, implementation="xla")
        attended = attended.reshape(batch, tokens, channels)
        return self.proj(attended)


class DINOv3MLP(nnx.Module):
    def __init__(self, cfg: DINOv3ViTL16Config, mp: jmp.Policy, *, rngs: nnx.Rngs):
        kwargs = dict(
            use_bias=True,
            dtype=mp.compute_dtype,
            param_dtype=mp.param_dtype,
            kernel_init=_trunc_normal(),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )
        self.fc1 = nnx.Linear(cfg.hidden_size, cfg.intermediate_size, **kwargs)
        self.fc2 = nnx.Linear(cfg.intermediate_size, cfg.hidden_size, **kwargs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc2(jax.nn.gelu(self.fc1(x), approximate=False))


class DINOv3Block(nnx.Module):
    def __init__(self, cfg: DINOv3ViTL16Config, mp: jmp.Policy, *, rngs: nnx.Rngs):
        self.norm1 = LayerNorm(
            cfg.hidden_size,
            epsilon=cfg.layer_norm_eps,
            param_dtype=mp.param_dtype,
            rngs=rngs,
        )
        self.attn = DINOv3Attention(cfg, mp, rngs=rngs)
        self.ls1 = nnx.Param(
            jnp.full((cfg.hidden_size,), cfg.layerscale_value, dtype=mp.param_dtype)
        )
        self.norm2 = LayerNorm(
            cfg.hidden_size,
            epsilon=cfg.layer_norm_eps,
            param_dtype=mp.param_dtype,
            rngs=rngs,
        )
        self.mlp = DINOv3MLP(cfg, mp, rngs=rngs)
        self.ls2 = nnx.Param(
            jnp.full((cfg.hidden_size,), cfg.layerscale_value, dtype=mp.param_dtype)
        )

    def __call__(self, x: jax.Array, rope: tuple[jax.Array, jax.Array]) -> jax.Array:
        x = x + self.attn(self.norm1(x), rope) * self.ls1.value
        return x + self.mlp(self.norm2(x)) * self.ls2.value


def _unwrap_state_dict(value) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("DINOv3 checkpoint must contain a mapping state_dict.")
    for key in ("state_dict", "model", "teacher"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            value = nested
    return value


def _canonical_state_dict(value) -> dict[str, object]:
    state = _unwrap_state_dict(value)
    canonical: dict[str, object] = {}
    prefixes = ("module.", "backbone.", "teacher.backbone.")
    for original_key, tensor in state.items():
        key = str(original_key)
        stripped = True
        while stripped:
            stripped = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    stripped = True
        canonical[key] = tensor
    return canonical


def _numpy(tensor) -> np.ndarray:
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu()
    if hasattr(tensor, "float"):
        tensor = tensor.float()
    return np.asarray(tensor.numpy() if hasattr(tensor, "numpy") else tensor)


def _required(state: Mapping[str, object], key: str) -> np.ndarray:
    if key not in state:
        raise KeyError(f"Official DINOv3 checkpoint is missing {key!r}.")
    return _numpy(state[key])


def _assign_linear(linear: nnx.Linear, state: Mapping[str, object], prefix: str) -> None:
    linear.kernel.value = jnp.asarray(_required(state, f"{prefix}.weight").T)
    if linear.bias is not None:
        linear.bias.value = jnp.asarray(_required(state, f"{prefix}.bias"))


def _assign_norm(norm: LayerNorm, state: Mapping[str, object], prefix: str) -> None:
    norm.norm.scale.value = jnp.asarray(_required(state, f"{prefix}.weight"))
    norm.norm.bias.value = jnp.asarray(_required(state, f"{prefix}.bias"))


class DINOv3ViTL16(nnx.Module):
    """The exact frozen DINOv3-L/16 backbone selected by RAEv2 K7."""

    def __init__(
        self,
        pretrained_path: str | Path | None = None,
        *,
        resolution: int = 256,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.float32,
        persisted_path: str | Path | None = None,
    ):
        if resolution != 256:
            raise ValueError(
                "The GitHub RAEv2 ImageNet K7 recipe uses DINOv3-L/16 at 256px."
            )
        # Validate this before allocating a 300M-parameter ViT. In addition to
        # producing a useful error, this prevents the official recipe from
        # silently continuing with random or DINOv2 weights.
        path = self.resolve_checkpoint_path(
            pretrained_path,
            persisted_path=persisted_path,
        )
        self.resolution = resolution
        self.patch_size = 16
        self.num_prefix_tokens = 5  # CLS + four storage/register tokens
        self.config = DINOv3ViTL16Config(image_size=resolution)
        self.mp = jmp.Policy(
            compute_dtype=dtype,
            param_dtype=param_dtype,
            # DINOv3 concatenates fp32 CLS/storage parameters with the bf16
            # patch projection. PyTorch promotes that sequence to fp32, and
            # its non-affine intermediate LayerNorm consequently returns fp32.
            output_dtype=jnp.float32,
        )
        rngs = nnx.Rngs(0)
        self.patch_embed = DINOv3PatchEmbed(self.config, self.mp, rngs=rngs)
        self.cls_token = nnx.Param(
            _trunc_normal()(rngs(), (1, 1, self.config.hidden_size), param_dtype)
        )
        self.storage_tokens = nnx.Param(
            _trunc_normal()(
                rngs(),
                (1, self.config.num_register_tokens, self.config.hidden_size),
                param_dtype,
            )
        )
        self.blocks = nnx.List(
            [
                DINOv3Block(self.config, self.mp, rngs=rngs)
                for _ in range(self.config.num_hidden_layers)
            ]
        )
        # Although the hub architecture declares fp32 RoPE, the released
        # checkpoint stores this persistent buffer in bfloat16.  PyTorch casts
        # those quantized values to fp32 while loading.  Recomputing the periods
        # from ``base`` therefore introduces a measurable representation drift.
        self.rope_periods = nnx.Variable(
            self.config.rope_theta
            ** (
                2.0
                * jnp.arange(
                    self.config.hidden_size // self.config.num_attention_heads // 4,
                    dtype=jnp.float32,
                )
                / (self.config.hidden_size // self.config.num_attention_heads // 2)
            )
        )

        self.load_pretrained(path)

    @staticmethod
    def resolve_checkpoint_path(
        path: str | Path | None,
        verify_checksum: bool = True,
        *,
        persisted_path: str | Path | None = None,
        cache_root: str | Path | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
    ) -> Path:
        return resolve_raev2_asset(
            DINOV3_VITL16_ASSET,
            path,
            persisted_path=persisted_path,
            cache_root=cache_root,
            verify_checksum=verify_checksum,
            force_download=force_download,
            local_files_only=local_files_only,
        )

    def load_pretrained(self, path: Path) -> None:
        import torch

        raw = torch.load(path, map_location="cpu", weights_only=False)
        state = _canonical_state_dict(raw)
        self.cls_token.value = jnp.asarray(_required(state, "cls_token"))
        self.storage_tokens.value = jnp.asarray(_required(state, "storage_tokens"))
        self.rope_periods[...] = jnp.asarray(_required(state, "rope_embed.periods"))
        patch_kernel = _required(state, "patch_embed.proj.weight")
        self.patch_embed.proj.kernel.value = jnp.asarray(
            patch_kernel.transpose(2, 3, 1, 0)
        )
        self.patch_embed.proj.bias.value = jnp.asarray(
            _required(state, "patch_embed.proj.bias")
        )

        for index, block in enumerate(self.blocks):
            prefix = f"blocks.{index}"
            _assign_norm(block.norm1, state, f"{prefix}.norm1")
            _assign_linear(block.attn.qkv, state, f"{prefix}.attn.qkv")
            # The released DINOv3 model masks the key third of the QKV bias.
            qkv_bias = block.attn.qkv.bias.value
            width = qkv_bias.shape[0] // 3
            if f"{prefix}.attn.qkv.bias_mask" in state:
                mask = jnp.asarray(_required(state, f"{prefix}.attn.qkv.bias_mask"))
                qkv_bias = qkv_bias * mask
            else:
                qkv_bias = qkv_bias.at[width : 2 * width].set(0)
            block.attn.qkv.bias.value = qkv_bias
            _assign_linear(block.attn.proj, state, f"{prefix}.attn.proj")
            block.ls1.value = jnp.asarray(_required(state, f"{prefix}.ls1.gamma"))
            _assign_norm(block.norm2, state, f"{prefix}.norm2")
            _assign_linear(block.mlp.fc1, state, f"{prefix}.mlp.fc1")
            _assign_linear(block.mlp.fc2, state, f"{prefix}.mlp.fc2")
            block.ls2.value = jnp.asarray(_required(state, f"{prefix}.ls2.gamma"))

    def encode(
        self,
        images: jax.Array,
        deterministic: bool = True,
        layers: Sequence[int] | None = None,
    ) -> jax.Array | tuple[jax.Array, list[jax.Array]]:
        del deterministic
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError(f"Expected BHWC RGB images, got {images.shape}.")
        patch_h = images.shape[1] // self.patch_size
        patch_w = images.shape[2] // self.patch_size
        rope = _rope_sincos(
            patch_h,
            patch_w,
            self.config.hidden_size // self.config.num_attention_heads,
            base=self.config.rope_theta,
            periods=self.rope_periods[...],
        )
        tokens = self.patch_embed(self.mp.cast_to_compute(images))
        batch = tokens.shape[0]
        cls = jnp.broadcast_to(self.cls_token.value, (batch, 1, tokens.shape[-1]))
        storage = jnp.broadcast_to(
            self.storage_tokens.value,
            (batch, self.config.num_register_tokens, tokens.shape[-1]),
        )
        tokens = jnp.concatenate((cls, storage, tokens), axis=1)

        requested = tuple(layers or ())
        if any(i < 0 or i >= self.config.num_hidden_layers for i in requested):
            raise ValueError(f"Invalid DINOv3 layer selection: {requested}.")
        requested_set = set(requested)
        captured: dict[int, jax.Array] = {}
        for index, block in enumerate(self.blocks):
            tokens = block(tokens, rope)
            if index in requested_set:
                captured[index] = self.mp.cast_to_output(tokens)
        final = self.mp.cast_to_output(tokens)
        if not layers:
            # The no-MLS path follows DINOv3's non-affine final LayerNorm.
            mean = jnp.mean(final, axis=-1, keepdims=True)
            var = jnp.mean(jnp.square(final - mean), axis=-1, keepdims=True)
            return (final - mean) * jax.lax.rsqrt(var + self.config.layer_norm_eps)
        return final, [captured[index] for index in requested]

    def __call__(
        self,
        images: jax.Array,
        deterministic: bool = True,
        layers: Sequence[int] | None = None,
    ) -> jax.Array | tuple[jax.Array, list[jax.Array]]:
        return self.encode(images, deterministic=deterministic, layers=layers)
