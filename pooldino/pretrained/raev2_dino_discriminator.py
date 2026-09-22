"""Exact frozen DINO-S/8 backbone used by RAEv2's stage-one discriminator.

The implementation follows ``src/stage1/disc/dinodisc.py`` at RAEv2 commit
``8a0d238f8dc3b261aba98b217f6c79c0182e8e94``.  In particular it keeps the
fused QKV projection, tanh-approximate GELU, 1e-6 LayerNorm epsilon, learned
absolute positions, and pre-final-norm activations used by the GAN heads.

The official profile never substitutes another DINO checkpoint.  It resolves
the DINO-S/8 checkpoint shipped in the RAEv2 model bundle through explicit,
environment, or shared-cache paths and validates its published SHA-256 before
constructing the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from pooldino.pretrained.raev2_assets import (
    RAEV2_DINO_S8_ASSET,
    raev2_asset_cache_path,
    resolve_raev2_asset,
)


RAEV2_GITHUB_COMMIT = "8a0d238f8dc3b261aba98b217f6c79c0182e8e94"
RAEV2_DINO_S8_FILENAME = RAEV2_DINO_S8_ASSET.filename
RAEV2_DINO_S8_SHA256 = RAEV2_DINO_S8_ASSET.sha256
DEFAULT_RAEV2_DINO_S8_PATH = raev2_asset_cache_path(RAEV2_DINO_S8_ASSET)


@dataclass(frozen=True)
class RAEv2DinoS8Config:
    image_size: int = 224
    patch_size: int = 8
    num_channels: int = 3
    hidden_size: int = 384
    intermediate_size: int = 1536
    num_hidden_layers: int = 12
    num_attention_heads: int = 6
    layer_norm_eps: float = 1e-6


def _canonical_state_dict(value) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("The RAEv2 DINO-S/8 checkpoint must contain a state dict.")
    for key in ("state_dict", "model", "teacher"):
        nested = value.get(key)
        if isinstance(nested, Mapping):
            value = nested
    state: dict[str, object] = {}
    for original_key, tensor in value.items():
        key = str(original_key)
        for prefix in ("module.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        state[key] = tensor
    return state


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def _take(state: dict[str, object], key: str) -> np.ndarray:
    try:
        return _numpy(state.pop(key))
    except KeyError as error:
        raise KeyError(f"Official RAEv2 DINO-S/8 checkpoint is missing {key!r}.") from error


class _FrozenLinear(nnx.Module):
    def __init__(
        self,
        state: dict[str, object],
        prefix: str,
        *,
        compute_dtype: jnp.dtype,
    ):
        self.kernel = nnx.Variable(jnp.asarray(_take(state, f"{prefix}.weight").T))
        self.bias = nnx.Variable(jnp.asarray(_take(state, f"{prefix}.bias")))
        self.compute_dtype = compute_dtype

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=self.compute_dtype)
        kernel = jnp.asarray(self.kernel[...], dtype=self.compute_dtype)
        bias = jnp.asarray(self.bias[...], dtype=self.compute_dtype)
        return jnp.matmul(x, kernel) + bias


class _FrozenLayerNorm(nnx.Module):
    def __init__(self, state: dict[str, object], prefix: str, *, epsilon: float):
        self.scale = nnx.Variable(jnp.asarray(_take(state, f"{prefix}.weight")))
        self.bias = nnx.Variable(jnp.asarray(_take(state, f"{prefix}.bias")))
        self.epsilon = epsilon

    def __call__(self, x: jax.Array) -> jax.Array:
        dtype = x.dtype
        x32 = jnp.asarray(x, dtype=jnp.float32)
        mean = jnp.mean(x32, axis=-1, keepdims=True)
        variance = jnp.mean(jnp.square(x32 - mean), axis=-1, keepdims=True)
        normalized = (x32 - mean) * jax.lax.rsqrt(variance + self.epsilon)
        normalized = normalized * self.scale[...] + self.bias[...]
        return normalized.astype(dtype)


class _FrozenPatchEmbed(nnx.Module):
    def __init__(
        self,
        state: dict[str, object],
        *,
        patch_size: int,
        compute_dtype: jnp.dtype,
    ):
        kernel = _take(state, "patch_embed.proj.weight").transpose(2, 3, 1, 0)
        self.kernel = nnx.Variable(jnp.asarray(kernel))
        self.bias = nnx.Variable(jnp.asarray(_take(state, "patch_embed.proj.bias")))
        self.patch_size = patch_size
        self.compute_dtype = compute_dtype

    def __call__(self, images: jax.Array) -> jax.Array:
        images = jnp.asarray(images, dtype=self.compute_dtype)
        kernel = jnp.asarray(self.kernel[...], dtype=self.compute_dtype)
        bias = jnp.asarray(self.bias[...], dtype=self.compute_dtype)
        patches = jax.lax.conv_general_dilated(
            images,
            kernel,
            window_strides=(self.patch_size, self.patch_size),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )
        patches = patches + bias
        return patches.reshape(patches.shape[0], -1, patches.shape[-1])


class _FrozenAttention(nnx.Module):
    def __init__(
        self,
        state: dict[str, object],
        prefix: str,
        cfg: RAEv2DinoS8Config,
        *,
        compute_dtype: jnp.dtype,
    ):
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.scale = self.head_dim**-0.5
        self.qkv = _FrozenLinear(state, f"{prefix}.qkv", compute_dtype=compute_dtype)
        # RAEv2 explicitly zeros the key-bias third after loading.
        qkv_bias = self.qkv.bias[...]
        width = qkv_bias.shape[0] // 3
        self.qkv.bias[...] = qkv_bias.at[width : 2 * width].set(0)
        self.proj = _FrozenLinear(state, f"{prefix}.proj", compute_dtype=compute_dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch,
            tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        q, k, v = (
            jnp.transpose(qkv[:, :, index], (0, 2, 1, 3))
            for index in range(3)
        )
        logits = jnp.matmul(q * self.scale, jnp.swapaxes(k, -1, -2))
        probabilities = jax.nn.softmax(logits, axis=-1)
        attended = jnp.matmul(probabilities, v)
        attended = jnp.transpose(attended, (0, 2, 1, 3)).reshape(
            batch,
            tokens,
            channels,
        )
        return self.proj(attended)


class _FrozenMLP(nnx.Module):
    def __init__(
        self,
        state: dict[str, object],
        prefix: str,
        *,
        compute_dtype: jnp.dtype,
    ):
        self.fc1 = _FrozenLinear(state, f"{prefix}.fc1", compute_dtype=compute_dtype)
        self.fc2 = _FrozenLinear(state, f"{prefix}.fc2", compute_dtype=compute_dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fc2(jax.nn.gelu(self.fc1(x), approximate=True))


class _FrozenBlock(nnx.Module):
    def __init__(
        self,
        state: dict[str, object],
        index: int,
        cfg: RAEv2DinoS8Config,
        *,
        compute_dtype: jnp.dtype,
    ):
        prefix = f"blocks.{index}"
        self.norm1 = _FrozenLayerNorm(
            state,
            f"{prefix}.norm1",
            epsilon=cfg.layer_norm_eps,
        )
        self.attn = _FrozenAttention(
            state,
            f"{prefix}.attn",
            cfg,
            compute_dtype=compute_dtype,
        )
        self.norm2 = _FrozenLayerNorm(
            state,
            f"{prefix}.norm2",
            epsilon=cfg.layer_norm_eps,
        )
        self.mlp = _FrozenMLP(
            state,
            f"{prefix}.mlp",
            compute_dtype=compute_dtype,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class RAEv2DinoS8(nnx.Module):
    """Frozen exact DINO-S/8 feature proxy used by ``DinoDisc``."""

    def __init__(
        self,
        checkpoint_path: str | Path | None,
        *,
        resolution: int = 224,
        dtype: jnp.dtype = jnp.bfloat16,
        verify_checksum: bool = True,
    ):
        path = self.resolve_checkpoint_path(checkpoint_path, verify_checksum)
        if resolution != 224:
            raise ValueError("RAEv2's official DINO discriminator is fixed at 224px.")

        import torch

        raw = torch.load(path, map_location="cpu", weights_only=False)
        state = _canonical_state_dict(raw)
        self.config = RAEv2DinoS8Config()
        self.resolution = resolution
        self.compute_dtype = dtype
        self.patch_embed = _FrozenPatchEmbed(
            state,
            patch_size=self.config.patch_size,
            compute_dtype=dtype,
        )
        self.cls_token = nnx.Variable(jnp.asarray(_take(state, "cls_token")))
        self.pos_embed = nnx.Variable(jnp.asarray(_take(state, "pos_embed")))
        self.blocks = nnx.List(
            [
                _FrozenBlock(
                    state,
                    index,
                    self.config,
                    compute_dtype=dtype,
                )
                for index in range(self.config.num_hidden_layers)
            ]
        )
        # Loaded for a strict checkpoint mapping even though RAEv2 intentionally
        # returns pre-final-norm discriminator activations.
        self.norm = _FrozenLayerNorm(state, "norm", epsilon=self.config.layer_norm_eps)
        if state:
            unexpected = ", ".join(sorted(state)[:12])
            raise ValueError(f"Unexpected official DINO-S/8 checkpoint keys: {unexpected}")

    @staticmethod
    def resolve_checkpoint_path(
        checkpoint_path: str | Path | None,
        verify_checksum: bool = True,
        *,
        persisted_path: str | Path | None = None,
        cache_root: str | Path | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
    ) -> Path:
        return resolve_raev2_asset(
            RAEV2_DINO_S8_ASSET,
            checkpoint_path,
            persisted_path=persisted_path,
            cache_root=cache_root,
            verify_checksum=verify_checksum,
            force_download=force_download,
            local_files_only=local_files_only,
        )

    def encode(
        self,
        images: jax.Array,
        deterministic: bool = True,
        capture_layers: Sequence[int] | None = None,
    ) -> jax.Array | tuple[jax.Array, list[jax.Array]]:
        del deterministic
        if images.shape[1:] != (224, 224, 3):
            raise ValueError(f"Expected DINO-S/8 input (B,224,224,3), got {images.shape}.")
        tokens = self.patch_embed(images)
        batch = tokens.shape[0]
        cls = jnp.broadcast_to(self.cls_token[...], (batch, 1, self.config.hidden_size))
        tokens = jnp.concatenate((cls, tokens), axis=1)
        tokens = tokens + self.pos_embed[...]

        requested = tuple(capture_layers or ())
        if any(index < 0 or index >= self.config.num_hidden_layers for index in requested):
            raise ValueError(f"Invalid DINO-S/8 layer selection: {requested}.")
        requested_set = set(requested)
        captured: dict[int, jax.Array] = {}
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if index in requested_set:
                captured[index] = tokens
        if not capture_layers:
            return tokens
        return tokens, [captured[index] for index in requested]

    def __call__(
        self,
        images: jax.Array,
        deterministic: bool = True,
        capture_layers: Sequence[int] | None = None,
    ) -> jax.Array | tuple[jax.Array, list[jax.Array]]:
        return self.encode(
            images,
            deterministic=deterministic,
            capture_layers=capture_layers,
        )


__all__ = [
    "DEFAULT_RAEV2_DINO_S8_PATH",
    "RAEV2_DINO_S8_FILENAME",
    "RAEV2_DINO_S8_SHA256",
    "RAEv2DinoS8",
    "RAEv2DinoS8Config",
]
