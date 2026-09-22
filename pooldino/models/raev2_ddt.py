"""Released-code-compatible RAEv2 DDT generator.

This module intentionally lives next to, rather than replacing, :mod:`dit_dh`.
``DiTDH`` predates the released RAEv2 implementation and uses AdaLN-Zero in
its encoder.  The released ``DiTwDDTHeadIG`` instead uses plain pre-norm
encoder blocks and represents time and class conditioning as tokens in the
encoder sequence.  Keeping the implementations separate both preserves old
checkpoints and makes parity experiments unambiguous.

The implementation follows ``nanovisionx/RAEv2`` commit
``8a0d238f8dc3b261aba98b217f6c79c0182e8e94``.  The only intentional extension
is that ``input_grid`` may be rectangular.  Square grids therefore reproduce
the released RoPE table, while pooled 2x4 and 4x2 layouts retain their actual
two-dimensional coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
from typing import Callable, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jmp
from einops import rearrange

from pooldino.models.repa import DenseRepaProjector
from pooldino.utils import Restorable


def _as_pair(value: int | tuple[int, int], *, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        pair = (value, value)
    else:
        if len(value) != 2:
            raise ValueError(f"{name} must contain exactly two values.")
        pair = (int(value[0]), int(value[1]))
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"{name} values must be positive, got {pair}.")
    return pair


@dataclass
class RAEv2DDTConfig:
    """Configuration of the released ImageNet ``DiTwDDTHeadIG`` model.

    Defaults are taken from
    ``configs/stage2/training/imagenet-dinov3l-k7.yaml`` in the released
    repository.  Inputs and outputs use the repository-wide flattened latent
    convention ``[batch, height * width, channels]``.
    """

    input_grid: tuple[int, int] = (16, 16)
    in_channels: int = 1024
    encoder_patch_size: int | tuple[int, int] = 1
    decoder_patch_size: int | tuple[int, int] = 1

    encoder_dim: int = 1440
    decoder_dim: int = 2048
    encoder_depth: int = 28
    decoder_depth: int = 2
    encoder_heads: int = 20
    decoder_heads: int = 16
    mlp_ratio: float = 4.0

    num_classes: int = 1000
    num_time_tokens: int = 4
    num_class_tokens: int = 8
    base_model_depth: int | None = 8
    rope_theta: float = 10_000.0

    def __post_init__(self):
        self.input_grid = _as_pair(self.input_grid, name="input_grid")
        self.encoder_patch_size = _as_pair(
            self.encoder_patch_size, name="encoder_patch_size"
        )
        self.decoder_patch_size = _as_pair(
            self.decoder_patch_size, name="decoder_patch_size"
        )
        for name, value in (
            ("in_channels", self.in_channels),
            ("encoder_dim", self.encoder_dim),
            ("decoder_dim", self.decoder_dim),
            ("encoder_depth", self.encoder_depth),
            ("decoder_depth", self.decoder_depth),
            ("encoder_heads", self.encoder_heads),
            ("decoder_heads", self.decoder_heads),
            ("num_time_tokens", self.num_time_tokens),
            ("num_class_tokens", self.num_class_tokens),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.encoder_dim % self.encoder_heads:
            raise ValueError("encoder_dim must be divisible by encoder_heads.")
        if self.decoder_dim % self.decoder_heads:
            raise ValueError("decoder_dim must be divisible by decoder_heads.")
        for name, dim in (
            ("encoder", self.encoder_dim // self.encoder_heads),
            ("decoder", self.decoder_dim // self.decoder_heads),
        ):
            if dim % 4:
                raise ValueError(f"{name} head dimension must be divisible by four for 2-D RoPE.")
        h, w = self.input_grid
        for name, patch in (
            ("encoder_patch_size", self.encoder_patch_size),
            ("decoder_patch_size", self.decoder_patch_size),
        ):
            if h % patch[0] or w % patch[1]:
                raise ValueError(f"{name}={patch} must divide input_grid={self.input_grid}.")
        enc_grid = (h // self.encoder_patch_size[0], w // self.encoder_patch_size[1])
        dec_grid = (h // self.decoder_patch_size[0], w // self.decoder_patch_size[1])
        if enc_grid != dec_grid:
            # The released decoder conditions each visual token on the
            # corresponding encoder token and cannot broadcast unequal grids.
            raise ValueError(
                "encoder and decoder patch sizes must produce the same token grid; "
                f"got {enc_grid} and {dec_grid}."
            )
        if (
            self.base_model_depth is not None
            and not 1 <= self.base_model_depth <= self.encoder_depth
        ):
            raise ValueError("base_model_depth must lie within the encoder depth.")


def _torch_default_uniform(fan_in: int) -> nnx.Initializer:
    """PyTorch ``nn.Linear``'s default Kaiming-uniform initializer.

    PyTorch calls ``kaiming_uniform_(a=sqrt(5))``, which simplifies to a
    uniform distribution with bound ``1 / sqrt(fan_in)``.
    """

    bound = 1.0 / math.sqrt(fan_in)

    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    return init


def _linear(
    in_features: int,
    out_features: int,
    mp: jmp.Policy,
    *,
    rngs: nnx.Rngs,
    kernel_axes: tuple[str | None, str | None],
    kernel_init: nnx.Initializer | None = None,
    bias_init: nnx.Initializer | None = None,
) -> nnx.Linear:
    kernel_init = kernel_init or _torch_default_uniform(in_features)
    bias_init = bias_init or _torch_default_uniform(in_features)
    return nnx.Linear(
        in_features,
        out_features,
        use_bias=True,
        dtype=mp.compute_dtype,
        param_dtype=mp.param_dtype,
        kernel_init=nnx.with_partitioning(kernel_init, kernel_axes),
        bias_init=bias_init,
        rngs=rngs,
    )


class RAEv2RMSNorm(nnx.Module):
    """RMSNorm matching ``model_utils.RMSNorm`` (including FP32 reduction)."""

    def __init__(self, dim: int, mp: jmp.Policy, *, rngs: nnx.Rngs, eps: float = 1e-6):
        self.eps = eps
        init = nnx.with_partitioning(nnx.initializers.ones_init(), ("model",))
        self.weight = nnx.Param(init(rngs.params(), (dim,), mp.param_dtype))

    def __call__(self, x: jax.Array) -> jax.Array:
        dtype = x.dtype
        x32 = x.astype(jnp.float32)
        x_norm = x32 * jax.lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + self.eps)
        return x_norm.astype(dtype) * self.weight[...]


def _rotate_half(x: jax.Array) -> jax.Array:
    original_shape = x.shape
    pairs = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x1, x2 = pairs[..., 0], pairs[..., 1]
    return jnp.stack((-x2, x1), axis=-1).reshape(original_shape)


class RAEv2RotaryEmbedding(nnx.Module):
    """Released two-dimensional RoPE with an explicit rectangular extension."""

    def __init__(
        self,
        head_dim: int,
        visual_grid: tuple[int, int],
        cond_len: int = 0,
        theta: float = 10_000.0,
    ):
        if head_dim % 4:
            raise ValueError("head_dim must be divisible by four for RAEv2 2-D RoPE.")
        height, width = _as_pair(visual_grid, name="visual_grid")
        half_dim = head_dim // 2
        frequencies = 1.0 / (
            theta
            ** (jnp.arange(0, half_dim, 2, dtype=jnp.float32) / float(half_dim))
        )
        row_angles = jnp.outer(jnp.arange(height, dtype=jnp.float32), frequencies)
        col_angles = jnp.outer(jnp.arange(width, dtype=jnp.float32), frequencies)
        visual_angles = jnp.concatenate(
            (
                jnp.broadcast_to(row_angles[:, None, :], (height, width, frequencies.size)),
                jnp.broadcast_to(col_angles[None, :, :], (height, width, frequencies.size)),
            ),
            axis=-1,
        ).reshape(height * width, half_dim)
        cond_angles = jnp.zeros((cond_len, half_dim), dtype=jnp.float32)
        angles = jnp.repeat(jnp.concatenate((visual_angles, cond_angles), axis=0), 2, axis=-1)
        self.freqs_cos = nnx.Variable(jnp.cos(angles))
        self.freqs_sin = nnx.Variable(jnp.sin(angles))
        self.visual_grid = (height, width)
        self.cond_len = cond_len

    def __call__(self, x: jax.Array) -> jax.Array:
        if x.shape[1] != self.freqs_cos.shape[0]:
            raise ValueError(
                f"RoPE was built for {self.freqs_cos.shape[0]} tokens, got {x.shape[1]}."
            )
        cos = self.freqs_cos[...][None, :, None, :]
        sin = self.freqs_sin[...][None, :, None, :]
        return x * cos + _rotate_half(x) * sin


class RAEv2NormAttention(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mp: jmp.Policy,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.implementation: Literal["cudnn", "xla"] = "xla"
        self.q = _linear(dim, dim, mp, rngs=rngs, kernel_axes=(None, "model"))
        self.k = _linear(dim, dim, mp, rngs=rngs, kernel_axes=(None, "model"))
        self.v = _linear(dim, dim, mp, rngs=rngs, kernel_axes=(None, "model"))
        self.proj = _linear(dim, dim, mp, rngs=rngs, kernel_axes=("model", None))
        self.q_norm = RAEv2RMSNorm(self.head_dim, mp, rngs=rngs)
        self.k_norm = RAEv2RMSNorm(self.head_dim, mp, rngs=rngs)

    def _get_attn_fn(self):
        """Return attention with the runtime-selected JAX backend."""

        return partial(
            jax.nn.dot_product_attention,
            implementation=self.implementation,
        )

    def __call__(
        self,
        x: jax.Array,
        rope: Callable[[jax.Array], jax.Array],
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        q = rearrange(self.q(x), "b n (h d) -> b n h d", h=self.num_heads)
        k = rearrange(self.k(x), "b n (h d) -> b n h d", h=self.num_heads)
        v = rearrange(self.v(x), "b n (h d) -> b n h d", h=self.num_heads)
        q = rope(self.q_norm(q)).astype(v.dtype)
        k = rope(self.k_norm(k)).astype(v.dtype)
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.ndim != 4:
                raise ValueError("attention_mask must have shape [B,N] or [B,1,1,N].")
            attention_mask = attention_mask.astype(jnp.bool_)
        out = self._get_attn_fn()(q, k, v, mask=attention_mask)
        return self.proj(rearrange(out, "b n h d -> b n (h d)"))


class RAEv2SwiGLUFFN(nnx.Module):
    def __init__(self, dim: int, hidden_dim: int, mp: jmp.Policy, *, rngs: nnx.Rngs):
        self.w1 = _linear(dim, hidden_dim, mp, rngs=rngs, kernel_axes=(None, "model"))
        self.w2 = _linear(dim, hidden_dim, mp, rngs=rngs, kernel_axes=(None, "model"))
        self.w3 = _linear(hidden_dim, dim, mp, rngs=rngs, kernel_axes=("model", None))

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.w3(nnx.silu(self.w1(x)) * self.w2(x))


class RAEv2DDTEncoderBlock(nnx.Module):
    """Plain released DDT encoder block; deliberately no AdaLN or gates."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        mp: jmp.Policy,
        *,
        rngs: nnx.Rngs,
    ):
        hidden_dim = int((2.0 / 3.0) * dim * mlp_ratio)
        self.norm1 = RAEv2RMSNorm(dim, mp, rngs=rngs)
        self.norm2 = RAEv2RMSNorm(dim, mp, rngs=rngs)
        self.attn = RAEv2NormAttention(dim, num_heads, mp, rngs=rngs)
        self.mlp = RAEv2SwiGLUFFN(dim, hidden_dim, mp, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        rope: Callable[[jax.Array], jax.Array],
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        x = x + self.attn(self.norm1(x), rope, attention_mask)
        return x + self.mlp(self.norm2(x))


def _modulate(x: jax.Array, shift: jax.Array, scale: jax.Array) -> jax.Array:
    return x * (1.0 + scale) + shift


class RAEv2DDTDecoderBlock(RAEv2DDTEncoderBlock):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        mp: jmp.Policy,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__(dim, num_heads, mlp_ratio, mp, rngs=rngs)
        self.adaln_modulation = nnx.Sequential(
            nnx.silu,
            _linear(
                dim,
                6 * dim,
                mp,
                rngs=rngs,
                kernel_axes=(None, "model"),
                kernel_init=nnx.initializers.zeros_init(),
                bias_init=nnx.initializers.zeros_init(),
            ),
        )

    def __call__(
        self,
        x: jax.Array,
        condition: jax.Array,
        rope: Callable[[jax.Array], jax.Array],
        attention_mask: jax.Array | None = None,
    ) -> jax.Array:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            self.adaln_modulation(condition), 6, axis=-1
        )
        x = x + gate_msa * self.attn(
            _modulate(self.norm1(x), shift_msa, scale_msa), rope, attention_mask
        )
        return x + gate_mlp * self.mlp(
            _modulate(self.norm2(x), shift_mlp, scale_mlp)
        )


class RAEv2DDTFinalLayer(nnx.Module):
    def __init__(
        self,
        dim: int,
        patch_area: int,
        out_channels: int,
        mp: jmp.Policy,
        *,
        rngs: nnx.Rngs,
    ):
        self.norm = RAEv2RMSNorm(dim, mp, rngs=rngs)
        self.linear = _linear(
            dim,
            patch_area * out_channels,
            mp,
            rngs=rngs,
            kernel_axes=("model", None),
            kernel_init=nnx.initializers.zeros_init(),
            bias_init=nnx.initializers.zeros_init(),
        )
        self.adaln_modulation = nnx.Sequential(
            nnx.silu,
            _linear(
                dim,
                2 * dim,
                mp,
                rngs=rngs,
                kernel_axes=(None, "model"),
                kernel_init=nnx.initializers.zeros_init(),
                bias_init=nnx.initializers.zeros_init(),
            ),
        )

    def __call__(self, x: jax.Array, condition: jax.Array) -> jax.Array:
        if condition.ndim < x.ndim:
            condition = condition[:, None, :]
        shift, scale = jnp.split(self.adaln_modulation(condition), 2, axis=-1)
        return self.linear(_modulate(self.norm(x), shift, scale))


class RAEv2GaussianFourierEmbedding(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_tokens: int,
        mp: jmp.Policy,
        *,
        rngs: nnx.Rngs,
        embedding_size: int = 256,
        scale: float = 1.0,
    ):
        # ``W`` is a requires_grad=False Parameter in the released module.
        self.W = nnx.Variable(
            jax.random.normal(rngs.params(), (embedding_size,), dtype=jnp.float32) * scale
        )
        self.mlp = nnx.Sequential(
            _linear(
                2 * embedding_size,
                dim,
                mp,
                rngs=rngs,
                kernel_axes=(None, "model"),
                kernel_init=nnx.initializers.normal(0.02),
            ),
            nnx.silu,
            _linear(
                dim,
                dim,
                mp,
                rngs=rngs,
                kernel_axes=("model", None),
                kernel_init=nnx.initializers.normal(0.02),
            ),
        )
        token_init = nnx.with_partitioning(
            nnx.initializers.normal(1.0 / math.sqrt(dim)), (None, "model")
        )
        self.learnable_tokens = nnx.Param(
            token_init(rngs.params(), (num_tokens, dim), mp.param_dtype)
        )

    def __call__(self, t: jax.Array) -> tuple[jax.Array, jax.Array]:
        angles = t[:, None] * self.W[...][None, :] * (2.0 * jnp.pi)
        base = self.mlp(jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1))
        base = base[:, None, :]
        return base, base + self.learnable_tokens[...][None, :, :]


class RAEv2LabelConditionEmbedder(nnx.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        num_tokens: int,
        mp: jmp.Policy,
        *,
        rngs: nnx.Rngs,
    ):
        embed_init = nnx.with_partitioning(nnx.initializers.normal(0.02), (None, "model"))
        self.embedding_table = nnx.Embed(
            num_classes + 1,
            dim,
            embedding_init=embed_init,
            param_dtype=mp.param_dtype,
            dtype=mp.compute_dtype,
            rngs=rngs,
        )
        token_init = nnx.with_partitioning(
            nnx.initializers.normal(1.0 / math.sqrt(dim)), (None, "model")
        )
        self.learnable_tokens = nnx.Param(
            token_init(rngs.params(), (num_tokens, dim), mp.param_dtype)
        )
        self.num_classes = num_classes

    def __call__(self, labels: jax.Array) -> jax.Array:
        if labels.ndim != 1:
            raise ValueError(f"labels must have shape [batch], got {labels.shape}.")
        return self.embedding_table(labels)[:, None, :] + self.learnable_tokens[...][None, :, :]


class RAEv2DDT(nnx.Module, Restorable):
    """JAX/NNX port of the released ``DiTwDDTHeadIG`` architecture."""

    def __init__(
        self,
        cfg: RAEv2DDTConfig,
        mp: jmp.Policy,
        *,
        rngs: nnx.Rngs,
        latent_mean: jax.Array | None = None,
        latent_std: jax.Array | None = None,
        self_repa_layer: int | None = None,
        self_repa_target_grid: tuple[int, int] | None = None,
    ):
        if self_repa_layer is not None and not 1 <= self_repa_layer <= cfg.encoder_depth:
            raise ValueError("self_repa_layer must lie within the encoder depth.")
        if (self_repa_layer is None) != (self_repa_target_grid is None):
            raise ValueError(
                "self_repa_layer and self_repa_target_grid must be provided together."
            )

        self.cfg = cfg
        self.mp = mp
        self.self_repa_layer = self_repa_layer
        self.input_grid = cfg.input_grid
        self.encoder_patch_size = cfg.encoder_patch_size
        self.decoder_patch_size = cfg.decoder_patch_size
        h, w = cfg.input_grid
        sph, spw = cfg.encoder_patch_size
        xph, xpw = cfg.decoder_patch_size
        self.encoder_grid = (h // sph, w // spw)
        self.decoder_grid = (h // xph, w // xpw)
        self.num_visual_tokens = self.encoder_grid[0] * self.encoder_grid[1]
        self.num_condition_tokens = cfg.num_time_tokens + cfg.num_class_tokens

        # Released PatchEmbed weights are overwritten with xavier-uniform and
        # zero bias immediately after module construction.
        self.s_embedder = _linear(
            cfg.in_channels * sph * spw,
            cfg.encoder_dim,
            mp,
            rngs=rngs,
            kernel_axes=(None, "model"),
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=nnx.initializers.zeros_init(),
        )
        self.x_embedder = _linear(
            cfg.in_channels * xph * xpw,
            cfg.decoder_dim,
            mp,
            rngs=rngs,
            kernel_axes=(None, "model"),
            kernel_init=nnx.initializers.xavier_uniform(),
            bias_init=nnx.initializers.zeros_init(),
        )
        if cfg.encoder_dim == cfg.decoder_dim:
            self.s_projector = None
        else:
            self.s_projector = _linear(
                cfg.encoder_dim,
                cfg.decoder_dim,
                mp,
                rngs=rngs,
                kernel_axes=(None, "model"),
            )

        self.t_embedder = RAEv2GaussianFourierEmbedding(
            cfg.encoder_dim, cfg.num_time_tokens, mp, rngs=rngs
        )
        self.ctx_embedder = RAEv2LabelConditionEmbedder(
            cfg.encoder_dim,
            cfg.num_classes,
            cfg.num_class_tokens,
            mp,
            rngs=rngs,
        )
        self.encoder_blocks = nnx.List(
            [
                RAEv2DDTEncoderBlock(
                    cfg.encoder_dim,
                    cfg.encoder_heads,
                    cfg.mlp_ratio,
                    mp,
                    rngs=rngs,
                )
                for _ in range(cfg.encoder_depth)
            ]
        )
        self.decoder_blocks = nnx.List(
            [
                RAEv2DDTDecoderBlock(
                    cfg.decoder_dim,
                    cfg.decoder_heads,
                    cfg.mlp_ratio,
                    mp,
                    rngs=rngs,
                )
                for _ in range(cfg.decoder_depth)
            ]
        )
        self.final_layer = RAEv2DDTFinalLayer(
            cfg.decoder_dim, xph * xpw, cfg.in_channels, mp, rngs=rngs
        )
        if cfg.base_model_depth is not None:
            self.base_final_layer = RAEv2DDTFinalLayer(
                cfg.encoder_dim, sph * spw, cfg.in_channels, mp, rngs=rngs
            )
        self.encoder_rope = RAEv2RotaryEmbedding(
            cfg.encoder_dim // cfg.encoder_heads,
            self.encoder_grid,
            self.num_condition_tokens,
            cfg.rope_theta,
        )
        self.decoder_rope = RAEv2RotaryEmbedding(
            cfg.decoder_dim // cfg.decoder_heads,
            self.decoder_grid,
            0,
            cfg.rope_theta,
        )

        if self_repa_layer is not None:
            assert self_repa_target_grid is not None
            target_grid = _as_pair(
                self_repa_target_grid,
                name="self_repa_target_grid",
            )
            if target_grid == self.encoder_grid:
                # Reference self-REPA is one PyTorch-default Linear applied to
                # block-eight visual tokens.  Keep that exact initialization
                # for pool1x1, where source and generator grids coincide.
                self.self_repa_projector = _linear(
                    cfg.encoder_dim,
                    cfg.in_channels,
                    mp,
                    rngs=rngs,
                    kernel_axes=("model", None),
                )
            else:
                # Spatially compressed latents are our research extension: a
                # coarse token predicts each frozen source token in its cell.
                self.self_repa_projector = DenseRepaProjector(
                    cfg.encoder_dim,
                    cfg.in_channels,
                    input_grid=self.encoder_grid,
                    target_grid=target_grid,
                    mp=mp,
                    rngs=rngs,
                )

        if latent_mean is not None or latent_std is not None:
            if latent_mean is None or latent_std is None:
                raise ValueError("latent_mean and latent_std must be provided together.")
            self.latent_mean = nnx.Variable(jnp.asarray(latent_mean, dtype=mp.param_dtype))
            self.latent_std = nnx.Variable(jnp.asarray(latent_std, dtype=mp.param_dtype))

    def normalize(self, z: jax.Array) -> jax.Array:
        if not hasattr(self, "latent_mean"):
            return z
        return (z - self.latent_mean[...]) / self.latent_std[...]

    def denormalize(self, z: jax.Array) -> jax.Array:
        if not hasattr(self, "latent_mean"):
            return z
        return z * self.latent_std[...] + self.latent_mean[...]

    def _patchify(self, x: jax.Array, patch: tuple[int, int]) -> jax.Array:
        if x.ndim != 3:
            raise ValueError(f"RAEv2DDT expects flattened [B,H*W,C] input, got {x.shape}.")
        h, w = self.input_grid
        if x.shape[1:] != (h * w, self.cfg.in_channels):
            raise ValueError(
                f"Expected input shape [B,{h * w},{self.cfg.in_channels}], got {x.shape}."
            )
        ph, pw = patch
        image = x.reshape(x.shape[0], h, w, self.cfg.in_channels)
        return rearrange(
            image,
            "b (gh ph) (gw pw) c -> b (gh gw) (c ph pw)",
            ph=ph,
            pw=pw,
        )

    def _unpatchify(
        self,
        x: jax.Array,
        patch: tuple[int, int],
        grid: tuple[int, int],
    ) -> jax.Array:
        ph, pw = patch
        gh, gw = grid
        image = rearrange(
            x,
            # This differs from PatchEmbed's channel-first flattened input:
            # released DDT.unpatchify reshapes the prediction as [p, p, c].
            "b (gh gw) (ph pw c) -> b (gh ph) (gw pw) c",
            gh=gh,
            gw=gw,
            ph=ph,
            pw=pw,
            c=self.cfg.in_channels,
        )
        return image.reshape(image.shape[0], -1, self.cfg.in_channels)

    def _attention_mask(
        self,
        batch_size: int,
        condition_attention_mask: jax.Array | None,
    ) -> jax.Array | None:
        if condition_attention_mask is None:
            return None
        if condition_attention_mask.shape[0] != batch_size:
            raise ValueError("Condition attention-mask batch does not match inputs.")
        if condition_attention_mask.shape[1] > self.num_condition_tokens:
            raise ValueError("Condition attention mask is longer than the condition sequence.")
        prefix = (
            self.num_visual_tokens
            + self.num_condition_tokens
            - condition_attention_mask.shape[1]
        )
        return jnp.concatenate(
            (
                jnp.ones((batch_size, prefix), dtype=jnp.bool_),
                condition_attention_mask.astype(jnp.bool_),
            ),
            axis=1,
        )

    def __call__(
        self,
        x: jax.Array,
        t: jax.Array,
        y: jax.Array,
        *,
        train: bool = True,
        return_base_model: bool = False,
        return_self_repa: bool = False,
        condition_attention_mask: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        del train  # CFG dropout is applied to labels by the released transport, not the model.
        if return_base_model and self.cfg.base_model_depth is None:
            raise ValueError(
                "return_base_model=True but this no-base RAEv2 DDT has no internal-guidance head."
            )
        if return_self_repa and self.self_repa_layer is None:
            raise ValueError(
                "return_self_repa=True but this model has no self-REPA projector."
            )
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError(f"t must have shape [batch], got {t.shape}.")
        if y.shape != t.shape:
            raise ValueError(f"y must have shape [batch], got {y.shape}.")

        s = self.s_embedder(self._patchify(x, self.encoder_patch_size))
        t_base, t_tokens = self.t_embedder(t)
        class_tokens = self.ctx_embedder(y)
        sequence = jnp.concatenate((s, t_tokens, class_tokens), axis=1)
        attention_mask = self._attention_mask(x.shape[0], condition_attention_mask)

        base_hidden = None
        self_repa_prediction = None
        for depth, block in enumerate(self.encoder_blocks, start=1):
            sequence = block(sequence, self.encoder_rope, attention_mask)
            visual = sequence[:, : self.num_visual_tokens]
            if return_base_model and depth == self.cfg.base_model_depth:
                base_hidden = visual
            if return_self_repa and depth == self.self_repa_layer:
                self_repa_prediction = self.self_repa_projector(visual)

        visual = nnx.silu(t_base + sequence[:, : self.num_visual_tokens])
        condition = visual if self.s_projector is None else self.s_projector(visual)

        x_tokens = self.x_embedder(self._patchify(x, self.decoder_patch_size))
        for block in self.decoder_blocks:
            x_tokens = block(x_tokens, condition, self.decoder_rope)
        prediction = self.final_layer(x_tokens, condition)
        prediction = self._unpatchify(
            prediction, self.decoder_patch_size, self.decoder_grid
        )

        output = {"x": prediction}
        if return_base_model:
            if base_hidden is None:
                raise RuntimeError("base_model_depth was not reached.")
            base_state = nnx.silu(t_base + base_hidden)
            base_prediction = self.base_final_layer(base_state, base_state)
            output["base_x"] = self._unpatchify(
                base_prediction, self.encoder_patch_size, self.encoder_grid
            )
        if return_self_repa:
            if self_repa_prediction is None:
                raise RuntimeError("self_repa_layer was not reached.")
            output["self_repa"] = self_repa_prediction
        return output


__all__ = [
    "RAEv2DDT",
    "RAEv2DDTConfig",
    "RAEv2DDTDecoderBlock",
    "RAEv2DDTEncoderBlock",
    "RAEv2DDTFinalLayer",
    "RAEv2GaussianFourierEmbedding",
    "RAEv2LabelConditionEmbedder",
    "RAEv2NormAttention",
    "RAEv2RMSNorm",
    "RAEv2RotaryEmbedding",
    "RAEv2SwiGLUFFN",
]
