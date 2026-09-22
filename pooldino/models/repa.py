"""Projection heads shared by pooled-latent REPA experiments."""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jmp


class DenseRepaProjector(nnx.Module):
    """Expand coarse hidden tokens into a row-major full-resolution token grid.

    Each coarse token predicts the ``pool_h * pool_w`` target tokens in its
    corresponding spatial cell. The explicit transpose is the token-space
    equivalent of anisotropic pixel shuffle and is important for 2x4/4x2
    pooling, where a plain reshape would silently scramble spatial positions.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        input_grid: tuple[int, int],
        target_grid: tuple[int, int],
        mp: jmp.Policy,
        rngs: nnx.Rngs,
    ):
        input_h, input_w = input_grid
        target_h, target_w = target_grid
        if min(input_h, input_w, target_h, target_w) <= 0:
            raise ValueError("REPA grid dimensions must be positive.")
        if target_h % input_h != 0 or target_w % input_w != 0:
            raise ValueError(
                f"target_grid={target_grid} must be divisible by input_grid={input_grid}."
            )

        self.input_grid = input_grid
        self.target_grid = target_grid
        self.expansion = (target_h // input_h, target_w // input_w)
        self.out_features = out_features

        row_kernel_init = nnx.with_partitioning(
            nnx.initializers.xavier_uniform(),
            ("model", None),
        )
        expansion_size = self.expansion[0] * self.expansion[1]
        self.linear = nnx.Linear(
            in_features,
            expansion_size * out_features,
            use_bias=True,
            kernel_init=row_kernel_init,
            bias_init=nnx.initializers.zeros_init(),
            param_dtype=mp.param_dtype,
            dtype=mp.compute_dtype,
            rngs=rngs,
        )

    def __call__(self, tokens: jax.Array) -> jax.Array:
        if tokens.ndim != 3:
            raise ValueError(f"Expected BTD hidden tokens, got {tokens.shape}.")
        input_h, input_w = self.input_grid
        if tokens.shape[1] != input_h * input_w:
            raise ValueError(
                f"input_grid={self.input_grid} has {input_h * input_w} tokens, "
                f"but the projector received {tokens.shape[1]}."
            )

        pool_h, pool_w = self.expansion
        batch = tokens.shape[0]
        projected = self.linear(tokens)
        projected = projected.reshape(
            batch,
            input_h,
            input_w,
            pool_h,
            pool_w,
            self.out_features,
        )
        projected = jnp.transpose(projected, (0, 1, 3, 2, 4, 5))
        return projected.reshape(batch, self.target_grid[0] * self.target_grid[1], -1)


__all__ = ["DenseRepaProjector"]
