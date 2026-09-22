"""Representation extraction for PoolDINO compression experiments.

This module centralizes the representation definition used by stage-1
compression, stage-2 flow matching, direct pixel decoding, and diagnostics.
It supports the final DINO layer as well as RAEv2-style multi-layer
aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import jax
import jax.numpy as jnp


Aggregation = Literal["mean", "sum"]


@dataclass(frozen=True)
class RepresentationConfig:
    """Definition of the frozen DINO patch representation.

    ``layers`` uses zero-based transformer-block indices. When it is empty,
    ``last_k`` layers are selected from the end of the backbone with
    ``layer_stride`` between consecutive selected layers. For example, a
    24-block ViT with ``last_k=7`` and ``layer_stride=2`` selects
    ``(11, 13, 15, 17, 19, 21, 23)``.
    """

    layers: tuple[int, ...] = ()
    last_k: int = 1
    layer_stride: int = 1
    aggregation: Aggregation = "mean"
    normalize_each: bool = True
    add_final_mean: bool = False
    eps: float = 1e-5

    def resolve_layers(self, num_layers: int) -> tuple[int, ...]:
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if self.layers:
            layers = tuple(int(i) for i in self.layers)
        else:
            if self.last_k <= 0:
                raise ValueError("last_k must be positive.")
            if self.layer_stride <= 0:
                raise ValueError("layer_stride must be positive.")
            first = num_layers - 1 - (self.last_k - 1) * self.layer_stride
            if first < 0:
                raise ValueError(
                    f"Cannot select last_k={self.last_k} with stride={self.layer_stride} "
                    f"from a {num_layers}-layer encoder."
                )
            layers = tuple(range(first, num_layers, self.layer_stride))

        if len(set(layers)) != len(layers):
            raise ValueError(f"Duplicate layer indices are not allowed: {layers}.")
        if any(i < 0 or i >= num_layers for i in layers):
            raise ValueError(
                f"Layer indices {layers} are invalid for a {num_layers}-layer encoder."
            )
        return layers


def layer_norm_no_affine(x: jax.Array, eps: float = 1e-5) -> jax.Array:
    """Layer-normalize the last dimension without learned affine parameters."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) * jax.lax.rsqrt(var + eps)


def _aggregate_layers(
    layers: Sequence[jax.Array],
    aggregation: Aggregation,
) -> jax.Array:
    if not layers:
        raise ValueError("At least one layer is required.")
    stacked = jnp.stack(tuple(layers), axis=0)
    if aggregation == "mean":
        return jnp.mean(stacked, axis=0)
    if aggregation == "sum":
        return jnp.sum(stacked, axis=0)
    raise ValueError(f"Unsupported aggregation: {aggregation!r}.")


def extract_representation(
    dino,
    images: jax.Array,
    *,
    layer_indices: tuple[int, ...],
    num_prefix_tokens: int,
    aggregation: Aggregation = "mean",
    normalize_each: bool = True,
    add_final_mean: bool = False,
    eps: float = 1e-5,
) -> tuple[jax.Array, jax.Array]:
    """Extract the experiment representation and the final-layer patch field.

    Args:
        dino: ``DinoWithRegisters``-compatible frozen backbone.
        images: Preprocessed images in BHWC format.
        layer_indices: Zero-based block indices to aggregate. An empty tuple
            means use the backbone's normal final output.
        num_prefix_tokens: Number of CLS/register tokens to discard.
        aggregation: Mean or sum across selected layers.
        normalize_each: Apply non-affine LayerNorm to every selected block.
        add_final_mean: Add the spatial mean of the last selected layer to all
            patch tokens. This reproduces the optional global signal used by
            the public RAEv2 DINOv3 implementation.
        eps: LayerNorm epsilon.

    Returns:
        ``(representation_patches, final_layer_patches)``.
    """
    representation, final_patches, final_mean = extract_representation_components(
        dino,
        images,
        layer_indices=layer_indices,
        num_prefix_tokens=num_prefix_tokens,
        aggregation=aggregation,
        normalize_each=normalize_each,
        eps=eps,
    )
    if add_final_mean and final_mean is not None:
        representation = representation + final_mean

    return representation, final_patches


def extract_representation_components(
    dino,
    images: jax.Array,
    *,
    layer_indices: tuple[int, ...],
    num_prefix_tokens: int,
    aggregation: Aggregation = "mean",
    normalize_each: bool = True,
    eps: float = 1e-5,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """Extract the local representation and optional final-layer global mean.

    ``final_mean`` is computed from the last selected normalized layer. It is
    ``None`` for the final-layer-only path so that existing ``-gmean`` behavior
    remains a no-op without MLS features.
    """
    if num_prefix_tokens < 0:
        raise ValueError("num_prefix_tokens must be non-negative.")

    if not layer_indices:
        final = dino(images)
        final_patches = final[:, num_prefix_tokens:]
        return final_patches, final_patches, None

    final, selected = dino(images, layers=layer_indices)
    if len(selected) != len(layer_indices):
        raise RuntimeError(
            f"Backbone returned {len(selected)} layers for request {layer_indices}."
        )

    selected_patches = []
    for activation in selected:
        patches = activation[:, num_prefix_tokens:]
        if normalize_each:
            patches = layer_norm_no_affine(patches, eps)
        selected_patches.append(patches)

    representation = _aggregate_layers(selected_patches, aggregation)
    final_mean = jnp.mean(selected_patches[-1], axis=1, keepdims=True)
    final_patches = final[:, num_prefix_tokens:]
    return representation, final_patches, final_mean


def representation_name(cfg: RepresentationConfig, num_layers: int | None = None) -> str:
    """Return a compact stable label for logs and result tables."""
    if cfg.layers:
        layer_part = ".".join(str(i) for i in cfg.layers)
        base = f"layers{layer_part}"
    elif cfg.last_k == 1:
        base = "final"
    else:
        base = f"k{cfg.last_k}s{cfg.layer_stride}"
    suffix = "sum" if cfg.aggregation == "sum" else "mean"
    if cfg.add_final_mean:
        suffix += "+global"
    if not cfg.normalize_each:
        suffix += "+raw"
    if num_layers is not None:
        resolved = ".".join(str(i) for i in cfg.resolve_layers(num_layers))
        return f"{base}-{suffix}[{resolved}]"
    return f"{base}-{suffix}"
