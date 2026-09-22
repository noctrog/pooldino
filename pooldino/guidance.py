"""Shared configuration/helpers for the PoolDINO paper implementation."""

from __future__ import annotations

from numbers import Real

import jax

def apply_repa_guidance(
    full_prediction: jax.Array,
    repa_prediction: jax.Array,
    scale: float,
) -> jax.Array:
    """One-pass self guidance: full + scale * (full - early)."""
    if isinstance(scale, Real) and scale < 0:
        raise ValueError("REPA guidance scale must be non-negative.")
    return full_prediction + scale * (full_prediction - repa_prediction)


def apply_internal_guidance(
    full_prediction: jax.Array,
    base_prediction: jax.Array,
    scale: float,
) -> jax.Array:
    """RAEv2 internal guidance: base + scale * (full - base).

    ``scale=1`` returns the full prediction. This differs from the legacy
    ``apply_repa_guidance`` convention, where scale zero returns the full
    prediction; both helpers remain explicit to prevent ambiguous sweeps.
    """
    if isinstance(scale, Real) and scale < 0:
        raise ValueError("Internal-guidance scale must be non-negative.")
    return base_prediction + scale * (full_prediction - base_prediction)

