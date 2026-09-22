"""Optax port of the GMuon recipe pinned by the released RAEv2 code.

The source of truth is ``nanovisionx/gmuon`` commit ``4863cc2``.  RAEv2
applies GMuon to every rank-two parameter and AdamW to every other parameter.
This module preserves that split and the pinned Polar Express / Gram
Newton--Schulz details, including the auto-tuned restart after iteration 2.
"""

from __future__ import annotations

import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax


# gram_newton_schulz/coefficients.py at 4863cc2.  Keep the construction used
# upstream rather than copying rounded decimal output.
_UNMODIFIED_POLAR_EXPRESS_COEFFICIENTS = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
)
_POLAR_EXPRESS_SAFETY_FACTOR = 1.05
POLAR_EXPRESS_COEFFICIENTS = tuple(
    (
        a / _POLAR_EXPRESS_SAFETY_FACTOR,
        b / _POLAR_EXPRESS_SAFETY_FACTOR**3,
        c / _POLAR_EXPRESS_SAFETY_FACTOR**5,
    )
    for a, b, c in _UNMODIFIED_POLAR_EXPRESS_COEFFICIENTS
)
GRAM_NEWTON_SCHULZ_RESTARTS = (2,)
GMUON_EPS = 1e-7


def _standard_newton_schulz(x: jax.Array) -> jax.Array:
    for a, b, c in POLAR_EXPRESS_COEFFICIENTS:
        gram = x @ x.T
        polynomial = b * gram + c * (gram @ gram)
        x = a * x + polynomial @ x
    return x


def _gram_newton_schulz(x: jax.Array) -> jax.Array:
    """Pinned Gram-NS loop, including the restart at zero-based index 2."""

    gram = x @ x.T
    identity = jnp.eye(gram.shape[-1], dtype=x.dtype)
    q = None
    last_index = len(POLAR_EXPRESS_COEFFICIENTS) - 1
    for index, (a, b, c) in enumerate(POLAR_EXPRESS_COEFFICIENTS):
        if index in GRAM_NEWTON_SCHULZ_RESTARTS and index != 0:
            assert q is not None
            x = q @ x
            gram = x @ x.T
            q = None

        z = b * gram + c * (gram @ gram)
        if index == 0 or index in GRAM_NEWTON_SCHULZ_RESTARTS:
            q = z + a * identity
        else:
            assert q is not None
            q = q @ z + a * q

        if (
            index < last_index
            and index + 1 not in GRAM_NEWTON_SCHULZ_RESTARTS
        ):
            rz = gram @ z + a * gram
            gram = z @ rz + a * rz

    assert q is not None
    return q @ x


def orthogonalize_gmuon(update: jax.Array) -> jax.Array:
    """Orthogonalize one rank-two momentum update as pinned GMuon does."""

    if update.ndim != 2:
        raise ValueError(f"GMuon requires a rank-two update, got {update.shape}.")

    # Upstream casts the Nesterov update to bf16, normalizes in fp32, and then
    # runs the matmuls in fp16 when custom kernels are disabled.
    original_dtype = jnp.bfloat16
    x = update.astype(original_dtype).astype(jnp.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (jnp.linalg.norm(x) + GMUON_EPS)
    x = x.astype(jnp.float16)

    # The pinned implementation selects Gram-NS for every non-square matrix
    # and standard NS only for a square matrix.
    if x.shape[0] != x.shape[1]:
        x = _gram_newton_schulz(x)
    else:
        x = _standard_newton_schulz(x)

    if transposed:
        x = x.T
    return x.astype(original_dtype)


class _ScaleByGMuonState(NamedTuple):
    count: jax.Array


def _scale_by_gmuon(
    learning_rate: optax.ScalarOrSchedule,
    *,
    weight_decay: float,
) -> optax.GradientTransformation:
    """Apply adjusted LR and decoupled decay to orthogonalized updates."""

    schedule = (
        learning_rate
        if callable(learning_rate)
        else lambda _: jnp.asarray(learning_rate)
    )

    def init_fn(params):
        del params
        return _ScaleByGMuonState(count=jnp.zeros([], dtype=jnp.int32))

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("GMuon requires parameters for decoupled weight decay.")
        base_lr = schedule(state.count)

        def scale_one(update, param):
            orthogonal = orthogonalize_gmuon(update)
            fan_out, fan_in = update.shape[-2:]
            adjusted_lr = base_lr * (0.2 * math.sqrt(max(fan_out, fan_in)))
            # Upstream performs this multiplication in the bf16 update buffer.
            scaled = -orthogonal * jnp.asarray(adjusted_lr, dtype=orthogonal.dtype)
            if weight_decay:
                scaled = scaled.astype(param.dtype) - (
                    jnp.asarray(base_lr * weight_decay, dtype=param.dtype) * param
                )
            return scaled

        scaled = jax.tree.map(scale_one, updates, params)
        return scaled, _ScaleByGMuonState(
            count=optax.safe_int32_increment(state.count)
        )

    return optax.GradientTransformation(init_fn, update_fn)


def gmuon(
    learning_rate: optax.ScalarOrSchedule,
    *,
    momentum: float = 0.95,
    nesterov: bool = True,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    """GMuon transform for a tree containing rank-two parameters only."""

    if not 0 <= momentum < 1:
        raise ValueError("momentum must lie in [0, 1).")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")
    return optax.chain(
        optax.trace(decay=momentum, nesterov=nesterov),
        _scale_by_gmuon(learning_rate, weight_decay=weight_decay),
    )


def rank_two_labels(params):
    """Return the exact RAEv2 optimizer partition for an arbitrary pytree."""

    return jax.tree.map(
        lambda param: "gmuon" if param.ndim == 2 else "adamw",
        params,
    )


def _path_names(path: tuple[object, ...]) -> tuple[object, ...]:
    names = []
    for key in path:
        if hasattr(key, "key"):
            names.append(key.key)
        elif hasattr(key, "name"):
            names.append(key.name)
    return tuple(names)


def raev2_ddt_optimizer_labels(params):
    """Match the official Conv2d-vs-Linear GMuon partition after flattening.

    The released patch embedders are 1x1 Conv2d weights and therefore rank
    four, which sends them to AdamW.  The JAX port stores the same coefficients
    as flattened rank-two Linear kernels, so these two paths need an explicit
    topology-preserving override.
    """

    def label(path, param):
        names = _path_names(path)
        is_flattened_patch_kernel = any(
            names[-2:] == (embedder, "kernel")
            or names[-3:] == (embedder, "kernel", "value")
            for embedder in ("s_embedder", "x_embedder")
        )
        if is_flattened_patch_kernel:
            return "adamw"
        return "gmuon" if param.ndim == 2 else "adamw"

    return jax.tree_util.tree_map_with_path(label, params)


def optimizer_partition_counts(
    params,
    label_fn: Callable = rank_two_labels,
) -> dict[str, int]:
    """Count tensors and scalar parameters assigned to each optimizer."""

    labels = label_fn(params)
    counts = {
        "gmuon_tensors": 0,
        "gmuon_parameters": 0,
        "adamw_tensors": 0,
        "adamw_parameters": 0,
    }
    param_leaves = jax.tree.leaves(params)
    label_leaves = jax.tree.leaves(labels)
    if len(param_leaves) != len(label_leaves):
        raise ValueError("Optimizer label tree does not match the parameter tree.")
    for param, optimizer_name in zip(param_leaves, label_leaves, strict=True):
        if optimizer_name not in ("gmuon", "adamw"):
            raise ValueError(f"Unknown optimizer partition label: {optimizer_name!r}.")
        counts[f"{optimizer_name}_tensors"] += 1
        counts[f"{optimizer_name}_parameters"] += int(param.size)
    return counts


def gmuon_with_adamw_fallback(
    learning_rate: optax.ScalarOrSchedule,
    *,
    adam_b1: float = 0.9,
    adam_b2: float = 0.95,
    adam_eps: float = 1e-8,
    momentum: float = 0.95,
    nesterov: bool = True,
    weight_decay: float = 0.0,
    label_fn: Callable = rank_two_labels,
) -> optax.GradientTransformation:
    """Rank-two GMuon plus AdamW fallback, matching RAEv2's split."""

    return optax.partition(
        {
            "gmuon": gmuon(
                learning_rate,
                momentum=momentum,
                nesterov=nesterov,
                weight_decay=weight_decay,
            ),
            "adamw": optax.adamw(
                learning_rate,
                b1=adam_b1,
                b2=adam_b2,
                eps=adam_eps,
                weight_decay=weight_decay,
            ),
        },
        label_fn,
    )


__all__ = [
    "GMUON_EPS",
    "GRAM_NEWTON_SCHULZ_RESTARTS",
    "POLAR_EXPRESS_COEFFICIENTS",
    "gmuon",
    "gmuon_with_adamw_fallback",
    "optimizer_partition_counts",
    "orthogonalize_gmuon",
    "raev2_ddt_optimizer_labels",
    "rank_two_labels",
]
