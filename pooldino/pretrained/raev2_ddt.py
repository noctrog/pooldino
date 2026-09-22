"""Official RAEv2 DDT checkpoint conversion and numerical parity helpers.

Torch is deliberately imported only by :func:`load_raev2_ddt_checkpoint` and
the comparison harness.  Training and ordinary JAX model imports therefore do
not acquire a PyTorch dependency at runtime.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from pooldino.models.raev2_ddt import (
    RAEv2DDT,
    RAEv2DDTDecoderBlock,
    RAEv2DDTEncoderBlock,
    RAEv2DDTFinalLayer,
)


@dataclass(frozen=True)
class RAEv2DDTConversionReport:
    loaded_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


@dataclass(frozen=True)
class ParityStatistic:
    shape: tuple[int, ...]
    max_abs: float
    mean_abs: float
    max_rel: float
    p99_abs: float
    rmse: float
    reference_rms: float
    nrmse: float
    cosine_similarity: float


def configure_raev2_ddt_parity_precision() -> None:
    """Force semantic-comparison matmuls to use strict FP32 accumulation.

    JAX's default GPU FP32 dot precision may use TF32.  That is a valid fast
    execution mode, but comparing it with Torch CPU FP32 makes harmless dot
    rounding accumulate across all 28 encoder blocks.  A parity diagnostic
    must choose strict accumulation itself rather than depend on the caller's
    environment.
    """

    jax.config.update("jax_default_matmul_precision", "highest")


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float32)


def _select_state_dict(checkpoint: Mapping[str, Any], component: str | None) -> dict[str, Any]:
    state: Mapping[str, Any] = checkpoint
    if component is not None and component in state and isinstance(state[component], Mapping):
        state = state[component]
    elif "state_dict" in state and isinstance(state["state_dict"], Mapping):
        state = state["state_dict"]
    elif "model" in state and isinstance(state["model"], Mapping):
        state = state["model"]

    result: dict[str, Any] = {}
    for original_key, value in state.items():
        if not hasattr(value, "shape"):
            continue
        key = str(original_key)
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        result[key] = value
    if not result:
        raise ValueError("Checkpoint contains no tensor-like state-dict entries.")
    return result


def _assign(variable, array: np.ndarray, *, key: str) -> None:
    if tuple(variable.shape) != tuple(array.shape):
        raise ValueError(
            f"Shape mismatch for {key}: checkpoint {array.shape}, JAX {variable.shape}."
        )
    variable[...] = jnp.asarray(array, dtype=variable.dtype)


def _pop(state: dict[str, Any], key: str) -> np.ndarray:
    try:
        return _to_numpy(state.pop(key))
    except KeyError as error:
        raise KeyError(f"Official RAEv2 checkpoint is missing required key '{key}'.") from error


def _assign_linear(state: dict[str, Any], prefix: str, linear) -> None:
    _assign(linear.kernel, _pop(state, f"{prefix}.weight").T, key=f"{prefix}.weight")
    _assign(linear.bias, _pop(state, f"{prefix}.bias"), key=f"{prefix}.bias")


def _assign_patch_embed(state: dict[str, Any], prefix: str, linear) -> None:
    weight = _pop(state, f"{prefix}.weight")
    if weight.ndim != 4:
        raise ValueError(f"{prefix}.weight must be a Conv2d tensor, got {weight.shape}.")
    weight = weight.reshape(weight.shape[0], -1).T
    _assign(linear.kernel, weight, key=f"{prefix}.weight")
    _assign(linear.bias, _pop(state, f"{prefix}.bias"), key=f"{prefix}.bias")


def _assign_norm(state: dict[str, Any], prefix: str, norm) -> None:
    _assign(norm.weight, _pop(state, f"{prefix}.weight"), key=f"{prefix}.weight")


def _assign_encoder_block(
    state: dict[str, Any],
    prefix: str,
    block: RAEv2DDTEncoderBlock,
) -> None:
    _assign_norm(state, f"{prefix}.norm1", block.norm1)
    _assign_norm(state, f"{prefix}.norm2", block.norm2)
    for name in ("q", "k", "v", "proj"):
        _assign_linear(state, f"{prefix}.attn.{name}", getattr(block.attn, name))
    _assign_norm(state, f"{prefix}.attn.q_norm", block.attn.q_norm)
    _assign_norm(state, f"{prefix}.attn.k_norm", block.attn.k_norm)
    for name in ("w1", "w2", "w3"):
        _assign_linear(state, f"{prefix}.mlp.{name}", getattr(block.mlp, name))


def _assign_decoder_block(
    state: dict[str, Any],
    prefix: str,
    block: RAEv2DDTDecoderBlock,
) -> None:
    _assign_encoder_block(state, prefix, block)
    _assign_linear(
        state,
        f"{prefix}.adaln_modulation.1",
        block.adaln_modulation.layers[1],
    )


def _assign_final_layer(
    state: dict[str, Any],
    prefix: str,
    layer: RAEv2DDTFinalLayer,
) -> None:
    _assign_norm(state, f"{prefix}.norm", layer.norm)
    _assign_linear(state, f"{prefix}.linear", layer.linear)
    _assign_linear(
        state,
        f"{prefix}.adaln_modulation.1",
        layer.adaln_modulation.layers[1],
    )


def _validate_rope(state: dict[str, Any], prefix: str, rope, atol: float = 2e-6) -> None:
    for suffix, local in (
        ("freqs_cos", np.asarray(rope.freqs_cos)),
        ("freqs_sin", np.asarray(rope.freqs_sin)),
    ):
        key = f"{prefix}.{suffix}"
        official = _pop(state, key)
        if official.shape != local.shape or not np.allclose(official, local, atol=atol, rtol=0.0):
            max_abs = (
                float(np.max(np.abs(official - local)))
                if official.shape == local.shape
                else None
            )
            raise ValueError(
                f"Fixed RoPE buffer mismatch for {key}: official={official.shape}, "
                f"JAX={local.shape}, max_abs={max_abs}."
            )


def convert_raev2_ddt_state_dict(
    checkpoint: Mapping[str, Any],
    model: RAEv2DDT,
    *,
    component: str | None = "ema",
    strict: bool = True,
) -> RAEv2DDTConversionReport:
    """Load an official DDT/EMA state dict into :class:`RAEv2DDT`.

    The official model stores patch embeddings as Conv2d kernels and all
    linears as ``[out, in]``.  This converter performs the required flattening
    and transposition, validates fixed RoPE buffers, and reports unused keys.
    """

    state = _select_state_dict(checkpoint, component)
    initial_keys = tuple(sorted(state))

    _assign_patch_embed(state, "s_embedder.proj", model.s_embedder)
    _assign_patch_embed(state, "x_embedder.proj", model.x_embedder)
    if model.s_projector is not None:
        _assign_linear(state, "s_projector", model.s_projector)

    _assign(model.t_embedder.W, _pop(state, "t_embedder.W"), key="t_embedder.W")
    _assign(
        model.t_embedder.learnable_tokens,
        _pop(state, "t_embedder.learnable_tokens"),
        key="t_embedder.learnable_tokens",
    )
    _assign_linear(state, "t_embedder.mlp.0", model.t_embedder.mlp.layers[0])
    _assign_linear(state, "t_embedder.mlp.2", model.t_embedder.mlp.layers[2])
    _assign(
        model.ctx_embedder.embedding_table.embedding,
        _pop(state, "ctx_embedder.embedding_table.weight"),
        key="ctx_embedder.embedding_table.weight",
    )
    _assign(
        model.ctx_embedder.learnable_tokens,
        _pop(state, "ctx_embedder.learnable_tokens"),
        key="ctx_embedder.learnable_tokens",
    )

    for index, block in enumerate(model.encoder_blocks):
        _assign_encoder_block(state, f"blocks.{index}", block)
    offset = len(model.encoder_blocks)
    for index, block in enumerate(model.decoder_blocks):
        _assign_decoder_block(state, f"blocks.{offset + index}", block)

    _assign_final_layer(state, "final_layer", model.final_layer)
    if model.cfg.base_model_depth is not None:
        _assign_final_layer(state, "base_final_layer", model.base_final_layer)
    _validate_rope(state, "enc_rope", model.encoder_rope)
    _validate_rope(state, "dec_rope", model.decoder_rope)

    if model.self_repa_layer is not None:
        projector = model.self_repa_projector
        # The released checkpoint format only contains the one-token-to-one-
        # token Linear head. A compressed-grid projector wraps its expanding
        # Linear as ``.linear`` and will fail the normal shape check if someone
        # attempts to load an incompatible official head into it.
        projector = getattr(projector, "linear", projector)
        _assign_linear(state, "repa_projector", projector)

    unexpected = tuple(sorted(state))
    if strict and unexpected:
        preview = ", ".join(unexpected[:12])
        if len(unexpected) > 12:
            preview += f", ... ({len(unexpected)} total)"
        raise ValueError(f"Unexpected official RAEv2 checkpoint keys: {preview}")
    loaded = tuple(key for key in initial_keys if key not in state)
    return RAEv2DDTConversionReport(loaded_keys=loaded, unexpected_keys=unexpected)


def load_raev2_ddt_checkpoint(
    path: str | Path,
    model: RAEv2DDT,
    *,
    component: str | None = "ema",
    strict: bool = True,
) -> RAEv2DDTConversionReport:
    """Load a local official checkpoint; this function lazily imports Torch."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - dependency is present in development
        raise ImportError("Loading a RAEv2 .pt checkpoint requires PyTorch.") from error
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Expected a mapping checkpoint, got {type(checkpoint).__name__}.")
    return convert_raev2_ddt_state_dict(
        checkpoint, model, component=component, strict=strict
    )


def _jax_intermediates(
    model: RAEv2DDT,
    x: jax.Array,
    t: jax.Array,
    labels: jax.Array,
) -> dict[str, np.ndarray]:
    if model.cfg.base_model_depth is None:
        raise ValueError("Layerwise IG parity requires base_model_depth to be enabled.")
    captures: dict[str, Any] = {}
    s = model.s_embedder(model._patchify(x, model.encoder_patch_size))
    captures["s_embed"] = s
    t_base, t_tokens = model.t_embedder(t)
    captures["time_base"] = t_base
    captures["time_tokens"] = t_tokens
    class_tokens = model.ctx_embedder(labels)
    captures["class_tokens"] = class_tokens
    sequence = jnp.concatenate((s, t_tokens, class_tokens), axis=1)
    captures["encoder_input"] = sequence
    base_hidden = None
    for index, block in enumerate(model.encoder_blocks):
        sequence = block(sequence, model.encoder_rope)
        captures[f"encoder.{index}"] = sequence
        if index + 1 == model.cfg.base_model_depth:
            base_hidden = sequence[:, : model.num_visual_tokens]
    visual = jax.nn.silu(t_base + sequence[:, : model.num_visual_tokens])
    condition = visual if model.s_projector is None else model.s_projector(visual)
    captures["condition"] = condition

    x_tokens = model.x_embedder(model._patchify(x, model.decoder_patch_size))
    captures["x_embed"] = x_tokens
    for index, block in enumerate(model.decoder_blocks):
        x_tokens = block(x_tokens, condition, model.decoder_rope)
        captures[f"decoder.{index}"] = x_tokens
    final_tokens = model.final_layer(x_tokens, condition)
    captures["final_tokens"] = final_tokens
    captures["output"] = model._unpatchify(
        final_tokens, model.decoder_patch_size, model.decoder_grid
    )

    if base_hidden is None:
        raise RuntimeError("base_model_depth was not reached in parity trace.")
    base_state = jax.nn.silu(t_base + base_hidden)
    captures["base_state"] = base_state
    base_tokens = model.base_final_layer(base_state, base_state)
    captures["base_final_tokens"] = base_tokens
    captures["base_output"] = model._unpatchify(
        base_tokens, model.encoder_patch_size, model.encoder_grid
    )
    return {key: np.asarray(value, dtype=np.float32) for key, value in captures.items()}


def _torch_intermediates(torch_model: Any, x: np.ndarray, t: np.ndarray, labels: np.ndarray):
    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise ImportError("The numerical parity harness requires PyTorch.") from error

    captures: dict[str, np.ndarray] = {}
    hooks = []

    def capture(name, transform=lambda value: value):
        def hook(_module, _args, output):
            value = transform(output)
            captures[name] = _to_numpy(value)

        return hook

    hooks.append(torch_model.s_embedder.register_forward_hook(capture("s_embed")))
    def capture_time(_module, _args, output):
        captures["time_base"] = _to_numpy(output[0])
        captures["time_tokens"] = _to_numpy(output[1])

    hooks.append(torch_model.t_embedder.register_forward_hook(capture_time))
    hooks.append(torch_model.ctx_embedder.register_forward_hook(capture("class_tokens")))
    for index in range(torch_model.num_enc_blocks):
        hooks.append(torch_model.blocks[index].register_forward_hook(capture(f"encoder.{index}")))
    hooks.append(torch_model.s_projector.register_forward_hook(capture("condition")))
    hooks.append(torch_model.x_embedder.register_forward_hook(capture("x_embed")))
    for index in range(torch_model.num_dec_blocks):
        block = torch_model.blocks[torch_model.num_enc_blocks + index]
        hooks.append(block.register_forward_hook(capture(f"decoder.{index}")))
    hooks.append(torch_model.final_layer.register_forward_hook(capture("final_tokens")))
    hooks.append(torch_model.base_final_layer.register_forward_hook(capture("base_final_tokens")))

    try:
        device = next(torch_model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    x_torch = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(device)
    t_torch = torch.from_numpy(t).to(device)
    labels_torch = torch.from_numpy(labels).to(device)
    @contextmanager
    def strict_float32_accumulation():
        old_matmul_precision = torch.get_float32_matmul_precision()
        old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        try:
            torch.set_float32_matmul_precision("highest")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            yield
        finally:
            torch.set_float32_matmul_precision(old_matmul_precision)
            torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = old_cudnn_tf32

    try:
        torch_model.eval()
        with strict_float32_accumulation(), torch.no_grad():
            output, base_output = torch_model(x_torch, t_torch, context=labels_torch)
    finally:
        for hook in hooks:
            hook.remove()

    captures["encoder_input"] = np.concatenate(
        (captures["s_embed"], captures["time_tokens"], captures["class_tokens"]), axis=1
    )
    base_hidden = captures[f"encoder.{torch_model.base_model_depth - 1}"]
    base_hidden = base_hidden[:, : captures["s_embed"].shape[1]]
    captures["base_state"] = np.asarray(
        torch.nn.functional.silu(
            torch.from_numpy(captures["time_base"] + base_hidden)
        )
    )
    captures["output"] = _to_numpy(output).transpose(0, 2, 3, 1).reshape(
        x.shape[0], -1, x.shape[-1]
    )
    captures["base_output"] = _to_numpy(base_output).transpose(0, 2, 3, 1).reshape(
        x.shape[0], -1, x.shape[-1]
    )
    return captures


def compare_raev2_ddt_torch_jax(
    torch_model: Any,
    jax_model: RAEv2DDT,
    x: np.ndarray,
    t: np.ndarray,
    labels: np.ndarray,
) -> dict[str, ParityStatistic]:
    """Compare every major released DDT stage using shared imported weights.

    ``x`` uses ``[B,H,W,C]`` layout; the helper performs the official NCHW and
    local flattened-token conversions.  The released implementation supports
    square inputs only, so full-checkpoint comparisons should use 16x16.
    """

    configure_raev2_ddt_parity_precision()
    x = np.asarray(x, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if x.ndim != 4 or x.shape[1:3] != jax_model.input_grid:
        raise ValueError(
            f"x must have [B,{jax_model.input_grid[0]},{jax_model.input_grid[1]},C] shape."
        )
    flat_x = x.reshape(x.shape[0], -1, x.shape[-1])
    jax_values = _jax_intermediates(
        jax_model,
        jnp.asarray(flat_x),
        jnp.asarray(t),
        jnp.asarray(labels),
    )
    torch_values = _torch_intermediates(torch_model, x, t, labels)

    if set(jax_values) != set(torch_values):
        missing = sorted(set(torch_values) - set(jax_values))
        extra = sorted(set(jax_values) - set(torch_values))
        raise RuntimeError(f"Parity trace keys differ: missing={missing}, extra={extra}.")
    stats: dict[str, ParityStatistic] = {}
    for key in jax_values:
        jax_value = jax_values[key]
        torch_value = torch_values[key]
        if jax_value.shape != torch_value.shape:
            raise ValueError(
                f"Shape mismatch at {key}: JAX={jax_value.shape}, Torch={torch_value.shape}."
            )
        difference = np.abs(jax_value - torch_value)
        difference64 = jax_value.astype(np.float64) - torch_value.astype(np.float64)
        torch64 = torch_value.astype(np.float64)
        jax64 = jax_value.astype(np.float64)
        rmse = float(np.sqrt(np.mean(np.square(difference64))))
        reference_rms = float(np.sqrt(np.mean(np.square(torch64))))
        nrmse = rmse / max(reference_rms, 1e-12)
        torch_norm = float(np.linalg.norm(torch64.reshape(-1)))
        jax_norm = float(np.linalg.norm(jax64.reshape(-1)))
        if torch_norm == 0.0 and jax_norm == 0.0:
            cosine_similarity = 1.0
        elif torch_norm == 0.0 or jax_norm == 0.0:
            cosine_similarity = 0.0
        else:
            cosine_similarity = float(
                np.vdot(torch64.reshape(-1), jax64.reshape(-1))
                / (torch_norm * jax_norm)
            )
        stats[key] = ParityStatistic(
            shape=tuple(jax_value.shape),
            max_abs=float(np.max(difference)),
            mean_abs=float(np.mean(difference)),
            max_rel=float(np.max(difference / (np.abs(torch_value) + 1e-8))),
            p99_abs=float(np.quantile(difference, 0.99)),
            rmse=rmse,
            reference_rms=reference_rms,
            nrmse=nrmse,
            cosine_similarity=cosine_similarity,
        )
    return stats


def assert_raev2_ddt_parity(
    statistics: Mapping[str, ParityStatistic],
    *,
    atol: float = 2e-5,
) -> None:
    failures = {
        name: stat.max_abs for name, stat in statistics.items() if stat.max_abs > atol
    }
    if failures:
        details = ", ".join(f"{name}={error:.3e}" for name, error in failures.items())
        raise AssertionError(f"RAEv2 DDT numerical parity failed (atol={atol}): {details}")


def assert_raev2_ddt_full_checkpoint_parity(
    statistics: Mapping[str, ParityStatistic],
    *,
    hidden_nrmse_atol: float = 5e-4,
    hidden_cosine_min: float = 0.9999,
    final_nrmse_atol: float = 1e-4,
    final_cosine_min: float = 0.99999,
    final_atol: float = 1e-4,
) -> None:
    """Robust gate for a converted, trained XL checkpoint.

    Deep residual networks can amplify a small accumulation-order difference
    into one sparse hidden-coordinate outlier.  Hidden ``max_abs`` is therefore
    reported but deliberately not gated.  Broad hidden drift is caught by
    normalized RMSE and cosine similarity.  The actual full/base predictions
    additionally retain a strict absolute-output bound.

    Strict key/shape/buffer conversion is enforced separately by
    :func:`convert_raev2_ddt_state_dict` with ``strict=True``.
    """

    if not statistics:
        raise AssertionError("RAEv2 DDT parity produced no stage statistics.")
    final_names = {
        "final_tokens",
        "base_final_tokens",
        "output",
        "base_output",
    }
    missing = {"output", "base_output"}.difference(statistics)
    if missing:
        raise AssertionError(
            f"RAEv2 DDT parity trace is missing final stages: {sorted(missing)}."
        )

    failures: list[str] = []
    for name, statistic in statistics.items():
        numeric = (
            statistic.max_abs,
            statistic.mean_abs,
            statistic.p99_abs,
            statistic.rmse,
            statistic.reference_rms,
            statistic.nrmse,
            statistic.cosine_similarity,
        )
        if not all(np.isfinite(value) for value in numeric):
            failures.append(f"{name}: non-finite statistic")
            continue

        is_final = name in final_names
        nrmse_limit = final_nrmse_atol if is_final else hidden_nrmse_atol
        cosine_limit = final_cosine_min if is_final else hidden_cosine_min
        if statistic.nrmse > nrmse_limit:
            failures.append(
                f"{name}: nrmse={statistic.nrmse:.3e}>{nrmse_limit:.3e}"
            )
        if statistic.cosine_similarity < cosine_limit:
            failures.append(
                f"{name}: cosine={statistic.cosine_similarity:.8f}"
                f"<{cosine_limit:.8f}"
            )
        if is_final and statistic.max_abs > final_atol:
            failures.append(
                f"{name}: max_abs={statistic.max_abs:.3e}>{final_atol:.3e}"
            )

    if failures:
        raise AssertionError(
            "RAEv2 DDT full-checkpoint parity failed: " + "; ".join(failures)
        )


__all__ = [
    "ParityStatistic",
    "RAEv2DDTConversionReport",
    "assert_raev2_ddt_full_checkpoint_parity",
    "assert_raev2_ddt_parity",
    "compare_raev2_ddt_torch_jax",
    "configure_raev2_ddt_parity_precision",
    "convert_raev2_ddt_state_dict",
    "load_raev2_ddt_checkpoint",
]
