"""Parity tests for the released RAEv2 stage-two recipe."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import asdict, replace
import sys
from types import SimpleNamespace

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest
from dacite import Config as DaciteConfig, from_dict

from pooldino.data import DataConfig
from pooldino.gmuon import (
    GRAM_NEWTON_SCHULZ_RESTARTS,
    POLAR_EXPRESS_COEFFICIENTS,
    gmuon_with_adamw_fallback,
    optimizer_partition_counts,
    orthogonalize_gmuon,
    raev2_ddt_optimizer_labels,
    rank_two_labels,
)
from pooldino.training import build_lr_schedule


def _import_or_skip_cv2_blocker(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        if "libGL.so.1" in str(error):
            pytest.skip(f"{module_name} is blocked by the local cv2 dependency: {error}")
        raise


def _torch_pinned_orthogonalize(update: np.ndarray) -> np.ndarray:
    torch = pytest.importorskip("torch")
    x = torch.as_tensor(update).to(torch.bfloat16).to(torch.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = (x / (x.norm() + 1e-7)).to(torch.float16)

    if x.shape[0] == x.shape[1]:
        for a, b, c in POLAR_EXPRESS_COEFFICIENTS:
            gram = x @ x.T
            polynomial = b * gram + c * (gram @ gram)
            x = a * x + polynomial @ x
    else:
        gram = x @ x.T
        identity = torch.eye(gram.shape[-1], dtype=x.dtype)
        q = None
        for index, (a, b, c) in enumerate(POLAR_EXPRESS_COEFFICIENTS):
            if index == 2:
                x = q @ x
                gram = x @ x.T
                q = None
            z = b * gram + c * (gram @ gram)
            q = z + a * identity if index in (0, 2) else q @ z + a * q
            if index < 4 and index + 1 != 2:
                rz = gram @ z + a * gram
                gram = z @ rz + a * rz
        x = q @ x

    if transposed:
        x = x.T
    return x.to(torch.bfloat16).to(torch.float32).numpy()


def test_pinned_polar_express_coefficients_and_restart():
    assert GRAM_NEWTON_SCHULZ_RESTARTS == (2,)
    assert len(POLAR_EXPRESS_COEFFICIENTS) == 5
    np.testing.assert_allclose(
        POLAR_EXPRESS_COEFFICIENTS[0],
        (7.892582874424408, -20.38301394587957, 13.555306149406924),
        rtol=0.0,
        atol=1e-14,
    )


@pytest.mark.parametrize("shape", [(2, 3), (2, 2), (3, 2)])
def test_gmuon_orthogonalization_tracks_literal_pytorch_reference(shape):
    rank = min(shape)
    update = np.zeros(shape, dtype=np.float32)
    update[:rank, :rank] = np.eye(rank, dtype=np.float32)
    actual = np.asarray(orthogonalize_gmuon(jnp.asarray(update)), dtype=np.float32)
    expected = _torch_pinned_orthogonalize(update)
    # CPU XLA and torch use different fp16 GEMM accumulation, so compare at
    # the resolution of two bf16 ULPs. GPU parity tests can use a tighter gate.
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2 / 128)


def test_gmuon_rank_partition_and_rms_adjusted_update():
    params = {
        "matrix": jnp.ones((2, 3), dtype=jnp.float32),
        "bias": jnp.ones((3,), dtype=jnp.float32),
        "scalar": jnp.asarray(1.0, dtype=jnp.float32),
    }
    labels = rank_two_labels(params)
    assert labels == {"matrix": "gmuon", "bias": "adamw", "scalar": "adamw"}

    learning_rate = 1e-3
    transform = gmuon_with_adamw_fallback(learning_rate, weight_decay=0.0)
    grads = {
        "matrix": jnp.eye(2, 3, dtype=jnp.float32),
        "bias": jnp.ones((3,), dtype=jnp.float32),
        "scalar": jnp.asarray(1.0, dtype=jnp.float32),
    }
    updates, _ = transform.update(grads, transform.init(params), params)

    # First-step pinned Nesterov input: M=g, U=.95*M+g=1.95*g.
    orthogonal = orthogonalize_gmuon(1.95 * grads["matrix"])
    adjusted_lr = learning_rate * 0.2 * math.sqrt(3)
    expected_matrix = -orthogonal * jnp.asarray(adjusted_lr, jnp.bfloat16)
    np.testing.assert_array_equal(
        np.asarray(updates["matrix"]),
        np.asarray(expected_matrix),
    )
    # AdamW fallback's first normalized update is -lr for a unit gradient.
    np.testing.assert_allclose(updates["bias"], -learning_rate, atol=1e-7)
    np.testing.assert_allclose(updates["scalar"], -learning_rate, atol=1e-7)


def test_exact_ddt_optimizer_partition_preserves_official_conv_rank_semantics():
    local_params = {
        "s_embedder": {
            "kernel": jnp.ones((4, 8), dtype=jnp.float32),
            "bias": jnp.ones((8,), dtype=jnp.float32),
        },
        "x_embedder": {
            "kernel": jnp.ones((4, 8), dtype=jnp.float32),
            "bias": jnp.ones((8,), dtype=jnp.float32),
        },
        "blocks": {
            "q_kernel": jnp.ones((8, 8), dtype=jnp.float32),
            "norm": jnp.ones((8,), dtype=jnp.float32),
        },
    }
    labels = raev2_ddt_optimizer_labels(local_params)

    # The corresponding GitHub parameters are Conv2d [out,in,1,1], hence
    # ndim=4 -> AdamW. Ordinary Linear weights remain ndim=2 -> GMuon.
    official_ndims = {
        "s_embedder.proj.weight": 4,
        "x_embedder.proj.weight": 4,
        "blocks.0.attn.q.weight": 2,
        "blocks.0.norm1.weight": 1,
    }
    official_labels = {
        name: "gmuon" if ndim == 2 else "adamw"
        for name, ndim in official_ndims.items()
    }
    assert labels["s_embedder"]["kernel"] == official_labels[
        "s_embedder.proj.weight"
    ]
    assert labels["x_embedder"]["kernel"] == official_labels[
        "x_embedder.proj.weight"
    ]
    assert labels["blocks"]["q_kernel"] == official_labels[
        "blocks.0.attn.q.weight"
    ]
    assert labels["blocks"]["norm"] == official_labels[
        "blocks.0.norm1.weight"
    ]
    # Generic/legacy behavior intentionally remains rank-based.
    assert rank_two_labels(local_params)["s_embedder"]["kernel"] == "gmuon"

    counts = optimizer_partition_counts(local_params, raev2_ddt_optimizer_labels)
    assert counts == {
        "gmuon_tensors": 1,
        "gmuon_parameters": 64,
        "adamw_tensors": 5,
        "adamw_parameters": 88,
    }


def test_exact_ddt_optimizer_partition_matches_nnx_param_paths():
    class ExactLikeModule(nnx.Module):
        def __init__(self, *, rngs: nnx.Rngs):
            self.s_embedder = nnx.Linear(4, 8, rngs=rngs)
            self.x_embedder = nnx.Linear(4, 8, rngs=rngs)
            self.other = nnx.Linear(8, 8, rngs=rngs)

    module = ExactLikeModule(rngs=nnx.Rngs(0))
    params = nnx.state(module, nnx.Param)
    labels = raev2_ddt_optimizer_labels(params)
    labels_by_path = {
        jax.tree_util.keystr(path): label
        for path, label in jax.tree_util.tree_leaves_with_path(labels)
    }

    # NNX descends through ``Param.value``. Keep this regression test pinned
    # to those real paths so a future tree-layout change cannot silently send
    # the flattened patch embedders back to GMuon.
    assert labels_by_path["['s_embedder']['kernel'].value"] == "adamw"
    assert labels_by_path["['x_embedder']['kernel'].value"] == "adamw"
    assert labels_by_path["['other']['kernel'].value"] == "gmuon"


def test_gmuon_partition_executes_through_nnx_optimizer():
    class TinyModule(nnx.Module):
        def __init__(self):
            self.matrix = nnx.Param(jnp.eye(2, 3, dtype=jnp.float32))
            self.bias = nnx.Param(jnp.zeros((3,), dtype=jnp.float32))

    module = TinyModule()
    optimizer = nnx.Optimizer(
        module,
        gmuon_with_adamw_fallback(1e-3),
        wrt=nnx.Param,
    )
    matrix_before = np.asarray(module.matrix[...]).copy()
    bias_before = np.asarray(module.bias[...]).copy()
    grads = nnx.grad(lambda model: jnp.sum(model.matrix) + jnp.sum(model.bias))(module)
    optimizer.update(module, grads)
    assert not np.array_equal(np.asarray(module.matrix[...]), matrix_before)
    assert not np.array_equal(np.asarray(module.bias[...]), bias_before)


def test_raev2_linear_schedule_holds_decays_then_holds():
    pooled_generator = _import_or_skip_cv2_blocker(
        "pooldino.pooled_generator"
    )
    cfg = pooled_generator.PooledGeneratorConfig()
    train = cfg.train.__class__(
        epochs=80,
        lr_start=2e-4,
        lr_peak=2e-4,
        lr_final=2e-5,
        warmup_epochs=25,
        lr_schedule="linear_decay",
        decay_end_epoch=50,
    )
    schedule = build_lr_schedule(train, total_updates=8000)
    assert float(schedule(0)) == pytest.approx(2e-4)
    assert float(schedule(2500)) == pytest.approx(2e-4)
    assert float(schedule(3750)) == pytest.approx(1.1e-4)
    assert float(schedule(5000)) == pytest.approx(2e-5)
    assert float(schedule(7999)) == pytest.approx(2e-5)


def test_exact_experiment_defaults_and_kappa_controls():
    pooled_generator = _import_or_skip_cv2_blocker(
        "pooldino.pooled_generator"
    )
    train_module = _import_or_skip_cv2_blocker(
        "pooldino.train_generator"
    )
    restored = SimpleNamespace(
        latent_spec=pooled_generator.PooledLatentSpec((4, 8), 1024),
        grid_hw=(16, 16),
        cfg=SimpleNamespace(stage1_profile="raev2_github"),
    )
    cfg = train_module.get_experiment(
        "raev2ddt",
        restored,
    )
    assert cfg.raev2_ddt.input_grid == (4, 8)
    assert (cfg.raev2_ddt.encoder_dim, cfg.raev2_ddt.encoder_heads) == (1440, 20)
    assert (cfg.raev2_ddt.decoder_dim, cfg.raev2_ddt.decoder_heads) == (2048, 16)
    assert cfg.raev2_ddt.base_model_depth == 8
    assert cfg.conditioning_dropout_prob == pytest.approx(0.1)
    assert cfg.xpred_denom_eps == pytest.approx(0.05)
    assert cfg.latent_norm_eps == pytest.approx(1e-5)
    assert cfg.time_dist_shift_dim == 262_144
    assert cfg.train.optimizer == "gmuon"
    assert cfg.train.weight_decay == 0
    assert cfg.train.epochs == 80
    assert cfg.train.warmup_epochs == 25
    assert cfg.train.decay_end_epoch == 50
    assert cfg.self_repa
    assert cfg.self_repa_layer == 8
    assert cfg.self_repa_coeff == pytest.approx(0.5)
    assert cfg.self_repa_target_grid == (16, 16)

    automatic = train_module.get_experiment(
        "raev2ddt-autokappa",
        restored,
    )
    assert automatic.time_dist_shift_dim is None
    assert automatic.kappa is None

    fixed = train_module.get_experiment(
        "raev2ddt-kappa8.0",
        restored,
    )
    assert fixed.time_dist_shift_dim is None
    assert fixed.kappa == pytest.approx(8.0)

    no_self_repa = train_module.get_experiment(
        "raev2ddt-nosrepa",
        restored,
    )
    assert not no_self_repa.self_repa

    self_repa = train_module.get_experiment(
        "raev2ddt-srepa0.25-srl7-srepacos0.1",
        restored,
    )
    assert self_repa.self_repa
    assert self_repa.self_repa_layer == 7
    assert self_repa.self_repa_coeff == pytest.approx(0.25)
    assert self_repa.self_repa_cosine_weight == pytest.approx(0.1)

    assert train_module.resolve_checkpoints_to_keep(cfg, None) == (20, 80)
    assert train_module.resolve_checkpoints_to_keep(cfg, (20, 80)) == (20, 80)
    extended = replace(cfg, train=replace(cfg.train, epochs=300))
    assert train_module.resolve_checkpoints_to_keep(
        extended,
        (20, 80, 150, 200, 250, 300),
    ) == (20, 80, 150, 200, 250, 300)
    with pytest.raises(ValueError, match="fixed to the 80-epoch RAEv2"):
        replace(cfg, train=replace(cfg.train, epochs=79))
    with pytest.raises(ValueError, match="must preserve checkpoints"):
        train_module.resolve_checkpoints_to_keep(cfg, (80,))
    with pytest.raises(ValueError, match="Could not parse"):
        train_module.get_experiment(
            "raev2ddt-ep200",
            restored,
        )


def test_official_time_shift_path_and_transport_clamp_match_source():
    train_module = _import_or_skip_cv2_blocker(
        "pooldino.train_generator"
    )
    raw_t = jnp.asarray([0.2, 0.5, 0.8], dtype=jnp.float32)
    shifted = train_module._shift_sampled_time(
        raw_t,
        8.0,
    )
    np.testing.assert_allclose(
        shifted,
        8.0 * raw_t / (1.0 + 7.0 * raw_t),
        rtol=0.0,
        atol=1e-7,
    )
    # This explicitly differs from the old local-time shift ordering.
    old_local_shift = raw_t / (8.0 - 7.0 * raw_t)
    assert not np.allclose(np.asarray(shifted), np.asarray(old_local_shift))

    data = jnp.zeros((2, 1, 1), dtype=jnp.float32)
    noise = jnp.ones_like(data)
    official_t = jnp.asarray([0.01, 0.5], dtype=jnp.float32)
    xt, target_drift, safe_t = train_module._raev2_training_path(
        data,
        noise,
        official_t,
        0.05,
    )
    np.testing.assert_allclose(xt[:, 0, 0], official_t, atol=1e-7)
    np.testing.assert_allclose(safe_t[:, 0, 0], [0.05, 0.5], atol=1e-7)
    # Below epsilon the official target is scaled by t / eps, rather than
    # using an unclamped constant noise-data velocity.
    np.testing.assert_allclose(target_drift[:, 0, 0], [0.2, 1.0], atol=1e-7)


def test_transport_dropout_is_applied_outside_exact_model():
    train_module = _import_or_skip_cv2_blocker(
        "pooldino.train_generator"
    )
    labels = jnp.asarray([1, 2, 3], dtype=jnp.int32)
    dropped, fraction = train_module._apply_transport_class_dropout(
        labels,
        jax.random.PRNGKey(0),
        1.0,
        1000,
    )
    np.testing.assert_array_equal(dropped, jnp.full_like(labels, 1000))
    assert float(fraction) == pytest.approx(1.0)

    untouched, fraction = train_module._apply_transport_class_dropout(
        labels,
        jax.random.PRNGKey(0),
        0.0,
        1000,
    )
    np.testing.assert_array_equal(untouched, labels)
    assert float(fraction) == 0.0


def test_exact_sampling_passes_complemented_time_to_ddt():
    evaluator = _import_or_skip_cv2_blocker(
        "pooldino.eval.gfid_pooled_decoder_adm"
    )

    class TimeEcho(nnx.Module):
        def __call__(
            self,
            state,
            t,
            labels,
            *,
            train,
            return_self_repa,
            return_base_model,
        ):
            del labels, train
            assert not return_self_repa
            assert not return_base_model
            clean = jnp.broadcast_to(jnp.square(t)[:, None, None], state.shape)
            return {"x": clean}

    state = jnp.zeros((2, 1, 1), dtype=jnp.float32)
    labels = jnp.asarray([1, 2], dtype=jnp.int32)
    # The evaluator passes the exact official model time, not a local time
    # recovered by complementing twice. clean=t^2 makes the result sensitive
    # to which coordinate reaches the model: .75 * (.75^2 / .75) = .5625.
    result = evaluator.integration_step(
        TimeEcho(),
        state,
        jnp.full((2,), 0.75, dtype=jnp.float32),
        jnp.asarray(0.75, dtype=jnp.float32),
        labels,
        jnp.full_like(labels, 1000),
        use_cfg=False,
        cfg_scale=0.0,
        use_self_repa=False,
        self_repa_scale=0.0,
        xpred_denom_eps=0.05,
    )
    np.testing.assert_allclose(result, 0.5625, rtol=0.0, atol=1e-6)


def test_exact_sampling_grid_is_complement_of_official_shifted_grid():
    evaluator = _import_or_skip_cv2_blocker(
        "pooldino.eval.gfid_pooled_decoder_adm"
    )
    steps = 5
    shift = 8.0
    local_t, model_t = evaluator._sampling_time_coordinates(
        steps,
        jnp.float32,
        time_shift=shift,
    )
    raw_official_t = jnp.linspace(1.0, 0.0, steps + 1, dtype=jnp.float32)
    official_t = shift * raw_official_t / (
        1.0 + (shift - 1.0) * raw_official_t
    )
    np.testing.assert_array_equal(model_t, official_t)
    np.testing.assert_allclose(1.0 - local_t, official_t, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(
        local_t[1:] - local_t[:-1],
        official_t[:-1] - official_t[1:],
        rtol=0.0,
        atol=1e-7,
    )


def test_named_raev2_sampling_protocols():
    evaluator = _import_or_skip_cv2_blocker(
        "pooldino.eval.gfid_pooled_decoder_adm"
    )
    base = evaluator.Config(generator_path="generator", pooled_decoder_path="decoder")
    train = evaluator.resolve_sampling_protocol(replace(base, protocol="raev2_train"))
    assert train.steps == 50
    assert train.per_class == 50
    assert train.seed == 42
    assert train.metric_backend == "raev2_fd"
    assert train.fd_metrics == ("fid", "inception_score")
    assert train.cfg_scale == pytest.approx(1.0)
    assert train.ig_scale is None

    cfg_sweep = evaluator.resolve_sampling_protocol(
        replace(
            base,
            protocol="raev2_cfg_sweep",
            steps=7,
            per_class=3,
            seed=0,
            cfg_scale=1.75,
            cfg_t_min=0.2,
            cfg_t_max=1.0,
            self_repa_guidance_scale=2.0,
            ig_scale=3.0,
        )
    )
    assert cfg_sweep.steps == 50
    assert cfg_sweep.per_class == 50
    assert cfg_sweep.seed == 42
    assert cfg_sweep.metric_backend == "raev2_fd"
    assert cfg_sweep.fd_metrics == ("fid", "inception_score")
    assert cfg_sweep.use_ema
    assert cfg_sweep.use_ema_pooled_decoder
    assert cfg_sweep.cfg_scale == pytest.approx(1.75)
    assert (cfg_sweep.cfg_t_min, cfg_sweep.cfg_t_max) == pytest.approx((0.2, 1.0))
    assert cfg_sweep.self_repa_guidance_scale is None
    assert cfg_sweep.ig_scale is None

    self_repa = evaluator.resolve_sampling_protocol(
        replace(
            base,
            protocol="raev2_self_repa",
            steps=7,
            per_class=3,
            seed=0,
            cfg_scale=2.0,
            cfg_t_min=0.1,
            cfg_t_max=1.0,
            self_repa_guidance_scale=0.75,
            self_repa_t_min=0.1,
            self_repa_t_max=0.8,
            ig_scale=3.0,
        )
    )
    assert self_repa.steps == 50
    assert self_repa.per_class == 50
    assert self_repa.seed == 42
    assert self_repa.metric_backend == "raev2_fd"
    assert self_repa.fd_metrics == ("fid", "inception_score")
    assert self_repa.use_ema
    assert self_repa.use_ema_pooled_decoder
    assert self_repa.cfg_scale == pytest.approx(2.0)
    assert (self_repa.cfg_t_min, self_repa.cfg_t_max) == pytest.approx((0.1, 1.0))
    assert self_repa.self_repa_guidance_scale == pytest.approx(0.75)
    assert (self_repa.self_repa_t_min, self_repa.self_repa_t_max) == pytest.approx(
        (0.1, 0.8)
    )
    assert self_repa.ig_scale is None

    cfg_sweep_100 = evaluator.resolve_sampling_protocol(
        replace(
            base,
            protocol="raev2_cfg_sweep_100",
            steps=7,
            cfg_scale=2.0,
            self_repa_guidance_scale=1.0,
            ig_scale=1.78,
        )
    )
    assert cfg_sweep_100.steps == 100
    assert cfg_sweep_100.fd_metrics == ("fid", "inception_score")
    assert cfg_sweep_100.self_repa_guidance_scale is None
    assert cfg_sweep_100.ig_scale is None

    self_repa_100 = evaluator.resolve_sampling_protocol(
        replace(
            base,
            protocol="raev2_self_repa_100",
            steps=7,
            cfg_scale=2.0,
            cfg_t_min=0.1,
            self_repa_guidance_scale=1.25,
            ig_scale=1.78,
        )
    )
    assert self_repa_100.steps == 100
    assert self_repa_100.fd_metrics == ("fid", "inception_score")
    assert self_repa_100.cfg_scale == pytest.approx(2.0)
    assert self_repa_100.self_repa_guidance_scale == pytest.approx(1.25)
    assert self_repa_100.ig_scale is None

    guided = evaluator.resolve_sampling_protocol(replace(base, protocol="raev2_ig"))
    assert guided.steps == 100
    assert guided.per_class == 50
    assert guided.seed == 42
    assert guided.metric_backend == "raev2_fd"
    assert guided.fd_metrics == ("fid", "fdr6", "mind6")
    assert guided.cfg_scale == pytest.approx(1.0)
    assert guided.ig_scale == pytest.approx(1.78)
    assert (guided.ig_t_min, guided.ig_t_max) == pytest.approx((0.0, 0.9))

    guided_with_cfg = evaluator.resolve_sampling_protocol(
        replace(
            base,
            protocol="raev2_ig",
            cfg_scale=2.75,
            cfg_t_min=0.1,
            cfg_t_max=1.0,
        )
    )
    assert guided_with_cfg.cfg_scale == pytest.approx(2.75)
    assert (guided_with_cfg.cfg_t_min, guided_with_cfg.cfg_t_max) == pytest.approx(
        (0.1, 1.0)
    )
    assert guided_with_cfg.ig_scale == pytest.approx(1.78)

    guided_with_scale_override = evaluator.resolve_sampling_protocol(
        replace(
            base,
            protocol="raev2_ig",
            ig_scale=1.5,
        )
    )
    assert guided_with_scale_override.ig_scale == pytest.approx(1.5)
    assert (
        guided_with_scale_override.ig_t_min,
        guided_with_scale_override.ig_t_max,
    ) == pytest.approx((0.0, 0.9))


@pytest.mark.parametrize("protocol", ["raev2_cfg_sweep", "raev2_cfg_sweep_100"])
def test_raev2_cfg_sweep_requires_an_active_or_neutral_cfg_scale(protocol: str):
    evaluator = _import_or_skip_cv2_blocker(
        "pooldino.eval.gfid_pooled_decoder_adm"
    )
    base = evaluator.Config(generator_path="generator", pooled_decoder_path="decoder")
    with pytest.raises(ValueError, match="--cfg-scale >= 1.0"):
        evaluator.resolve_sampling_protocol(
            replace(base, protocol=protocol, cfg_scale=None)
        )
    with pytest.raises(ValueError, match="--cfg-scale >= 1.0"):
        evaluator.resolve_sampling_protocol(
            replace(base, protocol=protocol, cfg_scale=0.9)
        )


@pytest.mark.parametrize("protocol", ["raev2_self_repa", "raev2_self_repa_100"])
def test_raev2_self_repa_protocol_requires_a_guidance_scale(protocol: str):
    evaluator = _import_or_skip_cv2_blocker(
        "pooldino.eval.gfid_pooled_decoder_adm"
    )
    base = evaluator.Config(generator_path="generator", pooled_decoder_path="decoder")
    with pytest.raises(ValueError, match="--self-repa-guidance-scale"):
        evaluator.resolve_sampling_protocol(
            replace(base, protocol=protocol)
        )


@pytest.mark.parametrize(
    "protocol", ["raev2_self_repa", "raev2_self_repa_100", "raev2_ig"]
)
def test_guided_raev2_protocols_reject_subneutral_explicit_cfg(protocol: str):
    evaluator = _import_or_skip_cv2_blocker(
        "pooldino.eval.gfid_pooled_decoder_adm"
    )
    base = evaluator.Config(generator_path="generator", pooled_decoder_path="decoder")
    with pytest.raises(ValueError, match="--cfg-scale >= 1.0"):
        evaluator.resolve_sampling_protocol(
            replace(
                base,
                protocol=protocol,
                cfg_scale=0.9,
                self_repa_guidance_scale=(
                    1.0 if protocol.startswith("raev2_self_repa") else None
                ),
            )
        )


def test_named_raev2_generation_uses_arrow_validation_condition_order(monkeypatch):
    evaluator = _import_or_skip_cv2_blocker(
        "pooldino.eval.gfid_pooled_decoder_adm"
    )
    # Keep the exact 50/class population while making its sequence observably
    # different from the legacy repeat(arange(1000), 50) construction.
    arrow_labels = np.roll(
        np.repeat(np.arange(1000, dtype=np.int32), 50),
        17,
    )

    class FakeArrowSource:
        def __init__(self, data_dir, split):
            assert data_dir == "/datasets/raev2"
            assert split == "validation"

        def __len__(self):
            return len(arrow_labels)

        def labels(self):
            return arrow_labels.copy()

    monkeypatch.setattr(evaluator, "_RAEv2HfImageDataSource", FakeArrowSource)
    data_cfg = SimpleNamespace(
        backend="raev2_hf",
        data_dir="/datasets/raev2",
        val_name="validation",
    )
    exact_cfg = evaluator.Config(
        generator_path="generator",
        pooled_decoder_path="decoder",
        protocol="raev2_train",
        per_class=50,
    )
    labels, provenance = evaluator.resolve_generation_labels(
        exact_cfg,
        data_cfg=data_cfg,
        num_classes=1000,
    )
    np.testing.assert_array_equal(labels, arrow_labels)
    assert provenance == "raev2_hf_validation_arrow_order"

    custom_cfg = replace(exact_cfg, protocol="custom", per_class=2)
    labels, provenance = evaluator.resolve_generation_labels(
        custom_cfg,
        data_cfg=SimpleNamespace(backend="tfds", data_dir=None),
        num_classes=3,
    )
    np.testing.assert_array_equal(labels, [0, 0, 1, 1, 2, 2])
    assert provenance == "balanced_class_repeat"


def test_named_raev2_generation_rejects_noncanonical_arrow_population(monkeypatch):
    evaluator = _import_or_skip_cv2_blocker(
        "pooldino.eval.gfid_pooled_decoder_adm"
    )

    class FakeArrowSource:
        def __init__(self, data_dir, split):
            pass

        def __len__(self):
            return 50_000

        def labels(self):
            # Correct cardinality but the wrong per-class population.
            return np.zeros(50_000, dtype=np.int32)

    monkeypatch.setattr(evaluator, "_RAEv2HfImageDataSource", FakeArrowSource)
    cfg = evaluator.Config(
        generator_path="generator",
        pooled_decoder_path="decoder",
        protocol="raev2_ig",
        per_class=50,
    )
    with pytest.raises(ValueError, match="exactly 50 examples"):
        evaluator.resolve_generation_labels(
            cfg,
            data_cfg=SimpleNamespace(
                backend="raev2_hf",
                data_dir="/datasets/raev2",
                val_name="validation",
            ),
            num_classes=1000,
        )


def test_named_raev2_generation_supports_tfds_validation_order(monkeypatch):
    evaluator = _import_or_skip_cv2_blocker(
        "pooldino.eval.gfid_pooled_decoder_adm"
    )
    tfds_labels = np.roll(
        np.repeat(np.arange(1000, dtype=np.int32), 50),
        29,
    )

    class FakeTfdsSource:
        def __len__(self):
            return len(tfds_labels)

        def __getitem__(self, index):
            return {"label": tfds_labels[index], "image": b"unused"}

    def fake_data_source(dataset, *, split, decoders):
        assert dataset == "imagenet2012"
        assert split == "validation"
        assert "image" in decoders
        return FakeTfdsSource()

    monkeypatch.setattr(evaluator.tfds, "data_source", fake_data_source)
    cfg = evaluator.Config(
        generator_path="generator",
        pooled_decoder_path="decoder",
        protocol="raev2_train",
        per_class=50,
    )
    labels, provenance = evaluator.resolve_generation_labels(
        cfg,
        data_cfg=SimpleNamespace(
            backend="tfds",
            data_dir=None,
            dataset="imagenet2012",
            val_name="validation",
        ),
        num_classes=1000,
    )
    np.testing.assert_array_equal(labels, tfds_labels)
    assert provenance == "tfds_validation_order"


def test_named_raev2_generation_prefers_packaged_validation_order(
    tmp_path, monkeypatch
):
    evaluator = _import_or_skip_cv2_blocker(
        "pooldino.eval.gfid_pooled_decoder_adm"
    )
    labels_path = tmp_path / evaluator.PACKAGED_CONDITION_LABELS
    labels = np.roll(
        np.repeat(np.arange(1000, dtype=np.int32), 50),
        41,
    )
    np.savez_compressed(
        labels_path,
        format_version=np.asarray(1, dtype=np.int64),
        dataset=np.asarray("imagenet2012"),
        split=np.asarray("validation"),
        labels=labels,
        labels_sha256=np.asarray(hashlib.sha256(labels.tobytes()).hexdigest()),
    )

    def fail_data_source(*args, **kwargs):
        raise AssertionError("TFDS must not be accessed when labels are packaged.")

    monkeypatch.setattr(evaluator.tfds, "data_source", fail_data_source)
    cfg = evaluator.Config(
        generator_path=tmp_path / "generator",
        pooled_decoder_path=tmp_path / "decoder-run",
        protocol="raev2_train",
        per_class=50,
    )
    assert evaluator.resolve_condition_labels_path(cfg) == labels_path
    actual, provenance = evaluator.resolve_generation_labels(
        cfg,
        data_cfg=SimpleNamespace(
            backend="tfds",
            data_dir=None,
            dataset="imagenet2012",
            val_name="validation",
        ),
        num_classes=1000,
        condition_labels_path=labels_path,
    )

    np.testing.assert_array_equal(actual, labels)
    assert provenance == f"packaged_tfds_validation_order:{labels_path}"


def test_packaged_validation_order_rejects_checksum_mismatch(tmp_path):
    evaluator = _import_or_skip_cv2_blocker(
        "pooldino.eval.gfid_pooled_decoder_adm"
    )
    labels_path = tmp_path / "labels.npz"
    labels = np.repeat(np.arange(1000, dtype=np.int32), 50)
    np.savez_compressed(
        labels_path,
        format_version=np.asarray(1, dtype=np.int64),
        dataset=np.asarray("imagenet2012"),
        split=np.asarray("validation"),
        labels=labels,
        labels_sha256=np.asarray("0" * 64),
    )
    cfg = evaluator.Config(
        generator_path="generator",
        pooled_decoder_path="decoder",
        protocol="raev2_train",
        per_class=50,
    )
    with pytest.raises(ValueError, match="checksum"):
        evaluator.resolve_generation_labels(
            cfg,
            data_cfg=SimpleNamespace(
                backend="tfds",
                dataset="imagenet2012",
                val_name="validation",
            ),
            num_classes=1000,
            condition_labels_path=labels_path,
        )


def test_raev2_metric_shuffle_matches_github_fixed_permutation():
    evaluator = _import_or_skip_cv2_blocker(
        "pooldino.eval.gfid_pooled_decoder_adm"
    )
    samples = np.arange(8, dtype=np.uint8).reshape(8, 1, 1, 1)
    shuffled = evaluator.permute_raev2_metric_samples(samples)
    np.testing.assert_array_equal(
        shuffled[:, 0, 0, 0],
        np.asarray([2, 4, 3, 6, 5, 0, 1, 7], dtype=np.uint8),
    )


def test_mind_reference_resolves_env_then_source_data_dir(tmp_path, monkeypatch):
    evaluator = _import_or_skip_cv2_blocker(
        "pooldino.eval.gfid_pooled_decoder_adm"
    )
    candidate = tmp_path / "imagenet-256-val.npz"
    candidate.touch()
    monkeypatch.delenv("NANOGEN_EVALS_REF_IMAGES", raising=False)
    assert evaluator.resolve_fd_reference_images(
        ("fid", "mind6"),
        reference_images=None,
        data_dir=tmp_path,
    ) == candidate

    env_candidate = tmp_path / "env-reference.npz"
    monkeypatch.setenv("NANOGEN_EVALS_REF_IMAGES", str(env_candidate))
    assert evaluator.resolve_fd_reference_images(
        ("mind6",),
        reference_images=None,
        data_dir=tmp_path,
    ) == env_candidate


def test_exact_experiment_rejects_a_legacy_stage1_decoder():
    pooled_generator = _import_or_skip_cv2_blocker(
        "pooldino.pooled_generator"
    )
    train_module = _import_or_skip_cv2_blocker(
        "pooldino.train_generator"
    )
    restored = SimpleNamespace(
        latent_spec=pooled_generator.PooledLatentSpec((4, 8), 1024),
        grid_hw=(16, 16),
        cfg=SimpleNamespace(stage1_profile="legacy"),
    )
    with pytest.raises(ValueError, match="stage1_profile='raev2_github'"):
        train_module.get_experiment(
            "raev2ddt",
            restored,
        )


def test_source_training_steps_drop_each_epoch_tail():
    pooled_generator = _import_or_skip_cv2_blocker(
        "pooldino.pooled_generator"
    )
    train_module = _import_or_skip_cv2_blocker(
        "pooldino.train_generator"
    )
    restored = SimpleNamespace(
        latent_spec=pooled_generator.PooledLatentSpec((4, 8), 1024),
        grid_hw=(16, 16),
        cfg=SimpleNamespace(stage1_profile="raev2_github"),
    )
    source = train_module.get_experiment(
        "raev2ddt",
        restored,
    )
    assert train_module.resolve_training_steps(
        1_281_167,
        source,
        grad_acc_steps=2,
    ) == (100_080, 1_251)
    epoch_semantics = train_module.resolve_train_epoch_semantics(
        source,
        grad_acc_steps=2,
    )
    assert epoch_semantics.global_batch_size == 1024
    assert epoch_semantics.source_world_size == 8
    assert epoch_semantics.grad_accum_steps == 2

def test_exact_generator_restore_never_falls_back_when_ema_is_missing(
    tmp_path,
    monkeypatch,
):
    pooled_generator = _import_or_skip_cv2_blocker(
        "pooldino.pooled_generator"
    )
    train_module = _import_or_skip_cv2_blocker(
        "pooldino.train_generator"
    )
    latent_spec = pooled_generator.PooledLatentSpec((4, 8), 1024)
    restored = SimpleNamespace(
        latent_spec=latent_spec,
        grid_hw=(16, 16),
        cfg=SimpleNamespace(stage1_profile="raev2_github"),
    )
    cfg = train_module.get_experiment(
        "raev2ddt",
        restored,
    )

    class FakeManager:
        def latest_step(self):
            return 1

        def all_steps(self):
            return [1]

        def restore(self, step, args):
            del step, args
            return {"config": asdict(cfg)}

        def close(self):
            pass

    monkeypatch.setattr(
        pooled_generator.ocp,
        "CheckpointManager",
        lambda *args, **kwargs: FakeManager(),
    )
    monkeypatch.setattr(
        pooled_generator,
        "restore_pooled_generator_item",
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyError("model_ema")),
    )
    with pytest.raises(ValueError, match="does not contain EMA weights"):
        pooled_generator.restore_pooled_generator(
            tmp_path,
            latent_spec=latent_spec,
            mesh=None,
            mp=None,
            use_ema=True,
        )


def test_exact_generator_restore_rejects_requested_experiment_drift():
    pooled_generator = _import_or_skip_cv2_blocker(
        "pooldino.pooled_generator"
    )
    train_module = _import_or_skip_cv2_blocker(
        "pooldino.train_generator"
    )
    restored_decoder = SimpleNamespace(
        latent_spec=pooled_generator.PooledLatentSpec((4, 8), 1024),
        grid_hw=(16, 16),
        cfg=SimpleNamespace(stage1_profile="raev2_github"),
    )
    target = train_module.get_experiment(
        "raev2ddt",
        restored_decoder,
    )
    checkpoint = replace(
        target,
        pooled_decoder_identity=pooled_generator.PooledDecoderIdentity(
            checkpoint_path="/decoder",
            checkpoint_step=10,
            use_ema=True,
            stage1_profile="raev2_github",
            config_sha256="a" * 64,
        ),
        optimizer_partition_counts={
            "gmuon_tensors": 3,
            "gmuon_parameters": 30,
            "adamw_tensors": 4,
            "adamw_parameters": 40,
        },
    )
    resolved = train_module.resolve_pooled_generator_restore_config(
        target,
        checkpoint,
    )
    assert resolved is checkpoint

    extended_target = replace(
        target,
        train=replace(target.train, epochs=300),
    )
    extended = train_module.resolve_pooled_generator_restore_config(
        extended_target,
        checkpoint,
    )
    assert extended.train.epochs == 300
    assert extended.pooled_decoder_identity == checkpoint.pooled_decoder_identity
    assert extended.optimizer_partition_counts == checkpoint.optimizer_partition_counts

    with pytest.raises(ValueError, match="shorter training horizon"):
        train_module.resolve_pooled_generator_restore_config(target, extended)

    with pytest.raises(ValueError, match="optimizer and learning-rate recipe"):
        replace(
            target,
            train=replace(target.train, lr_final=1e-5),
        )


def test_pooled_artifact_identities_roundtrip_and_reject_stats_drift(tmp_path):
    pooled_generator = _import_or_skip_cv2_blocker(
        "pooldino.pooled_generator"
    )
    decoder_identity = pooled_generator.PooledDecoderIdentity(
        checkpoint_path=str((tmp_path / "decoder").resolve()),
        checkpoint_step=123,
        use_ema=True,
        stage1_profile="raev2_github",
        config_sha256="a" * 64,
    )
    restored = SimpleNamespace(identity=decoder_identity)
    stats_path = tmp_path / "pooled_latent_stats.npz"

    def write_stats(scale: float) -> None:
        np.savez(
            stats_path,
            mean=np.full((2, 3), scale, dtype=np.float32),
            var=np.full((2, 3), 2.0, dtype=np.float32),
            count=np.asarray(17, dtype=np.int64),
            shape=np.asarray((2, 3), dtype=np.int64),
            decoder_path=np.asarray(decoder_identity.checkpoint_path),
            decoder_step=np.asarray(decoder_identity.checkpoint_step, dtype=np.int64),
            decoder_use_ema=np.asarray(decoder_identity.use_ema),
            decoder_stage1_profile=np.asarray(decoder_identity.stage1_profile),
            decoder_config_sha256=np.asarray(decoder_identity.config_sha256),
        )

    write_stats(1.0)
    _, _, stats_identity = pooled_generator.load_pooled_latent_stats(
        stats_path,
        expected_shape=(2, 3),
        expected_decoder_identity=decoder_identity,
    )
    cfg = pooled_generator.PooledGeneratorConfig()
    cfg = pooled_generator.bind_pooled_generator_artifacts(
        cfg,
        restored,
        stats_identity,
    )
    pooled_generator.validate_pooled_generator_artifacts(
        cfg,
        restored,
        stats_identity,
    )
    assert cfg.pooled_decoder_identity == decoder_identity
    assert cfg.latent_stats_identity.count == 17
    assert cfg.latent_stats_identity.shape == (2, 3)
    checkpoint_blob = json.loads(json.dumps(asdict(cfg)))
    restored_cfg = from_dict(
        pooled_generator.PooledGeneratorConfig,
        checkpoint_blob,
        config=DaciteConfig(cast=[tuple], strict=False),
    )
    assert restored_cfg.pooled_decoder_identity == decoder_identity
    assert restored_cfg.latent_stats_identity == stats_identity

    # Same path/shape/count is insufficient: changing the bytes changes the
    # persisted identity and must make resume/evaluation fail.
    write_stats(3.0)
    _, _, changed_identity = pooled_generator.load_pooled_latent_stats(
        stats_path,
        expected_shape=(2, 3),
        expected_decoder_identity=decoder_identity,
    )
    with pytest.raises(ValueError, match="Latent stats do not match"):
        pooled_generator.validate_pooled_generator_artifacts(
            cfg,
            restored,
            changed_identity,
        )


def test_pooled_artifact_identities_allow_workspace_relocation(tmp_path):
    pooled_generator = _import_or_skip_cv2_blocker(
        "pooldino.pooled_generator"
    )
    artifact_tail = (
        "output",
        "pooldino",
        "pooled-decoder",
        "repeatconv2x2-dinol-vitxl-raev2official-tfds",
    )
    original_root = tmp_path / "machine-a" / "pooldino-code"
    relocated_root = tmp_path / "machine-b" / "pooldino-code"
    original_decoder_path = original_root.joinpath(*artifact_tail)
    relocated_decoder_path = relocated_root.joinpath(*artifact_tail)

    checkpoint_decoder = pooled_generator.PooledDecoderIdentity(
        checkpoint_path=str(original_decoder_path),
        checkpoint_step=40032,
        use_ema=True,
        stage1_profile="raev2_github",
        config_sha256="a" * 64,
    )
    relocated_decoder = replace(
        checkpoint_decoder,
        checkpoint_path=str(relocated_decoder_path),
    )
    checkpoint_stats = pooled_generator.PooledLatentStatsIdentity(
        stats_path=str(original_decoder_path / "pooled_latent_stats.npz"),
        sha256="b" * 64,
        count=1_281_168,
        shape=(64, 1024),
    )
    relocated_stats = replace(
        checkpoint_stats,
        stats_path=str(relocated_decoder_path / "pooled_latent_stats.npz"),
    )
    cfg = replace(
        pooled_generator.PooledGeneratorConfig(),
        pooled_decoder_identity=checkpoint_decoder,
        latent_stats_identity=checkpoint_stats,
    )

    pooled_generator.validate_pooled_generator_artifacts(
        cfg,
        SimpleNamespace(identity=relocated_decoder),
        relocated_stats,
    )

    wrong_decoder = replace(
        relocated_decoder,
        checkpoint_path=str(relocated_decoder_path.parent / "repeatconv4x4-wrong-run"),
    )
    with pytest.raises(ValueError, match="checkpoint_path"):
        pooled_generator.validate_pooled_generator_artifacts(
            cfg,
            SimpleNamespace(identity=wrong_decoder),
            relocated_stats,
        )

    wrong_stats = replace(
        relocated_stats,
        stats_path=str(
            relocated_decoder_path.parent
            / "repeatconv4x4-wrong-run"
            / "pooled_latent_stats.npz"
        ),
    )
    with pytest.raises(ValueError, match="stats_path"):
        pooled_generator.validate_pooled_generator_artifacts(
            cfg,
            SimpleNamespace(identity=relocated_decoder),
            wrong_stats,
        )


def test_source_stats_allow_decoder_workspace_relocation(tmp_path):
    pooled_generator = _import_or_skip_cv2_blocker(
        "pooldino.pooled_generator"
    )
    relative_decoder_path = (
        "output/decoders/"
        "repeatconv2x2-dinol-vitxl-raev2official-tfds"
    )
    original_decoder_path = tmp_path / "machine-a" / relative_decoder_path
    relocated_decoder_path = tmp_path / "machine-b" / relative_decoder_path
    expected_decoder_identity = pooled_generator.PooledDecoderIdentity(
        checkpoint_path=str(relocated_decoder_path),
        checkpoint_step=40032,
        use_ema=True,
        stage1_profile="raev2_github",
        config_sha256="c" * 64,
    )
    stats_path = relocated_decoder_path / "pooled_latent_stats.npz"
    stats_path.parent.mkdir(parents=True)
    np.savez(
        stats_path,
        mean=np.zeros((2, 3), dtype=np.float32),
        var=np.ones((2, 3), dtype=np.float32),
        count=np.asarray(8, dtype=np.int64),
        shape=np.asarray((2, 3), dtype=np.int64),
        decoder_path=np.asarray(str(original_decoder_path)),
        decoder_step=np.asarray(40032, dtype=np.int64),
        decoder_use_ema=np.asarray(True),
        decoder_stage1_profile=np.asarray("raev2_github"),
        decoder_config_sha256=np.asarray("c" * 64),
    )

    pooled_generator.load_pooled_latent_stats(
        stats_path,
        expected_shape=(2, 3),
        expected_decoder_identity=expected_decoder_identity,
    )


def test_source_stats_require_decoder_and_shape_metadata(tmp_path):
    pooled_generator = _import_or_skip_cv2_blocker(
        "pooldino.pooled_generator"
    )
    stats_path = tmp_path / "old_stats.npz"
    np.savez(
        stats_path,
        mean=np.zeros((2, 3), dtype=np.float32),
        var=np.ones((2, 3), dtype=np.float32),
        count=np.asarray(8, dtype=np.int64),
    )
    with pytest.raises(ValueError, match="shape metadata"):
        pooled_generator.load_pooled_latent_stats(
            stats_path,
            expected_shape=(2, 3),
            require_source_metadata=True,
        )


def test_source_stats_validate_fixed_world_size_and_effective_count(tmp_path):
    pooled_generator = _import_or_skip_cv2_blocker(
        "pooldino.pooled_generator"
    )
    decoder_identity = pooled_generator.PooledDecoderIdentity(
        checkpoint_path=str((tmp_path / "decoder").resolve()),
        checkpoint_step=9,
        use_ema=True,
        stage1_profile="raev2_github",
        config_sha256="c" * 64,
    )
    stats_path = tmp_path / "source_stats.npz"

    def write(count: int) -> None:
        np.savez(
            stats_path,
            mean=np.zeros((2, 3), dtype=np.float32),
            var=np.ones((2, 3), dtype=np.float32),
            count=np.asarray(count, dtype=np.int64),
            shape=np.asarray((2, 3), dtype=np.int64),
            format_version=np.asarray(1, dtype=np.int64),
            source_sample_count=np.asarray(10, dtype=np.int64),
            sampler_world_size=np.asarray(8, dtype=np.int64),
            decoder_path=np.asarray(decoder_identity.checkpoint_path),
            decoder_step=np.asarray(decoder_identity.checkpoint_step, dtype=np.int64),
            decoder_use_ema=np.asarray(True),
            decoder_stage1_profile=np.asarray("raev2_github"),
            decoder_config_sha256=np.asarray(decoder_identity.config_sha256),
        )

    write(16)
    _, _, identity = pooled_generator.load_pooled_latent_stats(
        stats_path,
        expected_shape=(2, 3),
        expected_decoder_identity=decoder_identity,
        require_source_metadata=True,
    )
    assert identity.count == 16

    write(15)
    with pytest.raises(ValueError, match="padded source population 16"):
        pooled_generator.load_pooled_latent_stats(
            stats_path,
            expected_shape=(2, 3),
            expected_decoder_identity=decoder_identity,
            require_source_metadata=True,
        )


def test_host_float64_stats_and_exact_final_batch_trimming():
    stats_module = _import_or_skip_cv2_blocker(
        "pooldino.compute_stats"
    )
    values = np.asarray(
        [1e12 + 1.0, 1e12 + 2.0, 1e12 + 4.0, 1e12 + 8.0, 1e12 + 16.0],
        dtype=np.float64,
    ).reshape(5, 1, 1)
    count = 0
    mean = np.zeros((1, 1), dtype=np.float64)
    m2 = np.zeros_like(mean)
    count, mean, m2 = stats_module.update_host_running_stats(
        count,
        mean,
        m2,
        values[:2],
    )
    final_batch = stats_module.trim_host_stats_batch(values[2:], remaining=1)
    count, mean, m2 = stats_module.update_host_running_stats(
        count,
        mean,
        m2,
        final_batch,
    )
    assert count == 3
    np.testing.assert_allclose(mean, np.mean(values[:3], axis=0), rtol=0, atol=0)
    np.testing.assert_allclose(
        m2 / count,
        np.var(values[:3], axis=0),
        rtol=0,
        atol=2e-9,
    )

    # A partial final batch is padded for device sharding. The repeated device
    # entries must be removed before applying the max-samples limit.
    device_padded = np.arange(8, dtype=np.float64).reshape(8, 1, 1)
    valid = stats_module.trim_host_stats_batch(
        device_padded,
        remaining=10,
        valid_size=7,
    )
    np.testing.assert_array_equal(valid[:, 0, 0], np.arange(7))


def test_source_stats_pad_to_fixed_eight_rank_population_with_prefix_duplicates():
    stats_module = _import_or_skip_cv2_blocker(
        "pooldino.compute_stats"
    )
    population = np.arange(10, dtype=np.float64)
    padded = stats_module.pad_raev2_stats_population(population, world_size=8)
    assert len(padded) == 16
    np.testing.assert_array_equal(padded[:10], np.arange(10))
    np.testing.assert_array_equal(padded[10:], np.arange(6))
    assert stats_module.raev2_stats_padding_count(10, world_size=8) == 6


def test_pooled_stats_use_source_transform_and_keep_legacy_adm():
    stats_module = _import_or_skip_cv2_blocker(
        "pooldino.compute_stats"
    )
    decoder_module = _import_or_skip_cv2_blocker(
        "pooldino.augmentations.decoder"
    )
    data_cfg = DataConfig()
    aug_cfg = decoder_module.RAEDecoderAugConfig(crop_size=(256, 256))
    official = SimpleNamespace(
        cfg=SimpleNamespace(
            stage1_profile="raev2_github",
            backbone_resolution=256,
        ),
        aug_cfg=aug_cfg,
    )
    legacy = SimpleNamespace(
        cfg=SimpleNamespace(stage1_profile="legacy", backbone_resolution=224),
        aug_cfg=aug_cfg,
    )
    assert isinstance(
        stats_module.make_stats_augmentation(official, data_cfg),
        decoder_module.RAEv2GithubDecoderAugmentations,
    )
    assert isinstance(
        stats_module.make_stats_augmentation(legacy, data_cfg),
        stats_module.ADMCenterCropAugmentations,
    )


def test_source_stage2_encoder_policy_is_fp32_while_legacy_follows_ddt_policy():
    pooled_generator = _import_or_skip_cv2_blocker(
        "pooldino.pooled_generator"
    )
    ddt_policy = jmp.Policy(
        param_dtype=jnp.float32,
        compute_dtype=jnp.bfloat16,
        output_dtype=jnp.float32,
    )
    official_cfg = SimpleNamespace(stage1_profile="raev2_github")
    official_policy = pooled_generator.encoder_policy_for_restore(
        ddt_policy,
        official_cfg,
        source_encoder_fp32=True,
    )
    assert official_policy.compute_dtype == jnp.float32
    legacy_policy = pooled_generator.encoder_policy_for_restore(
        ddt_policy,
        SimpleNamespace(stage1_profile="legacy"),
        source_encoder_fp32=True,
    )
    assert legacy_policy.compute_dtype == jnp.bfloat16


def test_eval_precision_auto_matches_released_bf16_protocol():
    evaluator = _import_or_skip_cv2_blocker(
        "pooldino.eval.gfid_pooled_decoder_adm"
    )
    assert evaluator.resolve_eval_precision("auto") == "bf16"
    assert evaluator.resolve_eval_precision("fp32") == "fp32"


def test_exact_generator_keeps_restored_raev2_data_backend():
    pooled_generator = _import_or_skip_cv2_blocker(
        "pooldino.pooled_generator"
    )
    train_module = _import_or_skip_cv2_blocker(
        "pooldino.train_generator"
    )
    official_data = train_module.DataConfig(
        backend="raev2_hf",
        data_dir="data/imagenet-256",
        train_name="train",
        val_name="validation",
        num_workers=8,
    )
    restored = SimpleNamespace(
        latent_spec=pooled_generator.PooledLatentSpec((4, 8), 1024),
        grid_hw=(16, 16),
        data_cfg=official_data,
        cfg=SimpleNamespace(stage1_profile="raev2_github"),
    )
    cfg = train_module.get_experiment(
        "raev2ddt",
        restored,
    )
    resolved = train_module._resolve_generator_data_config(
        restored,
        dataset=None,
        num_workers=12,
    )
    assert resolved.backend == "raev2_hf"
    assert resolved.data_dir == "data/imagenet-256"
    assert resolved.num_workers == 12

    overridden = train_module._resolve_generator_data_config(
        restored,
        dataset="imagenet",
        num_workers=None,
    )
    assert overridden.backend == "tfds"
    assert overridden.dataset == "imagenet2012"


def test_pooled_generator_writes_success_marker(tmp_path):
    train_module = _import_or_skip_cv2_blocker(
        "pooldino.train_generator"
    )
    checkpoint_dir = tmp_path / "generator"

    train_module.write_success_marker(checkpoint_dir)

    assert (checkpoint_dir / "_SUCCESS").is_file()

    continuation_marker = tmp_path / "markers" / "_SUCCESS_E300"
    train_module.write_success_marker(checkpoint_dir, continuation_marker)
    assert continuation_marker.is_file()


def test_exact_generator_keeps_restored_raev2_tfds_backend():
    pooled_generator = _import_or_skip_cv2_blocker(
        "pooldino.pooled_generator"
    )
    train_module = _import_or_skip_cv2_blocker(
        "pooldino.train_generator"
    )
    tfds_data = train_module.DataConfig(
        backend="tfds",
        data_dir=None,
        dataset="imagenet2012",
        train_name="train",
        val_name="validation",
        num_workers=4,
    )
    restored = SimpleNamespace(
        latent_spec=pooled_generator.PooledLatentSpec((4, 8), 1024),
        grid_hw=(16, 16),
        data_cfg=tfds_data,
        cfg=SimpleNamespace(stage1_profile="raev2_github"),
    )
    cfg = train_module.get_experiment(
        "raev2ddt",
        restored,
    )
    resolved = train_module._resolve_generator_data_config(
        restored,
        dataset=None,
        num_workers=12,
    )
    assert resolved.backend == "tfds"
    assert resolved.data_dir is None
    assert resolved.dataset == "imagenet2012"
    assert resolved.num_workers == 12


def test_raev2_uint8_conversion_truncates_instead_of_rounding():
    evaluator = _import_or_skip_cv2_blocker(
        "pooldino.eval.gfid_pooled_decoder_adm"
    )
    images = np.full((1, 1, 1, 3), 1.6 / 255.0, dtype=np.float32)
    np.testing.assert_array_equal(
        evaluator._uint8(images, truncate=True),
        np.full((1, 1, 1, 3), 1, dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        evaluator._uint8(images, truncate=False),
        np.full((1, 1, 1, 3), 2, dtype=np.uint8),
    )


def test_raev2_fd_backend_delegates_exact_argument_contract(monkeypatch, tmp_path):
    evaluator = _import_or_skip_cv2_blocker(
        "pooldino.eval.gfid_pooled_decoder_adm"
    )
    captured = {}

    def compute_metrics(**kwargs):
        captured.update(kwargs)
        return {"fid": np.float64(1.25)}

    monkeypatch.setitem(
        sys.modules,
        "fd_evaluator",
        SimpleNamespace(compute_metrics=compute_metrics),
    )
    images = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    reference = tmp_path / "reference.npz"
    results = evaluator.compute_raev2_fd_metrics(
        images,
        metrics=("fid",),
        reference=reference,
        device="cpu",
        batch_size=16,
    )
    assert results == {"fid": 1.25}
    assert captured["images"] is images
    assert captured["metrics"] == ["fid"]
    assert captured["fid_reference"] == str(reference)
    assert captured["reference_feature_cache_key"] == "imagenet256_val"
    assert captured["device"] == "cpu"
    assert captured["batch_size"] == 16
