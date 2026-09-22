"""Focused tests for pooled-latent REPA and one-pass internal guidance."""

from __future__ import annotations

from dataclasses import asdict
import importlib
from types import SimpleNamespace

from dacite import Config as DaciteConfig, from_dict
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
import pytest

from pooldino.models.raev2_ddt import RAEv2DDT, RAEv2DDTConfig
from pooldino.models.repa import DenseRepaProjector


MP = jmp.Policy(
    param_dtype=jnp.float32,
    compute_dtype=jnp.float32,
    output_dtype=jnp.float32,
)


@pytest.fixture(autouse=True)
def single_device_model_mesh():
    """Provide the FSDP axis required by partitioned NNX initializers."""
    devices = np.asarray(jax.devices()[:1]).reshape(1, 1)
    mesh = jax.sharding.Mesh(devices, ("data", "model"))
    with jax.set_mesh(mesh):
        yield


def _import_or_skip_cv2_blocker(module_name: str):
    """Import a module while keeping unrelated REPA tests runnable without libGL."""
    try:
        return importlib.import_module(module_name)
    except ImportError as error:
        if "libGL.so.1" in str(error):
            pytest.skip(f"{module_name} is blocked by the local cv2 dependency: {error}")
        raise


@pytest.fixture(scope="module")
def compression_dit_module():
    return _import_or_skip_cv2_blocker("pooldino.guidance")


@pytest.mark.parametrize("pool_window", [(4, 4), (2, 4), (4, 2)])
def test_dense_repa_projector_preserves_anisotropic_row_major_order(pool_window):
    """Every coarse token must expand into its corresponding dense spatial cell."""
    pool_h, pool_w = pool_window
    input_h, input_w = 16 // pool_h, 16 // pool_w
    projector = DenseRepaProjector(
        1,
        1,
        input_grid=(input_h, input_w),
        target_grid=(16, 16),
        mp=MP,
        rngs=nnx.Rngs(0),
    )

    # Linear output slot k represents local row-major offset
    # (k // pool_w, k % pool_w). A distinctive coarse-token value makes a
    # transpose or plain-reshape ordering bug immediately visible.
    projector.linear.kernel[...] = jnp.ones_like(projector.linear.kernel[...])
    projector.linear.bias[...] = jnp.arange(
        pool_h * pool_w,
        dtype=jnp.float32,
    )
    coarse = (
        100.0
        * jnp.arange(input_h * input_w, dtype=jnp.float32).reshape(1, -1, 1)
    )

    actual = np.asarray(projector(coarse)).reshape(16, 16)
    expected = np.empty((16, 16), dtype=np.float32)
    for coarse_row in range(input_h):
        for coarse_col in range(input_w):
            coarse_value = 100.0 * (coarse_row * input_w + coarse_col)
            for local_row in range(pool_h):
                for local_col in range(pool_w):
                    expected[
                        coarse_row * pool_h + local_row,
                        coarse_col * pool_w + local_col,
                    ] = coarse_value + local_row * pool_w + local_col

    np.testing.assert_array_equal(actual, expected)


def test_guidance_neutral_points_and_scale_conventions_are_equivalent(
    compression_dit_module,
):
    full = jnp.asarray([[1.0, -2.0], [3.0, 4.0]], dtype=jnp.float32)
    base = jnp.asarray([[0.25, -1.5], [2.0, 6.0]], dtype=jnp.float32)

    legacy_neutral = compression_dit_module.apply_repa_guidance(full, base, 0.0)
    internal_neutral = compression_dit_module.apply_internal_guidance(full, base, 1.0)
    np.testing.assert_array_equal(legacy_neutral, full)
    np.testing.assert_array_equal(internal_neutral, full)

    # base + (1 + w) * (full - base) == full + w * (full - base)
    paper_weight = 0.78
    legacy = compression_dit_module.apply_repa_guidance(full, base, paper_weight)
    internal = compression_dit_module.apply_internal_guidance(
        full,
        base,
        1.0 + paper_weight,
    )
    np.testing.assert_allclose(internal, legacy, rtol=0.0, atol=1e-6)


def test_pooled_generator_config_defaults_and_head_properties():
    # Config lives in the lightweight restore/model module, not the cv2-heavy
    # train_pooled_generator entry point.
    pooled_generator = _import_or_skip_cv2_blocker(
        "pooldino.pooled_generator"
    )
    config_type = pooled_generator.PooledGeneratorConfig

    default = config_type()
    assert default.raev2_ddt.base_model_depth == 8
    assert default.base_model_depth == 8
    assert default.base_model_coeff == 1.0
    assert default.xpred_denom_eps == pytest.approx(0.05)
    assert default.self_repa
    assert default.self_repa_layer == 8
    assert default.self_repa_coeff == pytest.approx(0.5)
    assert default.self_repa_cosine_weight == 0.0
    assert default.self_repa_target_grid == (16, 16)
    assert default.train.epochs == 80
    assert default.train.optimizer == "gmuon"

    restored = from_dict(
        config_type,
        asdict(default),
        config=DaciteConfig(cast=[tuple], strict=False),
    )
    assert restored == default


def test_pooled_generator_experiment_exposes_raev2_defaults_and_overrides():
    pooled_generator = _import_or_skip_cv2_blocker(
        "pooldino.pooled_generator"
    )
    train_module = _import_or_skip_cv2_blocker(
        "pooldino.train_generator"
    )
    restored = SimpleNamespace(
        latent_spec=pooled_generator.PooledLatentSpec((4, 4), 1024),
        grid_hw=(16, 16),
        cfg=SimpleNamespace(stage1_profile="raev2_github"),
    )

    default = train_module.get_experiment(
        "raev2ddt",
        restored,
    )
    assert default.base_model_depth == 8
    assert default.base_model_coeff == 1.0
    assert default.xpred_denom_eps == 0.05
    assert default.self_repa
    assert default.self_repa_layer == 8
    assert default.self_repa_coeff == 0.5
    assert default.self_repa_target_grid == (16, 16)

    override = train_module.get_experiment(
        "raev2ddt-srepa0.25-srl7-srepacos0.1",
        restored,
    )
    assert override.self_repa_layer == 7
    assert override.self_repa_coeff == 0.25
    assert override.self_repa_cosine_weight == 0.1

    with pytest.raises(ValueError, match="Could not parse"):
        train_module.get_experiment("ditxl-base", restored)


def test_dense_target_is_prepool_field_plus_global_mean_and_stop_gradient():
    pooled_generator = _import_or_skip_cv2_blocker(
        "pooldino.pooled_generator"
    )

    first = jnp.asarray(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]
    )
    last = jnp.asarray(
        [[[2.0, 0.0], [4.0, 2.0], [6.0, 4.0], [8.0, 6.0]]]
    )

    class FakeDino:
        def __call__(self, images, *, layers):
            del layers
            scale = jnp.mean(images, axis=(1, 2, 3), keepdims=True).reshape(-1, 1, 1)
            selected = [scale * first, scale * last]
            return selected[-1], selected

    class MeanTokenizer:
        def __call__(self, patches, *, grid_hw):
            assert grid_hw == (2, 2)
            return jnp.mean(patches, axis=1, keepdims=True)

    def encode(images):
        return pooled_generator.encode_pooled_decoder_targets_impl(
            images,
            FakeDino(),
            MeanTokenizer(),
            backbone_resolution=2,
            num_prefix_tokens=0,
            layer_indices=(0, 1),
            aggregation="mean",
            normalize_each=False,
            add_final_mean=True,
            post_pool_norm=True,
            representation_eps=1e-5,
            grid_hw=(2, 2),
            pool_hw=(2, 2),
        )

    images = jnp.ones((1, 2, 2, 3), dtype=jnp.float32)
    _, dense_target = encode(images)
    local_field = (first + last) / 2.0
    expected = local_field + jnp.mean(last, axis=1, keepdims=True)
    np.testing.assert_allclose(dense_target, expected)

    dense_gradient = jax.grad(lambda value: jnp.sum(encode(value)[1]))(images)
    np.testing.assert_array_equal(dense_gradient, jnp.zeros_like(images))


def test_tiny_raev2_training_step_updates_base_head_and_has_finite_self_repa_loss():
    train_module = _import_or_skip_cv2_blocker(
        "pooldino.train_generator"
    )

    class FakeDino(nnx.Module):
        def __call__(self, images, *, layers):
            del layers
            scalar = jnp.mean(images, axis=-1).reshape(images.shape[0], 4, 1)
            patches = jnp.concatenate((scalar, 2.0 * scalar), axis=-1)
            return patches, (patches,)

    class MeanTokenizer(nnx.Module):
        def __call__(self, patches, *, grid_hw):
            assert grid_hw == (2, 2)
            return jnp.mean(patches, axis=1, keepdims=True)

    ddt_cfg = RAEv2DDTConfig(
        input_grid=(1, 1),
        in_channels=2,
        encoder_dim=16,
        decoder_dim=16,
        encoder_depth=2,
        decoder_depth=1,
        encoder_heads=4,
        decoder_heads=4,
        mlp_ratio=2.0,
        num_classes=10,
        num_time_tokens=4,
        num_class_tokens=8,
        base_model_depth=1,
    )

    def make_model(seed):
        return RAEv2DDT(
            ddt_cfg,
            MP,
            self_repa_layer=1,
            self_repa_target_grid=(2, 2),
            rngs=nnx.Rngs(seed),
        )

    model = make_model(4)
    model_ema = make_model(5)
    nnx.update(
        model_ema,
        jax.tree.map(lambda value: jnp.copy(value), nnx.state(model)),
    )
    optim = nnx.Optimizer(model, optax.sgd(1e-3), wrt=nnx.Param)
    initial_base_kernel = np.asarray(model.base_final_layer.linear.kernel[...]).copy()

    loss, metrics = train_module.train_step(
        model,
        model_ema,
        FakeDino(),
        MeanTokenizer(),
        optim,
        jnp.ones((2, 2, 2, 3), dtype=jnp.float32),
        jnp.asarray([1, 2], dtype=jnp.int32),
        jax.random.PRNGKey(0),
        0.99,
        should_update_ema=True,
        kappa=None,
        xpred_denom_eps=0.05,
        backbone_resolution=2,
        num_prefix_tokens=0,
        layer_indices=(0,),
        aggregation="mean",
        normalize_each=False,
        add_final_mean=False,
        post_pool_norm=False,
        representation_eps=1e-5,
        grid_hw=(2, 2),
        pool_hw=(2, 2),
        time_mu=0.0,
        time_sigma=1.0,
        base_model_coeff=1.0,
        use_self_repa=True,
        self_repa_coeff=0.5,
        self_repa_cosine_weight=0.0,
        conditioning_dropout_prob=0.1,
        num_classes=10,
    )

    assert np.isfinite(float(loss))
    assert np.isfinite(float(metrics["base_model_loss"]))
    assert np.isfinite(float(metrics["self_repa_loss"]))
    assert not np.array_equal(
        np.asarray(model.base_final_layer.linear.kernel[...]),
        initial_base_kernel,
    )
