import jax.numpy as jnp
import numpy as np
import pytest

from pooldino.train_decoder import (
    get_experiment,
    pool_patch_tokens,
    pooled_grid_shape,
    resolve_representation_layers,
)


def test_rectangular_average_pooling_is_row_major():
    patches = jnp.arange(16, dtype=jnp.float32).reshape(1, 16, 1)

    pooled_2x2 = np.asarray(
        pool_patch_tokens(patches, grid_hw=(4, 4), pool_hw=(2, 2))
    ).reshape(2, 2)
    np.testing.assert_allclose(
        pooled_2x2,
        np.asarray([[2.5, 4.5], [10.5, 12.5]], dtype=np.float32),
    )

    pooled_2x4 = np.asarray(
        pool_patch_tokens(patches, grid_hw=(4, 4), pool_hw=(2, 4))
    ).reshape(2, 1)
    np.testing.assert_allclose(
        pooled_2x4,
        np.asarray([[3.5], [11.5]], dtype=np.float32),
    )

    pooled_4x2 = np.asarray(
        pool_patch_tokens(patches, grid_hw=(4, 4), pool_hw=(4, 2))
    ).reshape(1, 2)
    np.testing.assert_allclose(
        pooled_4x2,
        np.asarray([[6.5, 8.5]], dtype=np.float32),
    )


def test_pooling_shape_validation():
    assert pooled_grid_shape((16, 16), (2, 2)) == (8, 8)
    assert pooled_grid_shape((16, 16), (2, 4)) == (8, 4)
    assert pooled_grid_shape((16, 16), (4, 2)) == (4, 8)
    assert pooled_grid_shape((16, 16), (4, 4)) == (4, 4)

    with pytest.raises(ValueError):
        pooled_grid_shape((16, 16), (3, 2))
    with pytest.raises(ValueError):
        pool_patch_tokens(
            jnp.zeros((1, 15, 2)),
            grid_hw=(4, 4),
            pool_hw=(2, 2),
        )


def test_final_layer_uses_normal_backbone_output():
    cfg = get_experiment("pool2x4-dinol-vitb")
    assert resolve_representation_layers(cfg.representation, 24) == ()


def test_dinov2_large_raev2_style_mls_selection():
    cfg = get_experiment("pool2x4-dinol-vitb-mls7-stride2-gmean")
    assert resolve_representation_layers(cfg.representation, 24) == (
        11,
        13,
        15,
        17,
        19,
        21,
        23,
    )
    assert cfg.representation.add_final_mean
    assert cfg.noise_tau == 0.8


def test_experiment_spec_covers_all_requested_pool_windows():
    expected = {
        "pool2x2-dinol-vitb": (2, 2),
        "pool4x4-dinol-vitb": (4, 4),
        "pool2x4-dinol-vitb": (2, 4),
        "pool4x2-dinol-vitb": (4, 2),
    }
    for name, pool_window in expected.items():
        cfg = get_experiment(name)
        assert cfg.pool_window == pool_window
        assert cfg.dino_name.endswith("large")


def test_noising_and_layer_aggregation_options_parse():
    cfg = get_experiment(
        "pool2x2-dinob-vits-mls6-stride2-sum-raw-noise0.2-e20-fp32"
    )
    assert cfg.pool_window == (2, 2)
    assert cfg.dino_name.endswith("base")
    assert cfg.representation.last_k == 6
    assert cfg.representation.layer_stride == 2
    assert cfg.representation.aggregation == "sum"
    assert not cfg.representation.normalize_each
    assert cfg.noise_tau == 0.2
    assert cfg.train.epochs == 20
    assert cfg.compute_dtype == "float32"
