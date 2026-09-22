from pathlib import Path

import jax.numpy as jnp
import numpy as np

from pooldino.eval.generate_pooled_random_projection import (
    Config as RandomProjectionConfig,
)
from pooldino.eval.generate_pooled_random_projection import generate_projection
from pooldino.eval.pooled_baseline_semantics import (
    average_projection,
    load_projection_artifact,
    projected_local_block_mean,
)


def test_average_projection_then_token_mean_equals_global_patch_mean():
    patches = jnp.arange(2 * 16 * 3, dtype=jnp.float32).reshape(2, 16, 3)
    expected = np.asarray(jnp.mean(patches, axis=1))
    for pool_hw in ((1, 1), (2, 2), (2, 4), (4, 2), (4, 4)):
        artifact = average_projection(pool_hw, channels=3)
        actual = projected_local_block_mean(
            patches,
            jnp.asarray(artifact.operator),
            jnp.asarray(artifact.center),
            grid_hw=(4, 4),
            pool_hw=pool_hw,
        )
        np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-6, atol=1e-6)


def test_projected_block_order_matches_row_major_local_blocks():
    patches = jnp.arange(16, dtype=jnp.float32).reshape(1, 16, 1)
    operator = jnp.asarray([[1.0, 10.0, 100.0, 1000.0]])
    actual = projected_local_block_mean(
        patches,
        operator,
        jnp.zeros((4,)),
        grid_hw=(4, 4),
        pool_hw=(2, 2),
    )
    blocks = np.asarray(
        [
            [0, 1, 4, 5],
            [2, 3, 6, 7],
            [8, 9, 12, 13],
            [10, 11, 14, 15],
        ],
        dtype=np.float32,
    )
    expected = np.mean(blocks @ np.asarray(operator).T, axis=0, keepdims=True)
    np.testing.assert_allclose(np.asarray(actual), expected)


def test_pca_artifact_loads_components_and_training_center(tmp_path: Path):
    components = np.eye(4, dtype=np.float32)
    center = np.arange(4, dtype=np.float32)
    path = tmp_path / "pca.npz"
    np.savez_compressed(path, pca_components=components, block_mean=center)
    artifact = load_projection_artifact(
        "pca",
        pool_hw=(1, 1),
        channels=4,
        path=path,
    )
    np.testing.assert_array_equal(artifact.operator, components)
    np.testing.assert_array_equal(artifact.center, center)
    assert artifact.sha256 is not None
    assert artifact.metadata["centered"] is True


def test_random_projection_is_persisted_deterministically_and_orthonormal(tmp_path: Path):
    center_path = tmp_path / "center.npz"
    np.savez_compressed(center_path, block_mean=np.arange(16, dtype=np.float32))
    first_path = tmp_path / "random-first.npz"
    second_path = tmp_path / "random-second.npz"
    for output_path in (first_path, second_path):
        generate_projection(
            RandomProjectionConfig(
                output_path=output_path,
                center_artifact=center_path,
                pool_hw=(2, 2),
                channels=4,
                seed=42,
            )
        )
    with np.load(first_path) as first, np.load(second_path) as second:
        np.testing.assert_array_equal(first["operator"], second["operator"])
        operator = first["operator"]
        np.testing.assert_allclose(operator @ operator.T, np.eye(4), atol=1e-5)
    artifact = load_projection_artifact(
        "random",
        pool_hw=(2, 2),
        channels=4,
        path=first_path,
    )
    assert artifact.metadata["seed"] == 42
    assert artifact.path == str(first_path.resolve())
